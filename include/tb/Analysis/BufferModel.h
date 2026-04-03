#ifndef MINI_TRITON_TB_ANALYSIS_BUFFERMODEL_H
#define MINI_TRITON_TB_ANALYSIS_BUFFERMODEL_H

#include "tb/Analysis/AccumulatorPlan.h"
#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/EpiloguePlan.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/MatmulRewritePlan.h"
#include "tb/IR/TBTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::tb {

enum class MemorySpace {
  Global,
  Shared,
  Registers,
};

enum class BufferRole {
  OperandA,
  OperandB,
  Output,
  Accumulator,
  EpilogueRelay,
  Descriptor,
  Generic,
};

struct BufferBacking {
  int64_t id = -1;
  BufferRole role = BufferRole::Generic;
  mlir::Type descType;
  int64_t depth = 1;
  int64_t aliasGroup = -1;
  bool stageIndexed = false;
  std::string debugName;
};

enum class ViewKind {
  FullBuffer,
  StageSlice,
  TileSlice,
  Fragment,
  Pack,
};

struct BufferView {
  int64_t id = -1;
  int64_t backing = -1;
  ViewKind kind = ViewKind::FullBuffer;
  int64_t stage = -1;
  int64_t bufferIndex = -1;
  int encoding = -1;
  llvm::SmallVector<int64_t, 4> offsets;
  llvm::SmallVector<int64_t, 4> shape;
  std::string debugName;
};

enum class BufferValueKind {
  GlobalTile,
  SharedTile,
  DotOperandFragment,
  AccumulatorFragment,
  EpilogueFragment,
};

struct ValueState {
  int64_t id = -1;
  BufferValueKind kind = BufferValueKind::SharedTile;
  int64_t definingOp = -1;
  llvm::SmallVector<int64_t, 8> users;
  int64_t ownerView = -1;
  int64_t loopDistance = 0;
  int64_t bundle = -1;
  int64_t firstUseOrdinal = -1;
  int64_t lastUseOrdinal = -1;
  std::string debugName;
};

enum class BufferOpKind {
  LoadA,
  LoadB,
  LocalLoadA,
  LocalLoadB,
  Mma,
  AccumulatorInit,
  AccumulatorStore,
};

// 中文标记：这里表达的是 pipeline/schedule 使用的语义类，
// 不是更低层的具体 materialization opcode。
enum class PipelineOpSemanticClass {
  AsyncProducer,
  SharedConsumer,
  TensorCoreCompute,
  EpilogueInit,
  EpilogueStore,
};

struct PipelineOp {
  struct IterationCoord {
    std::string axis;
    int64_t value = -1;
  };

  int64_t id = -1;
  BufferOpKind kind = BufferOpKind::Mma;
  llvm::SmallVector<int64_t, 8> inputs;
  llvm::SmallVector<int64_t, 8> outputs;
  llvm::SmallVector<IterationCoord, 8> iterationCoords;
};

struct BufferModel {
  llvm::SmallVector<BufferBacking, 8> backings;
  llvm::SmallVector<BufferView, 32> views;
  llvm::SmallVector<ValueState, 64> values;
  llvm::SmallVector<PipelineOp, 64> ops;
};

FailureOr<BufferModel> deriveBufferModel(const KernelConfig &config,
                                         const EncodingPlan &encodings,
                                         const AccumulatorPlan &accumulator,
                                         const EpiloguePlan &epilogue,
                                         const MatmulRewritePlan &rewrite,
                                         Operation *op);
DictionaryAttr buildBufferModelAttr(Builder &builder, const BufferModel &model);
FailureOr<BufferModel> parseBufferModelAttr(Operation *op);

StringRef stringifyBufferOpKind(BufferOpKind kind);
StringRef stringifyPipelineOpSemanticClass(PipelineOpSemanticClass semanticClass);
PipelineOpSemanticClass getPipelineOpSemanticClass(BufferOpKind kind);
bool isMainloopPipelineSemanticClass(PipelineOpSemanticClass semanticClass);

FailureOr<MemDescType> getMemDescType(Type type, Operation *op, StringRef role);
FailureOr<MemorySpace> getMemDescMemorySpace(Type type, Operation *op,
                                             StringRef role);
FailureOr<const BufferBacking *>
findUniqueBacking(const BufferModel &model, BufferRole role,
                  MemorySpace memorySpace, Operation *op,
                  StringRef contractName);
FailureOr<const BufferView *>
findUniqueFullBufferView(const BufferModel &model, BufferRole role,
                         MemorySpace memorySpace, Operation *op,
                         StringRef contractName);
Type getTypeForScalarKind(MLIRContext *context, ScalarKind kind);
int64_t getByteWidthForScalarKind(ScalarKind kind);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_BUFFERMODEL_H
