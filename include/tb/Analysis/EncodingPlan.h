#ifndef MINI_TRITON_TB_ANALYSIS_ENCODINGPLAN_H
#define MINI_TRITON_TB_ANALYSIS_ENCODINGPLAN_H

#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/MatmulSemantics.h"
#include "tb/Analysis/ProgramMappingPlan.h"
#include "tb/Analysis/TargetInfo.h"
#include "tb/IR/TBAttrs.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tb {

struct FragmentEncodingSpec {
  std::string role;
  llvm::SmallVector<int64_t, 4> instructionShape;
  llvm::SmallVector<int64_t, 4> repeatShape;
  llvm::SmallVector<int64_t, 4> logicalShape;
  llvm::SmallVector<int64_t, 4> registerShape;
  llvm::SmallVector<int64_t, 4> registerOrder;
  // 中文标记：operand fragment 的 lane->ldmatrix 访问也必须是合同真相，
  // 不能留给 lowering 再按指令形状临时重建。
  int64_t laneRowModulus = 0;
  int64_t laneColDivisor = 0;
  int64_t laneColStride = 0;
  int64_t valuesPerLane = 0;
  int64_t ldmatrixTileCount = 0;
  bool ldmatrixTranspose = false;
};

struct EncodingEntry {
  std::string name;
  mlir::Attribute encodingAttr;
};

struct SharedEncodingSpec {
  llvm::SmallVector<int64_t, 4> logicalShape;
  llvm::SmallVector<int64_t, 4> allocShape;
};

struct EncodingPlan {
  std::string contractModel;
  SharedEncodingSpec aSharedSpec;
  SharedEncodingSpec bSharedSpec;
  FragmentEncodingSpec fragmentA;
  FragmentEncodingSpec fragmentB;
  FragmentEncodingSpec fragmentAcc;
  llvm::SmallVector<EncodingEntry, 16> encodings;
  int aGlobal = -1;
  int bGlobal = -1;
  int aShared = -1;
  int bShared = -1;
  int aDot = -1;
  int bDot = -1;
  int acc = -1;
  int cStore = -1;
};

FailureOr<EncodingPlan> deriveEncodingPlan(const KernelConfig &config,
                                           const TargetInfo &target,
                                           const MatmulSemantics &semantics,
                                           const ProgramMappingPlan &programMapping,
                                           Operation *op);
DictionaryAttr buildEncodingPlanAttr(Builder &builder,
                                     const EncodingPlan &plan);
FailureOr<EncodingPlan> parseEncodingPlanAttr(Operation *op);

FailureOr<Attribute> getEncodingAttr(const EncodingPlan &plan, int encodingId,
                                     Operation *op, StringRef role);
FailureOr<BlockedEncodingAttr>
getBlockedEncodingAttr(const EncodingPlan &plan, int encodingId, Operation *op,
                       StringRef role);
FailureOr<SharedEncodingAttr>
getSharedEncodingAttr(const EncodingPlan &plan, int encodingId, Operation *op,
                      StringRef role);
FailureOr<DotOperandEncodingAttr>
getDotOperandEncodingAttr(const EncodingPlan &plan, int encodingId,
                          Operation *op, StringRef role);
FailureOr<AccumulatorEncodingAttr>
getAccumulatorEncodingAttr(const EncodingPlan &plan, int encodingId,
                           Operation *op, StringRef role);
FailureOr<const SharedEncodingSpec *>
getSharedEncodingSpec(const EncodingPlan &plan, int encodingId, Operation *op,
                      StringRef role);
FailureOr<llvm::SmallVector<int64_t, 2>>
getMmaWarpsPerCTA(const EncodingPlan &plan, Operation *op);
FailureOr<llvm::SmallVector<int64_t, 2>>
getAccumulatorTileShape(const EncodingPlan &plan, Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_ENCODINGPLAN_H
