#ifndef MINI_TRITON_TB_ANALYSIS_MATMULREWRITEPLAN_H
#define MINI_TRITON_TB_ANALYSIS_MATMULREWRITEPLAN_H

#include "tb/Analysis/AccumulatorPlan.h"
#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/EpiloguePlan.h"
#include "tb/Analysis/KernelConfig.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tb {

struct OperandFragmentPath {
  std::string role;
  llvm::SmallVector<int64_t, 4> instructionShape;
  llvm::SmallVector<int64_t, 4> fragmentShape;
  int64_t fragmentsPerKGroup = 0;
  int64_t consumerTileSpanM = 1;
  int64_t consumerTileSpanN = 1;
  bool usesLdMatrix = false;
};

struct MatmulRewritePlan {
  std::string contractModel;
  std::string mainloopKind;
  int64_t instructionM = 0;
  int64_t instructionN = 0;
  int64_t instructionK = 0;
  int64_t kGroups = 0;
  int64_t accTilesM = 0;
  int64_t accTilesN = 0;
  int64_t bGroupCount = 0;
  int64_t accumulatorFragments = 0;
  bool directAccumulatorInit = false;
  bool directAccumulatorStore = false;
  OperandFragmentPath aPath;
  OperandFragmentPath bPath;
};

FailureOr<MatmulRewritePlan> deriveMatmulRewritePlan(
    const KernelConfig &config, const EncodingPlan &encodings,
    const AccumulatorPlan &accumulator, const EpiloguePlan &epilogue,
    Operation *op);
DictionaryAttr buildMatmulRewritePlanAttr(Builder &builder,
                                          const MatmulRewritePlan &plan);
FailureOr<MatmulRewritePlan> parseMatmulRewritePlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_MATMULREWRITEPLAN_H
