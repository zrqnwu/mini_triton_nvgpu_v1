#ifndef MINI_TRITON_TB_ANALYSIS_WARPDECOMPOSITIONPLAN_H
#define MINI_TRITON_TB_ANALYSIS_WARPDECOMPOSITIONPLAN_H

#include "tb/Analysis/EpiloguePlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tb {

struct WarpTileCoverage {
  int64_t warpId = -1;
  int64_t coordM = -1;
  int64_t coordN = -1;
  int64_t tileBaseM = 0;
  int64_t tileBaseN = 0;
  int64_t tileRows = 0;
  int64_t tileCols = 0;
  llvm::SmallVector<int64_t, 8> mmaGroupIds;
  llvm::SmallVector<int64_t, 8> accumulatorPackIds;
  llvm::SmallVector<int64_t, 8> epiloguePackIds;
};

struct WarpDecompositionPlan {
  llvm::SmallVector<int64_t, 2> ctaTile;
  int64_t numWarps = 0;
  llvm::SmallVector<int64_t, 2> warpGrid;
  llvm::SmallVector<int64_t, 2> warpTile;
  std::string ownerScope = "per_warp_template";
  std::string warpOrder = "row_major";
  llvm::SmallVector<WarpTileCoverage, 8> warps;
};

FailureOr<WarpDecompositionPlan>
deriveWarpDecompositionPlan(const KernelConfig &config,
                            const EncodingPlan &encodings,
                            const AccumulatorPlan &accumulator,
                            const EpiloguePlan &epilogue, Operation *op);
DictionaryAttr buildWarpDecompositionPlanAttr(Builder &builder,
                                              const WarpDecompositionPlan &plan);
FailureOr<WarpDecompositionPlan> parseWarpDecompositionPlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_WARPDECOMPOSITIONPLAN_H
