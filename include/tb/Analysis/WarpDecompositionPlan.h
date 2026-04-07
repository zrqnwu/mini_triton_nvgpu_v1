#ifndef MINI_TRITON_TB_ANALYSIS_WARPDECOMPOSITIONPLAN_H
#define MINI_TRITON_TB_ANALYSIS_WARPDECOMPOSITIONPLAN_H

#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/EpiloguePlan.h"
#include "tb/Analysis/EpilogueReorderPlan.h"
#include "tb/Analysis/PersistentWorkPlan.h"
#include "tb/Analysis/ReductionPlan.h"

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
  llvm::SmallVector<int64_t, 8> sharedPackIds;
  llvm::SmallVector<int64_t, 16> reorderRowVectorIds;
  llvm::SmallVector<int64_t, 32> fragmentToRowVectorMap;
  llvm::SmallVector<int64_t, 16> rowVectorStoreOrder;
  llvm::SmallVector<int64_t, 16> rowVectorLoadOrder;
  llvm::SmallVector<int64_t, 2> reorderTileBase;
  bool reorderNeedsShared = false;
  llvm::SmallVector<int64_t, 8> reductionPartialIds;
  llvm::SmallVector<int64_t, 8> persistentBatchIds;
};

struct WarpDecompositionPlan {
  llvm::SmallVector<int64_t, 2> ctaTile;
  int64_t numWarps = 0;
  llvm::SmallVector<int64_t, 2> warpGrid;
  llvm::SmallVector<int64_t, 2> warpTile;
  std::string ownerScope = "per_warp_template";
  std::string warpOrder = "row_major";
  std::string landingOwner = "register_pack";
  std::string reorderOwner = "none";
  std::string reductionOwner = "none";
  std::string persistentOwner = "none";
  llvm::SmallVector<WarpTileCoverage, 8> warps;
};

FailureOr<WarpDecompositionPlan>
deriveWarpDecompositionPlan(const KernelConfig &config,
                            const EncodingPlan &encodings,
                            const AccumulatorPlan &accumulator,
                            const EpiloguePlan &epilogue,
                            const EpilogueReorderPlan &epilogueReorder,
                            const ReductionPlan &reduction,
                            const PersistentWorkPlan &persistentWork,
                            Operation *op);
DictionaryAttr buildWarpDecompositionPlanAttr(Builder &builder,
                                              const WarpDecompositionPlan &plan);
FailureOr<WarpDecompositionPlan> parseWarpDecompositionPlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_WARPDECOMPOSITIONPLAN_H
