#ifndef MINI_TRITON_TB_ANALYSIS_EPILOGUEREORDERPLAN_H
#define MINI_TRITON_TB_ANALYSIS_EPILOGUEREORDERPLAN_H

#include "tb/Analysis/AccumulatorPlan.h"
#include "tb/Analysis/EpiloguePlan.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/TargetInfo.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tb {

enum class EpilogueReorderKind {
  None,
  CTASharedRowReorder,
  CTASharedRelay,
};

struct EpilogueRowVectorMapping {
  int64_t rowVectorId = -1;
  int64_t packId = -1;
  int64_t warpOwner = -1;
  int64_t rowBase = 0;
  int64_t colBase = 0;
  int64_t rowOffset = 0;
  int64_t vectorWidth = 0;
  llvm::SmallVector<int64_t, 4> fragmentIds;
};

struct EpilogueReorderPlan {
  EpilogueReorderKind kind = EpilogueReorderKind::None;
  int64_t sharedTileRows = 0;
  int64_t sharedTileCols = 0;
  int64_t liveSlots = 0;
  int64_t initSharedStoreVectorWidth = 0;
  int64_t initSharedLoadVectorWidth = 0;
  int64_t storeSharedStoreVectorWidth = 0;
  int64_t storeSharedLoadVectorWidth = 0;
  int64_t workspaceBarrierCount = 0;
  bool requiresWarpSync = false;
  bool reorderNeededForInit = false;
  bool reorderNeededForStore = false;
  std::string ownerScope = "none";
  std::string workspaceSyncKind = "none";
  std::string reason;
  llvm::SmallVector<EpilogueRowVectorMapping, 32> rowVectors;
};

FailureOr<EpilogueReorderPlan>
deriveEpilogueReorderPlan(const KernelConfig &config, const TargetInfo &target,
                          const AccumulatorPlan &accumulator,
                          const EpiloguePlan &epilogue, Operation *op);
DictionaryAttr buildEpilogueReorderPlanAttr(Builder &builder,
                                            const EpilogueReorderPlan &plan);
FailureOr<EpilogueReorderPlan> parseEpilogueReorderPlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_EPILOGUEREORDERPLAN_H
