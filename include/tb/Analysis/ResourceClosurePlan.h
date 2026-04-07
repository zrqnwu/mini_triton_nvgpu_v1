#ifndef MINI_TRITON_TB_ANALYSIS_RESOURCECLOSUREPLAN_H
#define MINI_TRITON_TB_ANALYSIS_RESOURCECLOSUREPLAN_H

#include "tb/Analysis/AccumulatorPlan.h"
#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/EpiloguePlan.h"
#include "tb/Analysis/EpilogueReorderPlan.h"
#include "tb/Analysis/PersistentWorkPlan.h"
#include "tb/Analysis/ProgramMappingPlan.h"
#include "tb/Analysis/ReductionPlan.h"
#include "tb/Analysis/SharedWorkspacePlan.h"
#include "tb/Analysis/TargetInfo.h"
#include "tb/Analysis/TransportPlan.h"
#include "tb/Analysis/WarpDecompositionPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include <string>

namespace mlir::tb {

struct ResourceClosurePlan {
  int64_t estimatedAccumulatorRegs = 0;
  int64_t estimatedEpilogueRegs = 0;
  int64_t estimatedABShared = 0;
  int64_t estimatedEpilogueShared = 0;
  int64_t estimatedReductionScratch = 0;
  int64_t estimatedPersistentState = 0;
  int64_t estimatedTotalStaticShared = 0;
  int64_t estimatedTotalDynamicShared = 0;
  int64_t estimatedTotalRegs = 0;
  int64_t peakStaticSharedBytes = 0;
  int64_t workspaceTotalBytes = 0;
  int64_t workspaceAliasSavedBytes = 0;
  int64_t reorderSharedBytes = 0;
  int64_t reductionSharedBytes = 0;
  int64_t persistentSharedBytes = 0;
  int64_t mainloopSharedLiveBegin = 0;
  int64_t mainloopSharedLiveEnd = 0;
  int64_t epilogueReorderLiveBegin = 0;
  int64_t epilogueReorderLiveEnd = 0;
  int64_t staticSharedBudget = 0;
  int64_t dynamicSharedBudget = 0;
  std::string selectedMainlineKind;
  std::string selectedCLandingKind;
  std::string selectedBufferingKind;
  std::string selectedSharedWorkspacePolicy;
  std::string chosenLandingTradeoff;
  std::string chosenBufferingTradeoff;
  std::string reason;
};

FailureOr<ResourceClosurePlan>
deriveResourceClosurePlan(const KernelConfig &config, const TargetInfo &target,
                          const ProgramMappingPlan &programMapping,
                          const ReductionPlan &reduction,
                          const PersistentWorkPlan &persistentWork,
                          const EncodingPlan &encodings,
                          const TransportPlan &transport,
                          const AccumulatorPlan &accumulator,
                          const EpiloguePlan &epilogue,
                          const EpilogueReorderPlan &epilogueReorder,
                          const SharedWorkspacePlan &sharedWorkspace,
                          const WarpDecompositionPlan &warpPlan, Operation *op);
DictionaryAttr buildResourceClosurePlanAttr(Builder &builder,
                                            const ResourceClosurePlan &plan);
FailureOr<ResourceClosurePlan> parseResourceClosurePlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_RESOURCECLOSUREPLAN_H
