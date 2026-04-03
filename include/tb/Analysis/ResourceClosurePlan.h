#ifndef MINI_TRITON_TB_ANALYSIS_RESOURCECLOSUREPLAN_H
#define MINI_TRITON_TB_ANALYSIS_RESOURCECLOSUREPLAN_H

#include "tb/Analysis/TargetInfo.h"
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
  int64_t staticSharedBudget = 0;
  int64_t dynamicSharedBudget = 0;
  std::string chosenLandingTradeoff;
  std::string chosenBufferingTradeoff;
  std::string reason;
};

FailureOr<ResourceClosurePlan>
deriveResourceClosurePlan(const KernelConfig &config, const TargetInfo &target,
                          const EncodingPlan &encodings,
                          const AccumulatorPlan &accumulator,
                          const EpiloguePlan &epilogue,
                          const WarpDecompositionPlan &warpPlan, Operation *op);
DictionaryAttr buildResourceClosurePlanAttr(Builder &builder,
                                            const ResourceClosurePlan &plan);
FailureOr<ResourceClosurePlan> parseResourceClosurePlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_RESOURCECLOSUREPLAN_H
