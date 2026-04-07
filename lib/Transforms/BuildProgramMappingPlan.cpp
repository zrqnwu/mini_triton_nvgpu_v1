#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/PersistentWorkPlan.h"
#include "tb/Analysis/ProgramMappingPlan.h"
#include "tb/Analysis/ReductionPlan.h"
#include "tb/Analysis/TargetInfo.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBBUILDPROGRAMMAPPINGPLAN
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBBuildProgramMappingPlan
    : public impl::TBBuildProgramMappingPlanBase<TBBuildProgramMappingPlan> {
public:
  using impl::TBBuildProgramMappingPlanBase<
      TBBuildProgramMappingPlan>::TBBuildProgramMappingPlanBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    bool hadFailure = false;

    module.walk([&](MatmulOp op) {
      auto config = getKernelConfig(op);
      if (failed(config) ||
          failed(verifySupportedKernelConfig(*config, op.getOperation()))) {
        hadFailure = true;
        return;
      }

      auto target = getTargetInfo(op.getOperation());
      if (failed(target)) {
        hadFailure = true;
        return;
      }
      auto plan =
          deriveProgramMappingPlan(*config, *target, op.getOperation());
      if (failed(plan)) {
        hadFailure = true;
        return;
      }
      auto reductionPlan =
          deriveReductionPlan(*config, *plan, op.getOperation());
      if (failed(reductionPlan)) {
        hadFailure = true;
        return;
      }
      auto persistentWork =
          derivePersistentWorkPlan(*config, *target, *plan, op.getOperation());
      if (failed(persistentWork)) {
        hadFailure = true;
        return;
      }

      op->setAttr("tb.program_mapping_plan",
                  buildProgramMappingPlanAttr(builder, *plan));
      op->setAttr("tb.reduction_plan",
                  buildReductionPlanAttr(builder, *reductionPlan));
      op->setAttr("tb.persistent_work_plan",
                  buildPersistentWorkPlanAttr(builder, *persistentWork));
      if (failed(setModuleContextAttr(module, kTBNumCTAsAttrName,
                                      builder.getI64IntegerAttr(plan->numCTAs),
                                      op.getOperation()))) {
        hadFailure = true;
        return;
      }
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb
