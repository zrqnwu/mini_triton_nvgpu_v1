#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/LatencyPlan.h"
#include "tb/Analysis/LoopPlan.h"
#include "tb/Analysis/PipelinePlan.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBSCHEDULELOOPS
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBScheduleLoops : public impl::TBScheduleLoopsBase<TBScheduleLoops> {
public:
  using impl::TBScheduleLoopsBase<TBScheduleLoops>::TBScheduleLoopsBase;

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

      auto model = parseBufferModelAttr(op.getOperation());
      auto loopPlan = parseLoopPlanAttr(op.getOperation());
      auto latencyPlan = parseLatencyPlanAttr(op.getOperation());
      if (failed(model) || failed(loopPlan) || failed(latencyPlan)) {
        hadFailure = true;
        return;
      }

      auto pipelinePlan =
          derivePipelinePlan(*model, *loopPlan, *latencyPlan,
                             op.getOperation());
      if (failed(pipelinePlan)) {
        hadFailure = true;
        return;
      }

      op->removeAttr("tb.schedule_plan");
      op->setAttr("tb.pipeline_plan",
                  buildPipelinePlanAttr(builder, *pipelinePlan));
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb
