#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/LatencyPlan.h"
#include "tb/Analysis/LoopPlan.h"
#include "tb/Analysis/TargetInfo.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBASSIGNLATENCIES
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBAssignLatencies
    : public impl::TBAssignLatenciesBase<TBAssignLatencies> {
public:
  using impl::TBAssignLatenciesBase<TBAssignLatencies>::TBAssignLatenciesBase;

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
      if (failed(model) || failed(loopPlan)) {
        hadFailure = true;
        return;
      }

      auto target = getTargetInfo(op.getOperation());
      if (failed(target)) {
        hadFailure = true;
        return;
      }
      auto plan =
          deriveLatencyPlan(*config, *target, *model, *loopPlan,
                            op.getOperation());
      if (failed(plan)) {
        hadFailure = true;
        return;
      }

      op->removeAttr("tb.latency_plan");
      op->setAttr("tb.latency_plan", buildLatencyPlanAttr(builder, *plan));
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb
