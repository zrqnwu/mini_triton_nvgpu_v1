#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/LoopPlan.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBREGULARIZEKLOOP
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBRegularizeKLoop
    : public impl::TBRegularizeKLoopBase<TBRegularizeKLoop> {
public:
  using impl::TBRegularizeKLoopBase<TBRegularizeKLoop>::TBRegularizeKLoopBase;

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
      if (failed(model)) {
        hadFailure = true;
        return;
      }

      auto loopPlan = deriveLoopPlan(*model, op.getOperation());
      if (failed(loopPlan)) {
        hadFailure = true;
        return;
      }

      op->removeAttr("tb.loop_plan");
      op->setAttr("tb.loop_plan", buildLoopPlanAttr(builder, *loopPlan));
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb
