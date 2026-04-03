#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/TargetInfo.h"
#include "tb/Analysis/TransportPlan.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBBUILDTRANSPORTPLAN
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBBuildTransportPlan
    : public impl::TBBuildTransportPlanBase<TBBuildTransportPlan> {
public:
  using impl::TBBuildTransportPlanBase<
      TBBuildTransportPlan>::TBBuildTransportPlanBase;

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
      auto encodings = parseEncodingPlanAttr(op.getOperation());
      if (failed(target) || failed(encodings)) {
        hadFailure = true;
        return;
      }

      auto transport = deriveTransportPlan(*target, *encodings, op.getOperation());
      if (failed(transport)) {
        hadFailure = true;
        return;
      }

      op->setAttr("tb.transport_plan",
                  buildTransportPlanAttr(builder, *transport));
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb
