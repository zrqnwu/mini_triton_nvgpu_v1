#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/MatmulSemantics.h"
#include "tb/Analysis/TargetInfo.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBBUILDLAYOUTPLAN
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBBuildLayoutPlan
    : public impl::TBBuildLayoutPlanBase<TBBuildLayoutPlan> {
public:
  using impl::TBBuildLayoutPlanBase<
      TBBuildLayoutPlan>::TBBuildLayoutPlanBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    bool hadFailure = false;

    module.walk([&](MatmulOp op) {
      auto config = getKernelConfig(op);
      if (mlir::failed(config) ||
          mlir::failed(
              verifySupportedKernelConfig(*config, op.getOperation()))) {
        hadFailure = true;
        return;
      }

      auto semantics = parseMatmulSemanticsAttr(op.getOperation());
      auto programMapping = parseProgramMappingPlanAttr(op.getOperation());
      auto target = getTargetInfo(op.getOperation());
      if (failed(semantics) || failed(programMapping) || failed(target)) {
        hadFailure = true;
        return;
      }
      auto plan = deriveEncodingPlan(*config, *target, *semantics,
                                     *programMapping, op.getOperation());
      if (mlir::failed(plan)) {
        hadFailure = true;
        return;
      }

      op->removeAttr("tb.layout_plan");
      op->setAttr("tb.encoding_plan", buildEncodingPlanAttr(builder, *plan));
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb
