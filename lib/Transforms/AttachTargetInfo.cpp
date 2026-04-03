#include "tb/Analysis/TargetInfo.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBATTACHTARGETINFO
#include "tb/Transforms/Passes.h.inc"

namespace {

static FailureOr<int64_t> requireModuleI64Attr(ModuleOp module, StringRef name,
                                               Operation *anchorOp) {
  auto attr = dyn_cast_or_null<IntegerAttr>(module->getAttr(name));
  if (!attr) {
    anchorOp->emitError() << "missing module attr `" << name
                          << "` required by the explicit execution contract";
    return failure();
  }
  if (attr.getInt() <= 0) {
    anchorOp->emitError() << "module attr `" << name << "` must be positive";
    return failure();
  }
  return attr.getInt();
}

class TBAttachTargetInfo
    : public impl::TBAttachTargetInfoBase<TBAttachTargetInfo> {
public:
  using impl::TBAttachTargetInfoBase<
      TBAttachTargetInfo>::TBAttachTargetInfoBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    Builder builder(module.getContext());
    bool hadFailure = false;

    if (gpuArch.empty()) {
      module.emitError()
          << "tb-attach-target-info requires explicit `gpu-arch`; stage1 "
             "target ownership must not fall back to a hidden default chip";
      signalPassFailure();
      return;
    }
    std::string chip = gpuArch.getValue();
    std::string features =
        ptxFeatures.empty() ? std::string() : ptxFeatures.getValue();

    module.walk([&](MatmulOp op) {
      auto numWarps =
          requireModuleI64Attr(module, kTBNumWarpsAttrName, op.getOperation());
      auto requestedStages = requireModuleI64Attr(module, kTBRequestedStagesAttrName,
                                                  op.getOperation());
      if (failed(numWarps) || failed(requestedStages)) {
        hadFailure = true;
        return;
      }
      (void)*numWarps;
      (void)*requestedStages;
      auto target = deriveTargetInfoForArch(chip, features, op.getOperation());
      if (failed(target)) {
        hadFailure = true;
        return;
      }
      DictionaryAttr targetAttr = buildTargetInfoAttr(builder, *target);
      if (failed(setModuleContextAttr(module, kTBTargetModuleAttrName, targetAttr,
                                      op.getOperation())) ||
          failed(setModuleContextAttr(module, kTBThreadsPerWarpAttrName,
                                      builder.getI64IntegerAttr(
                                          target->threadsPerWarp),
                                      op.getOperation()))) {
        hadFailure = true;
        return;
      }
      op->removeAttr(kTBTargetOpAttrName);
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb
