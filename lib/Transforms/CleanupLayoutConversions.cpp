#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBCLEANUPLAYOUTCONVERSIONS
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBCleanupLayoutConversions
    : public impl::TBCleanupLayoutConversionsBase<
          TBCleanupLayoutConversions> {
public:
  using impl::TBCleanupLayoutConversionsBase<
      TBCleanupLayoutConversions>::TBCleanupLayoutConversionsBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<ConvertLayoutOp> toErase;
      module.walk([&](ConvertLayoutOp op) {
        Value replacement;
        if (op.getSource().getType() == op.getResult().getType()) {
          replacement = op.getSource();
        } else if (auto prev = op.getSource().getDefiningOp<ConvertLayoutOp>()) {
          if (prev.getSource().getType() == op.getResult().getType())
            replacement = prev.getSource();
        }
        if (!replacement)
          return;
        op.getResult().replaceAllUsesWith(replacement);
        toErase.push_back(op);
        changed = true;
      });
      for (ConvertLayoutOp op : llvm::reverse(toErase)) {
        if (op->use_empty())
          op.erase();
      }
    }

    SmallVector<ConvertLayoutOp> deadConversions;
    module.walk([&](ConvertLayoutOp op) {
      if (op->use_empty())
        deadConversions.push_back(op);
    });
    for (ConvertLayoutOp op : llvm::reverse(deadConversions))
      op.erase();
  }
};

} // namespace
} // namespace mlir::tb
