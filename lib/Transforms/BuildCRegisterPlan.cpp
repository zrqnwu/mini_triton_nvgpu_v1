#include "tb/Analysis/AccumulatorPlan.h"
#include "tb/Analysis/EpiloguePlan.h"
#include "tb/Analysis/EpilogueReorderPlan.h"
#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/PersistentWorkPlan.h"
#include "tb/Analysis/ProgramMappingPlan.h"
#include "tb/Analysis/ReductionPlan.h"
#include "tb/Analysis/ResourceClosurePlan.h"
#include "tb/Analysis/SharedWorkspacePlan.h"
#include "tb/Analysis/TargetInfo.h"
#include "tb/Analysis/TransportPlan.h"
#include "tb/Analysis/WarpDecompositionPlan.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tb {
#define GEN_PASS_DEF_TBBUILDCREGISTERPLAN
#include "tb/Transforms/Passes.h.inc"

namespace {

class TBBuildCRegisterPlan
    : public impl::TBBuildCRegisterPlanBase<TBBuildCRegisterPlan> {
public:
  using impl::TBBuildCRegisterPlanBase<
      TBBuildCRegisterPlan>::TBBuildCRegisterPlanBase;

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

      auto encodings = parseEncodingPlanAttr(op.getOperation());
      if (failed(encodings)) {
        hadFailure = true;
        return;
      }

      auto target = getTargetInfo(op.getOperation());
      if (failed(target)) {
        hadFailure = true;
        return;
      }
      auto transport = parseTransportPlanAttr(op.getOperation());
      auto programMapping = parseProgramMappingPlanAttr(op.getOperation());
      auto reduction = parseReductionPlanAttr(op.getOperation());
      auto persistentWork = parsePersistentWorkPlanAttr(op.getOperation());
      if (failed(transport) || failed(programMapping) || failed(reduction) ||
          failed(persistentWork)) {
        hadFailure = true;
        return;
      }
      auto accumulator =
          deriveAccumulatorPlan(*config, *target, *encodings, op.getOperation());
      if (failed(accumulator)) {
        hadFailure = true;
        return;
      }
      auto epilogue = deriveEpiloguePlan(*config, *target, *encodings,
                                         *accumulator, op.getOperation());
      if (failed(epilogue)) {
        hadFailure = true;
        return;
      }
      auto epilogueReorder = deriveEpilogueReorderPlan(
          *config, *target, *accumulator, *epilogue, op.getOperation());
      if (failed(epilogueReorder)) {
        hadFailure = true;
        return;
      }
      if (auto *store = std::get_if<DirectGlobalVectorPlan>(&epilogue->store)) {
        for (const DirectGlobalVectorPlan::Pack &pack : store->packs) {
          for (int64_t fragmentId : pack.fragmentIds) {
            if (fragmentId < 0 ||
                fragmentId >= static_cast<int64_t>(accumulator->packs.size())) {
              op.emitError()
                  << "direct epilogue pack references an invalid accumulator "
                     "fragment id";
              hadFailure = true;
              return;
            }
            accumulator->packs[static_cast<size_t>(fragmentId)].epiloguePack =
                pack.packId;
          }
        }
      }
      auto warpPlan = deriveWarpDecompositionPlan(
          *config, *encodings, *accumulator, *epilogue, *epilogueReorder,
          *reduction, *persistentWork, op.getOperation());
      if (failed(warpPlan)) {
        hadFailure = true;
        return;
      }
      auto sharedWorkspace = deriveSharedWorkspacePlan(
          *config, *target, *encodings, *transport, *epilogueReorder,
          *reduction, *persistentWork, op.getOperation());
      if (failed(sharedWorkspace)) {
        hadFailure = true;
        return;
      }
      auto resourcePlan = deriveResourceClosurePlan(
          *config, *target, *programMapping, *reduction, *persistentWork,
          *encodings, *transport, *accumulator, *epilogue, *epilogueReorder,
          *sharedWorkspace, *warpPlan, op.getOperation());
      if (failed(resourcePlan)) {
        hadFailure = true;
        return;
      }

      op->removeAttr("tb.c_register_plan");
      op->setAttr("tb.accumulator_plan",
                  buildAccumulatorPlanAttr(builder, *accumulator));
      op->setAttr("tb.epilogue_plan", buildEpiloguePlanAttr(builder, *epilogue));
      op->setAttr("tb.epilogue_reorder_plan",
                  buildEpilogueReorderPlanAttr(builder, *epilogueReorder));
      op->setAttr("tb.shared_workspace_plan",
                  buildSharedWorkspacePlanAttr(builder, *sharedWorkspace));
      op->setAttr("tb.warp_decomposition_plan",
                  buildWarpDecompositionPlanAttr(builder, *warpPlan));
      op->setAttr("tb.resource_closure_plan",
                  buildResourceClosurePlanAttr(builder, *resourcePlan));
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tb
