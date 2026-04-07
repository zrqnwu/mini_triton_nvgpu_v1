#include "tb/Analysis/ResourceClosurePlan.h"

#include "tb/Analysis/KernelConfig.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::tb;

namespace {

static FailureOr<int64_t> readI64Field(DictionaryAttr dict, StringRef name,
                                       Operation *op) {
  auto attr = dyn_cast_or_null<IntegerAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing integer field `" << name << "`";
    return failure();
  }
  return attr.getInt();
}

static FailureOr<StringRef> readStringField(DictionaryAttr dict, StringRef name,
                                            Operation *op) {
  auto attr = dyn_cast_or_null<StringAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing string field `" << name << "`";
    return failure();
  }
  return attr.getValue();
}

static LogicalResult validateResourceClosurePlan(
    const ResourceClosurePlan &plan, Operation *op) {
  if (plan.estimatedAccumulatorRegs <= 0 || plan.estimatedEpilogueRegs < 0 ||
      plan.estimatedABShared < 0 || plan.estimatedEpilogueShared < 0 ||
      plan.estimatedReductionScratch < 0 ||
      plan.estimatedPersistentState < 0 ||
      plan.estimatedTotalStaticShared < 0 ||
      plan.estimatedTotalDynamicShared < 0 || plan.estimatedTotalRegs <= 0 ||
      plan.peakStaticSharedBytes < 0 || plan.workspaceTotalBytes < 0 ||
      plan.workspaceAliasSavedBytes < 0 || plan.reorderSharedBytes < 0 ||
      plan.reductionSharedBytes < 0 || plan.persistentSharedBytes < 0 ||
      plan.mainloopSharedLiveEnd < plan.mainloopSharedLiveBegin ||
      plan.epilogueReorderLiveEnd < plan.epilogueReorderLiveBegin ||
      plan.staticSharedBudget <= 0 || plan.dynamicSharedBudget < 0 ||
      plan.selectedMainlineKind.empty() || plan.selectedCLandingKind.empty() ||
      plan.selectedBufferingKind.empty() ||
      plan.selectedSharedWorkspacePolicy.empty() ||
      plan.chosenLandingTradeoff.empty() ||
      plan.chosenBufferingTradeoff.empty() || plan.reason.empty()) {
    return op->emitError()
           << "resource closure plan must carry explicit positive closure truth";
  }
  return success();
}

static int64_t sumBytesForKinds(const SharedWorkspacePlan &workspace,
                                ArrayRef<SharedWorkspaceSegmentKind> kinds) {
  int64_t total = 0;
  for (const SharedWorkspaceSegment &segment : workspace.segments) {
    if (llvm::is_contained(kinds, segment.kind))
      total += segment.byteSize;
  }
  return total;
}

static int64_t maxBytesForKind(const SharedWorkspacePlan &workspace,
                               SharedWorkspaceSegmentKind kind) {
  int64_t result = 0;
  for (const SharedWorkspaceSegment &segment : workspace.segments) {
    if (segment.kind == kind)
      result = std::max(result, segment.byteSize);
  }
  return result;
}

static std::pair<int64_t, int64_t>
liveRangeForKinds(const SharedWorkspacePlan &workspace,
                  ArrayRef<SharedWorkspaceSegmentKind> kinds) {
  int64_t begin = std::numeric_limits<int64_t>::max();
  int64_t end = std::numeric_limits<int64_t>::min();
  bool found = false;
  for (const SharedWorkspaceSegment &segment : workspace.segments) {
    if (!llvm::is_contained(kinds, segment.kind))
      continue;
    found = true;
    begin = std::min(begin, segment.lifetimeBegin);
    end = std::max(end, segment.lifetimeEnd);
  }
  if (!found)
    return {0, 0};
  return {begin, end};
}

} // namespace

FailureOr<ResourceClosurePlan>
mlir::tb::deriveResourceClosurePlan(const KernelConfig &config,
                                    const TargetInfo &target,
                                    const ProgramMappingPlan &programMapping,
                                    const ReductionPlan &reduction,
                                    const PersistentWorkPlan &persistentWork,
                                    const EncodingPlan &encodings,
                                    const TransportPlan &transport,
                                    const AccumulatorPlan &accumulator,
                                    const EpiloguePlan &epilogue,
                                    const EpilogueReorderPlan &epilogueReorder,
                                    const SharedWorkspacePlan &sharedWorkspace,
                                    const WarpDecompositionPlan &warpPlan,
                                    Operation *op) {
  (void)config;
  (void)encodings;
  (void)transport;
  auto *store = std::get_if<DirectGlobalVectorPlan>(&epilogue.store);
  if (!store || accumulator.packs.empty() || warpPlan.warps.empty()) {
    op->emitError()
        << "resource closure requires direct epilogue plus warp decomposition";
    return failure();
  }

  ResourceClosurePlan plan;
  int64_t accumulatorValues =
      accumulator.registersPerWarp * accumulator.packs.front().elemCount;
  int64_t epilogueRows = store->laneAccess.rowOffsets.size();
  int64_t directPackValues = epilogueRows * store->vectorWidth;
  plan.estimatedAccumulatorRegs = accumulatorValues;
  plan.estimatedEpilogueRegs = directPackValues;
  plan.estimatedABShared = sumBytesForKinds(
      sharedWorkspace,
      {SharedWorkspaceSegmentKind::MainloopAStageBuffer,
       SharedWorkspaceSegmentKind::MainloopBStageBuffer});
  plan.reorderSharedBytes =
      maxBytesForKind(sharedWorkspace,
                      SharedWorkspaceSegmentKind::EpilogueReorderScratch);
  plan.reductionSharedBytes =
      maxBytesForKind(sharedWorkspace,
                      SharedWorkspaceSegmentKind::SplitKReductionScratch);
  plan.persistentSharedBytes =
      maxBytesForKind(sharedWorkspace,
                      SharedWorkspaceSegmentKind::PersistentStateScratch);
  plan.estimatedEpilogueShared = plan.reorderSharedBytes;
  plan.estimatedReductionScratch = plan.reductionSharedBytes;
  plan.estimatedPersistentState = plan.persistentSharedBytes;
  plan.workspaceTotalBytes = sharedWorkspace.totalBytes;
  plan.workspaceAliasSavedBytes = sharedWorkspace.aliasSavedBytes;
  plan.peakStaticSharedBytes = sharedWorkspace.peakBytes;
  plan.staticSharedBudget =
      std::min<int64_t>(target.maxSharedBytesPerCTA, 48 * 1024);
  plan.dynamicSharedBudget =
      std::max<int64_t>(target.maxSharedBytesPerCTA - plan.staticSharedBudget, 0);
  if (persistentWork.enabled) {
    plan.selectedMainlineKind = "persistent_tile_loop";
  } else if (reduction.requiresInterProgramReduction) {
    plan.selectedMainlineKind = "split_k";
  } else if (programMapping.mappingKind == ProgramMappingKind::GroupedTile) {
    plan.selectedMainlineKind = "grouped_tile";
  } else {
    plan.selectedMainlineKind = "single_tile";
  }

  auto mainloopLive = liveRangeForKinds(
      sharedWorkspace,
      {SharedWorkspaceSegmentKind::MainloopAStageBuffer,
       SharedWorkspaceSegmentKind::MainloopBStageBuffer});
  auto reorderLive = liveRangeForKinds(
      sharedWorkspace, {SharedWorkspaceSegmentKind::EpilogueReorderScratch});
  plan.mainloopSharedLiveBegin = mainloopLive.first;
  plan.mainloopSharedLiveEnd = mainloopLive.second;
  plan.epilogueReorderLiveBegin = reorderLive.first;
  plan.epilogueReorderLiveEnd = reorderLive.second;

  if (epilogue.targetLanding.kind != TargetLandingKind::RegisterPackGlobalVector) {
    op->emitError()
        << "resource closure currently expects final C landing to stay "
           "register-pack/global-vector while reorder ownership is carried by "
           "`tb.epilogue_reorder_plan`";
    return failure();
  }
  plan.selectedCLandingKind =
      epilogueReorder.kind == EpilogueReorderKind::None
          ? "register_direct_vector"
          : "register_direct_vector_with_cta_shared_reorder";
  plan.selectedBufferingKind =
      epilogueReorder.kind == EpilogueReorderKind::None
          ? "async_ab_plus_register_epilogue"
          : "async_ab_plus_cta_workspace_reorder";
  plan.selectedSharedWorkspacePolicy = sharedWorkspace.selectedPolicy;
  plan.chosenLandingTradeoff = "register_direct_vector";
  plan.chosenBufferingTradeoff = sharedWorkspace.selectedPolicy;
  plan.estimatedTotalStaticShared = sharedWorkspace.totalBytes;
  plan.estimatedTotalDynamicShared = 0;
  plan.estimatedTotalRegs = plan.estimatedAccumulatorRegs +
                            plan.estimatedEpilogueRegs +
                            plan.estimatedPersistentState;
  plan.reason =
      epilogueReorder.kind == EpilogueReorderKind::None
          ? "final C landing stays register-direct and shared closure now "
            "comes only from the unified CTA workspace used by the mainloop"
          : "final C landing stays register-direct while epilogue row reorder "
            "moves into the unified CTA workspace, so shared closure is the "
            "peak of aliased lifetimes instead of a semantic-role sum";
  if (plan.estimatedTotalStaticShared + plan.estimatedTotalDynamicShared >
      target.maxSharedBytesPerCTA) {
    op->emitError() << "resource closure exceeds target shared memory budget";
    return failure();
  }

  if (failed(validateResourceClosurePlan(plan, op)))
    return failure();
  return plan;
}

DictionaryAttr
mlir::tb::buildResourceClosurePlanAttr(Builder &builder,
                                       const ResourceClosurePlan &plan) {
  NamedAttrList attrs;
  attrs.set("estimated_accumulator_regs",
            builder.getI64IntegerAttr(plan.estimatedAccumulatorRegs));
  attrs.set("estimated_epilogue_regs",
            builder.getI64IntegerAttr(plan.estimatedEpilogueRegs));
  attrs.set("estimated_ab_shared",
            builder.getI64IntegerAttr(plan.estimatedABShared));
  attrs.set("estimated_epilogue_shared",
            builder.getI64IntegerAttr(plan.estimatedEpilogueShared));
  attrs.set("estimated_reduction_scratch",
            builder.getI64IntegerAttr(plan.estimatedReductionScratch));
  attrs.set("estimated_persistent_state",
            builder.getI64IntegerAttr(plan.estimatedPersistentState));
  attrs.set("estimated_total_static_shared",
            builder.getI64IntegerAttr(plan.estimatedTotalStaticShared));
  attrs.set("estimated_total_dynamic_shared",
            builder.getI64IntegerAttr(plan.estimatedTotalDynamicShared));
  attrs.set("estimated_total_regs",
            builder.getI64IntegerAttr(plan.estimatedTotalRegs));
  attrs.set("peak_static_shared_bytes",
            builder.getI64IntegerAttr(plan.peakStaticSharedBytes));
  attrs.set("workspace_total_bytes",
            builder.getI64IntegerAttr(plan.workspaceTotalBytes));
  attrs.set("workspace_alias_saved_bytes",
            builder.getI64IntegerAttr(plan.workspaceAliasSavedBytes));
  attrs.set("reorder_shared_bytes",
            builder.getI64IntegerAttr(plan.reorderSharedBytes));
  attrs.set("reduction_shared_bytes",
            builder.getI64IntegerAttr(plan.reductionSharedBytes));
  attrs.set("persistent_shared_bytes",
            builder.getI64IntegerAttr(plan.persistentSharedBytes));
  attrs.set("mainloop_shared_live_begin",
            builder.getI64IntegerAttr(plan.mainloopSharedLiveBegin));
  attrs.set("mainloop_shared_live_end",
            builder.getI64IntegerAttr(plan.mainloopSharedLiveEnd));
  attrs.set("epilogue_reorder_live_begin",
            builder.getI64IntegerAttr(plan.epilogueReorderLiveBegin));
  attrs.set("epilogue_reorder_live_end",
            builder.getI64IntegerAttr(plan.epilogueReorderLiveEnd));
  attrs.set("static_shared_budget",
            builder.getI64IntegerAttr(plan.staticSharedBudget));
  attrs.set("dynamic_shared_budget",
            builder.getI64IntegerAttr(plan.dynamicSharedBudget));
  attrs.set("selected_mainline_kind",
            builder.getStringAttr(plan.selectedMainlineKind));
  attrs.set("selected_c_landing_kind",
            builder.getStringAttr(plan.selectedCLandingKind));
  attrs.set("selected_buffering_kind",
            builder.getStringAttr(plan.selectedBufferingKind));
  attrs.set("selected_shared_workspace_policy",
            builder.getStringAttr(plan.selectedSharedWorkspacePolicy));
  attrs.set("chosen_landing_tradeoff",
            builder.getStringAttr(plan.chosenLandingTradeoff));
  attrs.set("chosen_buffering_tradeoff",
            builder.getStringAttr(plan.chosenBufferingTradeoff));
  attrs.set("reason", builder.getStringAttr(plan.reason));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<ResourceClosurePlan>
mlir::tb::parseResourceClosurePlanAttr(Operation *op) {
  auto root =
      dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.resource_closure_plan"));
  if (!root) {
    op->emitError() << "missing `tb.resource_closure_plan` attribute";
    return failure();
  }
  ResourceClosurePlan plan;
  auto estimatedAccumulatorRegs =
      readI64Field(root, "estimated_accumulator_regs", op);
  auto estimatedEpilogueRegs = readI64Field(root, "estimated_epilogue_regs", op);
  auto estimatedABShared = readI64Field(root, "estimated_ab_shared", op);
  auto estimatedEpilogueShared =
      readI64Field(root, "estimated_epilogue_shared", op);
  auto estimatedReductionScratch =
      readI64Field(root, "estimated_reduction_scratch", op);
  auto estimatedPersistentState =
      readI64Field(root, "estimated_persistent_state", op);
  auto estimatedTotalStaticShared =
      readI64Field(root, "estimated_total_static_shared", op);
  auto estimatedTotalDynamicShared =
      readI64Field(root, "estimated_total_dynamic_shared", op);
  auto estimatedTotalRegs = readI64Field(root, "estimated_total_regs", op);
  auto peakStaticSharedBytes =
      readI64Field(root, "peak_static_shared_bytes", op);
  auto workspaceTotalBytes = readI64Field(root, "workspace_total_bytes", op);
  auto workspaceAliasSavedBytes =
      readI64Field(root, "workspace_alias_saved_bytes", op);
  auto reorderSharedBytes = readI64Field(root, "reorder_shared_bytes", op);
  auto reductionSharedBytes = readI64Field(root, "reduction_shared_bytes", op);
  auto persistentSharedBytes =
      readI64Field(root, "persistent_shared_bytes", op);
  auto mainloopSharedLiveBegin =
      readI64Field(root, "mainloop_shared_live_begin", op);
  auto mainloopSharedLiveEnd =
      readI64Field(root, "mainloop_shared_live_end", op);
  auto epilogueReorderLiveBegin =
      readI64Field(root, "epilogue_reorder_live_begin", op);
  auto epilogueReorderLiveEnd =
      readI64Field(root, "epilogue_reorder_live_end", op);
  auto staticSharedBudget = readI64Field(root, "static_shared_budget", op);
  auto dynamicSharedBudget = readI64Field(root, "dynamic_shared_budget", op);
  auto selectedMainlineKind =
      readStringField(root, "selected_mainline_kind", op);
  auto selectedCLandingKind =
      readStringField(root, "selected_c_landing_kind", op);
  auto selectedBufferingKind =
      readStringField(root, "selected_buffering_kind", op);
  auto selectedSharedWorkspacePolicy =
      readStringField(root, "selected_shared_workspace_policy", op);
  auto chosenLandingTradeoff =
      readStringField(root, "chosen_landing_tradeoff", op);
  auto chosenBufferingTradeoff =
      readStringField(root, "chosen_buffering_tradeoff", op);
  auto reason = readStringField(root, "reason", op);
  if (failed(estimatedAccumulatorRegs) || failed(estimatedEpilogueRegs) ||
      failed(estimatedABShared) || failed(estimatedEpilogueShared) ||
      failed(estimatedReductionScratch) || failed(estimatedPersistentState) ||
      failed(estimatedTotalStaticShared) || failed(estimatedTotalDynamicShared) ||
      failed(estimatedTotalRegs) || failed(peakStaticSharedBytes) ||
      failed(workspaceTotalBytes) || failed(workspaceAliasSavedBytes) ||
      failed(reorderSharedBytes) || failed(reductionSharedBytes) ||
      failed(persistentSharedBytes) ||
      failed(mainloopSharedLiveBegin) || failed(mainloopSharedLiveEnd) ||
      failed(epilogueReorderLiveBegin) || failed(epilogueReorderLiveEnd) ||
      failed(staticSharedBudget) || failed(dynamicSharedBudget) ||
      failed(selectedMainlineKind) || failed(selectedCLandingKind) ||
      failed(selectedBufferingKind) || failed(selectedSharedWorkspacePolicy) ||
      failed(chosenLandingTradeoff) || failed(chosenBufferingTradeoff) ||
      failed(reason)) {
    op->emitError() << "malformed `tb.resource_closure_plan` attribute";
    return failure();
  }
  plan.estimatedAccumulatorRegs = *estimatedAccumulatorRegs;
  plan.estimatedEpilogueRegs = *estimatedEpilogueRegs;
  plan.estimatedABShared = *estimatedABShared;
  plan.estimatedEpilogueShared = *estimatedEpilogueShared;
  plan.estimatedReductionScratch = *estimatedReductionScratch;
  plan.estimatedPersistentState = *estimatedPersistentState;
  plan.estimatedTotalStaticShared = *estimatedTotalStaticShared;
  plan.estimatedTotalDynamicShared = *estimatedTotalDynamicShared;
  plan.estimatedTotalRegs = *estimatedTotalRegs;
  plan.peakStaticSharedBytes = *peakStaticSharedBytes;
  plan.workspaceTotalBytes = *workspaceTotalBytes;
  plan.workspaceAliasSavedBytes = *workspaceAliasSavedBytes;
  plan.reorderSharedBytes = *reorderSharedBytes;
  plan.reductionSharedBytes = *reductionSharedBytes;
  plan.persistentSharedBytes = *persistentSharedBytes;
  plan.mainloopSharedLiveBegin = *mainloopSharedLiveBegin;
  plan.mainloopSharedLiveEnd = *mainloopSharedLiveEnd;
  plan.epilogueReorderLiveBegin = *epilogueReorderLiveBegin;
  plan.epilogueReorderLiveEnd = *epilogueReorderLiveEnd;
  plan.staticSharedBudget = *staticSharedBudget;
  plan.dynamicSharedBudget = *dynamicSharedBudget;
  plan.selectedMainlineKind = selectedMainlineKind->str();
  plan.selectedCLandingKind = selectedCLandingKind->str();
  plan.selectedBufferingKind = selectedBufferingKind->str();
  plan.selectedSharedWorkspacePolicy = selectedSharedWorkspacePolicy->str();
  plan.chosenLandingTradeoff = chosenLandingTradeoff->str();
  plan.chosenBufferingTradeoff = chosenBufferingTradeoff->str();
  plan.reason = reason->str();
  if (failed(validateResourceClosurePlan(plan, op))) {
    op->emitError() << "malformed `tb.resource_closure_plan` attribute";
    return failure();
  }
  return plan;
}
