#include "tb/Analysis/ResourceClosurePlan.h"

#include "tb/Analysis/KernelConfig.h"

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

static int64_t getScalarByteWidth(ScalarKind kind) {
  switch (kind) {
  case ScalarKind::F16:
    return 2;
  case ScalarKind::F32:
    return 4;
  }
  llvm_unreachable("unknown scalar kind");
}

static LogicalResult validateResourceClosurePlan(
    const ResourceClosurePlan &plan, Operation *op) {
  if (plan.estimatedAccumulatorRegs <= 0 || plan.estimatedEpilogueRegs < 0 ||
      plan.estimatedABShared < 0 || plan.estimatedEpilogueShared < 0 ||
      plan.staticSharedBudget <= 0 || plan.dynamicSharedBudget < 0 ||
      plan.chosenLandingTradeoff.empty() ||
      plan.chosenBufferingTradeoff.empty() || plan.reason.empty()) {
    return op->emitError()
           << "resource closure plan must carry explicit positive closure truth";
  }
  return success();
}

} // namespace

FailureOr<ResourceClosurePlan>
mlir::tb::deriveResourceClosurePlan(const KernelConfig &config,
                                    const TargetInfo &target,
                                    const EncodingPlan &encodings,
                                    const AccumulatorPlan &accumulator,
                                    const EpiloguePlan &epilogue,
                                    const WarpDecompositionPlan &warpPlan,
                                    Operation *op) {
  (void)encodings;
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
  int64_t aBytes = getScalarByteWidth(config.aScalar);
  int64_t bBytes = getScalarByteWidth(config.bScalar);
  int64_t cBytes = getScalarByteWidth(config.cScalar);
  plan.estimatedAccumulatorRegs = accumulatorValues;
  plan.estimatedEpilogueRegs = directPackValues;
  plan.estimatedABShared =
      config.blockM * config.blockK * aBytes +
      config.blockK * config.blockN * bBytes;
  plan.staticSharedBudget =
      std::min<int64_t>(target.maxSharedBytesPerCTA, 48 * 1024);
  plan.dynamicSharedBudget =
      std::max<int64_t>(target.maxSharedBytesPerCTA - plan.staticSharedBudget, 0);

  if (epilogue.targetLanding.kind == TargetLandingKind::RegisterPackGlobalVector) {
    plan.estimatedEpilogueShared = 0;
    plan.chosenLandingTradeoff = "register_direct_vector";
    plan.chosenBufferingTradeoff = "async_ab_plus_register_epilogue";
    plan.reason =
        "direct owner stays in registers because direct packs are lane-local "
        "fixed-point fragments and do not require shared permutation";
  } else {
    int64_t packSlots = std::max<int64_t>(epilogue.targetLanding.sharedPackSlots, 1);
    int64_t sharedRows = std::max<int64_t>(epilogue.targetLanding.sharedTileRows, 1);
    int64_t sharedCols = std::max<int64_t>(epilogue.targetLanding.sharedTileCols, 1);
    plan.estimatedEpilogueShared =
        warpPlan.numWarps * packSlots * sharedRows * sharedCols * cBytes;
    plan.chosenLandingTradeoff = "shared_pack_vector";
    plan.chosenBufferingTradeoff = "async_ab_plus_shared_epilogue";
    plan.reason =
        "shared epilogue relay is required because direct lane-local packing "
        "is not available for this landing contract";
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
  attrs.set("static_shared_budget",
            builder.getI64IntegerAttr(plan.staticSharedBudget));
  attrs.set("dynamic_shared_budget",
            builder.getI64IntegerAttr(plan.dynamicSharedBudget));
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
  auto staticSharedBudget = readI64Field(root, "static_shared_budget", op);
  auto dynamicSharedBudget = readI64Field(root, "dynamic_shared_budget", op);
  auto chosenLandingTradeoff =
      readStringField(root, "chosen_landing_tradeoff", op);
  auto chosenBufferingTradeoff =
      readStringField(root, "chosen_buffering_tradeoff", op);
  auto reason = readStringField(root, "reason", op);
  if (failed(estimatedAccumulatorRegs) || failed(estimatedEpilogueRegs) ||
      failed(estimatedABShared) || failed(estimatedEpilogueShared) ||
      failed(staticSharedBudget) || failed(dynamicSharedBudget) ||
      failed(chosenLandingTradeoff) || failed(chosenBufferingTradeoff) ||
      failed(reason)) {
    op->emitError() << "malformed `tb.resource_closure_plan` attribute";
    return failure();
  }
  plan.estimatedAccumulatorRegs = *estimatedAccumulatorRegs;
  plan.estimatedEpilogueRegs = *estimatedEpilogueRegs;
  plan.estimatedABShared = *estimatedABShared;
  plan.estimatedEpilogueShared = *estimatedEpilogueShared;
  plan.staticSharedBudget = *staticSharedBudget;
  plan.dynamicSharedBudget = *dynamicSharedBudget;
  plan.chosenLandingTradeoff = chosenLandingTradeoff->str();
  plan.chosenBufferingTradeoff = chosenBufferingTradeoff->str();
  plan.reason = reason->str();
  if (failed(validateResourceClosurePlan(plan, op))) {
    op->emitError() << "malformed `tb.resource_closure_plan` attribute";
    return failure();
  }
  return plan;
}
