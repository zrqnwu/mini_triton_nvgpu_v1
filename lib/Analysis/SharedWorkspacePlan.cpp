#include "tb/Analysis/SharedWorkspacePlan.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::tb;

namespace {

static StringRef stringifySegmentKind(SharedWorkspaceSegmentKind kind) {
  switch (kind) {
  case SharedWorkspaceSegmentKind::MainloopAStageBuffer:
    return "mainloop_a_stage_buffer";
  case SharedWorkspaceSegmentKind::MainloopBStageBuffer:
    return "mainloop_b_stage_buffer";
  case SharedWorkspaceSegmentKind::EpilogueReorderScratch:
    return "epilogue_reorder_scratch";
  case SharedWorkspaceSegmentKind::SplitKReductionScratch:
    return "splitk_reduction_scratch";
  case SharedWorkspaceSegmentKind::PersistentStateScratch:
    return "persistent_state_scratch";
  }
  llvm_unreachable("unknown shared workspace segment kind");
}

static FailureOr<SharedWorkspaceSegmentKind>
parseSegmentKind(StringRef value, Operation *op) {
  if (value == "mainloop_a_stage_buffer")
    return SharedWorkspaceSegmentKind::MainloopAStageBuffer;
  if (value == "mainloop_b_stage_buffer")
    return SharedWorkspaceSegmentKind::MainloopBStageBuffer;
  if (value == "epilogue_reorder_scratch")
    return SharedWorkspaceSegmentKind::EpilogueReorderScratch;
  if (value == "splitk_reduction_scratch")
    return SharedWorkspaceSegmentKind::SplitKReductionScratch;
  if (value == "persistent_state_scratch")
    return SharedWorkspaceSegmentKind::PersistentStateScratch;
  op->emitError() << "unknown shared workspace segment kind `" << value << "`";
  return failure();
}

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

static FailureOr<bool> readBoolField(DictionaryAttr dict, StringRef name,
                                     Operation *op) {
  auto attr = dyn_cast_or_null<BoolAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing bool field `" << name << "`";
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

static int64_t product(ArrayRef<int64_t> shape) {
  int64_t result = 1;
  for (int64_t dim : shape)
    result *= dim;
  return result;
}

static int64_t alignTo(int64_t value, int64_t alignment) {
  alignment = std::max<int64_t>(alignment, 1);
  int64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

static DictionaryAttr buildSegmentAttr(Builder &builder,
                                       const SharedWorkspaceSegment &segment) {
  NamedAttrList attrs;
  attrs.set("kind", builder.getStringAttr(stringifySegmentKind(segment.kind)));
  attrs.set("name", builder.getStringAttr(segment.name));
  attrs.set("byte_offset", builder.getI64IntegerAttr(segment.byteOffset));
  attrs.set("byte_size", builder.getI64IntegerAttr(segment.byteSize));
  attrs.set("byte_alignment", builder.getI64IntegerAttr(segment.byteAlignment));
  attrs.set("logical_rows", builder.getI64IntegerAttr(segment.logicalRows));
  attrs.set("logical_cols", builder.getI64IntegerAttr(segment.logicalCols));
  attrs.set("stage_count", builder.getI64IntegerAttr(segment.stageCount));
  attrs.set("slot_count", builder.getI64IntegerAttr(segment.slotCount));
  attrs.set("warp_replicas", builder.getI64IntegerAttr(segment.warpReplicas));
  attrs.set("lifetime_begin", builder.getI64IntegerAttr(segment.lifetimeBegin));
  attrs.set("lifetime_end", builder.getI64IntegerAttr(segment.lifetimeEnd));
  attrs.set("alias_allowed", builder.getBoolAttr(segment.aliasAllowed));
  attrs.set("alias_set", builder.getI64IntegerAttr(segment.aliasSet));
  attrs.set("producer", builder.getStringAttr(segment.producer));
  attrs.set("consumer", builder.getStringAttr(segment.consumer));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<SharedWorkspaceSegment> parseSegmentAttr(DictionaryAttr dict,
                                                          Operation *op) {
  SharedWorkspaceSegment segment;
  auto kind = readStringField(dict, "kind", op);
  auto name = readStringField(dict, "name", op);
  auto byteOffset = readI64Field(dict, "byte_offset", op);
  auto byteSize = readI64Field(dict, "byte_size", op);
  auto byteAlignment = readI64Field(dict, "byte_alignment", op);
  auto logicalRows = readI64Field(dict, "logical_rows", op);
  auto logicalCols = readI64Field(dict, "logical_cols", op);
  auto stageCount = readI64Field(dict, "stage_count", op);
  auto slotCount = readI64Field(dict, "slot_count", op);
  auto warpReplicas = readI64Field(dict, "warp_replicas", op);
  auto lifetimeBegin = readI64Field(dict, "lifetime_begin", op);
  auto lifetimeEnd = readI64Field(dict, "lifetime_end", op);
  auto aliasAllowed = readBoolField(dict, "alias_allowed", op);
  auto aliasSet = readI64Field(dict, "alias_set", op);
  auto producer = readStringField(dict, "producer", op);
  auto consumer = readStringField(dict, "consumer", op);
  if (failed(kind) || failed(name) || failed(byteOffset) || failed(byteSize) ||
      failed(byteAlignment) || failed(logicalRows) || failed(logicalCols) ||
      failed(stageCount) || failed(slotCount) || failed(warpReplicas) ||
      failed(lifetimeBegin) || failed(lifetimeEnd) || failed(aliasAllowed) ||
      failed(aliasSet) || failed(producer) || failed(consumer)) {
    return failure();
  }
  auto parsedKind = parseSegmentKind(*kind, op);
  if (failed(parsedKind))
    return failure();
  segment.kind = *parsedKind;
  segment.name = name->str();
  segment.byteOffset = *byteOffset;
  segment.byteSize = *byteSize;
  segment.byteAlignment = *byteAlignment;
  segment.logicalRows = *logicalRows;
  segment.logicalCols = *logicalCols;
  segment.stageCount = *stageCount;
  segment.slotCount = *slotCount;
  segment.warpReplicas = *warpReplicas;
  segment.lifetimeBegin = *lifetimeBegin;
  segment.lifetimeEnd = *lifetimeEnd;
  segment.aliasAllowed = *aliasAllowed;
  segment.aliasSet = *aliasSet;
  segment.producer = producer->str();
  segment.consumer = consumer->str();
  return segment;
}

static LogicalResult validateSharedWorkspacePlan(const SharedWorkspacePlan &plan,
                                                 Operation *op) {
  if (plan.contractModel.empty() || plan.totalBytes <= 0 || plan.peakBytes <= 0 ||
      plan.aliasSavedBytes < 0 || plan.selectedPolicy.empty() ||
      plan.reason.empty() || plan.segments.empty()) {
    return op->emitError()
           << "shared workspace plan must carry explicit non-empty owner truth";
  }
  llvm::StringSet<> seenNames;
  for (const SharedWorkspaceSegment &segment : plan.segments) {
    if (segment.name.empty() || segment.byteSize <= 0 ||
        segment.byteAlignment <= 0 || segment.lifetimeEnd < segment.lifetimeBegin ||
        segment.producer.empty() || segment.consumer.empty()) {
      return op->emitError()
             << "shared workspace segment must carry positive geometry and "
                "explicit ownership";
    }
    if (!seenNames.insert(segment.name).second) {
      return op->emitError()
             << "shared workspace segment names must stay unique";
    }
  }
  return success();
}

} // namespace

FailureOr<SharedWorkspacePlan>
mlir::tb::deriveSharedWorkspacePlan(const KernelConfig &config,
                                    const TargetInfo &target,
                                    const EncodingPlan &encodings,
                                    const TransportPlan &transport,
                                    const EpilogueReorderPlan &reorder,
                                    const ReductionPlan &reduction,
                                    const PersistentWorkPlan &persistentWork,
                                    Operation *op) {
  SharedWorkspacePlan plan;
  plan.contractModel = "cta_shared_workspace_v1";
  plan.selectedPolicy = "single_cta_workspace_alias_by_lifetime";

  const int64_t aAlign =
      std::max<int64_t>(transport.operandA.asyncVectorBytes, 16);
  const int64_t bAlign =
      std::max<int64_t>(transport.operandB.asyncVectorBytes, 16);
  const int64_t cAlign = 16;
  const int64_t pAlign = 16;
  const int64_t requestedStages = std::max<int64_t>(config.requestedStages, 1);

  int64_t aBytes =
      requestedStages * product(encodings.aSharedSpec.allocShape) *
      getScalarByteWidth(config.aScalar);
  int64_t bBytes =
      requestedStages * product(encodings.bSharedSpec.allocShape) *
      getScalarByteWidth(config.bScalar);
  int64_t reorderBytes =
      reorder.kind == EpilogueReorderKind::None
          ? 0
          : config.numWarps * reorder.liveSlots * reorder.sharedTileRows *
                reorder.sharedTileCols * getScalarByteWidth(config.cScalar);
  int64_t reductionBytes =
      reduction.requiresInterProgramReduction &&
              reduction.scratchSpace == ReductionScratchSpace::Shared
          ? reduction.scratchBytes
          : 0;
  int64_t persistentBytes =
      persistentWork.enabled
          ? std::max<int64_t>(persistentWork.maxTilesPerProgram, 1) * 4
          : 0;

  int64_t dedicatedPrefix = 0;
  if (persistentBytes > 0) {
    SharedWorkspaceSegment persistent;
    persistent.kind = SharedWorkspaceSegmentKind::PersistentStateScratch;
    persistent.name = "persistent_state_scratch";
    persistent.byteOffset = 0;
    persistent.byteSize = persistentBytes;
    persistent.byteAlignment = pAlign;
    persistent.stageCount = 1;
    persistent.slotCount = 1;
    persistent.warpReplicas = 1;
    persistent.lifetimeBegin = 0;
    persistent.lifetimeEnd = 100;
    persistent.aliasAllowed = false;
    persistent.aliasSet = -1;
    persistent.producer = "persistent_tile_loop";
    persistent.consumer = "persistent_tile_loop";
    plan.segments.push_back(std::move(persistent));
    dedicatedPrefix = alignTo(persistentBytes, pAlign);
  }

  int64_t aliasRegionStart = alignTo(dedicatedPrefix, std::max(aAlign, bAlign));
  int64_t mainloopCursor = aliasRegionStart;

  SharedWorkspaceSegment aSegment;
  aSegment.kind = SharedWorkspaceSegmentKind::MainloopAStageBuffer;
  aSegment.name = "mainloop_a_stage_buffer";
  aSegment.byteOffset = mainloopCursor;
  aSegment.byteSize = aBytes;
  aSegment.byteAlignment = aAlign;
  aSegment.logicalRows = requestedStages * encodings.aSharedSpec.allocShape[0];
  aSegment.logicalCols = encodings.aSharedSpec.allocShape[1];
  aSegment.stageCount = requestedStages;
  aSegment.slotCount = 1;
  aSegment.warpReplicas = 1;
  aSegment.lifetimeBegin = 10;
  aSegment.lifetimeEnd = 40;
  aSegment.aliasAllowed = true;
  aSegment.aliasSet = 0;
  aSegment.producer = "cp_async_operand_a";
  aSegment.consumer = "ldmatrix_operand_a";
  plan.segments.push_back(aSegment);

  mainloopCursor = alignTo(aSegment.byteOffset + aSegment.byteSize, bAlign);
  SharedWorkspaceSegment bSegment;
  bSegment.kind = SharedWorkspaceSegmentKind::MainloopBStageBuffer;
  bSegment.name = "mainloop_b_stage_buffer";
  bSegment.byteOffset = mainloopCursor;
  bSegment.byteSize = bBytes;
  bSegment.byteAlignment = bAlign;
  bSegment.logicalRows = requestedStages * encodings.bSharedSpec.allocShape[0];
  bSegment.logicalCols = encodings.bSharedSpec.allocShape[1];
  bSegment.stageCount = requestedStages;
  bSegment.slotCount = 1;
  bSegment.warpReplicas = 1;
  bSegment.lifetimeBegin = 10;
  bSegment.lifetimeEnd = 40;
  bSegment.aliasAllowed = true;
  bSegment.aliasSet = 0;
  bSegment.producer = "cp_async_operand_b";
  bSegment.consumer = "ldmatrix_operand_b";
  plan.segments.push_back(bSegment);

  int64_t aliasRegionExtent = bSegment.byteOffset + bSegment.byteSize - aliasRegionStart;
  int64_t aliasSavingsTotal =
      aSegment.byteSize + bSegment.byteSize + persistentBytes;

  auto appendReorderSegment = [&](StringRef name, int64_t begin, int64_t end,
                                  StringRef producer,
                                  StringRef consumer) {
    if (reorderBytes <= 0)
      return;
    SharedWorkspaceSegment segment;
    segment.kind = SharedWorkspaceSegmentKind::EpilogueReorderScratch;
    segment.name = name.str();
    segment.byteOffset = alignTo(aliasRegionStart, cAlign);
    segment.byteSize = reorderBytes;
    segment.byteAlignment = cAlign;
    segment.logicalRows = reorder.sharedTileRows;
    segment.logicalCols = reorder.sharedTileCols;
    segment.stageCount = 1;
    segment.slotCount = reorder.liveSlots;
    segment.warpReplicas = config.numWarps;
    segment.lifetimeBegin = begin;
    segment.lifetimeEnd = end;
    segment.aliasAllowed = true;
    segment.aliasSet = 0;
    segment.producer = producer.str();
    segment.consumer = consumer.str();
    aliasRegionExtent =
        std::max(aliasRegionExtent,
                 segment.byteOffset + segment.byteSize - aliasRegionStart);
    aliasSavingsTotal += segment.byteSize;
    plan.segments.push_back(std::move(segment));
  };

  if (reorder.reorderNeededForInit) {
    appendReorderSegment("epilogue_init_reorder_scratch", 0, 9,
                         "global_vector_init", "accumulator_fragment_init");
  }
  if (reorder.reorderNeededForStore) {
    appendReorderSegment("epilogue_store_reorder_scratch", 50, 69,
                         "accumulator_fragment_store", "global_vector_store");
  }

  if (reductionBytes > 0) {
    SharedWorkspaceSegment segment;
    segment.kind = SharedWorkspaceSegmentKind::SplitKReductionScratch;
    segment.name = "splitk_reduction_scratch";
    segment.byteOffset = alignTo(aliasRegionStart, cAlign);
    segment.byteSize = reductionBytes;
    segment.byteAlignment = cAlign;
    segment.logicalRows = reduction.scratchRows;
    segment.logicalCols = reduction.scratchCols;
    segment.stageCount = 1;
    segment.slotCount = 1;
    segment.warpReplicas = 1;
    segment.lifetimeBegin = 70;
    segment.lifetimeEnd = 89;
    segment.aliasAllowed = true;
    segment.aliasSet = 0;
    segment.producer = "splitk_partial_write";
    segment.consumer = "splitk_final_reduce";
    aliasRegionExtent = std::max(aliasRegionExtent,
                                 segment.byteOffset + segment.byteSize -
                                     aliasRegionStart);
    aliasSavingsTotal += segment.byteSize;
    plan.segments.push_back(std::move(segment));
  }

  plan.totalBytes = aliasRegionStart + aliasRegionExtent;
  plan.peakBytes = plan.totalBytes;
  plan.aliasSavedBytes = std::max<int64_t>(aliasSavingsTotal - plan.totalBytes, 0);
  plan.reason =
      reorder.kind == EpilogueReorderKind::None
          ? "shared owner is unified as one CTA workspace even when C stays "
            "register-direct, so future epilogue/reduction/persistent paths "
            "do not reintroduce per-role shared ownership"
          : "shared owner is unified as one CTA workspace and epilogue row "
            "reorder scratch aliases the mainloop region by lifetime, matching "
            "the Triton-style single-CTA shared workspace model";

  if (plan.totalBytes > target.maxSharedBytesPerCTA) {
    op->emitError() << "shared workspace exceeds target shared memory budget";
    return failure();
  }
  if (failed(validateSharedWorkspacePlan(plan, op)))
    return failure();
  return plan;
}

DictionaryAttr
mlir::tb::buildSharedWorkspacePlanAttr(Builder &builder,
                                       const SharedWorkspacePlan &plan) {
  NamedAttrList attrs;
  attrs.set("contract_model", builder.getStringAttr(plan.contractModel));
  attrs.set("total_bytes", builder.getI64IntegerAttr(plan.totalBytes));
  attrs.set("peak_bytes", builder.getI64IntegerAttr(plan.peakBytes));
  attrs.set("alias_saved_bytes", builder.getI64IntegerAttr(plan.aliasSavedBytes));
  attrs.set("selected_policy", builder.getStringAttr(plan.selectedPolicy));
  attrs.set("reason", builder.getStringAttr(plan.reason));
  SmallVector<Attribute> segments;
  segments.reserve(plan.segments.size());
  for (const SharedWorkspaceSegment &segment : plan.segments)
    segments.push_back(buildSegmentAttr(builder, segment));
  attrs.set("segments", builder.getArrayAttr(segments));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<SharedWorkspacePlan>
mlir::tb::parseSharedWorkspacePlanAttr(Operation *op) {
  auto root =
      dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.shared_workspace_plan"));
  if (!root) {
    op->emitError() << "missing `tb.shared_workspace_plan` attribute";
    return failure();
  }
  auto contractModel = readStringField(root, "contract_model", op);
  auto totalBytes = readI64Field(root, "total_bytes", op);
  auto peakBytes = readI64Field(root, "peak_bytes", op);
  auto aliasSavedBytes = readI64Field(root, "alias_saved_bytes", op);
  auto selectedPolicy = readStringField(root, "selected_policy", op);
  auto reason = readStringField(root, "reason", op);
  auto segments = dyn_cast_or_null<ArrayAttr>(root.get("segments"));
  if (failed(contractModel) || failed(totalBytes) || failed(peakBytes) ||
      failed(aliasSavedBytes) || failed(selectedPolicy) || failed(reason) ||
      !segments) {
    op->emitError() << "malformed `tb.shared_workspace_plan` attribute";
    return failure();
  }

  SharedWorkspacePlan plan;
  plan.contractModel = contractModel->str();
  plan.totalBytes = *totalBytes;
  plan.peakBytes = *peakBytes;
  plan.aliasSavedBytes = *aliasSavedBytes;
  plan.selectedPolicy = selectedPolicy->str();
  plan.reason = reason->str();
  plan.segments.reserve(segments.size());
  for (Attribute attr : segments) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict) {
      op->emitError() << "`segments` must contain dictionary entries";
      return failure();
    }
    auto segment = parseSegmentAttr(dict, op);
    if (failed(segment))
      return failure();
    plan.segments.push_back(std::move(*segment));
  }
  if (failed(validateSharedWorkspacePlan(plan, op))) {
    op->emitError() << "malformed `tb.shared_workspace_plan` attribute";
    return failure();
  }
  return plan;
}

FailureOr<const SharedWorkspaceSegment *>
mlir::tb::findSharedWorkspaceSegment(const SharedWorkspacePlan &plan,
                                     SharedWorkspaceSegmentKind kind,
                                     StringRef name, Operation *op) {
  const SharedWorkspaceSegment *match = nullptr;
  for (const SharedWorkspaceSegment &segment : plan.segments) {
    if (segment.kind != kind)
      continue;
    if (!name.empty() && segment.name != name)
      continue;
    if (match) {
      op->emitError()
          << "shared workspace segment lookup matched multiple entries for kind `"
          << stringifySegmentKind(kind) << "`";
      return failure();
    }
    match = &segment;
  }
  if (!match) {
    op->emitError() << "missing shared workspace segment for kind `"
                    << stringifySegmentKind(kind) << "`";
    return failure();
  }
  return match;
}
