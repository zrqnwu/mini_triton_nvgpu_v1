#include "tb/Analysis/AsyncPlan.h"

#include "tb/Analysis/KernelConfig.h"
#include "tb/IR/TBOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::tb;

namespace {

struct Position {
  int64_t stage = -1;
  int64_t cluster = -1;
  int64_t order = -1;
};

static bool hasBoundaryOnM(const KernelConfig &config) {
  return (config.problemM % config.blockM) != 0;
}

static bool hasBoundaryOnN(const KernelConfig &config) {
  return (config.problemN % config.blockN) != 0;
}

static bool hasBoundaryOnK(const KernelConfig &config) {
  return (config.problemK % config.blockK) != 0;
}

static DenseI64ArrayAttr buildI64ArrayAttr(Builder &builder,
                                           ArrayRef<int64_t> values) {
  return builder.getDenseI64ArrayAttr(values);
}

static SmallVector<int64_t> parseI64Array(DenseI64ArrayAttr attr) {
  return SmallVector<int64_t>(attr.asArrayRef().begin(),
                              attr.asArrayRef().end());
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

static FailureOr<bool> readBoolField(DictionaryAttr dict, StringRef name,
                                     Operation *op) {
  auto attr = dyn_cast_or_null<BoolAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing bool field `" << name << "`";
    return failure();
  }
  return attr.getValue();
}

static bool readOptionalBoolField(DictionaryAttr dict, StringRef name,
                                  bool defaultValue) {
  auto attr = dyn_cast_or_null<BoolAttr>(dict.get(name));
  return attr ? attr.getValue() : defaultValue;
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

static std::string readOptionalStringField(DictionaryAttr dict, StringRef name,
                                           StringRef defaultValue) {
  auto attr = dyn_cast_or_null<StringAttr>(dict.get(name));
  return attr ? attr.getValue().str() : defaultValue.str();
}

static FailureOr<DenseI64ArrayAttr>
readDenseI64ArrayField(DictionaryAttr dict, StringRef name, Operation *op) {
  auto attr = dyn_cast_or_null<DenseI64ArrayAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing dense i64 array field `" << name << "`";
    return failure();
  }
  return attr;
}

static StringRef stringifyProducerKind(AsyncProducerKind kind) {
  switch (kind) {
  case AsyncProducerKind::CpAsync:
    return "cp_async";
  case AsyncProducerKind::TMA:
    return "tma";
  case AsyncProducerKind::BulkCopy:
    return "bulk_copy";
  case AsyncProducerKind::SyncCopyFallback:
    return "sync_copy_fallback";
  }
  llvm_unreachable("unknown async producer kind");
}

static FailureOr<AsyncProducerKind> parseProducerKind(StringRef value,
                                                      Operation *op) {
  if (value == "cp_async")
    return AsyncProducerKind::CpAsync;
  if (value == "tma")
    return AsyncProducerKind::TMA;
  if (value == "bulk_copy")
    return AsyncProducerKind::BulkCopy;
  if (value == "sync_copy_fallback")
    return AsyncProducerKind::SyncCopyFallback;
  op->emitError() << "unknown async producer kind `" << value << "`";
  return failure();
}

static StringRef stringifyBarrierKind(AsyncBarrierKind kind) {
  switch (kind) {
  case AsyncBarrierKind::AsyncGroup:
    return "async_group";
  case AsyncBarrierKind::CTA:
    return "cta_barrier";
  case AsyncBarrierKind::MBarrier:
    return "mbarrier";
  case AsyncBarrierKind::None:
    return "none";
  }
  llvm_unreachable("unknown async barrier kind");
}

static FailureOr<AsyncBarrierKind> parseBarrierKind(StringRef value,
                                                    Operation *op) {
  if (value == "async_group")
    return AsyncBarrierKind::AsyncGroup;
  if (value == "cta_barrier")
    return AsyncBarrierKind::CTA;
  if (value == "mbarrier")
    return AsyncBarrierKind::MBarrier;
  if (value == "none")
    return AsyncBarrierKind::None;
  op->emitError() << "unknown async barrier kind `" << value << "`";
  return failure();
}

static StringRef stringifyCachePolicy(AsyncCachePolicy policy) {
  switch (policy) {
  case AsyncCachePolicy::Default:
    return "default";
  case AsyncCachePolicy::BypassL1:
    return "bypass_l1";
  case AsyncCachePolicy::CacheAll:
    return "cache_all";
  }
  llvm_unreachable("unknown async cache policy");
}

static FailureOr<AsyncCachePolicy> parseCachePolicy(StringRef value,
                                                    Operation *op) {
  if (value == "default")
    return AsyncCachePolicy::Default;
  if (value == "bypass_l1")
    return AsyncCachePolicy::BypassL1;
  if (value == "cache_all")
    return AsyncCachePolicy::CacheAll;
  op->emitError() << "unknown async cache policy `" << value << "`";
  return failure();
}

static DictionaryAttr buildProducerAttr(Builder &builder,
                                        const AsyncProducer &producer) {
  NamedAttrList attrs;
  attrs.set("op_id", builder.getI64IntegerAttr(producer.opId));
  attrs.set("kind",
            builder.getStringAttr(stringifyProducerKind(producer.kind)));
  attrs.set("value_id", builder.getI64IntegerAttr(producer.valueId));
  attrs.set("src_view", builder.getI64IntegerAttr(producer.srcView));
  attrs.set("dst_view", builder.getI64IntegerAttr(producer.dstView));
  attrs.set("src_offsets", buildI64ArrayAttr(builder, producer.srcOffsets));
  attrs.set("group_id", builder.getI64IntegerAttr(producer.groupId));
  attrs.set("vec_bytes", builder.getI64IntegerAttr(producer.vecBytes));
  attrs.set("transaction_bytes",
            builder.getI64IntegerAttr(producer.transactionBytes));
  attrs.set("zero_fill", builder.getBoolAttr(producer.zeroFill));
  attrs.set("predicated", builder.getBoolAttr(producer.predicated));
  attrs.set("bypass_l1", builder.getBoolAttr(producer.bypassL1));
  attrs.set("barrier_kind",
            builder.getStringAttr(stringifyBarrierKind(producer.barrierKind)));
  attrs.set("cache_policy",
            builder.getStringAttr(stringifyCachePolicy(producer.cachePolicy)));
  attrs.set("legal", builder.getBoolAttr(producer.legal));
  attrs.set("reason", builder.getStringAttr(producer.reason));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<AsyncProducer> parseProducerAttr(DictionaryAttr dict,
                                                  Operation *op) {
  AsyncProducer producer;
  auto opId = readI64Field(dict, "op_id", op);
  auto kind = readStringField(dict, "kind", op);
  auto valueId = readI64Field(dict, "value_id", op);
  auto srcView = readI64Field(dict, "src_view", op);
  auto dstView = readI64Field(dict, "dst_view", op);
  auto srcOffsets = readDenseI64ArrayField(dict, "src_offsets", op);
  auto groupId = readI64Field(dict, "group_id", op);
  auto vecBytes = readI64Field(dict, "vec_bytes", op);
  auto legal = readBoolField(dict, "legal", op);
  auto reason = readStringField(dict, "reason", op);
  if (failed(opId) || failed(kind) || failed(valueId) || failed(srcView) ||
      failed(dstView) || failed(srcOffsets) || failed(groupId) ||
      failed(vecBytes) || failed(legal) || failed(reason)) {
    return failure();
  }
  auto parsedKind = parseProducerKind(*kind, op);
  if (failed(parsedKind))
    return failure();
  producer.opId = *opId;
  producer.kind = *parsedKind;
  producer.valueId = *valueId;
  producer.srcView = *srcView;
  producer.dstView = *dstView;
  producer.srcOffsets = parseI64Array(*srcOffsets);
  producer.groupId = *groupId;
  producer.vecBytes = *vecBytes;
  producer.transactionBytes =
      dyn_cast_or_null<IntegerAttr>(dict.get("transaction_bytes"))
          ? cast<IntegerAttr>(dict.get("transaction_bytes")).getInt()
          : producer.vecBytes;
  producer.zeroFill = readOptionalBoolField(dict, "zero_fill", false);
  producer.predicated = readOptionalBoolField(dict, "predicated", false);
  producer.bypassL1 = readOptionalBoolField(dict, "bypass_l1", false);
  auto parsedBarrierKind = parseBarrierKind(
      readOptionalStringField(dict, "barrier_kind", "async_group"), op);
  auto parsedCachePolicy = parseCachePolicy(
      readOptionalStringField(dict, "cache_policy", "default"), op);
  if (failed(parsedBarrierKind) || failed(parsedCachePolicy))
    return failure();
  producer.barrierKind = *parsedBarrierKind;
  producer.cachePolicy = *parsedCachePolicy;
  producer.legal = *legal;
  producer.reason = reason->str();
  return producer;
}

static DictionaryAttr buildGroupAttr(Builder &builder,
                                     const AsyncGroup &group) {
  NamedAttrList attrs;
  attrs.set("id", builder.getI64IntegerAttr(group.id));
  attrs.set("producers", buildI64ArrayAttr(builder, group.producers));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<AsyncGroup> parseGroupAttr(DictionaryAttr dict,
                                            Operation *op) {
  AsyncGroup group;
  auto id = readI64Field(dict, "id", op);
  auto producers = readDenseI64ArrayField(dict, "producers", op);
  if (failed(id) || failed(producers))
    return failure();
  group.id = *id;
  group.producers = parseI64Array(*producers);
  return group;
}

static DictionaryAttr buildWaitAttr(Builder &builder, const WaitInfo &wait) {
  NamedAttrList attrs;
  attrs.set("group_id", builder.getI64IntegerAttr(wait.groupId));
  attrs.set("before_op_id", builder.getI64IntegerAttr(wait.beforeOpId));
  attrs.set("required_stage", builder.getI64IntegerAttr(wait.requiredStage));
  attrs.set("required_cluster",
            builder.getI64IntegerAttr(wait.requiredCluster));
  attrs.set("required_order", builder.getI64IntegerAttr(wait.requiredOrder));
  attrs.set("needs_barrier", builder.getBoolAttr(wait.needsBarrier));
  attrs.set("reason", builder.getStringAttr(wait.reason));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<WaitInfo> parseWaitAttr(DictionaryAttr dict, Operation *op) {
  WaitInfo wait;
  auto groupId = readI64Field(dict, "group_id", op);
  auto beforeOpId = readI64Field(dict, "before_op_id", op);
  auto requiredStage = readI64Field(dict, "required_stage", op);
  auto requiredCluster = readI64Field(dict, "required_cluster", op);
  auto requiredOrder = readI64Field(dict, "required_order", op);
  auto needsBarrier = readBoolField(dict, "needs_barrier", op);
  auto reason = readStringField(dict, "reason", op);
  if (failed(groupId) || failed(beforeOpId) || failed(requiredStage) ||
      failed(requiredCluster) || failed(requiredOrder) ||
      failed(needsBarrier) || failed(reason)) {
    return failure();
  }
  wait.groupId = *groupId;
  wait.beforeOpId = *beforeOpId;
  wait.requiredStage = *requiredStage;
  wait.requiredCluster = *requiredCluster;
  wait.requiredOrder = *requiredOrder;
  wait.needsBarrier = *needsBarrier;
  wait.reason = reason->str();
  return wait;
}

static DictionaryAttr buildReuseFenceAttr(Builder &builder,
                                          const ReuseFence &fence) {
  NamedAttrList attrs;
  attrs.set("view_id", builder.getI64IntegerAttr(fence.viewId));
  attrs.set("backing", builder.getI64IntegerAttr(fence.backing));
  attrs.set("retiring_value_id",
            builder.getI64IntegerAttr(fence.retiringValueId));
  attrs.set("acquiring_value_id",
            builder.getI64IntegerAttr(fence.acquiringValueId));
  attrs.set("after_op_id", builder.getI64IntegerAttr(fence.afterOpId));
  attrs.set("required_after_stage",
            builder.getI64IntegerAttr(fence.requiredAfterStage));
  attrs.set("required_after_cluster",
            builder.getI64IntegerAttr(fence.requiredAfterCluster));
  attrs.set("required_after_order",
            builder.getI64IntegerAttr(fence.requiredAfterOrder));
  attrs.set("reason", builder.getStringAttr(fence.reason));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<ReuseFence> parseReuseFenceAttr(DictionaryAttr dict,
                                                 Operation *op) {
  ReuseFence fence;
  auto viewId = readI64Field(dict, "view_id", op);
  auto backing = readI64Field(dict, "backing", op);
  auto retiringValueId = readI64Field(dict, "retiring_value_id", op);
  auto acquiringValueId = readI64Field(dict, "acquiring_value_id", op);
  auto afterOpId = readI64Field(dict, "after_op_id", op);
  auto requiredAfterStage = readI64Field(dict, "required_after_stage", op);
  auto requiredAfterCluster = readI64Field(dict, "required_after_cluster", op);
  auto requiredAfterOrder = readI64Field(dict, "required_after_order", op);
  auto reason = readStringField(dict, "reason", op);
  if (failed(viewId) || failed(backing) || failed(retiringValueId) ||
      failed(acquiringValueId) || failed(afterOpId) ||
      failed(requiredAfterStage) || failed(requiredAfterCluster) ||
      failed(requiredAfterOrder) || failed(reason)) {
    return failure();
  }
  fence.viewId = *viewId;
  fence.backing = *backing;
  fence.retiringValueId = *retiringValueId;
  fence.acquiringValueId = *acquiringValueId;
  fence.afterOpId = *afterOpId;
  fence.requiredAfterStage = *requiredAfterStage;
  fence.requiredAfterCluster = *requiredAfterCluster;
  fence.requiredAfterOrder = *requiredAfterOrder;
  fence.reason = reason->str();
  return fence;
}

template <typename RangeT, typename BuildFn>
static ArrayAttr buildDictArrayAttr(Builder &builder, const RangeT &values,
                                    BuildFn buildElement) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (const auto &value : values)
    attrs.push_back(buildElement(builder, value));
  return builder.getArrayAttr(attrs);
}

template <typename T, typename ParseFn>
static FailureOr<SmallVector<T>>
parseDictArrayAttr(Attribute attr, StringRef name, Operation *op,
                   ParseFn parseElement) {
  auto array = dyn_cast_or_null<ArrayAttr>(attr);
  if (!array) {
    op->emitError() << "missing array field `" << name << "`";
    return failure();
  }
  SmallVector<T> values;
  values.reserve(array.size());
  for (Attribute element : array) {
    auto dict = dyn_cast<DictionaryAttr>(element);
    if (!dict) {
      op->emitError() << "field `" << name
                      << "` must contain only dictionary elements";
      return failure();
    }
    auto parsed = parseElement(dict, op);
    if (failed(parsed))
      return failure();
    values.push_back(std::move(*parsed));
  }
  return values;
}

static LogicalResult validateAsyncPlan(const AsyncPlan &plan, Operation *op) {
  DenseSet<int64_t> groupIds;
  DenseSet<int64_t> producerOpIds;
  for (const AsyncProducer &producer : plan.producers) {
    if (producer.dstView < 0)
      return op->emitError() << "async producer for value " << producer.valueId
                             << " must reference a concrete dst_view";
    if (producer.kind == AsyncProducerKind::CpAsync && producer.srcView < 0)
      return op->emitError()
             << "cp.async producer for value " << producer.valueId
             << " must reference a concrete src_view";
    if (producer.kind == AsyncProducerKind::CpAsync &&
        producer.srcOffsets.size() != 2) {
      return op->emitError()
             << "cp.async producer for value " << producer.valueId
             << " must carry rank-2 explicit src_offsets";
    }
    if (producer.opId >= 0 && !producerOpIds.insert(producer.opId).second &&
        producer.kind == AsyncProducerKind::CpAsync) {
      return op->emitError()
             << "duplicate cp.async producer op " << producer.opId;
    }
  }
  for (const AsyncGroup &group : plan.groups) {
    if (!groupIds.insert(group.id).second)
      return op->emitError() << "duplicate async group id " << group.id;
  }
  for (const ReuseFence &fence : plan.reuseFences) {
    if (fence.viewId < 0 || fence.backing < 0)
      return op->emitError()
             << "reuse fence must reference concrete view/backing ownership";
  }
  return success();
}

static bool isEarlierOrEqual(Position lhs, Position rhs) {
  if (lhs.stage != rhs.stage)
    return lhs.stage < rhs.stage;
  if (lhs.cluster != rhs.cluster)
    return lhs.cluster < rhs.cluster;
  return lhs.order <= rhs.order;
}

static bool isStrictlyEarlier(Position lhs, Position rhs) {
  if (lhs.stage != rhs.stage)
    return lhs.stage < rhs.stage;
  if (lhs.cluster != rhs.cluster)
    return lhs.cluster < rhs.cluster;
  return lhs.order < rhs.order;
}

static int64_t findIterationCoord(ArrayRef<PipelineOp::IterationCoord> coords,
                                  StringRef axis) {
  auto it = llvm::find_if(coords, [&](const PipelineOp::IterationCoord &coord) {
    return coord.axis == axis;
  });
  return it == coords.end() ? -1 : it->value;
}

static Position toPosition(const PipelinePlacement &placement) {
  return Position{placement.stage, placement.cluster, placement.order};
}

static FailureOr<std::pair<int64_t, Position>> findFirstUseFrontier(
    const ValueState &value,
    const DenseMap<int64_t, const PipelinePlacement *> &placementByOp,
    Operation *op) {
  Position firstUse = {-1, -1, -1};
  int64_t beforeOpId = -1;
  bool firstSet = false;
  for (int64_t userId : value.users) {
    auto userPlacementIt = placementByOp.find(userId);
    if (userPlacementIt == placementByOp.end()) {
      op->emitError() << "missing placement for async consumer op " << userId;
      return failure();
    }
    Position userPosition = toPosition(*userPlacementIt->second);
    if (!firstSet || isEarlierOrEqual(userPosition, firstUse)) {
      firstUse = userPosition;
      beforeOpId = userId;
      firstSet = true;
    }
  }
  if (!firstSet) {
    op->emitError() << "pipeline value " << value.id
                    << " must have at least one scheduled consumer";
    return failure();
  }
  return std::make_pair(beforeOpId, firstUse);
}

static FailureOr<Position> findRetireFrontier(
    const ValueState &value,
    const DenseMap<int64_t, const PipelinePlacement *> &placementByOp,
    Operation *op) {
  Position retire = {-1, -1, -1};
  bool retireSet = false;
  for (int64_t userId : value.users) {
    auto userPlacementIt = placementByOp.find(userId);
    if (userPlacementIt == placementByOp.end()) {
      op->emitError() << "missing placement for retiring consumer op "
                      << userId;
      return failure();
    }
    Position userPosition = toPosition(*userPlacementIt->second);
    if (!retireSet || isEarlierOrEqual(retire, userPosition)) {
      retire = userPosition;
      retireSet = true;
    }
  }
  if (retireSet)
    return retire;

  auto defPlacementIt = placementByOp.find(value.definingOp);
  if (defPlacementIt == placementByOp.end()) {
    op->emitError() << "missing placement for retiring producer op "
                    << value.definingOp;
    return failure();
  }
  return toPosition(*defPlacementIt->second);
}

static FailureOr<const ValueState *>
findStageOwnedValue(const StageBufferUse &use,
                    const DenseMap<int64_t, const ValueState *> &valuesById,
                    Operation *op) {
  const ValueState *matched = nullptr;
  for (const auto &[valueId, value] : valuesById) {
    (void)valueId;
    if (value->definingOp != use.producerOp || value->ownerView != use.viewId)
      continue;
    if (matched) {
      op->emitError() << "stage-owned buffer view " << use.viewId
                      << " has multiple values for producer op "
                      << use.producerOp;
      return failure();
    }
    matched = value;
  }
  if (!matched) {
    op->emitError() << "stage-owned buffer view " << use.viewId
                    << " is missing a matching pipeline value";
    return failure();
  }
  return matched;
}

} // namespace

FailureOr<AsyncPlan> mlir::tb::deriveAsyncPlan(const BufferModel &model,
                                               const TransportPlan &transport,
                                               const LatencyPlan &latencyPlan,
                                               const PipelinePlan &pipelinePlan,
                                               Operation *op) {
  auto matmul = dyn_cast<MatmulOp>(op);
  if (!matmul) {
    op->emitError() << "async plan can only be derived from tb.matmul";
    return failure();
  }
  auto config = getKernelConfig(matmul);
  if (failed(config))
    return failure();

  DenseMap<int64_t, const PipelinePlacement *> placementByOp;
  DenseMap<int64_t, const OpLatencyInfo *> latencyByOp;
  DenseMap<int64_t, const ValueState *> valuesById;
  DenseMap<int64_t, const PipelineOp *> opsById;
  DenseMap<int64_t, const BufferView *> viewsById;
  DenseMap<int64_t, const BufferBacking *> backingsById;
  for (const PipelinePlacement &placement : pipelinePlan.placements)
    placementByOp[placement.opId] = &placement;
  for (const OpLatencyInfo &entry : latencyPlan.ops)
    latencyByOp[entry.opId] = &entry;
  for (const ValueState &value : model.values)
    valuesById[value.id] = &value;
  for (const PipelineOp &pipelineOp : model.ops)
    opsById[pipelineOp.id] = &pipelineOp;
  for (const BufferView &view : model.views)
    viewsById[view.id] = &view;
  for (const BufferBacking &backing : model.backings)
    backingsById[backing.id] = &backing;

  auto findGlobalFullView = [&](BufferRole role,
                                StringRef contractName) -> FailureOr<int64_t> {
    auto view =
        findUniqueFullBufferView(model, role, MemorySpace::Global, op,
                                 contractName);
    if (failed(view))
      return failure();
    return (*view)->id;
  };

  struct AsyncCopyInfo {
    bool asyncEligible = false;
    int64_t vecBytes = 0;
    int64_t transactionBytes = 0;
    bool bypassL1 = false;
    AsyncCachePolicy cachePolicy = AsyncCachePolicy::Default;
  };

  auto getAsyncCopyInfo =
      [&](int64_t dstViewId) -> FailureOr<AsyncCopyInfo> {
    auto viewIt = viewsById.find(dstViewId);
    if (viewIt == viewsById.end()) {
      op->emitError() << "async producer references unknown dst_view "
                      << dstViewId;
      return failure();
    }
    auto backingIt = backingsById.find(viewIt->second->backing);
    if (backingIt == backingsById.end()) {
      op->emitError() << "async producer view " << dstViewId
                      << " references unknown backing";
      return failure();
    }
    auto descType = getMemDescType(backingIt->second->descType, op,
                                   backingIt->second->debugName);
    if (failed(descType))
      return failure();
    if (isa<SharedEncodingAttr>(descType->getEncoding())) {
      auto transportSpec = getTransportSpecForDstEncoding(
          transport, viewIt->second->encoding, op, backingIt->second->debugName);
      if (failed(transportSpec))
        return failure();
      if ((*transportSpec)->kind != "cp_async") {
        op->emitError() << "strict async mainline only accepts `cp_async` "
                           "transport kinds, but view "
                        << dstViewId << " carries `" << (*transportSpec)->kind
                        << "`";
        return failure();
      }
      AsyncCopyInfo info;
      info.asyncEligible = (*transportSpec)->asyncEligible;
      info.vecBytes = (*transportSpec)->asyncVectorBytes > 0
                          ? (*transportSpec)->asyncVectorBytes
                          : (*transportSpec)->vectorBytes;
      info.transactionBytes = (*transportSpec)->transactionBytes;
      info.bypassL1 = (*transportSpec)->bypassL1;
      auto cachePolicy =
          parseCachePolicy((*transportSpec)->cachePolicy, op);
      if (failed(cachePolicy))
        return failure();
      info.cachePolicy = *cachePolicy;
      return info;
    }
    op->emitError() << "strict async mainline only accepts shared dst encodings";
    return failure();
  };

  auto getProducerSrcOffsets =
      [&](const PipelineOp &producerOp, const BufferView &dstView)
      -> FailureOr<SmallVector<int64_t, 4>> {
    if (dstView.shape.size() < 2) {
      op->emitError() << "async producer dst_view " << dstView.id
                      << " must carry a rank-2 tile shape";
      return failure();
    }
    int64_t kGroup = findIterationCoord(producerOp.iterationCoords, "k_group");
    if (kGroup < 0) {
      op->emitError() << "async producer op " << producerOp.id
                      << " is missing a k_group coordinate";
      return failure();
    }

    SmallVector<int64_t, 4> srcOffsets(2, 0);
    switch (producerOp.kind) {
    case BufferOpKind::LoadA:
      srcOffsets[0] = 0;
      srcOffsets[1] = kGroup * dstView.shape[1];
      break;
    case BufferOpKind::LoadB:
      srcOffsets[0] = kGroup * dstView.shape[0];
      srcOffsets[1] = 0;
      break;
    default:
      op->emitError() << "async src offsets can only be derived for load_a/"
                         "load_b producers";
      return failure();
    }
    return srcOffsets;
  };

  auto globalAView = findGlobalFullView(BufferRole::OperandA,
                                        "async_global_operand_a_source");
  auto globalBView = findGlobalFullView(BufferRole::OperandB,
                                        "async_global_operand_b_source");
  if (failed(globalAView) || failed(globalBView))
    return failure();

  AsyncPlan plan;
  plan.producers.reserve(pipelinePlan.stageOwnedBuffers.size());
  plan.groups.reserve(pipelinePlan.stageOwnedBuffers.size());
  plan.waits.reserve(pipelinePlan.stageOwnedBuffers.size());
  plan.reuseFences.reserve(model.values.size());

  int64_t nextGroupId = 0;
  for (const StageBufferUse &use : pipelinePlan.stageOwnedBuffers) {
    auto producerOpIt = opsById.find(use.producerOp);
    if (producerOpIt == opsById.end()) {
      op->emitError() << "stage ownership references unknown producer op "
                      << use.producerOp;
      return failure();
    }
    const PipelineOp *producerOp = producerOpIt->second;
    if (producerOp->kind != BufferOpKind::LoadA &&
        producerOp->kind != BufferOpKind::LoadB) {
      op->emitError()
          << "stage-owned async producer must be defined by load_a/load_b";
      return failure();
    }

    auto value = findStageOwnedValue(use, valuesById, op);
    if (failed(value))
      return failure();

    auto dstViewIt = viewsById.find(use.viewId);
    if (dstViewIt == viewsById.end()) {
      op->emitError()
          << "stage-owned async producer references unknown dst_view "
          << use.viewId;
      return failure();
    }

    auto asyncInfo = getAsyncCopyInfo(use.viewId);
    if (failed(asyncInfo))
      return failure();
    auto latencyIt = latencyByOp.find(producerOp->id);
    if (latencyIt == latencyByOp.end()) {
      op->emitError() << "missing latency info for async producer op "
                      << producerOp->id;
      return failure();
    }
    bool asyncEligible = asyncInfo->asyncEligible && config->requestedStages > 1 &&
                         dstViewIt->second->kind == ViewKind::StageSlice &&
                         latencyIt->second->pipelineable;
    int64_t srcView = producerOp->kind == BufferOpKind::LoadA ? *globalAView
                                                              : *globalBView;
    int64_t producerIndex = static_cast<int64_t>(plan.producers.size());
    if (!asyncEligible) {
      op->emitError() << "strict async mainline does not allow synchronous "
                         "fallback for load op "
                      << producerOp->id;
      return failure();
    }

    auto srcOffsets = getProducerSrcOffsets(*producerOp, *dstViewIt->second);
    if (failed(srcOffsets))
      return failure();

    auto firstUse = findFirstUseFrontier(**value, placementByOp, op);
    if (failed(firstUse))
      return failure();
    if (use.firstConsumerOp >= 0 && use.firstConsumerOp != firstUse->first) {
      op->emitError() << "stage-owned buffer first-consumer frontier disagrees "
                         "with the scheduled first use for view "
                      << use.viewId;
      return failure();
    }

    int64_t groupId = nextGroupId++;
    AsyncProducer producer;
    producer.opId = producerOp->id;
    producer.kind = AsyncProducerKind::CpAsync;
    producer.valueId = (*value)->id;
    producer.srcView = srcView;
    producer.dstView = use.viewId;
    producer.srcOffsets = std::move(*srcOffsets);
    producer.groupId = groupId;
    producer.vecBytes = asyncInfo->vecBytes;
    producer.transactionBytes = asyncInfo->transactionBytes;
    bool needsBoundaryGuard =
        producerOp->kind == BufferOpKind::LoadA
            ? (hasBoundaryOnM(*config) || hasBoundaryOnK(*config))
            : (hasBoundaryOnK(*config) || hasBoundaryOnN(*config));
    producer.zeroFill = needsBoundaryGuard;
    producer.predicated = needsBoundaryGuard;
    producer.bypassL1 = asyncInfo->bypassL1;
    producer.barrierKind = AsyncBarrierKind::AsyncGroup;
    producer.cachePolicy = asyncInfo->cachePolicy;
    producer.legal = true;
    producer.reason = latencyIt->second->reason;
    plan.producers.push_back(std::move(producer));
    plan.groups.push_back({groupId, {producerIndex}});
    plan.waits.push_back(
        {groupId, firstUse->first, firstUse->second.stage,
         firstUse->second.cluster, firstUse->second.order,
         /*needsBarrier=*/true,
         "scheduled_first_use_frontier_of_stage_owned_async_value"});
  }

  DenseMap<int64_t, SmallVector<const ValueState *, 8>> valuesByOwnerView;
  for (const ValueState &value : model.values) {
    if (value.ownerView >= 0)
      valuesByOwnerView[value.ownerView].push_back(&value);
  }

  for (auto &[viewId, values] : valuesByOwnerView) {
    llvm::stable_sort(
        values, [&](const ValueState *lhs, const ValueState *rhs) {
          auto lhsPlacement = placementByOp.find(lhs->definingOp);
          auto rhsPlacement = placementByOp.find(rhs->definingOp);
          if (lhsPlacement == placementByOp.end() ||
              rhsPlacement == placementByOp.end())
            return lhs->definingOp < rhs->definingOp;
          return isEarlierOrEqual(toPosition(*lhsPlacement->second),
                                  toPosition(*rhsPlacement->second));
        });

    for (size_t i = 0, e = values.size(); i + 1 < e; ++i) {
      const ValueState *prevValue = values[i];
      const ValueState *nextValue = values[i + 1];
      auto nextBeginIt = placementByOp.find(nextValue->definingOp);
      if (nextBeginIt == placementByOp.end()) {
        op->emitError() << "missing placement for acquiring op "
                        << nextValue->definingOp;
        return failure();
      }

      auto retire = findRetireFrontier(*prevValue, placementByOp, op);
      if (failed(retire))
        return failure();
      Position nextBegin = toPosition(*nextBeginIt->second);
      bool sameOpCarry =
          llvm::is_contained(prevValue->users, nextValue->definingOp);
      bool legalReuse = sameOpCarry ? isEarlierOrEqual(*retire, nextBegin)
                                    : isStrictlyEarlier(*retire, nextBegin);
      if (!legalReuse) {
        op->emitError() << "view " << viewId << " reuse is illegal: value "
                        << nextValue->id << " begins before value "
                        << prevValue->id << " retires";
        return failure();
      }

      auto ownerViewIt = viewsById.find(viewId);
      if (ownerViewIt == viewsById.end()) {
        op->emitError() << "missing owner view " << viewId
                        << " for reuse fence";
        return failure();
      }
      plan.reuseFences.push_back(
          {viewId, ownerViewIt->second->backing, prevValue->id, nextValue->id,
           nextValue->definingOp, retire->stage, retire->cluster, retire->order,
           sameOpCarry ? "view_reuse_via_same_op_carry"
                       : "view_reuse_after_retire_frontier"});
    }
  }

  if (failed(validateAsyncPlan(plan, op)))
    return failure();
  return plan;
}

DictionaryAttr mlir::tb::buildAsyncPlanAttr(Builder &builder,
                                            const AsyncPlan &plan) {
  NamedAttrList attrs;
  attrs.set("producers",
            buildDictArrayAttr(builder, plan.producers, buildProducerAttr));
  attrs.set("groups", buildDictArrayAttr(builder, plan.groups, buildGroupAttr));
  attrs.set("waits", buildDictArrayAttr(builder, plan.waits, buildWaitAttr));
  attrs.set("reuse_fences",
            buildDictArrayAttr(builder, plan.reuseFences, buildReuseFenceAttr));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<AsyncPlan> mlir::tb::parseAsyncPlanAttr(Operation *op) {
  auto root = dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.async_plan"));
  if (!root) {
    op->emitError() << "missing `tb.async_plan` attribute";
    return failure();
  }

  auto producers = parseDictArrayAttr<AsyncProducer>(
      root.get("producers"), "producers", op, parseProducerAttr);
  auto groups = parseDictArrayAttr<AsyncGroup>(root.get("groups"), "groups", op,
                                               parseGroupAttr);
  auto waits = parseDictArrayAttr<WaitInfo>(root.get("waits"), "waits", op,
                                            parseWaitAttr);
  auto reuseFences = parseDictArrayAttr<ReuseFence>(
      root.get("reuse_fences"), "reuse_fences", op, parseReuseFenceAttr);
  if (failed(producers) || failed(groups) || failed(waits) ||
      failed(reuseFences)) {
    op->emitError() << "malformed `tb.async_plan` attribute";
    return failure();
  }

  AsyncPlan plan;
  plan.producers = std::move(*producers);
  plan.groups = std::move(*groups);
  plan.waits = std::move(*waits);
  plan.reuseFences = std::move(*reuseFences);
  if (failed(validateAsyncPlan(plan, op))) {
    op->emitError() << "malformed `tb.async_plan` attribute";
    return failure();
  }
  return plan;
}
