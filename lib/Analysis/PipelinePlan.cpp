#include "tb/Analysis/PipelinePlan.h"

#include "tb/Analysis/LatencyPlan.h"
#include "tb/IR/TBOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

using namespace mlir;
using namespace mlir::tb;

namespace {

struct TempPlacement {
  int64_t stage = 0;
  int64_t cluster = 0;
  std::string reason;
};

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

static DictionaryAttr buildPlacementAttr(Builder &builder,
                                         const PipelinePlacement &placement) {
  NamedAttrList attrs;
  attrs.set("op_id", builder.getI64IntegerAttr(placement.opId));
  attrs.set("stage", builder.getI64IntegerAttr(placement.stage));
  attrs.set("cluster", builder.getI64IntegerAttr(placement.cluster));
  attrs.set("order", builder.getI64IntegerAttr(placement.order));
  attrs.set("reason", builder.getStringAttr(placement.reason));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<PipelinePlacement> parsePlacementAttr(DictionaryAttr dict,
                                                       Operation *op) {
  PipelinePlacement placement;
  auto opId = readI64Field(dict, "op_id", op);
  auto stage = readI64Field(dict, "stage", op);
  auto cluster = readI64Field(dict, "cluster", op);
  auto order = readI64Field(dict, "order", op);
  auto reason = readStringField(dict, "reason", op);
  if (failed(opId) || failed(stage) || failed(cluster) || failed(order) ||
      failed(reason)) {
    return failure();
  }
  placement.opId = *opId;
  placement.stage = *stage;
  placement.cluster = *cluster;
  placement.order = *order;
  placement.reason = reason->str();
  return placement;
}

static DictionaryAttr buildStageOwnedBufferAttr(Builder &builder,
                                                const StageBufferUse &use) {
  NamedAttrList attrs;
  attrs.set("view_id", builder.getI64IntegerAttr(use.viewId));
  attrs.set("backing", builder.getI64IntegerAttr(use.backing));
  attrs.set("stage", builder.getI64IntegerAttr(use.stage));
  attrs.set("buffer_index", builder.getI64IntegerAttr(use.bufferIndex));
  attrs.set("producer_op", builder.getI64IntegerAttr(use.producerOp));
  attrs.set("first_consumer_op",
            builder.getI64IntegerAttr(use.firstConsumerOp));
  attrs.set("last_consumer_op", builder.getI64IntegerAttr(use.lastConsumerOp));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<StageBufferUse> parseStageOwnedBufferAttr(DictionaryAttr dict,
                                                           Operation *op) {
  StageBufferUse use;
  auto viewId = readI64Field(dict, "view_id", op);
  auto backing = readI64Field(dict, "backing", op);
  auto stage = readI64Field(dict, "stage", op);
  auto bufferIndex = readI64Field(dict, "buffer_index", op);
  auto producerOp = readI64Field(dict, "producer_op", op);
  auto firstConsumerOp = readI64Field(dict, "first_consumer_op", op);
  auto lastConsumerOp = readI64Field(dict, "last_consumer_op", op);
  if (failed(viewId) || failed(backing) || failed(stage) ||
      failed(bufferIndex) || failed(producerOp) || failed(firstConsumerOp) ||
      failed(lastConsumerOp)) {
    return failure();
  }
  use.viewId = *viewId;
  use.backing = *backing;
  use.stage = *stage;
  use.bufferIndex = *bufferIndex;
  use.producerOp = *producerOp;
  use.firstConsumerOp = *firstConsumerOp;
  use.lastConsumerOp = *lastConsumerOp;
  return use;
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

static LogicalResult validatePipelinePlan(const PipelinePlan &plan,
                                          Operation *op) {
  DenseSet<int64_t> placementOps;
  DenseSet<int64_t> stageOwnerKeys;
  for (const PipelinePlacement &placement : plan.placements) {
    if (!placementOps.insert(placement.opId).second)
      return op->emitError() << "duplicate placement for op " << placement.opId;
  }
  for (const StageBufferUse &use : plan.stageOwnedBuffers) {
    if (use.viewId < 0 || use.producerOp < 0)
      return op->emitError()
             << "stage ownership must carry concrete view_id and producer_op";
    int64_t key = (use.viewId << 32) ^ use.producerOp;
    if (!stageOwnerKeys.insert(key).second)
      return op->emitError()
             << "duplicate stage ownership record for view " << use.viewId
             << " and producer op " << use.producerOp;
  }
  return success();
}

static int64_t packStageClusterKey(int64_t stage, int64_t cluster) {
  return (stage << 32) ^ cluster;
}

static bool insertIfAbsent(DenseMap<int64_t, TempPlacement> &placements,
                           int64_t opId, int64_t stage, int64_t cluster,
                           StringRef reason) {
  if (placements.count(opId))
    return false;
  placements.try_emplace(opId, TempPlacement{stage, cluster, reason.str()});
  return true;
}

static bool isEarlierPlacement(const PipelinePlacement &lhs,
                               const PipelinePlacement &rhs) {
  if (lhs.stage != rhs.stage)
    return lhs.stage < rhs.stage;
  if (lhs.cluster != rhs.cluster)
    return lhs.cluster < rhs.cluster;
  return lhs.order < rhs.order;
}

static FailureOr<const OpLatencyInfo *>
findLatencyInfo(const DenseMap<int64_t, const OpLatencyInfo *> &latencyByOp,
                int64_t opId, Operation *op) {
  auto it = latencyByOp.find(opId);
  if (it == latencyByOp.end()) {
    op->emitError() << "missing latency info for pipeline op " << opId;
    return failure();
  }
  return it->second;
}

static FailureOr<int64_t>
requirePlacedStage(const DenseMap<int64_t, TempPlacement> &placements,
                   int64_t opId, Operation *op, StringRef role) {
  auto it = placements.find(opId);
  if (it == placements.end()) {
    op->emitError() << role << " op " << opId
                    << " must be placed before dependent scheduling";
    return failure();
  }
  return it->second.stage;
}

static FailureOr<int64_t>
findIterationCoord(ArrayRef<PipelineOp::IterationCoord> coords, StringRef axis,
                   int64_t opId, Operation *op) {
  auto it = llvm::find_if(coords, [&](const PipelineOp::IterationCoord &coord) {
    return coord.axis == axis;
  });
  if (it == coords.end()) {
    op->emitError() << "pipeline op " << opId
                    << " is missing iteration coordinate `" << axis << "`";
    return failure();
  }
  return it->value;
}

static FailureOr<int64_t> getSemanticCluster(const PipelineOp &pipelineOp,
                                             Operation *op) {
  switch (getPipelineOpSemanticClass(pipelineOp.kind)) {
  case PipelineOpSemanticClass::AsyncProducer:
    return int64_t{0};
  case PipelineOpSemanticClass::SharedConsumer:
    return int64_t{1};
  case PipelineOpSemanticClass::TensorCoreCompute:
    return int64_t{2};
  case PipelineOpSemanticClass::EpilogueInit:
  case PipelineOpSemanticClass::EpilogueStore:
    op->emitError() << "pipeline scheduler only accepts mainloop semantic "
                       "classes, but op "
                    << pipelineOp.id << " has semantic class `"
                    << stringifyPipelineOpSemanticClass(
                           getPipelineOpSemanticClass(pipelineOp.kind))
                    << "`";
    return failure();
  }
  llvm_unreachable("unknown pipeline semantic class");
}

static FailureOr<int64_t> getStageAnchor(const PipelineOp &pipelineOp,
                                         Operation *op) {
  switch (getPipelineOpSemanticClass(pipelineOp.kind)) {
  case PipelineOpSemanticClass::AsyncProducer:
    return findIterationCoord(pipelineOp.iterationCoords, "k_group",
                              pipelineOp.id, op);
  case PipelineOpSemanticClass::SharedConsumer:
  case PipelineOpSemanticClass::TensorCoreCompute:
    return int64_t{0};
  case PipelineOpSemanticClass::EpilogueInit:
  case PipelineOpSemanticClass::EpilogueStore:
    op->emitError() << "pipeline scheduler cannot anchor non-mainloop semantic "
                       "class `"
                    << stringifyPipelineOpSemanticClass(
                           getPipelineOpSemanticClass(pipelineOp.kind))
                    << "` for op " << pipelineOp.id;
    return failure();
  }
  llvm_unreachable("unknown pipeline semantic class");
}

static StringRef getPlacementReason(const PipelineOp &pipelineOp) {
  switch (getPipelineOpSemanticClass(pipelineOp.kind)) {
  case PipelineOpSemanticClass::AsyncProducer:
    return "async_issue_frontier_for_k_group";
  case PipelineOpSemanticClass::SharedConsumer:
    return "consumer_frontier_after_async_wait";
  case PipelineOpSemanticClass::TensorCoreCompute:
    return "mma_compute_frontier_after_operands_ready";
  case PipelineOpSemanticClass::EpilogueInit:
  case PipelineOpSemanticClass::EpilogueStore:
    return "unknown_pipeline_frontier";
  }
  llvm_unreachable("unknown pipeline semantic class");
}

static FailureOr<int64_t> computeEarliestStageFromInputs(
    const PipelineOp &pipelineOp,
    const DenseMap<int64_t, const ValueState *> &valuesById,
    const DenseMap<int64_t, TempPlacement> &placements,
    const DenseMap<int64_t, const OpLatencyInfo *> &latencyByOp,
    Operation *op) {
  int64_t earliestStage = 0;
  for (int64_t inputValueId : pipelineOp.inputs) {
    auto valueIt = valuesById.find(inputValueId);
    if (valueIt == valuesById.end()) {
      op->emitError() << "pipeline op " << pipelineOp.id
                      << " references unknown input value " << inputValueId;
      return failure();
    }
    const ValueState *value = valueIt->second;
    if (value->definingOp < 0)
      continue;
    auto predStage =
        requirePlacedStage(placements, value->definingOp, op, "producer");
    if (failed(predStage))
      return failure();
    auto predLatency = findLatencyInfo(latencyByOp, value->definingOp, op);
    if (failed(predLatency))
      return failure();
    earliestStage = std::max<int64_t>(
        earliestStage, *predStage + (*predLatency)->bufferDistance);
  }
  return earliestStage;
}

static LogicalResult scheduleSemanticMainline(
    const BufferModel &model, const LoopPlan &loopPlan,
    const LatencyPlan &latencyPlan, Operation *op,
    DenseMap<int64_t, TempPlacement> &placements, int64_t &numStages) {
  DenseMap<int64_t, const OpLatencyInfo *> latencyByOp;
  DenseMap<int64_t, const ValueState *> valuesById;
  DenseMap<int64_t, const PipelineOp *> opsById;
  DenseSet<int64_t> loopOwnedOps;
  int64_t maxStage = 0;

  for (const OpLatencyInfo &info : latencyPlan.ops)
    latencyByOp[info.opId] = &info;
  for (const PipelineOp &pipelineOp : model.ops)
    opsById[pipelineOp.id] = &pipelineOp;
  for (const ValueState &value : model.values)
    valuesById[value.id] = &value;

  if (loopPlan.iterations.empty()) {
    op->emitError() << "pipeline scheduling requires a non-empty loop_plan";
    return failure();
  }
  for (const LoopIterationPlan &iteration : loopPlan.iterations) {
    for (int64_t opId : iteration.asyncProducerOps) {
      if (!opsById.count(opId)) {
        op->emitError() << "loop_plan references unknown async op " << opId;
        return failure();
      }
      loopOwnedOps.insert(opId);
    }
    for (int64_t opId : iteration.consumerOps) {
      if (!opsById.count(opId)) {
        op->emitError() << "loop_plan references unknown consumer op " << opId;
        return failure();
      }
      loopOwnedOps.insert(opId);
    }
    for (int64_t opId : iteration.computeOps) {
      if (!opsById.count(opId)) {
        op->emitError() << "loop_plan references unknown compute op " << opId;
        return failure();
      }
      loopOwnedOps.insert(opId);
    }
  }
  if (loopOwnedOps.size() != model.ops.size()) {
    op->emitError()
        << "pipeline scheduler requires loop_plan to own every mainloop op "
           "exactly once";
    return failure();
  }

  for (const PipelineOp &pipelineOp : model.ops) {
    if (!loopOwnedOps.count(pipelineOp.id)) {
      op->emitError() << "pipeline op " << pipelineOp.id
                      << " is missing from tb.loop_plan";
      return failure();
    }
    if (failed(findLatencyInfo(latencyByOp, pipelineOp.id, op)))
      return failure();

    auto earliestStage = computeEarliestStageFromInputs(
        pipelineOp, valuesById, placements, latencyByOp, op);
    auto stageAnchor = getStageAnchor(pipelineOp, op);
    auto semanticCluster = getSemanticCluster(pipelineOp, op);
    if (failed(earliestStage) || failed(stageAnchor) ||
        failed(semanticCluster)) {
      return failure();
    }

    int64_t stage = std::max(*earliestStage, *stageAnchor);
    if (!insertIfAbsent(placements, pipelineOp.id, stage, *semanticCluster,
                        getPlacementReason(pipelineOp))) {
      op->emitError() << "duplicate placement for pipeline op "
                      << pipelineOp.id;
      return failure();
    }
    maxStage = std::max(maxStage, stage);
  }

  numStages = maxStage + 1;
  if (placements.size() != model.ops.size()) {
    op->emitError() << "pipeline scheduler must place every pipeline op";
    return failure();
  }
  return success();
}

static PipelinePlan
normalizeStageClusters(const BufferModel &model,
                       const DenseMap<int64_t, TempPlacement> &temp) {
  PipelinePlan plan;
  SmallVector<int64_t, 8> stageOrder;
  for (const PipelineOp &info : model.ops) {
    auto placementIt = temp.find(info.id);
    if (placementIt != temp.end())
      stageOrder.push_back(placementIt->second.stage);
  }
  llvm::sort(stageOrder);
  stageOrder.erase(std::unique(stageOrder.begin(), stageOrder.end()),
                   stageOrder.end());

  DenseMap<int64_t, int64_t> normalizedStage;
  int64_t nextStage = 0;
  for (int64_t stage : stageOrder)
    normalizedStage[stage] = nextStage++;
  plan.scheduledMaxStage = std::max<int64_t>(nextStage - 1, 0);

  DenseMap<int64_t, int64_t> originalOrderByOp;
  DenseMap<int64_t, llvm::SmallSetVector<int64_t, 4>> clustersByStage;
  for (const auto &it : llvm::enumerate(model.ops)) {
    const PipelineOp &info = it.value();
    originalOrderByOp[info.id] = static_cast<int64_t>(it.index());
    auto placementIt = temp.find(info.id);
    if (placementIt == temp.end())
      continue;
    int64_t stage = normalizedStage[placementIt->second.stage];
    clustersByStage[stage].insert(placementIt->second.cluster);
  }

  DenseMap<int64_t, DenseMap<int64_t, int64_t>> normalizedClusterByStage;
  for (auto &[stage, clusters] : clustersByStage) {
    int64_t clusterIndex = 0;
    for (int64_t cluster : clusters)
      normalizedClusterByStage[stage][cluster] = clusterIndex++;
  }

  SmallVector<PipelinePlacement, 32> sortedPlacements;
  sortedPlacements.reserve(model.ops.size());
  for (const PipelineOp &info : model.ops) {
    auto placementIt = temp.find(info.id);
    if (placementIt == temp.end())
      continue;

    PipelinePlacement placement;
    placement.opId = info.id;
    placement.stage = normalizedStage[placementIt->second.stage];
    placement.cluster =
        normalizedClusterByStage[placement.stage][placementIt->second.cluster];
    placement.reason = placementIt->second.reason;
    sortedPlacements.push_back(std::move(placement));
  }

  llvm::stable_sort(sortedPlacements, [&](const PipelinePlacement &lhs,
                                          const PipelinePlacement &rhs) {
    if (lhs.stage != rhs.stage)
      return lhs.stage < rhs.stage;
    if (lhs.cluster != rhs.cluster)
      return lhs.cluster < rhs.cluster;
    return originalOrderByOp[lhs.opId] < originalOrderByOp[rhs.opId];
  });

  DenseMap<int64_t, int64_t> nextOrder;
  for (PipelinePlacement &placement : sortedPlacements) {
    int64_t key = packStageClusterKey(placement.stage, placement.cluster);
    placement.order = nextOrder[key]++;
  }

  DenseMap<int64_t, PipelinePlacement> finalizedByOp;
  finalizedByOp.reserve(sortedPlacements.size());
  for (const PipelinePlacement &placement : sortedPlacements)
    finalizedByOp[placement.opId] = placement;

  plan.placements.reserve(sortedPlacements.size());
  for (const PipelineOp &info : model.ops) {
    auto it = finalizedByOp.find(info.id);
    if (it != finalizedByOp.end())
      plan.placements.push_back(it->second);
  }

  for (const ValueState &value : model.values) {
    auto ownerViewIt = llvm::find_if(model.views, [&](const BufferView &view) {
      return view.id == value.ownerView;
    });
    if (ownerViewIt == model.views.end() ||
        ownerViewIt->kind != ViewKind::StageSlice) {
      continue;
    }
    int64_t firstConsumerOp = -1;
    int64_t lastConsumerOp = -1;
    std::optional<PipelinePlacement> firstPlacement;
    std::optional<PipelinePlacement> lastPlacement;
    for (int64_t userId : value.users) {
      auto userIt = finalizedByOp.find(userId);
      if (userIt == finalizedByOp.end())
        continue;
      if (!firstPlacement ||
          isEarlierPlacement(userIt->second, *firstPlacement)) {
        firstPlacement = userIt->second;
        firstConsumerOp = userId;
      }
      if (!lastPlacement ||
          isEarlierPlacement(*lastPlacement, userIt->second)) {
        lastPlacement = userIt->second;
        lastConsumerOp = userId;
      }
    }
    plan.stageOwnedBuffers.push_back(
        {ownerViewIt->id, ownerViewIt->backing, ownerViewIt->stage,
         ownerViewIt->bufferIndex, value.definingOp, firstConsumerOp,
         lastConsumerOp});
  }
  return plan;
}

} // namespace

FailureOr<PipelinePlan>
mlir::tb::derivePipelinePlan(const BufferModel &model,
                             const LoopPlan &loopPlan,
                             const LatencyPlan &latencyPlan, Operation *op) {
  if (!isa<MatmulOp>(op)) {
    op->emitError() << "pipeline plan can only be derived from tb.matmul";
    return failure();
  }

  DenseMap<int64_t, TempPlacement> tempPlacements;
  int64_t numStages = 0;
  if (failed(scheduleSemanticMainline(model, loopPlan, latencyPlan, op,
                                      tempPlacements,
                                       numStages))) {
    return failure();
  }

  PipelinePlan plan = normalizeStageClusters(model, tempPlacements);
  if (failed(validatePipelinePlan(plan, op)))
    return failure();
  return plan;
}

DictionaryAttr mlir::tb::buildPipelinePlanAttr(Builder &builder,
                                               const PipelinePlan &plan) {
  NamedAttrList attrs;
  attrs.set("scheduled_max_stage",
            builder.getI64IntegerAttr(plan.scheduledMaxStage));
  attrs.set("placements",
            buildDictArrayAttr(builder, plan.placements, buildPlacementAttr));
  attrs.set("stage_owned_buffers",
            buildDictArrayAttr(builder, plan.stageOwnedBuffers,
                               buildStageOwnedBufferAttr));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<PipelinePlan> mlir::tb::parsePipelinePlanAttr(Operation *op) {
  auto root = dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.pipeline_plan"));
  if (!root) {
    op->emitError() << "missing `tb.pipeline_plan` attribute";
    return failure();
  }

  auto scheduledMaxStage = readI64Field(root, "scheduled_max_stage", op);
  auto placements = parseDictArrayAttr<PipelinePlacement>(
      root.get("placements"), "placements", op, parsePlacementAttr);
  auto stageOwnedBuffers = parseDictArrayAttr<StageBufferUse>(
      root.get("stage_owned_buffers"), "stage_owned_buffers", op,
      parseStageOwnedBufferAttr);
  if (failed(scheduledMaxStage) || failed(placements) ||
      failed(stageOwnedBuffers)) {
    op->emitError() << "malformed `tb.pipeline_plan` attribute";
    return failure();
  }

  PipelinePlan plan;
  plan.scheduledMaxStage = *scheduledMaxStage;
  plan.placements = std::move(*placements);
  plan.stageOwnedBuffers = std::move(*stageOwnedBuffers);
  if (failed(validatePipelinePlan(plan, op))) {
    op->emitError() << "malformed `tb.pipeline_plan` attribute";
    return failure();
  }
  return plan;
}
