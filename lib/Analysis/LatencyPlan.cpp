#include "tb/Analysis/LatencyPlan.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <optional>
#include <queue>

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

static FailureOr<bool> readBoolField(DictionaryAttr dict, StringRef name,
                                     Operation *op) {
  auto attr = dyn_cast_or_null<BoolAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing bool field `" << name << "`";
    return failure();
  }
  return attr.getValue();
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

static DictionaryAttr buildLatencyInfoAttr(Builder &builder,
                                           const OpLatencyInfo &info) {
  NamedAttrList attrs;
  attrs.set("op_id", builder.getI64IntegerAttr(info.opId));
  attrs.set("target_latency", builder.getI64IntegerAttr(info.targetLatency));
  attrs.set("self_latency", builder.getI64IntegerAttr(info.selfLatency));
  attrs.set("buffer_distance", builder.getI64IntegerAttr(info.bufferDistance));
  attrs.set("pipelineable", builder.getBoolAttr(info.pipelineable));
  attrs.set("acc_multi_buffer", builder.getBoolAttr(info.accMultiBuffer));
  attrs.set("reason", builder.getStringAttr(info.reason));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<OpLatencyInfo> parseLatencyInfoAttr(DictionaryAttr dict,
                                                     Operation *op) {
  OpLatencyInfo info;
  auto opId = readI64Field(dict, "op_id", op);
  auto targetLatency = readI64Field(dict, "target_latency", op);
  auto selfLatency = readI64Field(dict, "self_latency", op);
  auto bufferDistance = readI64Field(dict, "buffer_distance", op);
  auto pipelineable = readBoolField(dict, "pipelineable", op);
  auto accMultiBuffer = readBoolField(dict, "acc_multi_buffer", op);
  auto reason = readStringField(dict, "reason", op);
  if (failed(opId) || failed(targetLatency) || failed(selfLatency) ||
      failed(bufferDistance) || failed(pipelineable) ||
      failed(accMultiBuffer) || failed(reason)) {
    return failure();
  }

  info.opId = *opId;
  info.targetLatency = *targetLatency;
  info.selfLatency = *selfLatency;
  info.bufferDistance = *bufferDistance;
  info.pipelineable = *pipelineable;
  info.accMultiBuffer = *accMultiBuffer;
  info.reason = reason->str();
  return info;
}

static ArrayAttr buildLatencyArrayAttr(Builder &builder,
                                       ArrayRef<OpLatencyInfo> infos) {
  SmallVector<Attribute> attrs;
  attrs.reserve(infos.size());
  for (const OpLatencyInfo &info : infos)
    attrs.push_back(buildLatencyInfoAttr(builder, info));
  return builder.getArrayAttr(attrs);
}

static FailureOr<SmallVector<OpLatencyInfo, 32>>
parseLatencyArrayAttr(Attribute attr, StringRef name, Operation *op) {
  auto array = dyn_cast_or_null<ArrayAttr>(attr);
  if (!array) {
    op->emitError() << "missing array field `" << name << "`";
    return failure();
  }

  SmallVector<OpLatencyInfo, 32> infos;
  infos.reserve(array.size());
  for (Attribute elem : array) {
    auto dict = dyn_cast<DictionaryAttr>(elem);
    if (!dict) {
      op->emitError() << "field `" << name
                      << "` must contain only dictionary elements";
      return failure();
    }
    auto info = parseLatencyInfoAttr(dict, op);
    if (failed(info))
      return failure();
    infos.push_back(std::move(*info));
  }
  return infos;
}

static std::string makeGlobalLoadReason(bool pipelineable,
                                        int64_t indirectionLevel) {
  if (!pipelineable)
    return "global_load_not_pipelineable";
  return "global_load_stage_owned_to_mma_indirection_" +
         std::to_string(indirectionLevel);
}

static std::string makeLocalLoadReason(bool pipelineable) {
  if (!pipelineable)
    return "local_load_dead_end";
  return "local_load_to_mma_distance_0";
}

static std::string makeMmaReason(bool pipelineable) {
  if (!pipelineable)
    return "terminal_acc_value";
  return "acc_live_to_next_mma";
}

static FailureOr<std::optional<int64_t>> findUniqueDepthToFirstTarget(
    int64_t startOpId, BufferOpKind targetKind,
    const DenseMap<int64_t, const PipelineOp *> &opsById,
    const DenseMap<int64_t, const ValueState *> &valuesById, Operation *op) {
  auto startOpIt = opsById.find(startOpId);
  if (startOpIt == opsById.end())
    return std::optional<int64_t>();

  std::queue<std::pair<int64_t, int64_t>> queue;
  DenseMap<int64_t, int64_t> minDepthByOp;
  DenseSet<int64_t> depthsToMma;
  for (int64_t valueId : startOpIt->second->outputs) {
    auto valueIt = valuesById.find(valueId);
    if (valueIt == valuesById.end())
      continue;
    for (int64_t userId : valueIt->second->users)
      queue.push({userId, 0});
  }

  while (!queue.empty()) {
    auto [opId, depth] = queue.front();
    queue.pop();

    auto seenIt = minDepthByOp.find(opId);
    if (seenIt != minDepthByOp.end() && seenIt->second <= depth)
      continue;
    minDepthByOp[opId] = depth;

    auto opIt = opsById.find(opId);
    if (opIt == opsById.end())
      continue;

    const PipelineOp *user = opIt->second;
    if (user->kind == targetKind) {
      depthsToMma.insert(depth);
      continue;
    }

    int64_t nextDepth = depth + 1;
    for (int64_t outputValueId : user->outputs) {
      auto valueIt = valuesById.find(outputValueId);
      if (valueIt == valuesById.end())
        continue;
      for (int64_t nextUserId : valueIt->second->users)
        queue.push({nextUserId, nextDepth});
    }
  }

  if (depthsToMma.empty())
    return std::optional<int64_t>();
  if (depthsToMma.size() != 1) {
    op->emitError() << "pipeline producer-consumer chain reaches the target "
                       "with conflicting indirection depths";
    return failure();
  }
  return std::optional<int64_t>(*depthsToMma.begin());
}

static const BufferView *
findOwnerView(const ValueState &value,
              const DenseMap<int64_t, const BufferView *> &viewsById) {
  auto it = viewsById.find(value.ownerView);
  return it == viewsById.end() ? nullptr : it->second;
}

static const BufferBacking *findBackingForView(
    const BufferView *view,
    const DenseMap<int64_t, const BufferBacking *> &backingsById) {
  if (!view)
    return nullptr;
  auto it = backingsById.find(view->backing);
  return it == backingsById.end() ? nullptr : it->second;
}

} // namespace

FailureOr<LatencyPlan> mlir::tb::deriveLatencyPlan(const KernelConfig &config,
                                                   const TargetInfo &target,
                                                   const BufferModel &model,
                                                   const LoopPlan &loopPlan,
                                                   Operation *op) {
  LatencyPlan plan;
  DenseMap<int64_t, const PipelineOp *> opsById;
  DenseMap<int64_t, const ValueState *> valuesById;
  DenseMap<int64_t, const BufferView *> viewsById;
  DenseMap<int64_t, const BufferBacking *> backingsById;
  DenseMap<int64_t, int64_t> kGroupByOp;
  DenseSet<int64_t> loopOps;
  DenseSet<int64_t> carriedValueIds;
  opsById.reserve(model.ops.size());
  valuesById.reserve(model.values.size());
  for (const PipelineOp &info : model.ops)
    opsById[info.id] = &info;
  for (const ValueState &value : model.values)
    valuesById[value.id] = &value;
  for (const BufferView &view : model.views)
    viewsById[view.id] = &view;
  for (const BufferBacking &backing : model.backings)
    backingsById[backing.id] = &backing;
  for (const LoopIterationPlan &iteration : loopPlan.iterations) {
    for (int64_t opId : iteration.asyncProducerOps) {
      loopOps.insert(opId);
      kGroupByOp[opId] = iteration.kGroup;
    }
    for (int64_t opId : iteration.consumerOps) {
      loopOps.insert(opId);
      kGroupByOp[opId] = iteration.kGroup;
    }
    for (int64_t opId : iteration.computeOps) {
      loopOps.insert(opId);
      kGroupByOp[opId] = iteration.kGroup;
    }
  }
  for (const LoopCarriedValue &value : loopPlan.carriedValues)
    carriedValueIds.insert(value.valueId);
  if (loopOps.size() != model.ops.size()) {
    op->emitError() << "latency derivation requires loop_plan to cover every "
                       "pipeline op";
    return failure();
  }

  plan.ops.reserve(model.ops.size());
  for (const PipelineOp &info : model.ops) {
    if (!loopOps.count(info.id)) {
      op->emitError() << "latency derivation found pipeline op " << info.id
                      << " outside tb.loop_plan";
      return failure();
    }
    OpLatencyInfo latency;
    latency.opId = info.id;

    if (info.kind == BufferOpKind::LoadA || info.kind == BufferOpKind::LoadB) {
      auto indirectionLevel = findUniqueDepthToFirstTarget(
          info.id, BufferOpKind::Mma, opsById, valuesById, op);
      if (failed(indirectionLevel))
        return failure();

      const ValueState *outputValue =
          info.outputs.empty() ? nullptr
                               : valuesById.lookup(info.outputs.front());
      const BufferView *ownerView =
          outputValue ? findOwnerView(*outputValue, viewsById) : nullptr;
      const BufferBacking *ownerBacking =
          findBackingForView(ownerView, backingsById);
      bool stageOwnedOutput = ownerView && ownerBacking &&
                              ownerView->kind == ViewKind::StageSlice &&
                              ownerBacking->stageIndexed;
      if (stageOwnedOutput && ownerBacking->depth <= 0) {
        op->emitError() << "stage-owned async output must carry a positive "
                           "shared backing depth";
        return failure();
      }

      bool pipelineable = indirectionLevel->has_value() && stageOwnedOutput &&
                          target.supportsAsyncCopy &&
                          config.requestedStages > 1;
      latency.selfLatency = target.globalToSharedStageLatency;
      // 中文标记：stage-owned multibuffer 的 stage distance 是 depth - 1。
      // requestedStages=2 时，producer 需要领先 consumer 一整个 steady-state
      // stage，而不是两个 stage。
      latency.bufferDistance =
          pipelineable ? std::max<int64_t>(ownerBacking->depth - 1, 0) : 0;
      latency.targetLatency =
          pipelineable ? (latency.bufferDistance + **indirectionLevel)
                       : latency.selfLatency;
      latency.pipelineable = pipelineable;
      latency.accMultiBuffer = false;
      latency.reason = makeGlobalLoadReason(
          pipelineable,
          indirectionLevel->has_value() ? **indirectionLevel : -1);
      plan.ops.push_back(std::move(latency));
      continue;
    }

    if (info.kind == BufferOpKind::LocalLoadA ||
        info.kind == BufferOpKind::LocalLoadB) {
      auto indirectionLevel = findUniqueDepthToFirstTarget(
          info.id, BufferOpKind::Mma, opsById, valuesById, op);
      if (failed(indirectionLevel))
        return failure();

      bool hasConsumer = indirectionLevel->has_value();
      latency.selfLatency = target.sharedToRegisterStageLatency;
      // 中文标记：shared->register operand load 属于同一个 pipeline stage 内的
      // consumer cluster，不应该再强行跨一个 whole-stage。
      latency.bufferDistance = 0;
      latency.targetLatency =
          hasConsumer ? (latency.bufferDistance + **indirectionLevel) : 0;
      latency.pipelineable = false;
      latency.accMultiBuffer = false;
      latency.reason = makeLocalLoadReason(hasConsumer);
      plan.ops.push_back(std::move(latency));
      continue;
    }

    std::optional<int64_t> nextMmaDistance;
    bool hasCrossIterationMmaUser = false;
    for (int64_t outputValueId : info.outputs) {
      auto valueIt = valuesById.find(outputValueId);
      if (valueIt == valuesById.end()) {
        op->emitError() << "mma op " << info.id
                        << " references unknown output value " << outputValueId;
        return failure();
      }
      for (int64_t userId : valueIt->second->users) {
        auto userOpIt = opsById.find(userId);
        auto defKGroupIt = kGroupByOp.find(info.id);
        auto userKGroupIt = kGroupByOp.find(userId);
        if (userOpIt == opsById.end() || defKGroupIt == kGroupByOp.end() ||
            userKGroupIt == kGroupByOp.end()) {
          continue;
        }
        if (userOpIt->second->kind == BufferOpKind::Mma &&
            userKGroupIt->second > defKGroupIt->second) {
          hasCrossIterationMmaUser = true;
        }
      }
    }

    auto computeIt = llvm::find_if(loopPlan.carriedValues,
                                   [&](const LoopCarriedValue &value) {
                                     return value.definingOp == info.id;
                                   });
    if (computeIt != loopPlan.carriedValues.end()) {
      nextMmaDistance = computeIt->loopDistance;
    } else if (hasCrossIterationMmaUser) {
      op->emitError() << "mma op " << info.id
                      << " has cross-iteration mma users but is missing a "
                         "loop_plan carried value";
      return failure();
    }

    bool accMultiBuffer = nextMmaDistance.has_value() && config.requestedStages > 1;
    latency.selfLatency = target.mmaStageLatency;
    latency.bufferDistance = accMultiBuffer ? target.mmaStageLatency : 0;
    latency.targetLatency =
        accMultiBuffer ? (latency.bufferDistance + *nextMmaDistance) : 0;
    latency.pipelineable = false;
    latency.accMultiBuffer = accMultiBuffer;
    latency.reason = makeMmaReason(accMultiBuffer);
    plan.ops.push_back(std::move(latency));
  }

  return plan;
}

DictionaryAttr mlir::tb::buildLatencyPlanAttr(Builder &builder,
                                              const LatencyPlan &plan) {
  NamedAttrList attrs;
  attrs.set("ops", buildLatencyArrayAttr(builder, plan.ops));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<LatencyPlan> mlir::tb::parseLatencyPlanAttr(Operation *op) {
  auto root = dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.latency_plan"));
  if (!root) {
    op->emitError() << "missing `tb.latency_plan` attribute";
    return failure();
  }

  auto ops = parseLatencyArrayAttr(root.get("ops"), "ops", op);
  if (failed(ops))
    return failure();

  LatencyPlan plan;
  plan.ops = std::move(*ops);
  return plan;
}
