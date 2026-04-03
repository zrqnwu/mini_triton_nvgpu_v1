#include "tb/Analysis/LoopPlan.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::tb;

namespace {

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

static FailureOr<StringRef> readStringField(DictionaryAttr dict, StringRef name,
                                            Operation *op) {
  auto attr = dyn_cast_or_null<StringAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing string field `" << name << "`";
    return failure();
  }
  return attr.getValue();
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

static int64_t findIterationCoord(ArrayRef<PipelineOp::IterationCoord> coords,
                                  StringRef axis) {
  auto it = llvm::find_if(coords, [&](const PipelineOp::IterationCoord &coord) {
    return coord.axis == axis;
  });
  return it == coords.end() ? -1 : it->value;
}

static DictionaryAttr buildIterationAttr(Builder &builder,
                                         const LoopIterationPlan &iteration) {
  NamedAttrList attrs;
  attrs.set("k_group", builder.getI64IntegerAttr(iteration.kGroup));
  attrs.set("async_producer_ops",
            buildI64ArrayAttr(builder, iteration.asyncProducerOps));
  attrs.set("consumer_ops", buildI64ArrayAttr(builder, iteration.consumerOps));
  attrs.set("compute_ops", buildI64ArrayAttr(builder, iteration.computeOps));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<LoopIterationPlan> parseIterationAttr(DictionaryAttr dict,
                                                       Operation *op) {
  LoopIterationPlan iteration;
  auto kGroup = readI64Field(dict, "k_group", op);
  auto asyncProducerOps =
      readDenseI64ArrayField(dict, "async_producer_ops", op);
  auto consumerOps = readDenseI64ArrayField(dict, "consumer_ops", op);
  auto computeOps = readDenseI64ArrayField(dict, "compute_ops", op);
  if (failed(kGroup) || failed(asyncProducerOps) || failed(consumerOps) ||
      failed(computeOps)) {
    return failure();
  }
  iteration.kGroup = *kGroup;
  iteration.asyncProducerOps = parseI64Array(*asyncProducerOps);
  iteration.consumerOps = parseI64Array(*consumerOps);
  iteration.computeOps = parseI64Array(*computeOps);
  return iteration;
}

static DictionaryAttr buildCarriedValueAttr(Builder &builder,
                                            const LoopCarriedValue &value) {
  NamedAttrList attrs;
  attrs.set("value_id", builder.getI64IntegerAttr(value.valueId));
  attrs.set("defining_op", builder.getI64IntegerAttr(value.definingOp));
  attrs.set("owner_view", builder.getI64IntegerAttr(value.ownerView));
  attrs.set("loop_distance", builder.getI64IntegerAttr(value.loopDistance));
  attrs.set("users", buildI64ArrayAttr(builder, value.users));
  attrs.set("reason", builder.getStringAttr(value.reason));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<LoopCarriedValue> parseCarriedValueAttr(DictionaryAttr dict,
                                                         Operation *op) {
  LoopCarriedValue value;
  auto valueId = readI64Field(dict, "value_id", op);
  auto definingOp = readI64Field(dict, "defining_op", op);
  auto ownerView = readI64Field(dict, "owner_view", op);
  auto loopDistance = readI64Field(dict, "loop_distance", op);
  auto users = readDenseI64ArrayField(dict, "users", op);
  auto reason = readStringField(dict, "reason", op);
  if (failed(valueId) || failed(definingOp) || failed(ownerView) ||
      failed(loopDistance) || failed(users) || failed(reason)) {
    return failure();
  }
  value.valueId = *valueId;
  value.definingOp = *definingOp;
  value.ownerView = *ownerView;
  value.loopDistance = *loopDistance;
  value.users = parseI64Array(*users);
  value.reason = reason->str();
  return value;
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
static FailureOr<SmallVector<T, 0>>
parseDictArrayAttr(Attribute attr, StringRef name, Operation *op,
                   ParseFn parseElement) {
  auto array = dyn_cast_or_null<ArrayAttr>(attr);
  if (!array) {
    op->emitError() << "missing array field `" << name << "`";
    return failure();
  }
  SmallVector<T, 0> values;
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

static LogicalResult validateLoopPlan(const LoopPlan &plan, Operation *op) {
  if (!plan.singleMainLoop)
    return op->emitError() << "loop_plan must describe a single main loop";
  if (plan.loopAxis != "k_group")
    return op->emitError() << "loop_plan currently requires `k_group` as the "
                              "single main-loop axis";
  if (plan.iterationCount <= 0)
    return op->emitError() << "loop_plan iteration_count must be positive";
  if (plan.iterations.size() != static_cast<size_t>(plan.iterationCount))
    return op->emitError()
           << "loop_plan iterations must exactly match iteration_count";

  DenseSet<int64_t> seenOps;
  DenseSet<int64_t> seenValues;
  int64_t previousKGroup = -1;
  int64_t expectedAsyncCount = -1;
  int64_t expectedConsumerCount = -1;
  int64_t expectedComputeCount = -1;

  for (const LoopIterationPlan &iteration : plan.iterations) {
    if (iteration.kGroup != previousKGroup + 1) {
      return op->emitError()
             << "loop_plan k_group iterations must be contiguous from zero";
    }
    previousKGroup = iteration.kGroup;
    if (iteration.asyncProducerOps.empty() || iteration.consumerOps.empty() ||
        iteration.computeOps.empty()) {
      return op->emitError()
             << "loop_plan each iteration must carry producer/consumer/"
                "compute body slices";
    }
    if (expectedAsyncCount < 0) {
      expectedAsyncCount = iteration.asyncProducerOps.size();
      expectedConsumerCount = iteration.consumerOps.size();
      expectedComputeCount = iteration.computeOps.size();
    } else if (expectedAsyncCount !=
                   static_cast<int64_t>(iteration.asyncProducerOps.size()) ||
               expectedConsumerCount !=
                   static_cast<int64_t>(iteration.consumerOps.size()) ||
               expectedComputeCount !=
                   static_cast<int64_t>(iteration.computeOps.size())) {
      return op->emitError()
             << "loop_plan requires a regularized body shape across k_group "
                "iterations";
    }

    for (int64_t opId : iteration.asyncProducerOps) {
      if (!seenOps.insert(opId).second)
        return op->emitError() << "loop_plan duplicates op " << opId;
    }
    for (int64_t opId : iteration.consumerOps) {
      if (!seenOps.insert(opId).second)
        return op->emitError() << "loop_plan duplicates op " << opId;
    }
    for (int64_t opId : iteration.computeOps) {
      if (!seenOps.insert(opId).second)
        return op->emitError() << "loop_plan duplicates op " << opId;
    }
  }

  for (const LoopCarriedValue &value : plan.carriedValues) {
    if (value.valueId < 0 || value.definingOp < 0 || value.ownerView < 0)
      return op->emitError()
             << "loop_plan carried values must carry concrete ids";
    if (value.loopDistance <= 0)
      return op->emitError()
             << "loop_plan carried values must have positive loop distance";
    if (value.users.empty())
      return op->emitError()
             << "loop_plan carried value must have cross-iteration users";
    if (!seenValues.insert(value.valueId).second)
      return op->emitError() << "loop_plan duplicates carried value "
                             << value.valueId;
    if (value.reason.empty())
      return op->emitError()
             << "loop_plan carried value reason must not be empty";
  }
  return success();
}

} // namespace

FailureOr<LoopPlan> mlir::tb::deriveLoopPlan(const BufferModel &model,
                                             Operation *op) {
  DenseMap<int64_t, const PipelineOp *> opsById;
  DenseMap<int64_t, int64_t> originalOrderByOp;
  DenseMap<int64_t, LoopIterationPlan> iterationByKGroup;
  DenseMap<int64_t, int64_t> kGroupByOp;
  int64_t maxKGroup = -1;

  for (const auto &it : llvm::enumerate(model.ops)) {
    const PipelineOp &pipelineOp = it.value();
    opsById[pipelineOp.id] = &pipelineOp;
    originalOrderByOp[pipelineOp.id] = static_cast<int64_t>(it.index());

    int64_t kGroup = findIterationCoord(pipelineOp.iterationCoords, "k_group");
    if (kGroup < 0) {
      op->emitError() << "loop regularization requires every pipeline op to "
                         "carry the `k_group` iteration coordinate";
      return failure();
    }
    maxKGroup = std::max(maxKGroup, kGroup);
    kGroupByOp[pipelineOp.id] = kGroup;

    LoopIterationPlan &iteration = iterationByKGroup[kGroup];
    iteration.kGroup = kGroup;
    switch (getPipelineOpSemanticClass(pipelineOp.kind)) {
    case PipelineOpSemanticClass::AsyncProducer:
      iteration.asyncProducerOps.push_back(pipelineOp.id);
      break;
    case PipelineOpSemanticClass::SharedConsumer:
      iteration.consumerOps.push_back(pipelineOp.id);
      break;
    case PipelineOpSemanticClass::TensorCoreCompute:
      iteration.computeOps.push_back(pipelineOp.id);
      break;
    case PipelineOpSemanticClass::EpilogueInit:
    case PipelineOpSemanticClass::EpilogueStore:
      op->emitError()
          << "loop regularization only accepts mainloop semantic classes, but "
             "op "
          << pipelineOp.id << " has semantic class `"
          << stringifyPipelineOpSemanticClass(
                 getPipelineOpSemanticClass(pipelineOp.kind))
          << "`";
      return failure();
    }
  }

  if (maxKGroup < 0) {
    op->emitError() << "loop regularization requires a non-empty pipeline body";
    return failure();
  }

  LoopPlan plan;
  plan.loopAxis = "k_group";
  plan.iterationCount = maxKGroup + 1;
  plan.singleMainLoop = true;
  plan.iterations.reserve(plan.iterationCount);

  auto sortOps = [&](SmallVectorImpl<int64_t> &ops) {
    llvm::stable_sort(ops, [&](int64_t lhs, int64_t rhs) {
      return originalOrderByOp[lhs] < originalOrderByOp[rhs];
    });
  };

  for (int64_t kGroup = 0; kGroup <= maxKGroup; ++kGroup) {
    auto it = iterationByKGroup.find(kGroup);
    if (it == iterationByKGroup.end()) {
      op->emitError() << "loop regularization requires contiguous k_group "
                         "iterations, missing "
                      << kGroup;
      return failure();
    }
    sortOps(it->second.asyncProducerOps);
    sortOps(it->second.consumerOps);
    sortOps(it->second.computeOps);
    plan.iterations.push_back(it->second);
  }

  for (const ValueState &value : model.values) {
    if (value.loopDistance <= 0 || value.definingOp < 0)
      continue;

    auto defKGroupIt = kGroupByOp.find(value.definingOp);
    if (defKGroupIt == kGroupByOp.end()) {
      op->emitError() << "loop-carried value " << value.id
                      << " must be defined by a loop-regularized op";
      return failure();
    }

    SmallVector<int64_t, 8> carriedUsers;
    for (int64_t userId : value.users) {
      auto userKGroupIt = kGroupByOp.find(userId);
      if (userKGroupIt == kGroupByOp.end())
        continue;
      if (userKGroupIt->second > defKGroupIt->second)
        carriedUsers.push_back(userId);
    }
    if (carriedUsers.empty())
      continue;

    sortOps(carriedUsers);
    plan.carriedValues.push_back(
        {value.id, value.definingOp, value.ownerView, value.loopDistance,
         carriedUsers, "cross_iteration_value"});
  }

  if (failed(validateLoopPlan(plan, op))) {
    op->emitError() << "malformed `tb.loop_plan` attribute";
    return failure();
  }
  return plan;
}

DictionaryAttr mlir::tb::buildLoopPlanAttr(Builder &builder,
                                           const LoopPlan &plan) {
  NamedAttrList attrs;
  attrs.set("loop_axis", builder.getStringAttr(plan.loopAxis));
  attrs.set("iteration_count", builder.getI64IntegerAttr(plan.iterationCount));
  attrs.set("single_main_loop", builder.getBoolAttr(plan.singleMainLoop));
  attrs.set("iterations",
            buildDictArrayAttr(builder, plan.iterations, buildIterationAttr));
  attrs.set("carried_values", buildDictArrayAttr(builder, plan.carriedValues,
                                                 buildCarriedValueAttr));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<LoopPlan> mlir::tb::parseLoopPlanAttr(Operation *op) {
  auto root = dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.loop_plan"));
  if (!root) {
    op->emitError() << "missing `tb.loop_plan` attribute";
    return failure();
  }

  LoopPlan plan;
  auto loopAxis = readStringField(root, "loop_axis", op);
  auto iterationCount = readI64Field(root, "iteration_count", op);
  auto singleMainLoop = readBoolField(root, "single_main_loop", op);
  auto iterations = parseDictArrayAttr<LoopIterationPlan>(
      root.get("iterations"), "iterations", op, parseIterationAttr);
  auto carriedValues = parseDictArrayAttr<LoopCarriedValue>(
      root.get("carried_values"), "carried_values", op, parseCarriedValueAttr);
  if (failed(loopAxis) || failed(iterationCount) || failed(singleMainLoop) ||
      failed(iterations) || failed(carriedValues)) {
    return failure();
  }

  plan.loopAxis = loopAxis->str();
  plan.iterationCount = *iterationCount;
  plan.singleMainLoop = *singleMainLoop;
  plan.iterations = std::move(*iterations);
  plan.carriedValues = std::move(*carriedValues);
  if (failed(validateLoopPlan(plan, op))) {
    op->emitError() << "malformed `tb.loop_plan` attribute";
    return failure();
  }
  return plan;
}
