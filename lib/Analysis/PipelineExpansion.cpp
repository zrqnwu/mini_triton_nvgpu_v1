#include "tb/Analysis/PipelineExpansion.h"

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

static int64_t packStageClusterKey(int64_t stage, int64_t cluster) {
  return (stage << 32) ^ cluster;
}

static StringRef stringifyClusterKind(ExpandedClusterKind kind) {
  switch (kind) {
  case ExpandedClusterKind::AsyncIssue:
    return "async_issue";
  case ExpandedClusterKind::ConsumerWait:
    return "consumer_wait";
  case ExpandedClusterKind::MmaCompute:
    return "mma_compute";
  }
  llvm_unreachable("unknown pipeline expansion cluster kind");
}

static FailureOr<ExpandedClusterKind> parseClusterKind(StringRef value,
                                                       Operation *op) {
  if (value == "async_issue")
    return ExpandedClusterKind::AsyncIssue;
  if (value == "consumer_wait")
    return ExpandedClusterKind::ConsumerWait;
  if (value == "mma_compute")
    return ExpandedClusterKind::MmaCompute;
  op->emitError() << "unknown pipeline expansion cluster kind `" << value
                  << "`";
  return failure();
}

static DictionaryAttr buildClusterAttr(Builder &builder,
                                       const ExpandedCluster &cluster) {
  NamedAttrList attrs;
  attrs.set("ordinal", builder.getI64IntegerAttr(cluster.ordinal));
  attrs.set("stage", builder.getI64IntegerAttr(cluster.stage));
  attrs.set("cluster", builder.getI64IntegerAttr(cluster.cluster));
  attrs.set("k_group", builder.getI64IntegerAttr(cluster.kGroup));
  attrs.set("kind", builder.getStringAttr(stringifyClusterKind(cluster.kind)));
  attrs.set("op_ids", buildI64ArrayAttr(builder, cluster.opIds));
  attrs.set("wait_group_ids", buildI64ArrayAttr(builder, cluster.waitGroupIds));
  attrs.set("needs_barrier", builder.getBoolAttr(cluster.needsBarrier));
  attrs.set("reason", builder.getStringAttr(cluster.reason));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<ExpandedCluster> parseClusterAttr(DictionaryAttr dict,
                                                   Operation *op) {
  ExpandedCluster cluster;
  auto ordinal = readI64Field(dict, "ordinal", op);
  auto stage = readI64Field(dict, "stage", op);
  auto clusterId = readI64Field(dict, "cluster", op);
  auto kGroup = readI64Field(dict, "k_group", op);
  auto kind = readStringField(dict, "kind", op);
  auto opIds = readDenseI64ArrayField(dict, "op_ids", op);
  auto waitGroupIds = readDenseI64ArrayField(dict, "wait_group_ids", op);
  auto needsBarrier = readBoolField(dict, "needs_barrier", op);
  auto reason = readStringField(dict, "reason", op);
  if (failed(ordinal) || failed(stage) || failed(clusterId) || failed(kGroup) ||
      failed(kind) || failed(opIds) || failed(waitGroupIds) ||
      failed(needsBarrier) || failed(reason)) {
    return failure();
  }
  auto parsedKind = parseClusterKind(*kind, op);
  if (failed(parsedKind))
    return failure();
  cluster.ordinal = *ordinal;
  cluster.stage = *stage;
  cluster.cluster = *clusterId;
  cluster.kGroup = *kGroup;
  cluster.kind = *parsedKind;
  cluster.opIds = parseI64Array(*opIds);
  cluster.waitGroupIds = parseI64Array(*waitGroupIds);
  cluster.needsBarrier = *needsBarrier;
  cluster.reason = reason->str();
  return cluster;
}

static FailureOr<ExpandedClusterKind>
classifyClusterKind(ArrayRef<const PipelineOp *> ops, ArrayRef<const WaitInfo *> waits,
                    Operation *op, int64_t stage, int64_t cluster) {
  bool sawAsyncIssue = false;
  bool sawConsumer = false;
  bool sawMma = false;

  for (const PipelineOp *pipelineOp : ops) {
    switch (getPipelineOpSemanticClass(pipelineOp->kind)) {
    case PipelineOpSemanticClass::AsyncProducer:
      sawAsyncIssue = true;
      break;
    case PipelineOpSemanticClass::SharedConsumer:
      sawConsumer = true;
      break;
    case PipelineOpSemanticClass::TensorCoreCompute:
      sawMma = true;
      break;
    case PipelineOpSemanticClass::EpilogueInit:
    case PipelineOpSemanticClass::EpilogueStore:
      op->emitError() << "pipeline expansion only accepts mainloop semantic "
                         "classes, but stage "
                      << stage << " cluster " << cluster << " contains op kind "
                      << stringifyBufferOpKind(pipelineOp->kind) << " with "
                      << "semantic class `"
                      << stringifyPipelineOpSemanticClass(
                             getPipelineOpSemanticClass(pipelineOp->kind))
                      << "`";
      return failure();
    }
  }

  int kindCount =
      static_cast<int>(sawAsyncIssue) + static_cast<int>(sawConsumer) +
      static_cast<int>(sawMma);
  if (kindCount != 1) {
    op->emitError() << "pipeline expansion does not allow mixed semantic op "
                       "kinds inside one cluster at stage "
                    << stage << " cluster " << cluster;
    return failure();
  }

  if (sawAsyncIssue) {
    if (!waits.empty()) {
      op->emitError()
          << "async-issue cluster at stage " << stage << " cluster " << cluster
          << " must not carry explicit async waits";
      return failure();
    }
    return ExpandedClusterKind::AsyncIssue;
  }

  if (sawConsumer) {
    if (waits.empty()) {
      op->emitError() << "consumer cluster at stage " << stage << " cluster "
                      << cluster
                      << " must carry explicit async wait ownership";
      return failure();
    }
    return ExpandedClusterKind::ConsumerWait;
  }

  if (!waits.empty()) {
    op->emitError() << "mma cluster at stage " << stage << " cluster " << cluster
                    << " must not carry explicit async waits";
    return failure();
  }
  return ExpandedClusterKind::MmaCompute;
}

static LogicalResult validatePipelineExpansion(const PipelineExpansion &expansion,
                                               Operation *op) {
  if (expansion.clusters.empty())
    return op->emitError() << "pipeline_expansion must contain at least one "
                              "expanded cluster";

  DenseSet<int64_t> seenOps;
  DenseSet<int64_t> seenWaitGroups;
  int64_t prevStage = -1;
  int64_t prevCluster = -1;
  for (const auto &it : llvm::enumerate(expansion.clusters)) {
    const ExpandedCluster &cluster = it.value();
    if (cluster.ordinal != static_cast<int64_t>(it.index())) {
      return op->emitError()
             << "pipeline_expansion cluster ordinals must be contiguous";
    }
    if (cluster.stage < 0 || cluster.cluster < 0) {
      return op->emitError()
             << "pipeline_expansion clusters must carry non-negative stage/"
                "cluster ids";
    }
    if (cluster.kGroup < 0) {
      return op->emitError()
             << "pipeline_expansion clusters must carry a non-negative "
                "`k_group` owner";
    }
    if (cluster.opIds.empty()) {
      return op->emitError()
             << "pipeline_expansion clusters must carry explicit op order";
    }
    if (cluster.reason.empty()) {
      return op->emitError()
             << "pipeline_expansion cluster reason must not be empty";
    }
    if (cluster.stage < prevStage ||
        (cluster.stage == prevStage && cluster.cluster < prevCluster)) {
      return op->emitError()
             << "pipeline_expansion clusters must be ordered by stage/cluster";
    }
    prevStage = cluster.stage;
    prevCluster = cluster.cluster;

    for (int64_t opId : cluster.opIds) {
      if (!seenOps.insert(opId).second) {
        return op->emitError()
               << "pipeline_expansion duplicates op " << opId;
      }
    }
    for (int64_t groupId : cluster.waitGroupIds) {
      if (!seenWaitGroups.insert(groupId).second) {
        return op->emitError()
               << "pipeline_expansion duplicates wait group " << groupId;
      }
    }

    switch (cluster.kind) {
    case ExpandedClusterKind::AsyncIssue:
    case ExpandedClusterKind::MmaCompute:
      if (!cluster.waitGroupIds.empty()) {
        return op->emitError()
               << "pipeline_expansion wait ownership can only be attached to "
                  "consumer clusters";
      }
      if (cluster.needsBarrier) {
        return op->emitError()
               << "pipeline_expansion non-consumer cluster must not request a "
                  "barrier";
      }
      break;
    case ExpandedClusterKind::ConsumerWait:
      if (cluster.waitGroupIds.empty()) {
        return op->emitError()
               << "pipeline_expansion consumer cluster must carry wait groups";
      }
      if (!cluster.needsBarrier) {
        return op->emitError()
               << "pipeline_expansion consumer cluster must carry CTA barrier "
                  "ownership";
      }
      break;
    }
  }
  return success();
}

} // namespace

FailureOr<PipelineExpansion>
mlir::tb::derivePipelineExpansion(const BufferModel &model,
                                  const PipelinePlan &pipelinePlan,
                                  const AsyncPlan &asyncPlan, Operation *op) {
  DenseMap<int64_t, const PipelineOp *> opsById;
  DenseMap<int64_t, SmallVector<const PipelinePlacement *, 8>> placementsByKey;
  DenseMap<int64_t, SmallVector<const WaitInfo *, 4>> waitsByKey;
  DenseSet<int64_t> seenPlacedOps;

  for (const PipelineOp &pipelineOp : model.ops)
    opsById[pipelineOp.id] = &pipelineOp;

  for (const PipelinePlacement &placement : pipelinePlan.placements) {
    if (!opsById.count(placement.opId)) {
      op->emitError() << "pipeline plan references unknown op " << placement.opId;
      return failure();
    }
    if (!seenPlacedOps.insert(placement.opId).second) {
      op->emitError() << "pipeline expansion requires unique op placements";
      return failure();
    }
    placementsByKey[packStageClusterKey(placement.stage, placement.cluster)]
        .push_back(&placement);
  }

  for (const WaitInfo &wait : asyncPlan.waits)
    waitsByKey[packStageClusterKey(wait.requiredStage, wait.requiredCluster)]
        .push_back(&wait);

  if (seenPlacedOps.size() != model.ops.size()) {
    op->emitError()
        << "pipeline expansion requires every pipeline op to be scheduled";
    return failure();
  }

  SmallVector<std::pair<int64_t, int64_t>, 16> stageClusters;
  stageClusters.reserve(placementsByKey.size());
  for (const auto &[key, placements] : placementsByKey) {
    (void)key;
    if (placements.empty())
      continue;
    stageClusters.emplace_back(placements.front()->stage,
                               placements.front()->cluster);
  }
  llvm::sort(stageClusters);

  PipelineExpansion expansion;
  expansion.clusters.reserve(stageClusters.size());
  for (const auto &stageCluster : llvm::enumerate(stageClusters)) {
    int64_t stage = stageCluster.value().first;
    int64_t clusterId = stageCluster.value().second;
    int64_t clusterKey = packStageClusterKey(stage, clusterId);
    auto placementsIt = placementsByKey.find(clusterKey);
    if (placementsIt == placementsByKey.end()) {
      op->emitError() << "missing pipeline placement cluster for stage " << stage
                      << " cluster " << clusterId;
      return failure();
    }

    auto &placements = placementsIt->second;
    llvm::stable_sort(placements, [](const PipelinePlacement *lhs,
                                     const PipelinePlacement *rhs) {
      return lhs->order < rhs->order;
    });

    auto waitsIt = waitsByKey.find(clusterKey);
    ArrayRef<const WaitInfo *> waits =
        waitsIt == waitsByKey.end()
            ? ArrayRef<const WaitInfo *>{}
            : ArrayRef<const WaitInfo *>(waitsIt->second);
    if (waitsIt != waitsByKey.end()) {
      llvm::stable_sort(waitsIt->second, [](const WaitInfo *lhs,
                                            const WaitInfo *rhs) {
        return lhs->requiredOrder < rhs->requiredOrder;
      });
      waits = waitsIt->second;
    }

    SmallVector<const PipelineOp *, 32> pipelineOps;
    pipelineOps.reserve(placements.size());
    ExpandedCluster cluster;
    cluster.ordinal = stageCluster.index();
    cluster.stage = stage;
    cluster.cluster = clusterId;
    cluster.kGroup = -1;
    cluster.needsBarrier = false;
    cluster.reason = placements.front()->reason;
    for (const PipelinePlacement *placement : placements) {
      const PipelineOp *pipelineOp = opsById.lookup(placement->opId);
      pipelineOps.push_back(pipelineOp);
      cluster.opIds.push_back(placement->opId);
      if (!pipelineOp) {
        op->emitError() << "missing pipeline op " << placement->opId
                        << " while deriving pipeline_expansion";
        return failure();
      }
      int64_t opKGroup = findIterationCoord(pipelineOp->iterationCoords, "k_group");
      if (opKGroup < 0) {
        op->emitError() << "pipeline expansion op " << placement->opId
                        << " is missing a `k_group` coordinate";
        return failure();
      }
      if (cluster.kGroup < 0) {
        cluster.kGroup = opKGroup;
      } else if (cluster.kGroup != opKGroup) {
        op->emitError() << "pipeline expansion cluster at stage " << stage
                        << " cluster " << clusterId << " mixes k_group "
                        << cluster.kGroup << " and " << opKGroup;
        return failure();
      }
    }
    for (const WaitInfo *wait : waits) {
      if (!llvm::is_contained(cluster.opIds, wait->beforeOpId)) {
        op->emitError() << "expanded wait group " << wait->groupId
                        << " must anchor inside its consumer cluster";
        return failure();
      }
      cluster.waitGroupIds.push_back(wait->groupId);
      cluster.needsBarrier = cluster.needsBarrier || wait->needsBarrier;
    }

    auto kind = classifyClusterKind(pipelineOps, waits, op, stage, clusterId);
    if (failed(kind))
      return failure();
    cluster.kind = *kind;
    expansion.clusters.push_back(std::move(cluster));
  }

  for (const auto &[key, waits] : waitsByKey) {
    if (waits.empty())
      continue;
    if (!placementsByKey.count(key)) {
      const WaitInfo *wait = waits.front();
      op->emitError() << "wait group " << wait->groupId
                      << " has no expanded consumer cluster";
      return failure();
    }
  }

  if (failed(validatePipelineExpansion(expansion, op))) {
    op->emitError() << "malformed `tb.pipeline_expansion` attribute";
    return failure();
  }
  return expansion;
}

DictionaryAttr mlir::tb::buildPipelineExpansionAttr(
    Builder &builder, const PipelineExpansion &expansion) {
  SmallVector<Attribute> clusterAttrs;
  clusterAttrs.reserve(expansion.clusters.size());
  for (const ExpandedCluster &cluster : expansion.clusters)
    clusterAttrs.push_back(buildClusterAttr(builder, cluster));

  NamedAttrList attrs;
  attrs.set("clusters", builder.getArrayAttr(clusterAttrs));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<PipelineExpansion> mlir::tb::parsePipelineExpansionAttr(Operation *op) {
  auto root =
      dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.pipeline_expansion"));
  if (!root) {
    op->emitError() << "missing `tb.pipeline_expansion` attribute";
    return failure();
  }

  auto array = dyn_cast_or_null<ArrayAttr>(root.get("clusters"));
  if (!array) {
    op->emitError() << "missing array field `clusters`";
    return failure();
  }

  PipelineExpansion expansion;
  expansion.clusters.reserve(array.size());
  for (Attribute attr : array) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict) {
      op->emitError()
          << "`tb.pipeline_expansion.clusters` must contain only dictionaries";
      return failure();
    }
    auto cluster = parseClusterAttr(dict, op);
    if (failed(cluster))
      return failure();
    expansion.clusters.push_back(std::move(*cluster));
  }

  if (failed(validatePipelineExpansion(expansion, op))) {
    op->emitError() << "malformed `tb.pipeline_expansion` attribute";
    return failure();
  }
  return expansion;
}
