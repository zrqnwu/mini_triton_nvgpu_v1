#include "tb/Analysis/PipelineReady.h"

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

static LogicalResult validatePipelineReady(const PipelineReady &ready,
                                           Operation *op) {
  if (ready.scheduledMaxStage < 0)
    return op->emitError()
           << "pipeline_ready.scheduled_max_stage must be non-negative";
  if (ready.asyncGroups < 0)
    return op->emitError() << "pipeline_ready.async_groups must be non-negative";
  if (ready.requestedStages <= 0)
    return op->emitError()
           << "pipeline_ready.requested_stages must be positive";
  return success();
}

} // namespace

DictionaryAttr mlir::tb::buildPipelineReadyAttr(Builder &builder,
                                                const PipelineReady &ready) {
  NamedAttrList attrs;
  attrs.set("scheduled_max_stage",
            builder.getI64IntegerAttr(ready.scheduledMaxStage));
  attrs.set("async_groups", builder.getI64IntegerAttr(ready.asyncGroups));
  attrs.set("requested_stages", builder.getI64IntegerAttr(ready.requestedStages));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<PipelineReady> mlir::tb::parsePipelineReadyAttr(Operation *op) {
  auto root = dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.pipeline_ready"));
  if (!root) {
    op->emitError() << "missing `tb.pipeline_ready` attribute";
    return failure();
  }

  PipelineReady ready;
  auto scheduledMaxStage = readI64Field(root, "scheduled_max_stage", op);
  auto asyncGroups = readI64Field(root, "async_groups", op);
  auto requestedStages = readI64Field(root, "requested_stages", op);
  if (failed(scheduledMaxStage) || failed(asyncGroups) ||
      failed(requestedStages)) {
    op->emitError() << "malformed `tb.pipeline_ready` attribute";
    return failure();
  }

  ready.scheduledMaxStage = *scheduledMaxStage;
  ready.asyncGroups = *asyncGroups;
  ready.requestedStages = *requestedStages;
  if (failed(validatePipelineReady(ready, op))) {
    op->emitError() << "malformed `tb.pipeline_ready` attribute";
    return failure();
  }
  return ready;
}
