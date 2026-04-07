#include "tb/Analysis/PersistentWorkPlan.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::tb;

namespace {

static int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  return (lhs + rhs - 1) / rhs;
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

static StringRef stringifyPersistentWorkKind(PersistentWorkKind kind) {
  switch (kind) {
  case PersistentWorkKind::None:
    return "none";
  case PersistentWorkKind::TileStrideLoop:
    return "tile_stride_loop";
  }
  llvm_unreachable("unknown persistent work kind");
}

static FailureOr<PersistentWorkKind> parsePersistentWorkKind(StringRef value,
                                                             Operation *op) {
  if (value == "none")
    return PersistentWorkKind::None;
  if (value == "tile_stride_loop")
    return PersistentWorkKind::TileStrideLoop;
  op->emitError() << "unknown persistent work kind `" << value << "`";
  return failure();
}

static LogicalResult validatePersistentWorkPlan(const PersistentWorkPlan &plan,
                                                Operation *op) {
  if (plan.residentPrograms <= 0 || plan.programsPerWave <= 0 ||
      plan.tileBatchSize <= 0 || plan.maxTilesPerProgram <= 0 ||
      plan.totalWorkItems <= 0 || plan.loopStride <= 0 || plan.reason.empty()) {
    return op->emitError()
           << "persistent work plan must carry positive owner truth";
  }
  if (!plan.enabled) {
    if (plan.kind != PersistentWorkKind::None ||
        plan.requiresOuterSerialLoop || plan.ownerScope != "none" ||
        plan.completion != "single_tile") {
      return op->emitError()
             << "disabled persistent work plan must not carry loop ownership";
    }
    return success();
  }
  if (plan.kind != PersistentWorkKind::TileStrideLoop ||
      !plan.requiresOuterSerialLoop || plan.ownerScope == "none") {
    return op->emitError()
           << "enabled persistent work plan must carry explicit tile-loop ownership";
  }
  return success();
}

} // namespace

FailureOr<PersistentWorkPlan> mlir::tb::derivePersistentWorkPlan(
    const KernelConfig &config, const TargetInfo &target,
    const ProgramMappingPlan &programMapping, Operation *op) {
  (void)config;
  PersistentWorkPlan plan;
  plan.residentPrograms = std::max<int64_t>(programMapping.numCTAs, 1);
  plan.programsPerWave = plan.residentPrograms;
  plan.totalWorkItems = std::max<int64_t>(programMapping.totalPrograms, 1);
  plan.loopStride = plan.residentPrograms;
  plan.reason = "single-tile CTA launch keeps one program per tile";

  if (!programMapping.persistent) {
    plan.kind = PersistentWorkKind::None;
    plan.enabled = false;
    plan.tileBatchSize = 1;
    plan.maxTilesPerProgram = 1;
    plan.ownerScope = "none";
    if (failed(validatePersistentWorkPlan(plan, op)))
      return failure();
    return plan;
  }

  int64_t residentCap = target.numSms > 0 ? target.numSms : programMapping.numCTAs;
  plan.kind = PersistentWorkKind::TileStrideLoop;
  plan.enabled = true;
  plan.residentPrograms =
      std::max<int64_t>(1, std::min<int64_t>(programMapping.numCTAs, residentCap));
  plan.programsPerWave = plan.residentPrograms;
  plan.tileBatchSize = std::max<int64_t>(1, programMapping.programsPerTile);
  plan.totalWorkItems = std::max<int64_t>(programMapping.totalPrograms, 1);
  plan.loopStride = plan.residentPrograms;
  plan.maxTilesPerProgram = ceilDiv(plan.totalWorkItems, plan.residentPrograms);
  plan.requiresOuterSerialLoop = true;
  plan.tileOrder = "row_major";
  plan.ownerScope = "cta_persistent_loop";
  plan.completion = "exhaust_program_stride";
  plan.reason =
      "persistent launch keeps resident CTAs fixed and advances each program "
      "through a row-major tile stride loop";
  if (failed(validatePersistentWorkPlan(plan, op)))
    return failure();
  return plan;
}

DictionaryAttr mlir::tb::buildPersistentWorkPlanAttr(
    Builder &builder, const PersistentWorkPlan &plan) {
  NamedAttrList attrs;
  attrs.set("kind", builder.getStringAttr(stringifyPersistentWorkKind(plan.kind)));
  attrs.set("enabled", builder.getBoolAttr(plan.enabled));
  attrs.set("resident_programs", builder.getI64IntegerAttr(plan.residentPrograms));
  attrs.set("programs_per_wave", builder.getI64IntegerAttr(plan.programsPerWave));
  attrs.set("tile_batch_size", builder.getI64IntegerAttr(plan.tileBatchSize));
  attrs.set("max_tiles_per_program",
            builder.getI64IntegerAttr(plan.maxTilesPerProgram));
  attrs.set("total_work_items", builder.getI64IntegerAttr(plan.totalWorkItems));
  attrs.set("loop_stride", builder.getI64IntegerAttr(plan.loopStride));
  attrs.set("requires_outer_serial_loop",
            builder.getBoolAttr(plan.requiresOuterSerialLoop));
  attrs.set("tile_order", builder.getStringAttr(plan.tileOrder));
  attrs.set("owner_scope", builder.getStringAttr(plan.ownerScope));
  attrs.set("completion", builder.getStringAttr(plan.completion));
  attrs.set("reason", builder.getStringAttr(plan.reason));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<PersistentWorkPlan>
mlir::tb::parsePersistentWorkPlanAttr(Operation *op) {
  auto root =
      dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.persistent_work_plan"));
  if (!root) {
    op->emitError() << "missing `tb.persistent_work_plan` attribute";
    return failure();
  }

  PersistentWorkPlan plan;
  auto kind = readStringField(root, "kind", op);
  auto enabled = readBoolField(root, "enabled", op);
  auto residentPrograms = readI64Field(root, "resident_programs", op);
  auto programsPerWave = readI64Field(root, "programs_per_wave", op);
  auto tileBatchSize = readI64Field(root, "tile_batch_size", op);
  auto maxTilesPerProgram = readI64Field(root, "max_tiles_per_program", op);
  auto totalWorkItems = readI64Field(root, "total_work_items", op);
  auto loopStride = readI64Field(root, "loop_stride", op);
  auto requiresOuterSerialLoop =
      readBoolField(root, "requires_outer_serial_loop", op);
  auto tileOrder = readStringField(root, "tile_order", op);
  auto ownerScope = readStringField(root, "owner_scope", op);
  auto completion = readStringField(root, "completion", op);
  auto reason = readStringField(root, "reason", op);
  if (failed(kind) || failed(enabled) || failed(residentPrograms) ||
      failed(programsPerWave) || failed(tileBatchSize) ||
      failed(maxTilesPerProgram) || failed(totalWorkItems) ||
      failed(loopStride) || failed(requiresOuterSerialLoop) ||
      failed(tileOrder) || failed(ownerScope) || failed(completion) ||
      failed(reason)) {
    op->emitError() << "malformed `tb.persistent_work_plan` attribute";
    return failure();
  }

  auto parsedKind = parsePersistentWorkKind(*kind, op);
  if (failed(parsedKind))
    return failure();
  plan.kind = *parsedKind;
  plan.enabled = *enabled;
  plan.residentPrograms = *residentPrograms;
  plan.programsPerWave = *programsPerWave;
  plan.tileBatchSize = *tileBatchSize;
  plan.maxTilesPerProgram = *maxTilesPerProgram;
  plan.totalWorkItems = *totalWorkItems;
  plan.loopStride = *loopStride;
  plan.requiresOuterSerialLoop = *requiresOuterSerialLoop;
  plan.tileOrder = tileOrder->str();
  plan.ownerScope = ownerScope->str();
  plan.completion = completion->str();
  plan.reason = reason->str();
  if (failed(validatePersistentWorkPlan(plan, op))) {
    op->emitError() << "malformed `tb.persistent_work_plan` attribute";
    return failure();
  }
  return plan;
}
