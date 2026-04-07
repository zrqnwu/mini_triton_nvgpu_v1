#include "tb/Analysis/ReductionPlan.h"

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

static StringRef stringifyReductionPlanKind(ReductionPlanKind kind) {
  switch (kind) {
  case ReductionPlanKind::None:
    return "none";
  case ReductionPlanKind::SplitKSerial:
    return "split_k_serial";
  case ReductionPlanKind::SplitKAtomic:
    return "split_k_atomic";
  }
  llvm_unreachable("unknown reduction plan kind");
}

static FailureOr<ReductionPlanKind> parseReductionPlanKind(StringRef value,
                                                           Operation *op) {
  if (value == "none")
    return ReductionPlanKind::None;
  if (value == "split_k_serial")
    return ReductionPlanKind::SplitKSerial;
  if (value == "split_k_atomic")
    return ReductionPlanKind::SplitKAtomic;
  op->emitError() << "unknown reduction plan kind `" << value << "`";
  return failure();
}

static StringRef stringifyReductionScratchSpace(ReductionScratchSpace space) {
  switch (space) {
  case ReductionScratchSpace::None:
    return "none";
  case ReductionScratchSpace::Global:
    return "global";
  case ReductionScratchSpace::Shared:
    return "shared";
  }
  llvm_unreachable("unknown reduction scratch space");
}

static FailureOr<ReductionScratchSpace>
parseReductionScratchSpace(StringRef value, Operation *op) {
  if (value == "none")
    return ReductionScratchSpace::None;
  if (value == "global")
    return ReductionScratchSpace::Global;
  if (value == "shared")
    return ReductionScratchSpace::Shared;
  op->emitError() << "unknown reduction scratch space `" << value << "`";
  return failure();
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

static LogicalResult validateReductionPlan(const ReductionPlan &plan,
                                          Operation *op) {
  if (plan.splitK <= 0 || plan.kTilesPerProgram < 0 ||
      plan.remainderKTiles < 0 || plan.partialTileRows < 0 ||
      plan.partialTileCols < 0 || plan.producerProgramsPerTile <= 0 ||
      plan.finalReducerPrograms <= 0 || plan.scratchRows < 0 ||
      plan.scratchCols < 0 || plan.scratchBytes < 0 || plan.reason.empty()) {
    return op->emitError()
           << "reduction plan must carry explicit non-negative owner truth";
  }
  if (plan.kind == ReductionPlanKind::None) {
    if (plan.splitK != 1 || plan.requiresInterProgramReduction ||
        plan.scratchSpace != ReductionScratchSpace::None ||
        plan.scratchBytes != 0 || plan.partialOwner != "none" ||
        plan.finalExecutor != "none") {
      return op->emitError()
             << "non-split reduction plan must not carry split-k payload";
    }
    return success();
  }
  if (plan.splitK <= 1 || plan.kTilesPerProgram <= 0 ||
      plan.partialTileRows <= 0 || plan.partialTileCols <= 0 ||
      !plan.requiresInterProgramReduction || plan.partialOwner == "none" ||
      plan.finalExecutor == "none") {
    return op->emitError()
           << "split-k reduction plan must carry explicit producer/final owner truth";
  }
  if (plan.scratchSpace == ReductionScratchSpace::None || plan.scratchBytes <= 0) {
    return op->emitError()
           << "split-k reduction plan must carry concrete scratch ownership";
  }
  return success();
}

} // namespace

FailureOr<ReductionPlan>
mlir::tb::deriveReductionPlan(const KernelConfig &config,
                              const ProgramMappingPlan &programMapping,
                              Operation *op) {
  ReductionPlan plan;
  plan.partialTileRows = config.blockM;
  plan.partialTileCols = config.blockN;
  plan.kTilesPerProgram = programMapping.problemTilesK;
  plan.reason = "single-program CTA fully owns the K reduction for each tile";

  if (programMapping.splitK <= 1) {
    plan.kind = ReductionPlanKind::None;
    plan.splitK = 1;
    plan.producerProgramsPerTile = 1;
    plan.finalReducerPrograms = 1;
    plan.partialOwner = "none";
    plan.finalExecutor = "none";
    if (failed(validateReductionPlan(plan, op)))
      return failure();
    return plan;
  }

  plan.splitK = programMapping.splitK;
  plan.kTilesPerProgram = ceilDiv(programMapping.problemTilesK, plan.splitK);
  plan.remainderKTiles = programMapping.problemTilesK % plan.splitK;
  plan.producerProgramsPerTile = programMapping.programsPerTile;
  plan.finalReducerPrograms = 1;
  plan.requiresInterProgramReduction = true;
  plan.partialOwner = "per_program_partial_tile";
  plan.scratchSpace = ReductionScratchSpace::Global;
  plan.scratchRows = config.blockM;
  plan.scratchCols = config.blockN;
  plan.scratchBytes = config.blockM * config.blockN * getScalarByteWidth(config.cScalar) *
                      programMapping.splitK;
  switch (programMapping.reductionMode) {
  case ReductionMode::SplitKSerial:
    plan.kind = ReductionPlanKind::SplitKSerial;
    plan.finalExecutor = "serial_tail_program";
    plan.reason =
        "split-k serial maps multiple programs to one tile and keeps a single "
        "final reducer contract";
    break;
  case ReductionMode::SplitKParallel:
    plan.kind = ReductionPlanKind::SplitKAtomic;
    plan.finalExecutor = "global_atomic_accumulate";
    plan.reason =
        "split-k parallel maps multiple programs to one tile and reduces via "
        "global atomic accumulation";
    break;
  case ReductionMode::None:
    op->emitError() << "split-k program mapping requires a reduction mode";
    return failure();
  }

  if (failed(validateReductionPlan(plan, op)))
    return failure();
  return plan;
}

DictionaryAttr mlir::tb::buildReductionPlanAttr(Builder &builder,
                                                const ReductionPlan &plan) {
  NamedAttrList attrs;
  attrs.set("kind", builder.getStringAttr(stringifyReductionPlanKind(plan.kind)));
  attrs.set("scratch_space",
            builder.getStringAttr(stringifyReductionScratchSpace(plan.scratchSpace)));
  attrs.set("split_k", builder.getI64IntegerAttr(plan.splitK));
  attrs.set("k_tiles_per_program",
            builder.getI64IntegerAttr(plan.kTilesPerProgram));
  attrs.set("remainder_k_tiles",
            builder.getI64IntegerAttr(plan.remainderKTiles));
  attrs.set("partial_tile_rows",
            builder.getI64IntegerAttr(plan.partialTileRows));
  attrs.set("partial_tile_cols",
            builder.getI64IntegerAttr(plan.partialTileCols));
  attrs.set("producer_programs_per_tile",
            builder.getI64IntegerAttr(plan.producerProgramsPerTile));
  attrs.set("final_reducer_programs",
            builder.getI64IntegerAttr(plan.finalReducerPrograms));
  attrs.set("scratch_rows", builder.getI64IntegerAttr(plan.scratchRows));
  attrs.set("scratch_cols", builder.getI64IntegerAttr(plan.scratchCols));
  attrs.set("scratch_bytes", builder.getI64IntegerAttr(plan.scratchBytes));
  attrs.set("requires_inter_program_reduction",
            builder.getBoolAttr(plan.requiresInterProgramReduction));
  attrs.set("partial_owner", builder.getStringAttr(plan.partialOwner));
  attrs.set("final_executor", builder.getStringAttr(plan.finalExecutor));
  attrs.set("reason", builder.getStringAttr(plan.reason));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<ReductionPlan> mlir::tb::parseReductionPlanAttr(Operation *op) {
  auto root = dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.reduction_plan"));
  if (!root) {
    op->emitError() << "missing `tb.reduction_plan` attribute";
    return failure();
  }

  ReductionPlan plan;
  auto kind = readStringField(root, "kind", op);
  auto scratchSpace = readStringField(root, "scratch_space", op);
  auto splitK = readI64Field(root, "split_k", op);
  auto kTilesPerProgram = readI64Field(root, "k_tiles_per_program", op);
  auto remainderKTiles = readI64Field(root, "remainder_k_tiles", op);
  auto partialTileRows = readI64Field(root, "partial_tile_rows", op);
  auto partialTileCols = readI64Field(root, "partial_tile_cols", op);
  auto producerProgramsPerTile =
      readI64Field(root, "producer_programs_per_tile", op);
  auto finalReducerPrograms = readI64Field(root, "final_reducer_programs", op);
  auto scratchRows = readI64Field(root, "scratch_rows", op);
  auto scratchCols = readI64Field(root, "scratch_cols", op);
  auto scratchBytes = readI64Field(root, "scratch_bytes", op);
  auto requiresInterProgramReduction =
      readBoolField(root, "requires_inter_program_reduction", op);
  auto partialOwner = readStringField(root, "partial_owner", op);
  auto finalExecutor = readStringField(root, "final_executor", op);
  auto reason = readStringField(root, "reason", op);
  if (failed(kind) || failed(scratchSpace) || failed(splitK) ||
      failed(kTilesPerProgram) || failed(remainderKTiles) ||
      failed(partialTileRows) || failed(partialTileCols) ||
      failed(producerProgramsPerTile) || failed(finalReducerPrograms) ||
      failed(scratchRows) || failed(scratchCols) || failed(scratchBytes) ||
      failed(requiresInterProgramReduction) || failed(partialOwner) ||
      failed(finalExecutor) || failed(reason)) {
    op->emitError() << "malformed `tb.reduction_plan` attribute";
    return failure();
  }

  auto parsedKind = parseReductionPlanKind(*kind, op);
  auto parsedScratchSpace = parseReductionScratchSpace(*scratchSpace, op);
  if (failed(parsedKind) || failed(parsedScratchSpace))
    return failure();

  plan.kind = *parsedKind;
  plan.scratchSpace = *parsedScratchSpace;
  plan.splitK = *splitK;
  plan.kTilesPerProgram = *kTilesPerProgram;
  plan.remainderKTiles = *remainderKTiles;
  plan.partialTileRows = *partialTileRows;
  plan.partialTileCols = *partialTileCols;
  plan.producerProgramsPerTile = *producerProgramsPerTile;
  plan.finalReducerPrograms = *finalReducerPrograms;
  plan.scratchRows = *scratchRows;
  plan.scratchCols = *scratchCols;
  plan.scratchBytes = *scratchBytes;
  plan.requiresInterProgramReduction = *requiresInterProgramReduction;
  plan.partialOwner = partialOwner->str();
  plan.finalExecutor = finalExecutor->str();
  plan.reason = reason->str();
  if (failed(validateReductionPlan(plan, op))) {
    op->emitError() << "malformed `tb.reduction_plan` attribute";
    return failure();
  }
  return plan;
}
