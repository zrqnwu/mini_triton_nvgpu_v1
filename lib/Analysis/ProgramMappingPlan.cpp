#include "tb/Analysis/ProgramMappingPlan.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::tb;

namespace {

static int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  return (lhs + rhs - 1) / rhs;
}

static DenseI64ArrayAttr buildI64ArrayAttr(Builder &builder,
                                           ArrayRef<int64_t> values) {
  return builder.getDenseI64ArrayAttr(values);
}

static SmallVector<int64_t> parseI64Array(DenseI64ArrayAttr attr) {
  return SmallVector<int64_t>(attr.asArrayRef().begin(), attr.asArrayRef().end());
}

static bool isPermutation(ArrayRef<int64_t> values) {
  SmallVector<bool, 8> seen(values.size(), false);
  for (int64_t value : values) {
    if (value < 0 || value >= static_cast<int64_t>(values.size()) ||
        seen[value]) {
      return false;
    }
    seen[value] = true;
  }
  return true;
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

static FailureOr<DenseI64ArrayAttr> readDenseI64ArrayField(DictionaryAttr dict,
                                                           StringRef name,
                                                           Operation *op) {
  auto attr = dyn_cast_or_null<DenseI64ArrayAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing dense i64 array field `" << name << "`";
    return failure();
  }
  return attr;
}

static FailureOr<int64_t> readOptionalModuleI64Attr(Operation *op, StringRef name,
                                                    int64_t defaultValue) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return defaultValue;
  if (auto attr = dyn_cast_or_null<IntegerAttr>(module->getAttr(name)))
    return attr.getInt();
  if (module->hasAttr(name)) {
    op->emitError() << "module attr `" << name << "` must be an i64 attribute";
    return failure();
  }
  return defaultValue;
}

static FailureOr<bool> readOptionalModuleBoolAttr(Operation *op, StringRef name,
                                                  bool defaultValue) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return defaultValue;
  if (auto attr = dyn_cast_or_null<BoolAttr>(module->getAttr(name)))
    return attr.getValue();
  if (module->hasAttr(name)) {
    op->emitError() << "module attr `" << name << "` must be a bool attribute";
    return failure();
  }
  return defaultValue;
}

static FailureOr<std::string> readOptionalModuleStringAttr(Operation *op,
                                                           StringRef name,
                                                           StringRef defaultValue) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return defaultValue.str();
  if (auto attr = dyn_cast_or_null<StringAttr>(module->getAttr(name)))
    return attr.getValue().str();
  if (module->hasAttr(name)) {
    op->emitError() << "module attr `" << name << "` must be a string attribute";
    return failure();
  }
  return defaultValue.str();
}

static StringRef stringifyProgramMappingKind(ProgramMappingKind kind) {
  switch (kind) {
  case ProgramMappingKind::Tile:
    return "tile";
  case ProgramMappingKind::GroupedTile:
    return "grouped_tile";
  case ProgramMappingKind::SplitK:
    return "split_k";
  case ProgramMappingKind::PersistentTile:
    return "persistent_tile";
  }
  llvm_unreachable("unknown program mapping kind");
}

static FailureOr<ProgramMappingKind> parseProgramMappingKind(StringRef value,
                                                             Operation *op) {
  if (value == "tile")
    return ProgramMappingKind::Tile;
  if (value == "grouped_tile")
    return ProgramMappingKind::GroupedTile;
  if (value == "split_k")
    return ProgramMappingKind::SplitK;
  if (value == "persistent_tile")
    return ProgramMappingKind::PersistentTile;
  op->emitError() << "unknown program mapping kind `" << value << "`";
  return failure();
}

static StringRef stringifyLaunchOrder(ProgramLaunchOrder order) {
  switch (order) {
  case ProgramLaunchOrder::RowMajor:
    return "row_major";
  case ProgramLaunchOrder::GroupedM:
    return "grouped_m";
  case ProgramLaunchOrder::Persistent:
    return "persistent";
  }
  llvm_unreachable("unknown launch order");
}

static FailureOr<ProgramLaunchOrder> parseLaunchOrder(StringRef value,
                                                      Operation *op) {
  if (value == "row_major")
    return ProgramLaunchOrder::RowMajor;
  if (value == "grouped_m")
    return ProgramLaunchOrder::GroupedM;
  if (value == "persistent")
    return ProgramLaunchOrder::Persistent;
  op->emitError() << "unknown program launch order `" << value << "`";
  return failure();
}

static StringRef stringifySwizzleKind(ProgramSwizzleKind kind) {
  switch (kind) {
  case ProgramSwizzleKind::None:
    return "none";
  case ProgramSwizzleKind::GroupedM:
    return "grouped_m";
  }
  llvm_unreachable("unknown swizzle kind");
}

static FailureOr<ProgramSwizzleKind> parseSwizzleKind(StringRef value,
                                                      Operation *op) {
  if (value == "none")
    return ProgramSwizzleKind::None;
  if (value == "grouped_m")
    return ProgramSwizzleKind::GroupedM;
  op->emitError() << "unknown program swizzle kind `" << value << "`";
  return failure();
}

static StringRef stringifyReductionMode(ReductionMode mode) {
  switch (mode) {
  case ReductionMode::None:
    return "none";
  case ReductionMode::SplitKSerial:
    return "split_k_serial";
  case ReductionMode::SplitKParallel:
    return "split_k_parallel";
  }
  llvm_unreachable("unknown reduction mode");
}

static FailureOr<ReductionMode> parseReductionMode(StringRef value,
                                                   Operation *op) {
  if (value == "none")
    return ReductionMode::None;
  if (value == "split_k_serial")
    return ReductionMode::SplitKSerial;
  if (value == "split_k_parallel")
    return ReductionMode::SplitKParallel;
  op->emitError() << "unknown reduction mode `" << value << "`";
  return failure();
}

static LogicalResult validateProgramMappingPlan(const ProgramMappingPlan &plan,
                                               Operation *op) {
  if (plan.problemTilesM <= 0 || plan.problemTilesN <= 0 ||
      plan.problemTilesK <= 0)
    return op->emitError() << "program mapping must carry positive problem tile coverage";
  if (plan.tileM <= 0 || plan.tileN <= 0 || plan.tileK <= 0)
    return op->emitError() << "program mapping tile shape must be positive";
  if (plan.groupM <= 0 || plan.groupTileSpanM <= 0 || plan.groupTileSpanN <= 0)
    return op->emitError() << "program mapping group spans must be positive";
  if (plan.programsPerLaunchGroup <= 0 || plan.launchGroupCount <= 0 ||
      plan.totalPrograms <= 0) {
    return op->emitError() << "program mapping launch formula must be positive";
  }
  if (plan.programsPerLaunchGroup !=
      plan.groupTileSpanM * plan.groupTileSpanN) {
    return op->emitError() << "program mapping launch-group width must equal "
                              "group_tile_span_m * group_tile_span_n";
  }
  if (plan.totalPrograms >
      plan.launchGroupCount * plan.programsPerLaunchGroup) {
    return op->emitError() << "program mapping total_programs must not exceed "
                              "launch_group_count * programs_per_launch_group";
  }
  if (plan.numCTAs <= 0)
    return op->emitError() << "program mapping num_ctas must be positive";
  if (plan.ctasPerCGA.size() != 3 || plan.ctaSplitNum.size() != 3 ||
      plan.ctaOrder.size() != 3) {
    return op->emitError() << "program mapping cga metadata must have rank 3";
  }
  if (llvm::any_of(plan.ctasPerCGA, [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(plan.ctaSplitNum, [](int64_t value) { return value <= 0; })) {
    return op->emitError() << "program mapping cga metadata must be positive";
  }
  if (!isPermutation(plan.ctaOrder))
    return op->emitError() << "program mapping cta_order must be a permutation";

  switch (plan.mappingKind) {
  case ProgramMappingKind::Tile:
    if (plan.groupM != 1 || plan.groupTileSpanM != 1 ||
        plan.groupTileSpanN != plan.problemTilesN ||
        plan.launchOrder != ProgramLaunchOrder::RowMajor ||
        plan.swizzleKind != ProgramSwizzleKind::None ||
        plan.launchGroupCount != plan.problemTilesM ||
        plan.totalPrograms != plan.problemTilesM * plan.problemTilesN) {
      return op->emitError()
             << "tile mapping contract is inconsistent with its launch formula";
    }
    break;
  case ProgramMappingKind::GroupedTile:
    if (plan.totalPrograms != plan.problemTilesM * plan.problemTilesN) {
      return op->emitError()
             << "grouped_tile total_programs must equal problem_tiles_m * "
                "problem_tiles_n";
    }
    if (plan.groupM <= 1 || plan.groupTileSpanM != plan.groupM ||
        plan.groupTileSpanN != plan.problemTilesN ||
        plan.launchOrder != ProgramLaunchOrder::GroupedM ||
        plan.swizzleKind != ProgramSwizzleKind::GroupedM ||
        plan.launchGroupCount != ceilDiv(plan.problemTilesM, plan.groupM)) {
      return op->emitError()
             << "grouped_tile mapping contract is inconsistent with its launch formula";
    }
    if (plan.launchGroupCount > 1 &&
        plan.totalPrograms <=
            (plan.launchGroupCount - 1) * plan.programsPerLaunchGroup) {
      return op->emitError()
             << "grouped_tile total_programs must include a non-empty tail "
                "or an exact final launch group";
    }
    break;
  case ProgramMappingKind::SplitK:
    if (plan.splitK <= 1 || plan.persistent ||
        plan.reductionMode == ReductionMode::None ||
        plan.programsPerTile != plan.splitK ||
        plan.groupM != 1 ||
        plan.groupTileSpanM != 1 ||
        plan.groupTileSpanN != plan.problemTilesN * plan.splitK ||
        plan.programsPerLaunchGroup != plan.groupTileSpanN ||
        plan.launchOrder != ProgramLaunchOrder::RowMajor ||
        plan.swizzleKind != ProgramSwizzleKind::None ||
        plan.launchGroupCount != plan.problemTilesM ||
        plan.totalPrograms !=
            plan.problemTilesM * plan.problemTilesN * plan.splitK) {
      return op->emitError()
             << "split_k mapping contract is inconsistent with its launch formula";
    }
    break;
  case ProgramMappingKind::PersistentTile:
    if (!plan.persistent || plan.splitK != 1 ||
        plan.reductionMode != ReductionMode::None ||
        plan.programsPerTile != 1 || plan.groupM != 1 ||
        plan.groupTileSpanM != 1 ||
        plan.groupTileSpanN != plan.problemTilesN ||
        plan.programsPerLaunchGroup != plan.groupTileSpanN ||
        plan.launchOrder != ProgramLaunchOrder::Persistent ||
        plan.swizzleKind != ProgramSwizzleKind::None ||
        plan.launchGroupCount != plan.problemTilesM ||
        plan.totalPrograms != plan.problemTilesM * plan.problemTilesN ||
        plan.numCTAs > plan.totalPrograms) {
      return op->emitError()
             << "persistent_tile mapping contract is inconsistent with its work formula";
    }
    break;
  }
  return success();
}

} // namespace

FailureOr<ProgramMappingPlan>
mlir::tb::deriveProgramMappingPlan(const KernelConfig &config,
                                   const TargetInfo &target, Operation *op) {
  auto requestedNumCTAs = readOptionalModuleI64Attr(op, kTBNumCTAsAttrName, 1);
  auto requestedSplitK = readOptionalModuleI64Attr(op, kTBSplitKModuleAttrName, 1);
  auto requestedPersistent =
      readOptionalModuleBoolAttr(op, kTBPersistentModuleAttrName, false);
  auto requestedProgramsPerTile = readOptionalModuleI64Attr(
      op, kTBProgramsPerTileModuleAttrName, 0);
  auto requestedReductionMode =
      readOptionalModuleStringAttr(op, kTBReductionModeModuleAttrName, "");
  if (failed(requestedNumCTAs))
    return failure();
  if (failed(requestedSplitK) || failed(requestedPersistent) ||
      failed(requestedProgramsPerTile) || failed(requestedReductionMode)) {
    return failure();
  }

  ProgramMappingPlan plan;
  plan.problemTilesM = ceilDiv(config.problemM, config.blockM);
  plan.problemTilesN = ceilDiv(config.problemN, config.blockN);
  plan.problemTilesK = ceilDiv(config.problemK, config.blockK);
  plan.tileM = config.blockM;
  plan.tileN = config.blockN;
  plan.tileK = config.blockK;
  plan.groupM = std::max<int64_t>(1, config.groupM);
  plan.groupTileSpanM = 1;
  plan.groupTileSpanN = plan.problemTilesN;
  plan.programsPerLaunchGroup = plan.groupTileSpanM * plan.groupTileSpanN;
  plan.launchGroupCount = plan.problemTilesM;
  plan.totalPrograms = plan.problemTilesM * plan.problemTilesN;
  plan.splitK = std::max<int64_t>(*requestedSplitK, 1);
  plan.persistent = *requestedPersistent;
  plan.programsPerTile = 1;
  plan.numCTAs = *requestedNumCTAs;
  plan.ctasPerCGA = {plan.numCTAs, 1, 1};
  plan.ctaSplitNum = {1, 1, 1};
  plan.ctaOrder = {0, 1, 2};

  if (plan.problemTilesM <= 0 || plan.problemTilesN <= 0 || plan.problemTilesK <= 0) {
    op->emitError() << "program mapping requires positive tile coverage on M/N/K";
    return failure();
  }
  if (plan.splitK <= 0) {
    op->emitError() << "module attr `" << kTBSplitKModuleAttrName
                    << "` must be positive";
    return failure();
  }
  if (plan.persistent && plan.splitK != 1) {
    op->emitError() << "stage1 does not allow persistent and split-k in the same "
                       "program mapping contract yet";
    return failure();
  }
  if (plan.numCTAs <= 0) {
    op->emitError() << "module attr `" << kTBNumCTAsAttrName
                    << "` must be positive";
    return failure();
  }
  if (*requestedProgramsPerTile > 0)
    plan.programsPerTile = *requestedProgramsPerTile;

  if (plan.persistent) {
    plan.mappingKind = ProgramMappingKind::PersistentTile;
    plan.launchOrder = ProgramLaunchOrder::Persistent;
    plan.swizzleKind = ProgramSwizzleKind::None;
    plan.groupM = 1;
    plan.groupTileSpanM = 1;
    plan.groupTileSpanN = plan.problemTilesN;
    plan.programsPerLaunchGroup = plan.groupTileSpanN;
    plan.launchGroupCount = plan.problemTilesM;
    plan.totalPrograms = plan.problemTilesM * plan.problemTilesN;
    plan.programsPerTile = 1;
    plan.reductionMode = ReductionMode::None;
    plan.numCTAs = std::min<int64_t>(plan.numCTAs, plan.totalPrograms);
  } else if (plan.splitK > 1) {
    plan.mappingKind = ProgramMappingKind::SplitK;
    plan.launchOrder = ProgramLaunchOrder::RowMajor;
    plan.swizzleKind = ProgramSwizzleKind::None;
    plan.groupM = 1;
    plan.groupTileSpanM = 1;
    plan.groupTileSpanN = plan.problemTilesN * plan.splitK;
    plan.programsPerLaunchGroup = plan.groupTileSpanN;
    plan.launchGroupCount = plan.problemTilesM;
    plan.totalPrograms = plan.problemTilesM * plan.problemTilesN * plan.splitK;
    plan.programsPerTile = plan.splitK;
    if (requestedReductionMode->empty() ||
        *requestedReductionMode == "split_k_serial") {
      plan.reductionMode = ReductionMode::SplitKSerial;
    } else if (*requestedReductionMode == "split_k_parallel") {
      plan.reductionMode = ReductionMode::SplitKParallel;
    } else {
      op->emitError()
          << "module attr `" << kTBReductionModeModuleAttrName
          << "` must be `split_k_serial` or `split_k_parallel`";
      return failure();
    }
  } else if (plan.groupM > 1 && plan.problemTilesM > 1 &&
             plan.problemTilesN > 1) {
    plan.mappingKind = ProgramMappingKind::GroupedTile;
    plan.launchOrder = ProgramLaunchOrder::GroupedM;
    plan.swizzleKind = ProgramSwizzleKind::GroupedM;
    plan.groupTileSpanM = plan.groupM;
    plan.groupTileSpanN = plan.problemTilesN;
    plan.programsPerLaunchGroup = plan.groupTileSpanM * plan.groupTileSpanN;
    plan.launchGroupCount = ceilDiv(plan.problemTilesM, plan.groupM);
    plan.totalPrograms = plan.problemTilesM * plan.problemTilesN;
  } else {
    plan.mappingKind = ProgramMappingKind::Tile;
    plan.launchOrder = ProgramLaunchOrder::RowMajor;
    plan.swizzleKind = ProgramSwizzleKind::None;
    plan.groupM = 1;
    plan.groupTileSpanM = 1;
    plan.groupTileSpanN = plan.problemTilesN;
    plan.programsPerLaunchGroup = plan.groupTileSpanN;
    plan.launchGroupCount = plan.problemTilesM;
    plan.totalPrograms = plan.problemTilesM * plan.problemTilesN;
  }

  if (target.numSms > 0 && config.numWarps > target.maxWarpsPerCTA) {
    op->emitError() << "num_warps exceeds target max warps per CTA";
    return failure();
  }
  if (plan.persistent && target.numSms > 0 && plan.numCTAs > target.numSms) {
    plan.numCTAs = target.numSms;
    plan.ctasPerCGA = {plan.numCTAs, 1, 1};
  }

  if (failed(validateProgramMappingPlan(plan, op)))
    return failure();
  return plan;
}

DictionaryAttr mlir::tb::buildProgramMappingPlanAttr(
    Builder &builder, const ProgramMappingPlan &plan) {
  NamedAttrList attrs;
  attrs.set("mapping_kind",
            builder.getStringAttr(stringifyProgramMappingKind(plan.mappingKind)));
  attrs.set("problem_tiles_m", builder.getI64IntegerAttr(plan.problemTilesM));
  attrs.set("problem_tiles_n", builder.getI64IntegerAttr(plan.problemTilesN));
  attrs.set("problem_tiles_k", builder.getI64IntegerAttr(plan.problemTilesK));
  attrs.set("tile_m", builder.getI64IntegerAttr(plan.tileM));
  attrs.set("tile_n", builder.getI64IntegerAttr(plan.tileN));
  attrs.set("tile_k", builder.getI64IntegerAttr(plan.tileK));
  attrs.set("group_m", builder.getI64IntegerAttr(plan.groupM));
  attrs.set("group_tile_span_m",
            builder.getI64IntegerAttr(plan.groupTileSpanM));
  attrs.set("group_tile_span_n",
            builder.getI64IntegerAttr(plan.groupTileSpanN));
  attrs.set("programs_per_launch_group",
            builder.getI64IntegerAttr(plan.programsPerLaunchGroup));
  attrs.set("launch_group_count",
            builder.getI64IntegerAttr(plan.launchGroupCount));
  attrs.set("total_programs", builder.getI64IntegerAttr(plan.totalPrograms));
  attrs.set("split_k", builder.getI64IntegerAttr(plan.splitK));
  attrs.set("launch_order",
            builder.getStringAttr(stringifyLaunchOrder(plan.launchOrder)));
  attrs.set("swizzle_kind",
            builder.getStringAttr(stringifySwizzleKind(plan.swizzleKind)));
  attrs.set("persistent", builder.getBoolAttr(plan.persistent));
  attrs.set("reduction_mode",
            builder.getStringAttr(stringifyReductionMode(plan.reductionMode)));
  attrs.set("programs_per_tile",
            builder.getI64IntegerAttr(plan.programsPerTile));
  attrs.set("num_ctas", builder.getI64IntegerAttr(plan.numCTAs));
  attrs.set("ctas_per_cga", buildI64ArrayAttr(builder, plan.ctasPerCGA));
  attrs.set("cta_split_num", buildI64ArrayAttr(builder, plan.ctaSplitNum));
  attrs.set("cta_order", buildI64ArrayAttr(builder, plan.ctaOrder));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<ProgramMappingPlan> mlir::tb::parseProgramMappingPlanAttr(
    Operation *op) {
  auto root =
      dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.program_mapping_plan"));
  if (!root) {
    op->emitError() << "missing `tb.program_mapping_plan` attribute";
    return failure();
  }

  ProgramMappingPlan plan;
  auto mappingKind = readStringField(root, "mapping_kind", op);
  auto problemTilesM = readI64Field(root, "problem_tiles_m", op);
  auto problemTilesN = readI64Field(root, "problem_tiles_n", op);
  auto problemTilesK = readI64Field(root, "problem_tiles_k", op);
  auto tileM = readI64Field(root, "tile_m", op);
  auto tileN = readI64Field(root, "tile_n", op);
  auto tileK = readI64Field(root, "tile_k", op);
  auto groupM = readI64Field(root, "group_m", op);
  auto groupTileSpanM = readI64Field(root, "group_tile_span_m", op);
  auto groupTileSpanN = readI64Field(root, "group_tile_span_n", op);
  auto programsPerLaunchGroup =
      readI64Field(root, "programs_per_launch_group", op);
  auto launchGroupCount = readI64Field(root, "launch_group_count", op);
  auto totalPrograms = readI64Field(root, "total_programs", op);
  auto splitK = readI64Field(root, "split_k", op);
  auto launchOrder = readStringField(root, "launch_order", op);
  auto swizzleKind = readStringField(root, "swizzle_kind", op);
  auto persistent = readBoolField(root, "persistent", op);
  auto reductionMode = readStringField(root, "reduction_mode", op);
  auto programsPerTile = readI64Field(root, "programs_per_tile", op);
  auto numCTAs = readI64Field(root, "num_ctas", op);
  auto ctasPerCGA = readDenseI64ArrayField(root, "ctas_per_cga", op);
  auto ctaSplitNum = readDenseI64ArrayField(root, "cta_split_num", op);
  auto ctaOrder = readDenseI64ArrayField(root, "cta_order", op);
  if (failed(mappingKind) || failed(problemTilesM) || failed(problemTilesN) ||
      failed(problemTilesK) || failed(tileM) || failed(tileN) || failed(tileK) ||
      failed(groupM) || failed(groupTileSpanM) || failed(groupTileSpanN) ||
      failed(programsPerLaunchGroup) || failed(launchGroupCount) ||
      failed(totalPrograms) || failed(splitK) || failed(launchOrder) ||
      failed(swizzleKind) || failed(persistent) || failed(reductionMode) ||
      failed(programsPerTile) || failed(numCTAs) || failed(ctasPerCGA) ||
      failed(ctaSplitNum) || failed(ctaOrder)) {
    op->emitError() << "malformed `tb.program_mapping_plan` attribute";
    return failure();
  }

  auto parsedMappingKind = parseProgramMappingKind(*mappingKind, op);
  auto parsedLaunchOrder = parseLaunchOrder(*launchOrder, op);
  auto parsedSwizzleKind = parseSwizzleKind(*swizzleKind, op);
  auto parsedReductionMode = parseReductionMode(*reductionMode, op);
  if (failed(parsedMappingKind) || failed(parsedLaunchOrder) ||
      failed(parsedSwizzleKind) || failed(parsedReductionMode)) {
    return failure();
  }

  plan.mappingKind = *parsedMappingKind;
  plan.problemTilesM = *problemTilesM;
  plan.problemTilesN = *problemTilesN;
  plan.problemTilesK = *problemTilesK;
  plan.tileM = *tileM;
  plan.tileN = *tileN;
  plan.tileK = *tileK;
  plan.groupM = *groupM;
  plan.groupTileSpanM = *groupTileSpanM;
  plan.groupTileSpanN = *groupTileSpanN;
  plan.programsPerLaunchGroup = *programsPerLaunchGroup;
  plan.launchGroupCount = *launchGroupCount;
  plan.totalPrograms = *totalPrograms;
  plan.splitK = *splitK;
  plan.launchOrder = *parsedLaunchOrder;
  plan.swizzleKind = *parsedSwizzleKind;
  plan.persistent = *persistent;
  plan.reductionMode = *parsedReductionMode;
  plan.programsPerTile = *programsPerTile;
  plan.numCTAs = *numCTAs;
  plan.ctasPerCGA = parseI64Array(*ctasPerCGA);
  plan.ctaSplitNum = parseI64Array(*ctaSplitNum);
  plan.ctaOrder = parseI64Array(*ctaOrder);
  if (failed(validateProgramMappingPlan(plan, op))) {
    op->emitError() << "malformed `tb.program_mapping_plan` attribute";
    return failure();
  }
  return plan;
}
