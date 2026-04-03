#include "tb/Analysis/AccumulatorPlan.h"

#include "tb/Analysis/KernelConfig.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::tb;

namespace {

static DenseI64ArrayAttr buildI64ArrayAttr(Builder &builder,
                                           ArrayRef<int64_t> values) {
  return builder.getDenseI64ArrayAttr(values);
}

static SmallVector<int64_t> parseI64Array(DenseI64ArrayAttr attr) {
  return SmallVector<int64_t>(attr.asArrayRef().begin(), attr.asArrayRef().end());
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

static DictionaryAttr buildLaneAccessAttr(Builder &builder,
                                          const LaneAccessPattern &pattern) {
  NamedAttrList attrs;
  attrs.set("lane_row_group_size",
            builder.getI64IntegerAttr(pattern.laneRowGroupSize));
  attrs.set("lane_col_group_size",
            builder.getI64IntegerAttr(pattern.laneColGroupSize));
  attrs.set("lane_col_stride",
            builder.getI64IntegerAttr(pattern.laneColStride));
  attrs.set("row_offsets", buildI64ArrayAttr(builder, pattern.rowOffsets));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<LaneAccessPattern> parseLaneAccessAttr(DictionaryAttr dict,
                                                        Operation *op) {
  LaneAccessPattern pattern;
  auto laneRowGroupSize = readI64Field(dict, "lane_row_group_size", op);
  auto laneColGroupSize = readI64Field(dict, "lane_col_group_size", op);
  auto laneColStride = readI64Field(dict, "lane_col_stride", op);
  auto rowOffsets = readDenseI64ArrayField(dict, "row_offsets", op);
  if (failed(laneRowGroupSize) || failed(laneColGroupSize) ||
      failed(laneColStride) || failed(rowOffsets)) {
    return failure();
  }
  pattern.laneRowGroupSize = *laneRowGroupSize;
  pattern.laneColGroupSize = *laneColGroupSize;
  pattern.laneColStride = *laneColStride;
  pattern.rowOffsets = parseI64Array(*rowOffsets);
  return pattern;
}

static DictionaryAttr buildPackAttr(Builder &builder,
                                    const AccumulatorPack &pack) {
  NamedAttrList attrs;
  attrs.set("pack_id", builder.getI64IntegerAttr(pack.packId));
  attrs.set("row_base", builder.getI64IntegerAttr(pack.rowBase));
  attrs.set("col_base", builder.getI64IntegerAttr(pack.colBase));
  attrs.set("rows", builder.getI64IntegerAttr(pack.rows));
  attrs.set("cols", builder.getI64IntegerAttr(pack.cols));
  attrs.set("elem_count", builder.getI64IntegerAttr(pack.elemCount));
  attrs.set("vector_width", builder.getI64IntegerAttr(pack.vectorWidth));
  attrs.set("warp_owner", builder.getI64IntegerAttr(pack.warpOwner));
  attrs.set("mma_group", builder.getI64IntegerAttr(pack.mmaGroup));
  attrs.set("epilogue_pack", builder.getI64IntegerAttr(pack.epiloguePack));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<AccumulatorPack> parsePackAttr(DictionaryAttr dict,
                                                Operation *op) {
  AccumulatorPack pack;
  auto packId = readI64Field(dict, "pack_id", op);
  auto rowBase = readI64Field(dict, "row_base", op);
  auto colBase = readI64Field(dict, "col_base", op);
  auto rows = readI64Field(dict, "rows", op);
  auto cols = readI64Field(dict, "cols", op);
  auto elemCount = readI64Field(dict, "elem_count", op);
  auto vectorWidth = readI64Field(dict, "vector_width", op);
  auto warpOwner = readI64Field(dict, "warp_owner", op);
  auto mmaGroup = readI64Field(dict, "mma_group", op);
  auto epiloguePack = readI64Field(dict, "epilogue_pack", op);
  if (failed(packId) || failed(rowBase) || failed(colBase) || failed(rows) ||
      failed(cols) || failed(elemCount) || failed(vectorWidth) ||
      failed(warpOwner) || failed(mmaGroup) || failed(epiloguePack)) {
    return failure();
  }
  pack.packId = *packId;
  pack.rowBase = *rowBase;
  pack.colBase = *colBase;
  pack.rows = *rows;
  pack.cols = *cols;
  pack.elemCount = *elemCount;
  pack.vectorWidth = *vectorWidth;
  pack.warpOwner = *warpOwner;
  pack.mmaGroup = *mmaGroup;
  pack.epiloguePack = *epiloguePack;
  return pack;
}

static LogicalResult validateAccumulatorPlan(const AccumulatorPlan &plan,
                                             Operation *op) {
  if (plan.encoding < 0)
    return op->emitError() << "accumulator plan must reference a valid encoding";
  if (plan.registersPerWarp <= 0)
    return op->emitError()
           << "accumulator plan registers_per_warp must be positive";
  if (plan.laneAccess.laneRowGroupSize <= 0 ||
      plan.laneAccess.laneColGroupSize <= 0 ||
      plan.laneAccess.laneColStride <= 0 ||
      plan.laneAccess.rowOffsets.empty()) {
    return op->emitError()
           << "accumulator plan must carry explicit positive lane access truth";
  }
  if (plan.packs.empty())
    return op->emitError() << "accumulator plan must carry packs";
  if (plan.multiBufferDepth <= 0)
    return op->emitError()
           << "accumulator plan multi_buffer_depth must be positive";
  if (plan.ownerScope != "per_warp_template") {
    return op->emitError() << "accumulator plan currently requires "
                              "`per_warp_template` owner scope";
  }

  llvm::DenseSet<int64_t> seenPackIds;
  int64_t expectedVectorWidth = -1;
  for (const AccumulatorPack &pack : plan.packs) {
    if (pack.packId < 0 || !seenPackIds.insert(pack.packId).second) {
      return op->emitError()
             << "accumulator plan pack ids must be unique and non-negative";
    }
    if (pack.rows <= 0 || pack.cols <= 0 || pack.elemCount <= 0 ||
        pack.vectorWidth <= 0) {
      return op->emitError()
             << "accumulator plan pack geometry must be positive";
    }
    if (pack.warpOwner < 0 || pack.mmaGroup < 0 || pack.epiloguePack < -1) {
      return op->emitError()
             << "accumulator plan owner metadata must be non-negative";
    }
    if (expectedVectorWidth < 0)
      expectedVectorWidth = pack.vectorWidth;
    if (pack.vectorWidth != expectedVectorWidth) {
      return op->emitError()
             << "accumulator plan packs must keep a uniform vector width";
    }
  }

  return success();
}

} // namespace

FailureOr<AccumulatorPlan>
mlir::tb::deriveAccumulatorPlan(const KernelConfig &config,
                                const TargetInfo &target,
                                const EncodingPlan &encodings, Operation *op) {
  (void)target;
  if (config.cScalar != ScalarKind::F32) {
    op->emitError() << "V1 direct C path expects fp32 accumulators";
    return failure();
  }
  auto warpTileShape = getAccumulatorTileShape(encodings, op);
  if (failed(getAccumulatorEncodingAttr(encodings, encodings.acc, op, "acc")) ||
      failed(warpTileShape) ||
      encodings.fragmentAcc.instructionShape.size() < 2) {
    op->emitError() << "malformed accumulator encoding plan";
    return failure();
  }

  constexpr int64_t kVectorWidth = 2;
  int64_t packRows = encodings.fragmentAcc.instructionShape.front();
  int64_t packCols = encodings.fragmentAcc.instructionShape[1];
  if (packRows <= 0 || packCols <= 0) {
    op->emitError() << "invalid accumulator pack shape";
    return failure();
  }
  if ((*warpTileShape)[0] % packRows != 0 ||
      (*warpTileShape)[1] % packCols != 0) {
    op->emitError()
        << "warp tile is incompatible with the accumulator pack grid";
    return failure();
  }

  AccumulatorPlan accumulator;
  accumulator.encoding = encodings.acc;
  accumulator.registersPerWarp =
      ((*warpTileShape)[0] / packRows) * ((*warpTileShape)[1] / packCols);
  accumulator.ownerScope = "per_warp_template";
  accumulator.laneAccess = {/*laneRowGroupSize=*/4,
                            /*laneColGroupSize=*/4,
                            /*laneColStride=*/kVectorWidth,
                            /*rowOffsets=*/{0, packRows / 2}};
  accumulator.liveAcrossStages = false;
  accumulator.multiBufferDepth = 1;

  int64_t packId = 0;
  for (int64_t rowBase = 0; rowBase < (*warpTileShape)[0]; rowBase += packRows) {
    for (int64_t colBase = 0; colBase < (*warpTileShape)[1];
         colBase += packCols) {
      int64_t mmaGroup = (rowBase / packRows) * ((*warpTileShape)[1] / packCols) +
                         (colBase / packCols);
      accumulator.packs.push_back({packId++, rowBase, colBase, packRows,
                                   packCols, packRows * packCols,
                                   kVectorWidth,
                                   /*warpOwner=*/0, mmaGroup,
                                   /*epiloguePack=*/-1});
    }
  }
  if (failed(validateAccumulatorPlan(accumulator, op)))
    return failure();
  return accumulator;
}

DictionaryAttr mlir::tb::buildAccumulatorPlanAttr(Builder &builder,
                                                  const AccumulatorPlan &plan) {
  NamedAttrList attrs;
  attrs.set("encoding", builder.getI64IntegerAttr(plan.encoding));
  attrs.set("registers_per_warp", builder.getI64IntegerAttr(plan.registersPerWarp));
  attrs.set("owner_scope", builder.getStringAttr(plan.ownerScope));
  attrs.set("lane_access", buildLaneAccessAttr(builder, plan.laneAccess));
  SmallVector<Attribute> packs;
  packs.reserve(plan.packs.size());
  for (const AccumulatorPack &pack : plan.packs)
    packs.push_back(buildPackAttr(builder, pack));
  attrs.set("packs", builder.getArrayAttr(packs));
  attrs.set("live_across_stages", builder.getBoolAttr(plan.liveAcrossStages));
  attrs.set("multi_buffer_depth", builder.getI64IntegerAttr(plan.multiBufferDepth));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<AccumulatorPlan> mlir::tb::parseAccumulatorPlanAttr(Operation *op) {
  auto root =
      dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.accumulator_plan"));
  if (!root) {
    op->emitError() << "missing `tb.accumulator_plan` attribute";
    return failure();
  }
  AccumulatorPlan plan;
  auto encoding = readI64Field(root, "encoding", op);
  auto registersPerWarp = readI64Field(root, "registers_per_warp", op);
  auto ownerScope = readStringField(root, "owner_scope", op);
  auto laneAccess = dyn_cast_or_null<DictionaryAttr>(root.get("lane_access"));
  auto packsAttr = dyn_cast_or_null<ArrayAttr>(root.get("packs"));
  auto liveAcrossStages = readBoolField(root, "live_across_stages", op);
  auto multiBufferDepth = readI64Field(root, "multi_buffer_depth", op);
  if (failed(encoding) || failed(registersPerWarp) || failed(ownerScope) ||
      !laneAccess || !packsAttr || failed(liveAcrossStages) ||
      failed(multiBufferDepth)) {
    op->emitError() << "malformed `tb.accumulator_plan` attribute";
    return failure();
  }
  auto parsedLaneAccess = parseLaneAccessAttr(laneAccess, op);
  if (failed(parsedLaneAccess))
    return failure();
  plan.encoding = static_cast<int>(*encoding);
  plan.registersPerWarp = *registersPerWarp;
  plan.ownerScope = ownerScope->str();
  plan.laneAccess = std::move(*parsedLaneAccess);
  plan.liveAcrossStages = *liveAcrossStages;
  plan.multiBufferDepth = *multiBufferDepth;
  plan.packs.reserve(packsAttr.size());
  for (Attribute attr : packsAttr) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict) {
      op->emitError() << "`packs` must contain dictionary entries";
      return failure();
    }
    auto pack = parsePackAttr(dict, op);
    if (failed(pack))
      return failure();
    plan.packs.push_back(std::move(*pack));
  }
  if (failed(validateAccumulatorPlan(plan, op))) {
    op->emitError() << "malformed `tb.accumulator_plan` attribute";
    return failure();
  }
  return plan;
}
