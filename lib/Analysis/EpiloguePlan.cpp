#include "tb/Analysis/EpiloguePlan.h"

#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/KernelConfig.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::tb;

namespace {

static bool hasBoundaryOnM(const KernelConfig &config) {
  return (config.problemM % config.blockM) != 0;
}

static bool hasBoundaryOnN(const KernelConfig &config) {
  return (config.problemN % config.blockN) != 0;
}

static bool hasScalarTailOnN(const KernelConfig &config, int64_t vectorWidth) {
  if (!hasBoundaryOnN(config) || vectorWidth <= 0)
    return false;
  int64_t remainderN = config.problemN % config.blockN;
  if (remainderN == 0)
    return false;
  return (remainderN % vectorWidth) != 0;
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

static bool isSupportedGlobalVectorWidth(int64_t vectorWidth) {
  return vectorWidth == 2 || vectorWidth == 4;
}

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

static FailureOr<StringRef> readStringField(DictionaryAttr dict, StringRef name,
                                            Operation *op) {
  auto attr = dyn_cast_or_null<StringAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing string field `" << name << "`";
    return failure();
  }
  return attr.getValue();
}

static bool readOptionalBoolField(DictionaryAttr dict, StringRef name,
                                  bool defaultValue) {
  auto attr = dyn_cast_or_null<BoolAttr>(dict.get(name));
  return attr ? attr.getValue() : defaultValue;
}

static std::string readOptionalStringField(DictionaryAttr dict, StringRef name,
                                           StringRef defaultValue) {
  auto attr = dyn_cast_or_null<StringAttr>(dict.get(name));
  return attr ? attr.getValue().str() : defaultValue.str();
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

static StringRef stringifyInitMode(AccumulatorInitMode mode) {
  switch (mode) {
  case AccumulatorInitMode::Zero:
    return "zero";
  case AccumulatorInitMode::DirectGlobalVector:
    return "direct_global_vector";
  case AccumulatorInitMode::SharedRelay:
    return "shared_relay";
  }
  llvm_unreachable("unknown init mode");
}

static FailureOr<AccumulatorInitMode> parseInitMode(StringRef value,
                                                    Operation *op) {
  if (value == "zero")
    return AccumulatorInitMode::Zero;
  if (value == "direct_global_vector")
    return AccumulatorInitMode::DirectGlobalVector;
  if (value == "shared_relay")
    return AccumulatorInitMode::SharedRelay;
  op->emitError() << "unknown accumulator init mode `" << value << "`";
  return failure();
}

static StringRef stringifyStoreMode(AccumulatorStoreMode mode) {
  switch (mode) {
  case AccumulatorStoreMode::DirectGlobalVector:
    return "direct_global_vector";
  case AccumulatorStoreMode::SharedRelay:
    return "shared_relay";
  }
  llvm_unreachable("unknown store mode");
}

static FailureOr<AccumulatorStoreMode> parseStoreMode(StringRef value,
                                                      Operation *op) {
  if (value == "direct_global_vector")
    return AccumulatorStoreMode::DirectGlobalVector;
  if (value == "shared_relay")
    return AccumulatorStoreMode::SharedRelay;
  op->emitError() << "unknown accumulator store mode `" << value << "`";
  return failure();
}

static StringRef stringifyTargetLandingKind(TargetLandingKind kind) {
  switch (kind) {
  case TargetLandingKind::None:
    return "none";
  case TargetLandingKind::RegisterPackGlobalVector:
    return "register_pack_global_vector";
  case TargetLandingKind::SharedPackThenGlobalVector:
    return "shared_pack_then_global_vector";
  case TargetLandingKind::SharedRelayThenGlobalVector:
    return "shared_relay_then_global_vector";
  }
  llvm_unreachable("unknown epilogue target landing kind");
}

static FailureOr<TargetLandingKind> parseTargetLandingKind(StringRef value,
                                                           Operation *op) {
  if (value == "none")
    return TargetLandingKind::None;
  if (value == "register_pack_global_vector")
    return TargetLandingKind::RegisterPackGlobalVector;
  if (value == "shared_pack_then_global_vector")
    return TargetLandingKind::SharedPackThenGlobalVector;
  if (value == "shared_relay_then_global_vector")
    return TargetLandingKind::SharedRelayThenGlobalVector;
  op->emitError() << "unknown epilogue target landing kind `" << value << "`";
  return failure();
}

static StringRef stringifyExprKind(EpilogueExprKind kind) {
  switch (kind) {
  case EpilogueExprKind::LoadBias:
    return "load_bias";
  case EpilogueExprKind::Add:
    return "add";
  case EpilogueExprKind::Convert:
    return "convert";
  case EpilogueExprKind::Activation:
    return "activation";
  case EpilogueExprKind::Clamp:
    return "clamp";
  }
  llvm_unreachable("unknown epilogue expr kind");
}

static FailureOr<EpilogueExprKind> parseExprKind(StringRef value,
                                                 Operation *op) {
  if (value == "load_bias")
    return EpilogueExprKind::LoadBias;
  if (value == "add")
    return EpilogueExprKind::Add;
  if (value == "convert")
    return EpilogueExprKind::Convert;
  if (value == "activation")
    return EpilogueExprKind::Activation;
  if (value == "clamp")
    return EpilogueExprKind::Clamp;
  op->emitError() << "unknown epilogue expr kind `" << value << "`";
  return failure();
}

static StringRef stringifyScalarKindLocal(ScalarKind kind) {
  return stringifyScalarKind(kind);
}

static FailureOr<ScalarKind> parseScalarKindLocal(StringRef value,
                                                  Operation *op) {
  if (value == "f16")
    return ScalarKind::F16;
  if (value == "f32")
    return ScalarKind::F32;
  op->emitError() << "unknown epilogue expr scalar kind `" << value << "`";
  return failure();
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
  if (failed(packId) || failed(rowBase) || failed(colBase) || failed(rows) ||
      failed(cols) || failed(elemCount) || failed(vectorWidth))
    return failure();
  pack.packId = *packId;
  pack.rowBase = *rowBase;
  pack.colBase = *colBase;
  pack.rows = *rows;
  pack.cols = *cols;
  pack.elemCount = *elemCount;
  pack.vectorWidth = *vectorWidth;
  return pack;
}

static DictionaryAttr buildDirectPlanAttr(Builder &builder,
                                          const DirectGlobalVectorPlan &plan) {
  auto buildDirectPackAttr = [&](const DirectGlobalVectorPlan::Pack &pack) {
    NamedAttrList attrs;
    attrs.set("pack_id", builder.getI64IntegerAttr(pack.packId));
    attrs.set("row_base", builder.getI64IntegerAttr(pack.rowBase));
    attrs.set("col_base", builder.getI64IntegerAttr(pack.colBase));
    attrs.set("rows", builder.getI64IntegerAttr(pack.rows));
    attrs.set("cols", builder.getI64IntegerAttr(pack.cols));
    attrs.set("vector_width", builder.getI64IntegerAttr(pack.vectorWidth));
    attrs.set("warp_owner", builder.getI64IntegerAttr(pack.warpOwner));
    attrs.set("fragment_ids", buildI64ArrayAttr(builder, pack.fragmentIds));
    return builder.getDictionaryAttr(attrs);
  };

  NamedAttrList attrs;
  attrs.set("owner_scope", builder.getStringAttr(plan.ownerScope));
  attrs.set("vector_width", builder.getI64IntegerAttr(plan.vectorWidth));
  attrs.set("boundary_aware", builder.getBoolAttr(plan.boundaryAware));
  attrs.set("scalar_tail", builder.getBoolAttr(plan.scalarTail));
  attrs.set("lane_access", buildLaneAccessAttr(builder, plan.laneAccess));
  SmallVector<Attribute> packs;
  packs.reserve(plan.packs.size());
  for (const DirectGlobalVectorPlan::Pack &pack : plan.packs)
    packs.push_back(buildDirectPackAttr(pack));
  attrs.set("packs", builder.getArrayAttr(packs));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<DirectGlobalVectorPlan> parseDirectPlanAttr(DictionaryAttr dict,
                                                             Operation *op) {
  auto parseDirectPackAttr =
      [&](DictionaryAttr dictAttr) -> FailureOr<DirectGlobalVectorPlan::Pack> {
    DirectGlobalVectorPlan::Pack pack;
    auto packId = readI64Field(dictAttr, "pack_id", op);
    auto rowBase = readI64Field(dictAttr, "row_base", op);
    auto colBase = readI64Field(dictAttr, "col_base", op);
    auto rows = readI64Field(dictAttr, "rows", op);
    auto cols = readI64Field(dictAttr, "cols", op);
    auto vectorWidth = readI64Field(dictAttr, "vector_width", op);
    auto warpOwner = readI64Field(dictAttr, "warp_owner", op);
    auto fragmentIds = readDenseI64ArrayField(dictAttr, "fragment_ids", op);
    if (failed(packId) || failed(rowBase) || failed(colBase) || failed(rows) ||
        failed(cols) || failed(vectorWidth) || failed(warpOwner) ||
        failed(fragmentIds)) {
      return failure();
    }
    pack.packId = *packId;
    pack.rowBase = *rowBase;
    pack.colBase = *colBase;
    pack.rows = *rows;
    pack.cols = *cols;
    pack.vectorWidth = *vectorWidth;
    pack.warpOwner = *warpOwner;
    pack.fragmentIds = parseI64Array(*fragmentIds);
    return pack;
  };

  DirectGlobalVectorPlan plan;
  plan.ownerScope = readOptionalStringField(dict, "owner_scope",
                                            "per_warp_template");
  auto vectorWidth = readI64Field(dict, "vector_width", op);
  auto boundaryAware = readOptionalBoolField(dict, "boundary_aware", false);
  auto scalarTail = readOptionalBoolField(dict, "scalar_tail", false);
  auto laneAccess = dyn_cast_or_null<DictionaryAttr>(dict.get("lane_access"));
  auto packs = dyn_cast_or_null<ArrayAttr>(dict.get("packs"));
  if (failed(vectorWidth) || !laneAccess || !packs) {
    op->emitError() << "malformed direct global vector epilogue payload";
    return failure();
  }
  auto parsedLaneAccess = parseLaneAccessAttr(laneAccess, op);
  if (failed(parsedLaneAccess))
    return failure();
  plan.vectorWidth = *vectorWidth;
  plan.boundaryAware = boundaryAware;
  plan.scalarTail = scalarTail;
  plan.laneAccess = std::move(*parsedLaneAccess);
  plan.packs.reserve(packs.size());
  for (Attribute attr : packs) {
    auto dictAttr = dyn_cast<DictionaryAttr>(attr);
    if (!dictAttr) {
      op->emitError() << "`packs` must contain dictionary entries";
      return failure();
    }
    auto pack = parseDirectPackAttr(dictAttr);
    if (failed(pack))
      return failure();
    plan.packs.push_back(std::move(*pack));
  }
  return plan;
}

static DictionaryAttr buildSharedRelayPlanAttr(Builder &builder,
                                               const SharedRelayPlan &plan) {
  NamedAttrList attrs;
  attrs.set("relay_encoding", builder.getI64IntegerAttr(plan.relayEncoding));
  attrs.set("logical_shape", buildI64ArrayAttr(builder, plan.logicalShape));
  attrs.set("alloc_shape", buildI64ArrayAttr(builder, plan.allocShape));
  SmallVector<Attribute> packs;
  packs.reserve(plan.packs.size());
  for (const AccumulatorPack &pack : plan.packs)
    packs.push_back(buildPackAttr(builder, pack));
  attrs.set("packs", builder.getArrayAttr(packs));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<SharedRelayPlan> parseSharedRelayPlanAttr(DictionaryAttr dict,
                                                           Operation *op) {
  SharedRelayPlan plan;
  auto relayEncoding = readI64Field(dict, "relay_encoding", op);
  auto logicalShape = readDenseI64ArrayField(dict, "logical_shape", op);
  auto allocShape = readDenseI64ArrayField(dict, "alloc_shape", op);
  auto packs = dyn_cast_or_null<ArrayAttr>(dict.get("packs"));
  if (failed(relayEncoding) || failed(logicalShape) || failed(allocShape) ||
      !packs) {
    op->emitError() << "malformed shared relay payload";
    return failure();
  }
  plan.relayEncoding = static_cast<int>(*relayEncoding);
  plan.logicalShape = parseI64Array(*logicalShape);
  plan.allocShape = parseI64Array(*allocShape);
  plan.packs.reserve(packs.size());
  for (Attribute attr : packs) {
    auto dictAttr = dyn_cast<DictionaryAttr>(attr);
    if (!dictAttr) {
      op->emitError() << "`packs` must contain dictionary entries";
      return failure();
    }
    auto pack = parsePackAttr(dictAttr, op);
    if (failed(pack))
      return failure();
    plan.packs.push_back(std::move(*pack));
  }
  return plan;
}

static DictionaryAttr buildExprAttr(Builder &builder,
                                    const EpilogueExprOp &expr) {
  NamedAttrList attrs;
  attrs.set("kind", builder.getStringAttr(stringifyExprKind(expr.kind)));
  attrs.set("input_scalar",
            builder.getStringAttr(stringifyScalarKindLocal(expr.inputScalar)));
  attrs.set("output_scalar",
            builder.getStringAttr(stringifyScalarKindLocal(expr.outputScalar)));
  attrs.set("vector_width", builder.getI64IntegerAttr(expr.vectorWidth));
  attrs.set("aux", builder.getStringAttr(expr.aux));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<EpilogueExprOp> parseExprAttr(DictionaryAttr dict,
                                               Operation *op) {
  EpilogueExprOp expr;
  auto kind = readStringField(dict, "kind", op);
  auto inputScalar = readStringField(dict, "input_scalar", op);
  auto outputScalar = readStringField(dict, "output_scalar", op);
  auto vectorWidth = readI64Field(dict, "vector_width", op);
  if (failed(kind) || failed(inputScalar) || failed(outputScalar) ||
      failed(vectorWidth)) {
    return failure();
  }
  auto parsedKind = parseExprKind(*kind, op);
  auto parsedInputScalar = parseScalarKindLocal(*inputScalar, op);
  auto parsedOutputScalar = parseScalarKindLocal(*outputScalar, op);
  if (failed(parsedKind) || failed(parsedInputScalar) ||
      failed(parsedOutputScalar)) {
    return failure();
  }
  expr.kind = *parsedKind;
  expr.inputScalar = *parsedInputScalar;
  expr.outputScalar = *parsedOutputScalar;
  expr.vectorWidth = *vectorWidth;
  expr.aux = readOptionalStringField(dict, "aux", "");
  return expr;
}

static DictionaryAttr buildTargetLandingAttr(Builder &builder,
                                             const TargetLandingPlan &plan) {
  NamedAttrList attrs;
  attrs.set("kind",
            builder.getStringAttr(stringifyTargetLandingKind(plan.kind)));
  attrs.set("global_vector_width",
            builder.getI64IntegerAttr(plan.globalVectorWidth));
  attrs.set("global_access_bytes",
            builder.getI64IntegerAttr(plan.globalAccessBytes));
  attrs.set("producer_fragment_rows",
            builder.getI64IntegerAttr(plan.producerFragmentRows));
  attrs.set("producer_fragment_cols",
            builder.getI64IntegerAttr(plan.producerFragmentCols));
  attrs.set("direct_pack_rows",
            builder.getI64IntegerAttr(plan.directPackRows));
  attrs.set("direct_pack_cols",
            builder.getI64IntegerAttr(plan.directPackCols));
  attrs.set("warp_batching_group",
            builder.getI64IntegerAttr(plan.warpBatchingGroup));
  attrs.set("required_shared_bytes",
            builder.getI64IntegerAttr(plan.requiredSharedBytes));
  attrs.set("expected_register_footprint",
            builder.getI64IntegerAttr(plan.expectedRegisterFootprint));
  attrs.set("shared_tile_rows",
            builder.getI64IntegerAttr(plan.sharedTileRows));
  attrs.set("shared_tile_cols",
            builder.getI64IntegerAttr(plan.sharedTileCols));
  attrs.set("shared_pack_slots",
            builder.getI64IntegerAttr(plan.sharedPackSlots));
  attrs.set("init_shared_store_vector_width",
            builder.getI64IntegerAttr(plan.initSharedStoreVectorWidth));
  attrs.set("init_shared_load_vector_width",
            builder.getI64IntegerAttr(plan.initSharedLoadVectorWidth));
  attrs.set("store_shared_store_vector_width",
            builder.getI64IntegerAttr(plan.storeSharedStoreVectorWidth));
  attrs.set("store_shared_load_vector_width",
            builder.getI64IntegerAttr(plan.storeSharedLoadVectorWidth));
  attrs.set("use_shared_pack_for_init",
            builder.getBoolAttr(plan.useSharedPackForInit));
  attrs.set("use_shared_pack_for_store",
            builder.getBoolAttr(plan.useSharedPackForStore));
  attrs.set("required_sync_kind", builder.getStringAttr(plan.requiredSyncKind));
  attrs.set("reason", builder.getStringAttr(plan.reason));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<TargetLandingPlan> parseTargetLandingAttr(DictionaryAttr dict,
                                                           Operation *op) {
  TargetLandingPlan plan;
  auto kind = readStringField(dict, "kind", op);
  auto globalVectorWidth = readI64Field(dict, "global_vector_width", op);
  auto globalAccessBytes = readI64Field(dict, "global_access_bytes", op);
  auto producerFragmentRows =
      readI64Field(dict, "producer_fragment_rows", op);
  auto producerFragmentCols =
      readI64Field(dict, "producer_fragment_cols", op);
  auto directPackRows = readI64Field(dict, "direct_pack_rows", op);
  auto directPackCols = readI64Field(dict, "direct_pack_cols", op);
  auto warpBatchingGroup = readI64Field(dict, "warp_batching_group", op);
  auto requiredSharedBytes = readI64Field(dict, "required_shared_bytes", op);
  auto expectedRegisterFootprint =
      readI64Field(dict, "expected_register_footprint", op);
  auto sharedTileRows = readI64Field(dict, "shared_tile_rows", op);
  auto sharedTileCols = readI64Field(dict, "shared_tile_cols", op);
  auto sharedPackSlots = readI64Field(dict, "shared_pack_slots", op);
  auto initSharedStoreVectorWidth =
      readI64Field(dict, "init_shared_store_vector_width", op);
  auto initSharedLoadVectorWidth =
      readI64Field(dict, "init_shared_load_vector_width", op);
  auto storeSharedStoreVectorWidth =
      readI64Field(dict, "store_shared_store_vector_width", op);
  auto storeSharedLoadVectorWidth =
      readI64Field(dict, "store_shared_load_vector_width", op);
  auto useSharedPackForInit = readOptionalBoolField(
      dict, "use_shared_pack_for_init", false);
  auto useSharedPackForStore = readOptionalBoolField(
      dict, "use_shared_pack_for_store", false);
  auto requiredSyncKind = readStringField(dict, "required_sync_kind", op);
  auto reason = readStringField(dict, "reason", op);
  if (failed(kind) || failed(globalVectorWidth) || failed(globalAccessBytes) ||
      failed(producerFragmentRows) || failed(producerFragmentCols) ||
      failed(directPackRows) || failed(directPackCols) ||
      failed(warpBatchingGroup) || failed(requiredSharedBytes) ||
      failed(expectedRegisterFootprint) ||
      failed(sharedTileRows) || failed(sharedTileCols) ||
      failed(sharedPackSlots) ||
      failed(initSharedStoreVectorWidth) ||
      failed(initSharedLoadVectorWidth) ||
      failed(storeSharedStoreVectorWidth) ||
      failed(storeSharedLoadVectorWidth) || failed(requiredSyncKind) ||
      failed(reason)) {
    return failure();
  }
  auto parsedKind = parseTargetLandingKind(*kind, op);
  if (failed(parsedKind))
    return failure();
  plan.kind = *parsedKind;
  plan.globalVectorWidth = *globalVectorWidth;
  plan.globalAccessBytes = *globalAccessBytes;
  plan.producerFragmentRows = *producerFragmentRows;
  plan.producerFragmentCols = *producerFragmentCols;
  plan.directPackRows = *directPackRows;
  plan.directPackCols = *directPackCols;
  plan.warpBatchingGroup = *warpBatchingGroup;
  plan.requiredSharedBytes = *requiredSharedBytes;
  plan.expectedRegisterFootprint = *expectedRegisterFootprint;
  plan.sharedTileRows = *sharedTileRows;
  plan.sharedTileCols = *sharedTileCols;
  plan.sharedPackSlots = *sharedPackSlots;
  plan.initSharedStoreVectorWidth = *initSharedStoreVectorWidth;
  plan.initSharedLoadVectorWidth = *initSharedLoadVectorWidth;
  plan.storeSharedStoreVectorWidth = *storeSharedStoreVectorWidth;
  plan.storeSharedLoadVectorWidth = *storeSharedLoadVectorWidth;
  plan.useSharedPackForInit = useSharedPackForInit;
  plan.useSharedPackForStore = useSharedPackForStore;
  plan.requiredSyncKind = requiredSyncKind->str();
  plan.reason = reason->str();
  return plan;
}

static DictionaryAttr buildVariantAttr(
    Builder &builder,
    const std::variant<std::monostate, DirectGlobalVectorPlan, SharedRelayPlan>
        &variant) {
  NamedAttrList attrs;
  if (std::holds_alternative<std::monostate>(variant)) {
    attrs.set("kind", builder.getStringAttr("none"));
  } else if (auto *direct = std::get_if<DirectGlobalVectorPlan>(&variant)) {
    attrs.set("kind", builder.getStringAttr("direct_global_vector"));
    attrs.set("payload", buildDirectPlanAttr(builder, *direct));
  } else {
    auto *relay = std::get_if<SharedRelayPlan>(&variant);
    attrs.set("kind", builder.getStringAttr("shared_relay"));
    attrs.set("payload", buildSharedRelayPlanAttr(builder, *relay));
  }
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<std::variant<std::monostate, DirectGlobalVectorPlan,
                              SharedRelayPlan>>
parseVariantAttr(DictionaryAttr dict, Operation *op) {
  auto kind = readStringField(dict, "kind", op);
  if (failed(kind))
    return failure();
  if (*kind == "none")
    return std::variant<std::monostate, DirectGlobalVectorPlan, SharedRelayPlan>(
        std::monostate{});
  auto payload = dyn_cast_or_null<DictionaryAttr>(dict.get("payload"));
  if (!payload) {
    op->emitError() << "missing payload for epilogue variant `" << *kind << "`";
    return failure();
  }
  if (*kind == "direct_global_vector") {
    auto parsed = parseDirectPlanAttr(payload, op);
    if (failed(parsed))
      return failure();
    return std::variant<std::monostate, DirectGlobalVectorPlan, SharedRelayPlan>(
        std::move(*parsed));
  }
  if (*kind == "shared_relay") {
    auto parsed = parseSharedRelayPlanAttr(payload, op);
    if (failed(parsed))
      return failure();
    return std::variant<std::monostate, DirectGlobalVectorPlan, SharedRelayPlan>(
        std::move(*parsed));
  }
  op->emitError() << "unknown epilogue variant kind `" << *kind << "`";
  return failure();
}

static LogicalResult validateDirectGlobalPlan(const DirectGlobalVectorPlan &plan,
                                              Operation *op, StringRef role) {
  if (plan.vectorWidth <= 0)
    return op->emitError() << role << " direct-global plan must carry a "
                                     "positive vector width";
  if (plan.ownerScope != "per_warp_template") {
    return op->emitError() << role
                           << " direct-global owner scope must currently stay "
                              "`per_warp_template`";
  }
  if (plan.scalarTail && !plan.boundaryAware) {
    return op->emitError()
           << role << " direct-global scalar tail requires boundary awareness";
  }
  if (plan.laneAccess.laneRowGroupSize <= 0 ||
      plan.laneAccess.laneColGroupSize <= 0 ||
      plan.laneAccess.laneColStride <= 0 ||
      plan.laneAccess.rowOffsets.empty()) {
    return op->emitError() << role << " direct-global plan must carry "
                                     "explicit positive lane access metadata";
  }
  if (plan.packs.empty())
    return op->emitError() << role << " direct-global plan must carry packs";

  llvm::DenseSet<int64_t> seenFragmentIds;
  for (const DirectGlobalVectorPlan::Pack &pack : plan.packs) {
    if (pack.packId < 0 || pack.rows <= 0 || pack.cols <= 0 ||
        pack.vectorWidth != plan.vectorWidth || pack.fragmentIds.empty() ||
        pack.warpOwner < 0) {
      return op->emitError() << role
                             << " direct-global pack geometry is malformed";
    }
    for (int64_t fragmentId : pack.fragmentIds) {
      if (fragmentId < 0 || !seenFragmentIds.insert(fragmentId).second) {
        return op->emitError()
               << role << " direct-global fragment coverage must be unique";
      }
    }
  }
  return success();
}

static LogicalResult validateSharedRelayPlan(const SharedRelayPlan &plan,
                                             Operation *op, StringRef role) {
  if (plan.relayEncoding < 0)
    return op->emitError() << role
                           << " shared relay plan must reference an encoding";
  if (plan.logicalShape.empty() || plan.allocShape.empty() ||
      plan.logicalShape.size() != plan.allocShape.size()) {
    return op->emitError() << role
                           << " shared relay plan must carry matching "
                              "logical/alloc shapes";
  }
  if (plan.packs.empty())
    return op->emitError() << role << " shared relay plan must carry packs";
  return success();
}

static LogicalResult validateTargetLandingPlan(
    const TargetLandingPlan &plan, const AccumulatorPlan &accumulator,
    const EpiloguePlan &epilogue, Operation *op) {
  bool needsTargetLanding =
      epilogue.initMode == AccumulatorInitMode::DirectGlobalVector ||
      epilogue.storeMode == AccumulatorStoreMode::DirectGlobalVector;
  if (!needsTargetLanding) {
    if (plan.kind != TargetLandingKind::None) {
      return op->emitError()
             << "non-direct epilogue must not carry a target landing payload";
    }
    return success();
  }

  if (plan.kind != TargetLandingKind::SharedPackThenGlobalVector &&
      plan.kind != TargetLandingKind::SharedRelayThenGlobalVector &&
      plan.kind != TargetLandingKind::RegisterPackGlobalVector) {
    return op->emitError()
           << "direct-global epilogue must carry an explicit register/shared "
              "target landing contract";
  }
  if (accumulator.packs.empty() || accumulator.packs.front().vectorWidth <= 0) {
    return op->emitError()
           << "target landing requires explicit accumulator fragment width";
  }
  if (plan.globalVectorWidth <= 0 || plan.globalAccessBytes <= 0 ||
      plan.producerFragmentRows <= 0 || plan.producerFragmentCols <= 0 ||
      plan.directPackRows <= 0 || plan.directPackCols <= 0 ||
      plan.warpBatchingGroup <= 0 || plan.expectedRegisterFootprint <= 0 ||
      plan.requiredSharedBytes < 0 || plan.requiredSyncKind.empty() ||
      plan.reason.empty()) {
    return op->emitError() << "target landing must carry explicit owner and "
                              "resource metadata";
  }
  auto *init = std::get_if<DirectGlobalVectorPlan>(&epilogue.init);
  auto *store = std::get_if<DirectGlobalVectorPlan>(&epilogue.store);
  if (!init || !store || init->packs.empty() || store->packs.empty()) {
    return op->emitError()
           << "target landing requires direct init/store payloads";
  }
  if (accumulator.ownerScope != init->ownerScope ||
      accumulator.ownerScope != store->ownerScope) {
    return op->emitError()
           << "target landing requires accumulator/direct owner scopes to stay "
              "aligned";
  }
  int64_t fragmentVectorWidth = accumulator.packs.front().vectorWidth;
  if (plan.globalVectorWidth != init->vectorWidth ||
      plan.globalVectorWidth != store->vectorWidth) {
    return op->emitError()
           << "target landing global vector width must match the logical "
              "direct-global vector width";
  }
  if (plan.producerFragmentRows != accumulator.packs.front().rows ||
      plan.producerFragmentCols != accumulator.packs.front().cols ||
      plan.directPackRows != init->packs.front().rows ||
      plan.directPackCols != init->packs.front().cols ||
      plan.directPackRows != store->packs.front().rows ||
      plan.directPackCols != store->packs.front().cols) {
    return op->emitError()
           << "target landing fragment/direct-pack geometry must match the "
              "logical epilogue ownership";
  }
  if (plan.kind == TargetLandingKind::RegisterPackGlobalVector) {
    if (plan.sharedTileRows != 0 || plan.sharedTileCols != 0 ||
        plan.sharedPackSlots != 0 ||
        plan.initSharedStoreVectorWidth != 0 ||
        plan.initSharedLoadVectorWidth != 0 ||
        plan.storeSharedStoreVectorWidth != 0 ||
        plan.storeSharedLoadVectorWidth != 0 ||
        plan.useSharedPackForInit || plan.useSharedPackForStore ||
        plan.requiredSharedBytes != 0 || plan.requiredSyncKind != "none") {
      return op->emitError()
             << "register-pack landing must not carry shared-pack payload";
    }
    return success();
  }
  if (plan.kind == TargetLandingKind::SharedRelayThenGlobalVector) {
    if (plan.sharedTileRows <= 0 || plan.sharedTileCols <= 0 ||
        plan.sharedPackSlots <= 0 ||
        plan.initSharedStoreVectorWidth <= 0 ||
        plan.initSharedLoadVectorWidth <= 0 ||
        plan.storeSharedStoreVectorWidth <= 0 ||
        plan.storeSharedLoadVectorWidth <= 0 ||
        plan.requiredSharedBytes <= 0 || plan.requiredSyncKind == "none") {
      return op->emitError()
             << "shared-relay landing must carry explicit shared relay "
                "materialization metadata";
    }
    return success();
  }
  if (plan.sharedTileRows <= 0 || plan.sharedTileCols <= 0 ||
      plan.sharedPackSlots <= 0 ||
      plan.initSharedStoreVectorWidth <= 0 ||
      plan.initSharedLoadVectorWidth <= 0 ||
      plan.storeSharedStoreVectorWidth <= 0 ||
      plan.storeSharedLoadVectorWidth <= 0) {
    return op->emitError()
           << "shared-pack landing must carry positive shared materialization "
              "metadata";
  }
  if (plan.sharedTileRows != plan.directPackRows ||
      plan.sharedTileCols != plan.directPackCols ||
      plan.initSharedStoreVectorWidth != plan.globalVectorWidth ||
      plan.storeSharedLoadVectorWidth != plan.globalVectorWidth ||
      plan.initSharedLoadVectorWidth != fragmentVectorWidth ||
      plan.storeSharedStoreVectorWidth != fragmentVectorWidth ||
      !plan.useSharedPackForInit || !plan.useSharedPackForStore ||
      plan.requiredSharedBytes <= 0 || plan.requiredSyncKind != "warp") {
    return op->emitError()
           << "shared-pack landing metadata is inconsistent with the direct "
              "vector bridge";
  }
  return success();
}

static LogicalResult validateEpiloguePlan(const EpiloguePlan &plan,
                                          Operation *op) {
  if (plan.initMode == AccumulatorInitMode::DirectGlobalVector) {
    auto *direct = std::get_if<DirectGlobalVectorPlan>(&plan.init);
    if (!direct) {
      return op->emitError()
             << "direct_global_vector init mode must carry direct payload";
    }
    if (failed(validateDirectGlobalPlan(*direct, op, "init")))
      return failure();
  } else if (plan.initMode == AccumulatorInitMode::SharedRelay) {
    auto *relay = std::get_if<SharedRelayPlan>(&plan.init);
    if (!relay) {
      return op->emitError() << "shared_relay init mode must carry relay payload";
    }
    if (failed(validateSharedRelayPlan(*relay, op, "init")))
      return failure();
  }

  if (plan.storeMode == AccumulatorStoreMode::DirectGlobalVector) {
    auto *direct = std::get_if<DirectGlobalVectorPlan>(&plan.store);
    if (!direct) {
      return op->emitError()
             << "direct_global_vector store mode must carry direct payload";
    }
    if (failed(validateDirectGlobalPlan(*direct, op, "store")))
      return failure();
  } else if (plan.storeMode == AccumulatorStoreMode::SharedRelay) {
    auto *relay = std::get_if<SharedRelayPlan>(&plan.store);
    if (!relay) {
      return op->emitError()
             << "shared_relay store mode must carry relay payload";
    }
    if (failed(validateSharedRelayPlan(*relay, op, "store")))
      return failure();
  }

  for (const EpilogueExprOp &expr : plan.exprs) {
    if (expr.vectorWidth < 0) {
      return op->emitError()
             << "epilogue expr vector widths must be non-negative";
    }
  }
  return success();
}

static LogicalResult validateEpiloguePlan(const EpiloguePlan &plan,
                                          const AccumulatorPlan &accumulator,
                                          Operation *op) {
  if (failed(validateEpiloguePlan(plan, op)))
    return failure();
  if (failed(validateTargetLandingPlan(plan.targetLanding, accumulator, plan,
                                       op))) {
    return failure();
  }
  return success();
}

static EpiloguePlan deriveDirectEpiloguePlan(
    const KernelConfig &config, const TargetInfo &target,
    const AccumulatorPlan &accumulator, Operation *op) {
  (void)target;
  if (accumulator.packs.empty()) {
    op->emitError() << "direct epilogue requires accumulator fragments";
    return EpiloguePlan();
  }
  if (accumulator.laneAccess.rowOffsets.empty()) {
    op->emitError() << "direct epilogue requires explicit accumulator lane rows";
    return EpiloguePlan();
  }

  int64_t fragmentVectorWidth = accumulator.packs.front().vectorWidth;
  if (fragmentVectorWidth <= 0) {
    op->emitError() << "direct epilogue requires positive fragment vector width";
    return EpiloguePlan();
  }
  int64_t directVectorWidth = accumulator.laneAccess.laneColGroupSize > 0
                                  ? accumulator.laneAccess.laneColGroupSize
                                  : fragmentVectorWidth;
  if (directVectorWidth < fragmentVectorWidth ||
      directVectorWidth % fragmentVectorWidth != 0) {
    op->emitError() << "direct epilogue vector width must be derived from the "
                       "accumulator lane/layout truth and stay divisible by "
                       "the fragment vector width";
    return EpiloguePlan();
  }
  int64_t fragmentsPerDirectPack = directVectorWidth / fragmentVectorWidth;
  if (fragmentsPerDirectPack <= 0) {
    op->emitError() << "direct epilogue requires a positive pack factor";
    return EpiloguePlan();
  }
  int64_t fragmentRows = accumulator.packs.front().rows;
  int64_t fragmentCols = accumulator.packs.front().cols;
  int64_t directPackCols = fragmentCols * fragmentsPerDirectPack;
  if (fragmentRows <= 0 || fragmentCols <= 0 || directPackCols <= 0) {
    op->emitError() << "direct epilogue requires positive fixed-point fragment "
                       "and direct-pack geometry";
    return EpiloguePlan();
  }

  EpiloguePlan epilogue;
  epilogue.initMode = AccumulatorInitMode::DirectGlobalVector;
  epilogue.storeMode = AccumulatorStoreMode::DirectGlobalVector;
  DirectGlobalVectorPlan init;
  DirectGlobalVectorPlan store;
  init.ownerScope = accumulator.ownerScope;
  store.ownerScope = accumulator.ownerScope;
  init.vectorWidth = directVectorWidth;
  store.vectorWidth = init.vectorWidth;
  bool needsBoundaryGuard = hasBoundaryOnM(config) || hasBoundaryOnN(config);
  init.boundaryAware = needsBoundaryGuard;
  store.boundaryAware = needsBoundaryGuard;
  bool needsScalarTail = hasScalarTailOnN(config, directVectorWidth);
  init.scalarTail = needsScalarTail;
  store.scalarTail = needsScalarTail;
  init.laneAccess = accumulator.laneAccess;
  store.laneAccess = accumulator.laneAccess;
  init.laneAccess.laneColStride = init.vectorWidth;
  store.laneAccess.laneColStride = store.vectorWidth;

  using GroupKey = std::tuple<int64_t, int64_t, int64_t>;
  std::map<GroupKey, SmallVector<const AccumulatorPack *, 4>> fragmentGroups;
  for (const AccumulatorPack &fragment : accumulator.packs) {
    if (fragment.rows != fragmentRows || fragment.cols != fragmentCols ||
        fragment.vectorWidth != fragmentVectorWidth) {
      op->emitError() << "direct epilogue expects a uniform accumulator "
                         "fragment grid";
      return EpiloguePlan();
    }
    if (fragment.colBase < 0 || fragment.rowBase < 0 ||
        (fragment.colBase % fragmentCols) != 0) {
      op->emitError() << "direct epilogue fragment grid must be aligned to the "
                         "accumulator fragment fixed point";
      return EpiloguePlan();
    }
    int64_t groupColBase = (fragment.colBase / directPackCols) * directPackCols;
    fragmentGroups[GroupKey{fragment.warpOwner, fragment.rowBase, groupColBase}]
        .push_back(&fragment);
  }

  int64_t packId = 0;
  for (auto &it : fragmentGroups) {
    auto &group = it.second;
    std::sort(group.begin(), group.end(),
              [](const AccumulatorPack *lhs, const AccumulatorPack *rhs) {
                return lhs->colBase < rhs->colBase;
              });
    if (group.size() != static_cast<size_t>(fragmentsPerDirectPack)) {
      op->emitError()
          << "direct epilogue fixed-point group does not contain the required "
             "number of accumulator fragments";
      return EpiloguePlan();
    }

    DirectGlobalVectorPlan::Pack pack;
    const auto [warpOwner, rowBase, colBase] = it.first;
    pack.packId = packId++;
    pack.rowBase = rowBase;
    pack.colBase = colBase;
    pack.rows = fragmentRows;
    pack.cols = directPackCols;
    pack.vectorWidth = init.vectorWidth;
    pack.warpOwner = warpOwner;
    for (int64_t ordinal = 0; ordinal < fragmentsPerDirectPack; ++ordinal) {
      const AccumulatorPack &fragment = *group[static_cast<size_t>(ordinal)];
      if (fragment.warpOwner != warpOwner || fragment.rowBase != rowBase ||
          fragment.colBase != colBase + ordinal * fragmentCols) {
        op->emitError()
            << "direct epilogue fixed-point group is not contiguous in the "
               "accumulator ownership grid";
        return EpiloguePlan();
      }
      pack.fragmentIds.push_back(fragment.packId);
    }
    init.packs.push_back(pack);
    store.packs.push_back(pack);
  }
  epilogue.init = std::move(init);
  epilogue.store = std::move(store);
  return epilogue;
}

static FailureOr<TargetLandingPlan>
deriveTargetLandingPlan(const KernelConfig &config, const TargetInfo &target,
                        const AccumulatorPlan &accumulator,
                        const EpiloguePlan &epilogue, Operation *op) {
  (void)target;
  auto *init = std::get_if<DirectGlobalVectorPlan>(&epilogue.init);
  auto *store = std::get_if<DirectGlobalVectorPlan>(&epilogue.store);
  if (!init || !store || init->packs.empty() || store->packs.empty() ||
      accumulator.packs.empty()) {
    op->emitError()
        << "target landing requires direct init/store plus accumulator packs";
    return failure();
  }
  if (init->packs.size() != store->packs.size()) {
    op->emitError()
        << "target landing requires init/store direct packs to stay aligned";
    return failure();
  }

  int64_t cElementBytes = getScalarByteWidth(config.cScalar);
  int64_t firstWarpOwner = init->packs.front().warpOwner;
  llvm::DenseSet<int64_t> distinctPackRows;
  int64_t packsInOwnerTemplate = 0;
  for (const DirectGlobalVectorPlan::Pack &pack : init->packs) {
    if (pack.warpOwner == firstWarpOwner) {
      distinctPackRows.insert(pack.rowBase);
      ++packsInOwnerTemplate;
    }
  }
  int64_t packRowCount = static_cast<int64_t>(distinctPackRows.size());
  if (packRowCount <= 0 || packsInOwnerTemplate <= 0 ||
      packsInOwnerTemplate % packRowCount != 0) {
    op->emitError() << "target landing could not derive a stable direct-pack "
                       "batch shape from the full pack grid";
    return failure();
  }
  int64_t packsPerWarpRow = packsInOwnerTemplate / packRowCount;
  int64_t fragmentVectorWidth = accumulator.packs.front().vectorWidth;
  if (init->vectorWidth != store->vectorWidth) {
    op->emitError() << "target landing requires init/store direct vector widths "
                       "to stay aligned";
    return failure();
  }
  if (!isSupportedGlobalVectorWidth(init->vectorWidth)) {
    op->emitError() << "target landing currently requires vector<2xf32> or "
                       "vector<4xf32> direct-global C rows";
    return failure();
  }
  bool needsBoundaryGuard = init->boundaryAware || store->boundaryAware;
  bool canUseBoundaryAwareDirect =
      !needsBoundaryGuard ||
      (init->vectorWidth == fragmentVectorWidth &&
       store->vectorWidth == fragmentVectorWidth);

  TargetLandingPlan plan;
  plan.kind = TargetLandingKind::RegisterPackGlobalVector;
  plan.globalVectorWidth = init->vectorWidth;
  plan.globalAccessBytes = plan.globalVectorWidth * cElementBytes;
  plan.producerFragmentRows = accumulator.packs.front().rows;
  plan.producerFragmentCols = accumulator.packs.front().cols;
  plan.directPackRows = init->packs.front().rows;
  plan.directPackCols = init->packs.front().cols;
  plan.warpBatchingGroup = packsPerWarpRow;
  plan.expectedRegisterFootprint =
      accumulator.registersPerWarp * accumulator.packs.front().elemCount +
      init->vectorWidth * init->laneAccess.rowOffsets.size();
  plan.requiredSharedBytes = 0;
  plan.sharedTileRows = 0;
  plan.sharedTileCols = 0;
  plan.sharedPackSlots = 0;
  plan.initSharedStoreVectorWidth = 0;
  plan.initSharedLoadVectorWidth = 0;
  plan.storeSharedStoreVectorWidth = 0;
  plan.storeSharedLoadVectorWidth = 0;
  plan.useSharedPackForInit = false;
  plan.useSharedPackForStore = false;
  plan.requiredSyncKind = "none";
  plan.reason =
      !config.exactTile && !canUseBoundaryAwareDirect
          ? "final C landing stays register-pack/global-vector; any shared "
            "row reorder now lives in tb.epilogue_reorder_plan and the "
            "unified CTA workspace instead of this landing contract"
          : (config.exactTile
                 ? "exact_tile direct_global_vector packs are derived from the "
                   "fixed-point accumulator ownership grid and lower as "
                   "register pack/unpack without shared relay"
                 : "boundary-aware direct_global_vector keeps the final C "
                   "landing in register packs because every direct pack "
                   "already matches one accumulator fragment width and can "
                   "be masked lane-locally");
  return plan;
}

} // namespace

FailureOr<EpiloguePlan> mlir::tb::deriveEpiloguePlan(
    const KernelConfig &config, const TargetInfo &target,
    const EncodingPlan &encodings, const AccumulatorPlan &accumulator,
    Operation *op) {
  (void)encodings;
  if (config.cScalar != ScalarKind::F32) {
    op->emitError() << "V1 direct epilogue expects fp32 accumulator/output";
    return failure();
  }
  if (accumulator.packs.empty()) {
    op->emitError() << "epilogue plan requires accumulator packs";
    return failure();
  }
  auto epilogue = deriveDirectEpiloguePlan(config, target, accumulator, op);
  if (std::holds_alternative<std::monostate>(epilogue.init) ||
      std::holds_alternative<std::monostate>(epilogue.store)) {
    return failure();
  }
  auto targetLanding =
      deriveTargetLandingPlan(config, target, accumulator, epilogue, op);
  if (failed(targetLanding))
    return failure();
  epilogue.targetLanding = *targetLanding;
  if (failed(validateEpiloguePlan(epilogue, accumulator, op)))
    return failure();
  return epilogue;
}

DictionaryAttr mlir::tb::buildEpiloguePlanAttr(Builder &builder,
                                               const EpiloguePlan &plan) {
  NamedAttrList attrs;
  attrs.set("init_mode", builder.getStringAttr(stringifyInitMode(plan.initMode)));
  attrs.set("store_mode",
            builder.getStringAttr(stringifyStoreMode(plan.storeMode)));
  attrs.set("init", buildVariantAttr(builder, plan.init));
  attrs.set("store", buildVariantAttr(builder, plan.store));
  attrs.set("target_landing", buildTargetLandingAttr(builder, plan.targetLanding));
  SmallVector<Attribute> exprs;
  exprs.reserve(plan.exprs.size());
  for (const EpilogueExprOp &expr : plan.exprs)
    exprs.push_back(buildExprAttr(builder, expr));
  attrs.set("exprs", builder.getArrayAttr(exprs));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<EpiloguePlan> mlir::tb::parseEpiloguePlanAttr(Operation *op) {
  auto root = dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.epilogue_plan"));
  if (!root) {
    op->emitError() << "missing `tb.epilogue_plan` attribute";
    return failure();
  }
  auto initMode = readStringField(root, "init_mode", op);
  auto storeMode = readStringField(root, "store_mode", op);
  auto init = dyn_cast_or_null<DictionaryAttr>(root.get("init"));
  auto store = dyn_cast_or_null<DictionaryAttr>(root.get("store"));
  auto targetLanding =
      dyn_cast_or_null<DictionaryAttr>(root.get("target_landing"));
  if (failed(initMode) || failed(storeMode) || !init || !store ||
      !targetLanding) {
    op->emitError() << "malformed `tb.epilogue_plan` attribute";
    return failure();
  }
  auto parsedInitMode = parseInitMode(*initMode, op);
  auto parsedStoreMode = parseStoreMode(*storeMode, op);
  auto parsedInit = parseVariantAttr(init, op);
  auto parsedStore = parseVariantAttr(store, op);
  auto parsedTargetLanding = parseTargetLandingAttr(targetLanding, op);
  if (failed(parsedInitMode) || failed(parsedStoreMode) || failed(parsedInit) ||
      failed(parsedStore) || failed(parsedTargetLanding)) {
    return failure();
  }
  EpiloguePlan plan;
  plan.initMode = *parsedInitMode;
  plan.storeMode = *parsedStoreMode;
  plan.init = std::move(*parsedInit);
  plan.store = std::move(*parsedStore);
  plan.targetLanding = *parsedTargetLanding;
  if (auto exprs = dyn_cast_or_null<ArrayAttr>(root.get("exprs"))) {
    plan.exprs.reserve(exprs.size());
    for (Attribute attr : exprs) {
      auto dict = dyn_cast<DictionaryAttr>(attr);
      if (!dict) {
        op->emitError() << "`exprs` must contain dictionary entries";
        return failure();
      }
      auto expr = parseExprAttr(dict, op);
      if (failed(expr))
        return failure();
      plan.exprs.push_back(std::move(*expr));
    }
  }
  AccumulatorPlan accumulator;
  auto parsedAccumulator = parseAccumulatorPlanAttr(op);
  if (failed(parsedAccumulator))
    return failure();
  if (failed(validateEpiloguePlan(plan, *parsedAccumulator, op))) {
    op->emitError() << "malformed `tb.epilogue_plan` attribute";
    return failure();
  }
  return plan;
}
