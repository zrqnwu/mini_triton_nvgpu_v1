#include "tb/Analysis/WarpDecompositionPlan.h"

#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/KernelConfig.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>

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

static FailureOr<StringRef> readStringField(DictionaryAttr dict, StringRef name,
                                            Operation *op) {
  auto attr = dyn_cast_or_null<StringAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing string field `" << name << "`";
    return failure();
  }
  return attr.getValue();
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

static DictionaryAttr buildWarpTileAttr(Builder &builder,
                                        const WarpTileCoverage &tile) {
  NamedAttrList attrs;
  attrs.set("warp_id", builder.getI64IntegerAttr(tile.warpId));
  attrs.set("coord_m", builder.getI64IntegerAttr(tile.coordM));
  attrs.set("coord_n", builder.getI64IntegerAttr(tile.coordN));
  attrs.set("tile_base_m", builder.getI64IntegerAttr(tile.tileBaseM));
  attrs.set("tile_base_n", builder.getI64IntegerAttr(tile.tileBaseN));
  attrs.set("tile_rows", builder.getI64IntegerAttr(tile.tileRows));
  attrs.set("tile_cols", builder.getI64IntegerAttr(tile.tileCols));
  attrs.set("mma_group_ids", buildI64ArrayAttr(builder, tile.mmaGroupIds));
  attrs.set("accumulator_pack_ids",
            buildI64ArrayAttr(builder, tile.accumulatorPackIds));
  attrs.set("epilogue_pack_ids",
            buildI64ArrayAttr(builder, tile.epiloguePackIds));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<WarpTileCoverage> parseWarpTileAttr(DictionaryAttr dict,
                                                     Operation *op) {
  WarpTileCoverage tile;
  auto warpId = readI64Field(dict, "warp_id", op);
  auto coordM = readI64Field(dict, "coord_m", op);
  auto coordN = readI64Field(dict, "coord_n", op);
  auto tileBaseM = readI64Field(dict, "tile_base_m", op);
  auto tileBaseN = readI64Field(dict, "tile_base_n", op);
  auto tileRows = readI64Field(dict, "tile_rows", op);
  auto tileCols = readI64Field(dict, "tile_cols", op);
  auto mmaGroupIds = readDenseI64ArrayField(dict, "mma_group_ids", op);
  auto accumulatorPackIds =
      readDenseI64ArrayField(dict, "accumulator_pack_ids", op);
  auto epiloguePackIds = readDenseI64ArrayField(dict, "epilogue_pack_ids", op);
  if (failed(warpId) || failed(coordM) || failed(coordN) ||
      failed(tileBaseM) || failed(tileBaseN) || failed(tileRows) ||
      failed(tileCols) || failed(mmaGroupIds) || failed(accumulatorPackIds) ||
      failed(epiloguePackIds)) {
    return failure();
  }
  tile.warpId = *warpId;
  tile.coordM = *coordM;
  tile.coordN = *coordN;
  tile.tileBaseM = *tileBaseM;
  tile.tileBaseN = *tileBaseN;
  tile.tileRows = *tileRows;
  tile.tileCols = *tileCols;
  tile.mmaGroupIds = parseI64Array(*mmaGroupIds);
  tile.accumulatorPackIds = parseI64Array(*accumulatorPackIds);
  tile.epiloguePackIds = parseI64Array(*epiloguePackIds);
  return tile;
}

static bool isTemplateLocalCoverage(int64_t rowBase, int64_t colBase,
                                    int64_t rows, int64_t cols,
                                    ArrayRef<int64_t> warpTile) {
  return rowBase >= 0 && colBase >= 0 && rows > 0 && cols > 0 &&
         rowBase + rows <= warpTile[0] && colBase + cols <= warpTile[1];
}

static LogicalResult validateWarpDecompositionPlan(
    const WarpDecompositionPlan &plan, Operation *op) {
  if (plan.ctaTile.size() != 2 || plan.warpGrid.size() != 2 ||
      plan.warpTile.size() != 2) {
    return op->emitError()
           << "warp decomposition must carry rank-2 CTA/grid/tile shapes";
  }
  if (plan.numWarps <= 0 || plan.warpOrder.empty()) {
    return op->emitError()
           << "warp decomposition must carry positive warp metadata";
  }
  if (plan.ownerScope != "per_warp_template") {
    return op->emitError() << "warp decomposition currently requires "
                              "`per_warp_template` owner scope";
  }
  if (plan.warps.size() != static_cast<size_t>(plan.numWarps)) {
    return op->emitError()
           << "warp decomposition warp list size must equal num_warps";
  }
  llvm::DenseSet<int64_t> seenWarpIds;
  for (const WarpTileCoverage &tile : plan.warps) {
    llvm::DenseSet<int64_t> seenMmaGroups;
    llvm::DenseSet<int64_t> seenAccumulatorPacks;
    llvm::DenseSet<int64_t> seenEpiloguePacks;
    if (tile.warpId < 0 || !seenWarpIds.insert(tile.warpId).second) {
      return op->emitError()
             << "warp decomposition warp ids must be unique and non-negative";
    }
    if (tile.coordM < 0 || tile.coordN < 0 || tile.tileRows <= 0 ||
        tile.tileCols <= 0 || tile.mmaGroupIds.empty() ||
        tile.accumulatorPackIds.empty() ||
        tile.epiloguePackIds.empty()) {
      return op->emitError()
             << "warp decomposition must carry explicit positive warp coverage";
    }
    for (int64_t mmaGroupId : tile.mmaGroupIds) {
      if (mmaGroupId < 0 || !seenMmaGroups.insert(mmaGroupId).second) {
        return op->emitError()
               << "warp decomposition mma-group coverage must be unique and "
                  "non-negative";
      }
    }
    for (int64_t packId : tile.accumulatorPackIds) {
      if (packId < 0 || !seenAccumulatorPacks.insert(packId).second) {
        return op->emitError()
               << "warp decomposition accumulator-pack coverage must be "
                  "unique and non-negative";
      }
    }
    for (int64_t packId : tile.epiloguePackIds) {
      if (packId < 0 || !seenEpiloguePacks.insert(packId).second) {
        return op->emitError()
               << "warp decomposition epilogue-pack coverage must be unique "
                  "and non-negative";
      }
    }
  }
  return success();
}

} // namespace

FailureOr<WarpDecompositionPlan>
mlir::tb::deriveWarpDecompositionPlan(const KernelConfig &config,
                                      const EncodingPlan &encodings,
                                      const AccumulatorPlan &accumulator,
                                      const EpiloguePlan &epilogue,
                                      Operation *op) {
  auto warpGrid = getMmaWarpsPerCTA(encodings, op);
  auto warpTile = getAccumulatorTileShape(encodings, op);
  auto *store = std::get_if<DirectGlobalVectorPlan>(&epilogue.store);
  if (failed(warpGrid) || failed(warpTile) || !store || store->packs.empty()) {
    op->emitError() << "warp decomposition requires direct epilogue ownership";
    return failure();
  }
  if ((*warpGrid)[0] * (*warpGrid)[1] != config.numWarps) {
    op->emitError() << "warp decomposition warp grid disagrees with num_warps";
    return failure();
  }
  if ((*warpGrid)[0] * (*warpTile)[0] != config.blockM ||
      (*warpGrid)[1] * (*warpTile)[1] != config.blockN) {
    op->emitError() << "warp decomposition warp geometry does not cover CTA tile";
    return failure();
  }

  WarpDecompositionPlan plan;
  plan.ctaTile = {config.blockM, config.blockN};
  plan.numWarps = config.numWarps;
  plan.warpGrid.assign(warpGrid->begin(), warpGrid->end());
  plan.warpTile.assign(warpTile->begin(), warpTile->end());
  plan.ownerScope = accumulator.ownerScope;

  if (plan.ownerScope != store->ownerScope) {
    op->emitError() << "warp decomposition requires accumulator and direct "
                       "epilogue owner scopes to stay aligned";
    return failure();
  }

  llvm::DenseSet<int64_t> seenMmaGroupIds;
  SmallVector<int64_t, 8> mmaGroupIds;
  SmallVector<int64_t, 8> accumulatorPackIds;
  accumulatorPackIds.reserve(accumulator.packs.size());
  for (const AccumulatorPack &pack : accumulator.packs) {
    if (!isTemplateLocalCoverage(pack.rowBase, pack.colBase, pack.rows,
                                 pack.cols, *warpTile)) {
      op->emitError() << "accumulator pack " << pack.packId
                      << " does not fit within the warp-template tile";
      return failure();
    }
    accumulatorPackIds.push_back(pack.packId);
    if (seenMmaGroupIds.insert(pack.mmaGroup).second)
      mmaGroupIds.push_back(pack.mmaGroup);
  }
  std::sort(mmaGroupIds.begin(), mmaGroupIds.end());

  SmallVector<int64_t, 8> epiloguePackIds;
  epiloguePackIds.reserve(store->packs.size());
  for (const DirectGlobalVectorPlan::Pack &pack : store->packs) {
    if (!isTemplateLocalCoverage(pack.rowBase, pack.colBase, pack.rows,
                                 pack.cols, *warpTile)) {
      op->emitError() << "epilogue direct pack " << pack.packId
                      << " does not fit within the warp-template tile";
      return failure();
    }
    epiloguePackIds.push_back(pack.packId);
  }

  if (mmaGroupIds.empty() || accumulatorPackIds.empty() || epiloguePackIds.empty()) {
    op->emitError() << "warp decomposition requires non-empty template-local "
                       "mma/accumulator/epilogue coverage";
    return failure();
  }

  for (int64_t warpId = 0; warpId < config.numWarps; ++warpId) {
    int64_t coordM = warpId / (*warpGrid)[1];
    int64_t coordN = warpId % (*warpGrid)[1];
    WarpTileCoverage tile;
    tile.warpId = warpId;
    tile.coordM = coordM;
    tile.coordN = coordN;
    tile.tileBaseM = coordM * (*warpTile)[0];
    tile.tileBaseN = coordN * (*warpTile)[1];
    tile.tileRows = (*warpTile)[0];
    tile.tileCols = (*warpTile)[1];
    tile.mmaGroupIds = mmaGroupIds;
    tile.accumulatorPackIds = accumulatorPackIds;
    tile.epiloguePackIds = epiloguePackIds;
    plan.warps.push_back(std::move(tile));
  }

  if (failed(validateWarpDecompositionPlan(plan, op)))
    return failure();
  return plan;
}

DictionaryAttr
mlir::tb::buildWarpDecompositionPlanAttr(Builder &builder,
                                         const WarpDecompositionPlan &plan) {
  NamedAttrList attrs;
  attrs.set("cta_tile", buildI64ArrayAttr(builder, plan.ctaTile));
  attrs.set("num_warps", builder.getI64IntegerAttr(plan.numWarps));
  attrs.set("warp_grid", buildI64ArrayAttr(builder, plan.warpGrid));
  attrs.set("warp_tile", buildI64ArrayAttr(builder, plan.warpTile));
  attrs.set("owner_scope", builder.getStringAttr(plan.ownerScope));
  attrs.set("warp_order", builder.getStringAttr(plan.warpOrder));
  SmallVector<Attribute> warps;
  warps.reserve(plan.warps.size());
  for (const WarpTileCoverage &tile : plan.warps)
    warps.push_back(buildWarpTileAttr(builder, tile));
  attrs.set("warps", builder.getArrayAttr(warps));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<WarpDecompositionPlan>
mlir::tb::parseWarpDecompositionPlanAttr(Operation *op) {
  auto root =
      dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.warp_decomposition_plan"));
  if (!root) {
    op->emitError() << "missing `tb.warp_decomposition_plan` attribute";
    return failure();
  }
  auto ctaTile = readDenseI64ArrayField(root, "cta_tile", op);
  auto numWarps = readI64Field(root, "num_warps", op);
  auto warpGrid = readDenseI64ArrayField(root, "warp_grid", op);
  auto warpTile = readDenseI64ArrayField(root, "warp_tile", op);
  auto warpOrder = readStringField(root, "warp_order", op);
  auto warpsAttr = dyn_cast_or_null<ArrayAttr>(root.get("warps"));
  if (failed(ctaTile) || failed(numWarps) || failed(warpGrid) ||
      failed(warpTile) || failed(warpOrder) || !warpsAttr) {
    op->emitError() << "malformed `tb.warp_decomposition_plan` attribute";
    return failure();
  }

  WarpDecompositionPlan plan;
  plan.ctaTile = parseI64Array(*ctaTile);
  plan.numWarps = *numWarps;
  plan.warpGrid = parseI64Array(*warpGrid);
  plan.warpTile = parseI64Array(*warpTile);
  plan.ownerScope = readOptionalStringField(root, "owner_scope",
                                            "per_warp_template");
  plan.warpOrder = warpOrder->str();
  plan.warps.reserve(warpsAttr.size());
  for (Attribute attr : warpsAttr) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict) {
      op->emitError() << "`warps` must contain dictionary entries";
      return failure();
    }
    auto tile = parseWarpTileAttr(dict, op);
    if (failed(tile))
      return failure();
    plan.warps.push_back(std::move(*tile));
  }
  if (failed(validateWarpDecompositionPlan(plan, op))) {
    op->emitError() << "malformed `tb.warp_decomposition_plan` attribute";
    return failure();
  }
  return plan;
}
