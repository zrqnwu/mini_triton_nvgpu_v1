#include "tb/Analysis/EpilogueReorderPlan.h"

#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <map>
#include <tuple>

using namespace mlir;
using namespace mlir::tb;

namespace {

static StringRef stringifyEpilogueReorderKind(EpilogueReorderKind kind) {
  switch (kind) {
  case EpilogueReorderKind::None:
    return "none";
  case EpilogueReorderKind::CTASharedRowReorder:
    return "cta_shared_row_reorder";
  case EpilogueReorderKind::CTASharedRelay:
    return "cta_shared_relay";
  }
  llvm_unreachable("unknown epilogue reorder kind");
}

static FailureOr<EpilogueReorderKind>
parseEpilogueReorderKind(StringRef value, Operation *op) {
  if (value == "none")
    return EpilogueReorderKind::None;
  if (value == "cta_shared_row_reorder")
    return EpilogueReorderKind::CTASharedRowReorder;
  if (value == "cta_shared_relay")
    return EpilogueReorderKind::CTASharedRelay;
  op->emitError() << "unknown epilogue reorder kind `" << value << "`";
  return failure();
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

static DictionaryAttr buildRowVectorAttr(Builder &builder,
                                         const EpilogueRowVectorMapping &row) {
  NamedAttrList attrs;
  attrs.set("row_vector_id", builder.getI64IntegerAttr(row.rowVectorId));
  attrs.set("pack_id", builder.getI64IntegerAttr(row.packId));
  attrs.set("warp_owner", builder.getI64IntegerAttr(row.warpOwner));
  attrs.set("row_base", builder.getI64IntegerAttr(row.rowBase));
  attrs.set("col_base", builder.getI64IntegerAttr(row.colBase));
  attrs.set("row_offset", builder.getI64IntegerAttr(row.rowOffset));
  attrs.set("vector_width", builder.getI64IntegerAttr(row.vectorWidth));
  attrs.set("fragment_ids", buildI64ArrayAttr(builder, row.fragmentIds));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<EpilogueRowVectorMapping>
parseRowVectorAttr(DictionaryAttr dict, Operation *op) {
  EpilogueRowVectorMapping row;
  auto rowVectorId = readI64Field(dict, "row_vector_id", op);
  auto packId = readI64Field(dict, "pack_id", op);
  auto warpOwner = readI64Field(dict, "warp_owner", op);
  auto rowBase = readI64Field(dict, "row_base", op);
  auto colBase = readI64Field(dict, "col_base", op);
  auto rowOffset = readI64Field(dict, "row_offset", op);
  auto vectorWidth = readI64Field(dict, "vector_width", op);
  auto fragmentIds = readDenseI64ArrayField(dict, "fragment_ids", op);
  if (failed(rowVectorId) || failed(packId) || failed(warpOwner) ||
      failed(rowBase) || failed(colBase) || failed(rowOffset) ||
      failed(vectorWidth) || failed(fragmentIds)) {
    return failure();
  }
  row.rowVectorId = *rowVectorId;
  row.packId = *packId;
  row.warpOwner = *warpOwner;
  row.rowBase = *rowBase;
  row.colBase = *colBase;
  row.rowOffset = *rowOffset;
  row.vectorWidth = *vectorWidth;
  row.fragmentIds = parseI64Array(*fragmentIds);
  return row;
}

static LogicalResult validateEpilogueReorderPlan(const EpilogueReorderPlan &plan,
                                                 Operation *op) {
  if (plan.reason.empty() || plan.ownerScope.empty() ||
      plan.workspaceSyncKind.empty()) {
    return op->emitError()
           << "epilogue reorder plan must carry explicit owner/sync/reason truth";
  }
  if (plan.kind == EpilogueReorderKind::None) {
    if (plan.sharedTileRows != 0 || plan.sharedTileCols != 0 ||
        plan.liveSlots != 0 || plan.initSharedStoreVectorWidth != 0 ||
        plan.initSharedLoadVectorWidth != 0 ||
        plan.storeSharedStoreVectorWidth != 0 ||
        plan.storeSharedLoadVectorWidth != 0 ||
        plan.workspaceBarrierCount != 0 || plan.requiresWarpSync ||
        plan.reorderNeededForInit || plan.reorderNeededForStore ||
        plan.ownerScope != "none" || plan.workspaceSyncKind != "none" ||
        !plan.rowVectors.empty()) {
      return op->emitError()
             << "empty epilogue reorder plan must not carry shared reorder payload";
    }
    return success();
  }

  if (plan.sharedTileRows <= 0 || plan.sharedTileCols <= 0 ||
      plan.liveSlots <= 0 || plan.initSharedStoreVectorWidth <= 0 ||
      plan.initSharedLoadVectorWidth <= 0 ||
      plan.storeSharedStoreVectorWidth <= 0 ||
      plan.storeSharedLoadVectorWidth <= 0 ||
      plan.workspaceBarrierCount <= 0 || !plan.requiresWarpSync ||
      !plan.reorderNeededForInit || !plan.reorderNeededForStore ||
      plan.ownerScope != "cta_shared_workspace" ||
      plan.workspaceSyncKind != "cta" || plan.rowVectors.empty()) {
    return op->emitError()
           << "cta-shared epilogue reorder must carry complete shared owner truth";
  }

  llvm::DenseSet<int64_t> rowIds;
  for (const EpilogueRowVectorMapping &row : plan.rowVectors) {
    if (row.rowVectorId < 0 || !rowIds.insert(row.rowVectorId).second ||
        row.packId < 0 || row.warpOwner < 0 || row.vectorWidth <= 0 ||
        row.fragmentIds.empty()) {
      return op->emitError()
             << "epilogue row-vector mapping must carry unique positive ids";
    }
  }
  return success();
}

static FailureOr<int64_t>
derivePacksPerWarpRow(const AccumulatorPlan &accumulator,
                      const EpiloguePlan &epilogue, Operation *op) {
  auto *init = std::get_if<DirectGlobalVectorPlan>(&epilogue.init);
  auto *store = std::get_if<DirectGlobalVectorPlan>(&epilogue.store);
  if (!init || !store || init->packs.empty() || store->packs.empty() ||
      accumulator.packs.empty()) {
    op->emitError() << "epilogue reorder requires direct init/store payloads";
    return failure();
  }

  int64_t directPackCols = init->packs.front().cols;
  int64_t fragmentCols = accumulator.packs.front().cols;
  if (directPackCols <= 0 || fragmentCols <= 0 ||
      directPackCols % fragmentCols != 0) {
    op->emitError()
        << "epilogue reorder requires a direct pack width aligned to fragments";
    return failure();
  }

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
    op->emitError() << "epilogue reorder could not derive a stable warp-row "
                       "batch shape from direct packs";
    return failure();
  }
  return packsInOwnerTemplate / packRowCount;
}

} // namespace

FailureOr<EpilogueReorderPlan>
mlir::tb::deriveEpilogueReorderPlan(const KernelConfig &config,
                                    const TargetInfo &target,
                                    const AccumulatorPlan &accumulator,
                                    const EpiloguePlan &epilogue,
                                    Operation *op) {
  (void)target;
  auto *init = std::get_if<DirectGlobalVectorPlan>(&epilogue.init);
  auto *store = std::get_if<DirectGlobalVectorPlan>(&epilogue.store);
  if (!init || !store || init->packs.empty() || store->packs.empty() ||
      accumulator.packs.empty()) {
    op->emitError() << "epilogue reorder requires direct init/store payloads";
    return failure();
  }

  int64_t packsPerWarpRow = 0;
  auto derivedPacksPerWarpRow =
      derivePacksPerWarpRow(accumulator, epilogue, op);
  if (failed(derivedPacksPerWarpRow))
    return failure();
  packsPerWarpRow = *derivedPacksPerWarpRow;

  int64_t fragmentVectorWidth = accumulator.packs.front().vectorWidth;
  bool needsBoundaryGuard = init->boundaryAware || store->boundaryAware;
  bool canUseBoundaryAwareDirect =
      !needsBoundaryGuard ||
      (init->vectorWidth == fragmentVectorWidth &&
       store->vectorWidth == fragmentVectorWidth);
  bool needsSharedReorder = !config.exactTile && !canUseBoundaryAwareDirect;

  EpilogueReorderPlan plan;
  if (!needsSharedReorder) {
    plan.kind = EpilogueReorderKind::None;
    plan.ownerScope = "none";
    plan.workspaceSyncKind = "none";
    plan.reason =
        config.exactTile
            ? "exact-tile direct-global landing does not need an epilogue "
              "reorder owner"
            : "boundary-aware direct-global landing stays lane-local and does "
              "not need CTA shared row reorder";
    if (failed(validateEpilogueReorderPlan(plan, op)))
      return failure();
    return plan;
  }

  plan.kind = EpilogueReorderKind::CTASharedRowReorder;
  plan.sharedTileRows = init->packs.front().rows;
  plan.sharedTileCols = init->packs.front().cols;
  plan.liveSlots = config.numWarps == 1 ? packsPerWarpRow : 1;
  plan.initSharedStoreVectorWidth = init->vectorWidth;
  plan.initSharedLoadVectorWidth = fragmentVectorWidth;
  plan.storeSharedStoreVectorWidth = fragmentVectorWidth;
  plan.storeSharedLoadVectorWidth = store->vectorWidth;
  plan.workspaceBarrierCount = 2;
  plan.requiresWarpSync = true;
  plan.reorderNeededForInit = true;
  plan.reorderNeededForStore = true;
  plan.ownerScope = "cta_shared_workspace";
  plan.workspaceSyncKind = "cta";
  plan.reason =
      config.numWarps == 1
          ? "single-warp general-shape keeps the final landing as direct "
            "global vectors but inserts a CTA-workspace row reorder owner "
            "so the full warp-row batch can stay live without leaving the "
            "scratch allocation inside TargetLandingPlan"
          : "multi-warp general-shape keeps the final landing as direct "
            "global vectors but moves row reorder to a CTA workspace owner "
            "with one live slot per warp-row batch, so shared scratch no "
            "longer scales with the old semantic-role split";

  int64_t nextRowVectorId = 0;
  for (const DirectGlobalVectorPlan::Pack &pack : store->packs) {
    for (int64_t rowOffset : store->laneAccess.rowOffsets) {
      EpilogueRowVectorMapping row;
      row.rowVectorId = nextRowVectorId++;
      row.packId = pack.packId;
      row.warpOwner = pack.warpOwner;
      row.rowBase = pack.rowBase + rowOffset;
      row.colBase = pack.colBase;
      row.rowOffset = rowOffset;
      row.vectorWidth = store->vectorWidth;
      row.fragmentIds.assign(pack.fragmentIds.begin(), pack.fragmentIds.end());
      plan.rowVectors.push_back(std::move(row));
    }
  }

  if (failed(validateEpilogueReorderPlan(plan, op)))
    return failure();
  return plan;
}

DictionaryAttr
mlir::tb::buildEpilogueReorderPlanAttr(Builder &builder,
                                       const EpilogueReorderPlan &plan) {
  NamedAttrList attrs;
  attrs.set("kind", builder.getStringAttr(stringifyEpilogueReorderKind(plan.kind)));
  attrs.set("shared_tile_rows", builder.getI64IntegerAttr(plan.sharedTileRows));
  attrs.set("shared_tile_cols", builder.getI64IntegerAttr(plan.sharedTileCols));
  attrs.set("live_slots", builder.getI64IntegerAttr(plan.liveSlots));
  attrs.set("init_shared_store_vector_width",
            builder.getI64IntegerAttr(plan.initSharedStoreVectorWidth));
  attrs.set("init_shared_load_vector_width",
            builder.getI64IntegerAttr(plan.initSharedLoadVectorWidth));
  attrs.set("store_shared_store_vector_width",
            builder.getI64IntegerAttr(plan.storeSharedStoreVectorWidth));
  attrs.set("store_shared_load_vector_width",
            builder.getI64IntegerAttr(plan.storeSharedLoadVectorWidth));
  attrs.set("workspace_barrier_count",
            builder.getI64IntegerAttr(plan.workspaceBarrierCount));
  attrs.set("requires_warp_sync", builder.getBoolAttr(plan.requiresWarpSync));
  attrs.set("reorder_needed_for_init",
            builder.getBoolAttr(plan.reorderNeededForInit));
  attrs.set("reorder_needed_for_store",
            builder.getBoolAttr(plan.reorderNeededForStore));
  attrs.set("owner_scope", builder.getStringAttr(plan.ownerScope));
  attrs.set("workspace_sync_kind", builder.getStringAttr(plan.workspaceSyncKind));
  attrs.set("reason", builder.getStringAttr(plan.reason));
  SmallVector<Attribute> rowVectors;
  rowVectors.reserve(plan.rowVectors.size());
  for (const EpilogueRowVectorMapping &row : plan.rowVectors)
    rowVectors.push_back(buildRowVectorAttr(builder, row));
  attrs.set("row_vectors", builder.getArrayAttr(rowVectors));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<EpilogueReorderPlan>
mlir::tb::parseEpilogueReorderPlanAttr(Operation *op) {
  auto root =
      dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.epilogue_reorder_plan"));
  if (!root) {
    op->emitError() << "missing `tb.epilogue_reorder_plan` attribute";
    return failure();
  }
  auto kind = readStringField(root, "kind", op);
  auto sharedTileRows = readI64Field(root, "shared_tile_rows", op);
  auto sharedTileCols = readI64Field(root, "shared_tile_cols", op);
  auto liveSlots = readI64Field(root, "live_slots", op);
  auto initSharedStoreVectorWidth =
      readI64Field(root, "init_shared_store_vector_width", op);
  auto initSharedLoadVectorWidth =
      readI64Field(root, "init_shared_load_vector_width", op);
  auto storeSharedStoreVectorWidth =
      readI64Field(root, "store_shared_store_vector_width", op);
  auto storeSharedLoadVectorWidth =
      readI64Field(root, "store_shared_load_vector_width", op);
  auto workspaceBarrierCount =
      readI64Field(root, "workspace_barrier_count", op);
  auto ownerScope = readStringField(root, "owner_scope", op);
  auto workspaceSyncKind = readStringField(root, "workspace_sync_kind", op);
  auto reason = readStringField(root, "reason", op);
  auto rowVectors = dyn_cast_or_null<ArrayAttr>(root.get("row_vectors"));
  if (failed(kind) || failed(sharedTileRows) || failed(sharedTileCols) ||
      failed(liveSlots) || failed(initSharedStoreVectorWidth) ||
      failed(initSharedLoadVectorWidth) ||
      failed(storeSharedStoreVectorWidth) ||
      failed(storeSharedLoadVectorWidth) ||
      failed(workspaceBarrierCount) || failed(ownerScope) ||
      failed(workspaceSyncKind) || failed(reason) || !rowVectors) {
    op->emitError() << "malformed `tb.epilogue_reorder_plan` attribute";
    return failure();
  }
  auto parsedKind = parseEpilogueReorderKind(*kind, op);
  if (failed(parsedKind))
    return failure();

  EpilogueReorderPlan plan;
  plan.kind = *parsedKind;
  plan.sharedTileRows = *sharedTileRows;
  plan.sharedTileCols = *sharedTileCols;
  plan.liveSlots = *liveSlots;
  plan.initSharedStoreVectorWidth = *initSharedStoreVectorWidth;
  plan.initSharedLoadVectorWidth = *initSharedLoadVectorWidth;
  plan.storeSharedStoreVectorWidth = *storeSharedStoreVectorWidth;
  plan.storeSharedLoadVectorWidth = *storeSharedLoadVectorWidth;
  plan.workspaceBarrierCount = *workspaceBarrierCount;
  plan.requiresWarpSync =
      readOptionalBoolField(root, "requires_warp_sync", false);
  plan.reorderNeededForInit =
      readOptionalBoolField(root, "reorder_needed_for_init", false);
  plan.reorderNeededForStore =
      readOptionalBoolField(root, "reorder_needed_for_store", false);
  plan.ownerScope = ownerScope->str();
  plan.workspaceSyncKind = workspaceSyncKind->str();
  plan.reason = reason->str();
  plan.rowVectors.reserve(rowVectors.size());
  for (Attribute attr : rowVectors) {
    auto dict = dyn_cast<DictionaryAttr>(attr);
    if (!dict) {
      op->emitError() << "`row_vectors` must contain dictionary entries";
      return failure();
    }
    auto row = parseRowVectorAttr(dict, op);
    if (failed(row))
      return failure();
    plan.rowVectors.push_back(std::move(*row));
  }
  if (failed(validateEpilogueReorderPlan(plan, op))) {
    op->emitError() << "malformed `tb.epilogue_reorder_plan` attribute";
    return failure();
  }
  return plan;
}
