#include "tb/Analysis/MatmulRewritePlan.h"

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

static DictionaryAttr buildOperandPathAttr(Builder &builder,
                                           const OperandFragmentPath &path) {
  NamedAttrList attrs;
  attrs.set("role", builder.getStringAttr(path.role));
  attrs.set("instruction_shape",
            buildI64ArrayAttr(builder, path.instructionShape));
  attrs.set("fragment_shape", buildI64ArrayAttr(builder, path.fragmentShape));
  attrs.set("fragments_per_k_group",
            builder.getI64IntegerAttr(path.fragmentsPerKGroup));
  attrs.set("consumer_tile_span_m",
            builder.getI64IntegerAttr(path.consumerTileSpanM));
  attrs.set("consumer_tile_span_n",
            builder.getI64IntegerAttr(path.consumerTileSpanN));
  attrs.set("uses_ldmatrix", builder.getBoolAttr(path.usesLdMatrix));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<OperandFragmentPath> parseOperandPathAttr(DictionaryAttr dict,
                                                           Operation *op) {
  OperandFragmentPath path;
  auto role = readStringField(dict, "role", op);
  auto instructionShape =
      readDenseI64ArrayField(dict, "instruction_shape", op);
  auto fragmentShape = readDenseI64ArrayField(dict, "fragment_shape", op);
  auto fragmentsPerKGroup = readI64Field(dict, "fragments_per_k_group", op);
  auto consumerTileSpanM = readI64Field(dict, "consumer_tile_span_m", op);
  auto consumerTileSpanN = readI64Field(dict, "consumer_tile_span_n", op);
  auto usesLdMatrix = readBoolField(dict, "uses_ldmatrix", op);
  if (failed(role) || failed(instructionShape) || failed(fragmentShape) ||
      failed(fragmentsPerKGroup) || failed(consumerTileSpanM) ||
      failed(consumerTileSpanN) || failed(usesLdMatrix)) {
    return failure();
  }
  path.role = role->str();
  path.instructionShape = parseI64Array(*instructionShape);
  path.fragmentShape = parseI64Array(*fragmentShape);
  path.fragmentsPerKGroup = *fragmentsPerKGroup;
  path.consumerTileSpanM = *consumerTileSpanM;
  path.consumerTileSpanN = *consumerTileSpanN;
  path.usesLdMatrix = *usesLdMatrix;
  return path;
}

static FailureOr<int64_t> deriveBConsumerTileSpan(const EncodingPlan &encodings,
                                                  Operation *op) {
  if (encodings.fragmentB.instructionShape.size() != 2 ||
      encodings.fragmentAcc.repeatShape.size() != 2 ||
      encodings.fragmentB.logicalShape.size() != 2) {
    op->emitError() << "B fragment encoding is missing layout-driven rewrite "
                       "geometry";
    return failure();
  }
  int64_t instructionN = encodings.fragmentB.instructionShape.back();
  int64_t warpTileN = encodings.fragmentB.logicalShape.back();
  int64_t accTilesN = encodings.fragmentAcc.repeatShape.back();
  if (instructionN <= 0 || warpTileN <= 0 || accTilesN <= 0 ||
      warpTileN != accTilesN * instructionN) {
    op->emitError() << "B fragment encoding does not align with the "
                       "accumulator N tile grid";
    return failure();
  }

  int64_t tileSpanN = 1;
  if (encodings.fragmentB.ldmatrixTranspose &&
      encodings.fragmentB.ldmatrixTileCount > 0) {
    tileSpanN = std::max<int64_t>(1, encodings.fragmentB.ldmatrixTileCount / 2);
  }
  tileSpanN = std::min(tileSpanN, accTilesN);
  if (tileSpanN <= 0 || accTilesN % tileSpanN != 0) {
    op->emitError() << "B fragment group span must evenly cover the "
                       "accumulator N tile grid";
    return failure();
  }
  return tileSpanN;
}

static LogicalResult validateMatmulRewritePlan(const MatmulRewritePlan &plan,
                                               Operation *op) {
  if (plan.contractModel.empty() || plan.mainloopKind.empty())
    return op->emitError() << "matmul rewrite plan must carry contract/mainloop";
  if (plan.instructionM <= 0 || plan.instructionN <= 0 || plan.instructionK <= 0)
    return op->emitError() << "matmul rewrite instruction shape must be positive";
  if (plan.kGroups <= 0 || plan.accTilesM <= 0 || plan.accTilesN <= 0 ||
      plan.bGroupCount <= 0 || plan.accumulatorFragments <= 0) {
    return op->emitError() << "matmul rewrite geometry must be positive";
  }
  if (plan.aPath.fragmentsPerKGroup != plan.accTilesM)
    return op->emitError() << "A fragment path must match accTilesM";
  if (plan.aPath.consumerTileSpanM != 1 || plan.aPath.consumerTileSpanN != 1)
    return op->emitError() << "A fragment path must stay aligned to one "
                              "consumer tile";
  if (plan.bPath.fragmentsPerKGroup != plan.bGroupCount)
    return op->emitError() << "B fragment path must match B fragment-group "
                              "count";
  if (plan.bPath.consumerTileSpanM != 1 || plan.bPath.consumerTileSpanN <= 0 ||
      plan.bPath.fragmentsPerKGroup * plan.bPath.consumerTileSpanN !=
          plan.accTilesN) {
    return op->emitError() << "B fragment path must be derived from encoding "
                              "group span, not a lowering-side heuristic";
  }
  return success();
}

} // namespace

FailureOr<MatmulRewritePlan> mlir::tb::deriveMatmulRewritePlan(
    const KernelConfig &config, const EncodingPlan &encodings,
    const AccumulatorPlan &accumulator, const EpiloguePlan &epilogue,
    Operation *op) {
  (void)accumulator;
  MatmulRewritePlan plan;
  plan.contractModel = "triton_layout_driven_matmul_rewrite_v2";
  plan.mainloopKind = "tensor_core_ldmatrix_mma_async";
  if (encodings.fragmentA.instructionShape.size() < 2 ||
      encodings.fragmentB.instructionShape.size() < 2 ||
      encodings.fragmentAcc.repeatShape.size() < 2) {
    op->emitError() << "encoding plan is missing fragment rewrite geometry";
    return failure();
  }

  plan.instructionM = encodings.fragmentA.instructionShape.front();
  plan.instructionN = encodings.fragmentB.instructionShape.back();
  plan.instructionK = encodings.fragmentA.instructionShape.back();
  if (plan.instructionK <= 0) {
    op->emitError() << "matmul rewrite requires a positive instruction_k";
    return failure();
  }
  plan.kGroups = ceilDiv(config.problemK, plan.instructionK);
  plan.accTilesM = encodings.fragmentAcc.repeatShape.front();
  plan.accTilesN = encodings.fragmentAcc.repeatShape.back();
  auto bTileSpanN = deriveBConsumerTileSpan(encodings, op);
  if (failed(bTileSpanN))
    return failure();
  plan.bGroupCount = plan.accTilesN / *bTileSpanN;
  plan.accumulatorFragments = plan.accTilesM * plan.accTilesN;
  plan.directAccumulatorInit =
      epilogue.initMode == AccumulatorInitMode::DirectGlobalVector;
  plan.directAccumulatorStore =
      epilogue.storeMode == AccumulatorStoreMode::DirectGlobalVector;

  plan.aPath = {"a",
                SmallVector<int64_t, 4>(encodings.fragmentA.instructionShape),
                SmallVector<int64_t, 4>(encodings.fragmentA.logicalShape),
                plan.accTilesM,
                /*consumerTileSpanM=*/1,
                /*consumerTileSpanN=*/1,
                /*usesLdMatrix=*/true};
  plan.bPath = {"b",
                SmallVector<int64_t, 4>(encodings.fragmentB.instructionShape),
                SmallVector<int64_t, 4>(encodings.fragmentB.logicalShape),
                plan.bGroupCount,
                /*consumerTileSpanM=*/1,
                /*consumerTileSpanN=*/*bTileSpanN,
                /*usesLdMatrix=*/true};

  if (failed(validateMatmulRewritePlan(plan, op)))
    return failure();
  return plan;
}

DictionaryAttr mlir::tb::buildMatmulRewritePlanAttr(
    Builder &builder, const MatmulRewritePlan &plan) {
  NamedAttrList attrs;
  attrs.set("contract_model", builder.getStringAttr(plan.contractModel));
  attrs.set("mainloop_kind", builder.getStringAttr(plan.mainloopKind));
  attrs.set("instruction_m", builder.getI64IntegerAttr(plan.instructionM));
  attrs.set("instruction_n", builder.getI64IntegerAttr(plan.instructionN));
  attrs.set("instruction_k", builder.getI64IntegerAttr(plan.instructionK));
  attrs.set("k_groups", builder.getI64IntegerAttr(plan.kGroups));
  attrs.set("acc_tiles_m", builder.getI64IntegerAttr(plan.accTilesM));
  attrs.set("acc_tiles_n", builder.getI64IntegerAttr(plan.accTilesN));
  attrs.set("b_group_count", builder.getI64IntegerAttr(plan.bGroupCount));
  attrs.set("accumulator_fragments",
            builder.getI64IntegerAttr(plan.accumulatorFragments));
  attrs.set("direct_accumulator_init",
            builder.getBoolAttr(plan.directAccumulatorInit));
  attrs.set("direct_accumulator_store",
            builder.getBoolAttr(plan.directAccumulatorStore));
  attrs.set("a_path", buildOperandPathAttr(builder, plan.aPath));
  attrs.set("b_path", buildOperandPathAttr(builder, plan.bPath));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<MatmulRewritePlan> mlir::tb::parseMatmulRewritePlanAttr(
    Operation *op) {
  auto root =
      dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.matmul_rewrite"));
  if (!root) {
    op->emitError() << "missing `tb.matmul_rewrite` attribute";
    return failure();
  }

  MatmulRewritePlan plan;
  auto contractModel = readStringField(root, "contract_model", op);
  auto mainloopKind = readStringField(root, "mainloop_kind", op);
  auto instructionM = readI64Field(root, "instruction_m", op);
  auto instructionN = readI64Field(root, "instruction_n", op);
  auto instructionK = readI64Field(root, "instruction_k", op);
  auto kGroups = readI64Field(root, "k_groups", op);
  auto accTilesM = readI64Field(root, "acc_tiles_m", op);
  auto accTilesN = readI64Field(root, "acc_tiles_n", op);
  auto bGroupCount = readI64Field(root, "b_group_count", op);
  auto accumulatorFragments =
      readI64Field(root, "accumulator_fragments", op);
  auto directAccumulatorInit =
      readBoolField(root, "direct_accumulator_init", op);
  auto directAccumulatorStore =
      readBoolField(root, "direct_accumulator_store", op);
  auto aPath = dyn_cast_or_null<DictionaryAttr>(root.get("a_path"));
  auto bPath = dyn_cast_or_null<DictionaryAttr>(root.get("b_path"));
  if (failed(contractModel) || failed(mainloopKind) || failed(instructionM) ||
      failed(instructionN) || failed(instructionK) || failed(kGroups) ||
      failed(accTilesM) || failed(accTilesN) || failed(bGroupCount) ||
      failed(accumulatorFragments) || failed(directAccumulatorInit) ||
      failed(directAccumulatorStore) || !aPath || !bPath) {
    op->emitError() << "malformed `tb.matmul_rewrite` attribute";
    return failure();
  }

  auto parsedAPath = parseOperandPathAttr(aPath, op);
  auto parsedBPath = parseOperandPathAttr(bPath, op);
  if (failed(parsedAPath) || failed(parsedBPath))
    return failure();

  plan.contractModel = contractModel->str();
  plan.mainloopKind = mainloopKind->str();
  plan.instructionM = *instructionM;
  plan.instructionN = *instructionN;
  plan.instructionK = *instructionK;
  plan.kGroups = *kGroups;
  plan.accTilesM = *accTilesM;
  plan.accTilesN = *accTilesN;
  plan.bGroupCount = *bGroupCount;
  plan.accumulatorFragments = *accumulatorFragments;
  plan.directAccumulatorInit = *directAccumulatorInit;
  plan.directAccumulatorStore = *directAccumulatorStore;
  plan.aPath = std::move(*parsedAPath);
  plan.bPath = std::move(*parsedBPath);
  if (failed(validateMatmulRewritePlan(plan, op))) {
    op->emitError() << "malformed `tb.matmul_rewrite` attribute";
    return failure();
  }
  return plan;
}
