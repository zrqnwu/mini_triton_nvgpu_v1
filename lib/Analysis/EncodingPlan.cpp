#include "tb/Analysis/EncodingPlan.h"

#include "tb/Analysis/KernelConfig.h"
#include "tb/IR/TBAttrs.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <tuple>

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

static FailureOr<ArrayAttr> readArrayField(DictionaryAttr dict, StringRef name,
                                           Operation *op) {
  auto attr = dyn_cast_or_null<ArrayAttr>(dict.get(name));
  if (!attr) {
    op->emitError() << "missing array field `" << name << "`";
    return failure();
  }
  return attr;
}

static DictionaryAttr buildSharedSpecAttr(Builder &builder,
                                          const SharedEncodingSpec &spec) {
  NamedAttrList attrs;
  attrs.set("logical_shape", buildI64ArrayAttr(builder, spec.logicalShape));
  attrs.set("alloc_shape", buildI64ArrayAttr(builder, spec.allocShape));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<SharedEncodingSpec> parseSharedSpecAttr(DictionaryAttr dict,
                                                         Operation *op) {
  SharedEncodingSpec spec;
  auto logicalShape = readDenseI64ArrayField(dict, "logical_shape", op);
  auto allocShape = readDenseI64ArrayField(dict, "alloc_shape", op);
  if (failed(logicalShape) || failed(allocShape)) {
    return failure();
  }
  spec.logicalShape = parseI64Array(*logicalShape);
  spec.allocShape = parseI64Array(*allocShape);
  return spec;
}

static LogicalResult validateSharedSpec(const SharedEncodingSpec &spec,
                                        Operation *op, StringRef role) {
  if (spec.logicalShape.empty())
    return op->emitError() << role << " shared spec must carry a logical shape";
  if (spec.logicalShape.size() != spec.allocShape.size()) {
    return op->emitError()
           << role << " shared spec alloc_shape rank must match logical_shape";
  }
  if (llvm::any_of(spec.logicalShape, [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(spec.allocShape, [](int64_t value) { return value <= 0; })) {
    return op->emitError() << role
                           << " shared spec shapes must be strictly positive";
  }
  return success();
}

static DictionaryAttr buildFragmentAttr(Builder &builder,
                                        const FragmentEncodingSpec &spec) {
  NamedAttrList attrs;
  attrs.set("role", builder.getStringAttr(spec.role));
  attrs.set("instruction_shape", buildI64ArrayAttr(builder, spec.instructionShape));
  attrs.set("repeat_shape", buildI64ArrayAttr(builder, spec.repeatShape));
  attrs.set("logical_shape", buildI64ArrayAttr(builder, spec.logicalShape));
  attrs.set("register_shape", buildI64ArrayAttr(builder, spec.registerShape));
  attrs.set("register_order", buildI64ArrayAttr(builder, spec.registerOrder));
  attrs.set("lane_row_modulus",
            builder.getI64IntegerAttr(spec.laneRowModulus));
  attrs.set("lane_col_divisor",
            builder.getI64IntegerAttr(spec.laneColDivisor));
  attrs.set("lane_col_stride",
            builder.getI64IntegerAttr(spec.laneColStride));
  attrs.set("values_per_lane", builder.getI64IntegerAttr(spec.valuesPerLane));
  attrs.set("ldmatrix_tile_count",
            builder.getI64IntegerAttr(spec.ldmatrixTileCount));
  attrs.set("ldmatrix_transpose",
            builder.getBoolAttr(spec.ldmatrixTranspose));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<FragmentEncodingSpec> parseFragmentAttr(DictionaryAttr dict,
                                                         Operation *op) {
  FragmentEncodingSpec spec;
  auto role = readStringField(dict, "role", op);
  auto instructionShape = readDenseI64ArrayField(dict, "instruction_shape", op);
  auto repeatShape = readDenseI64ArrayField(dict, "repeat_shape", op);
  auto logicalShape = readDenseI64ArrayField(dict, "logical_shape", op);
  auto registerShape = readDenseI64ArrayField(dict, "register_shape", op);
  auto registerOrder = readDenseI64ArrayField(dict, "register_order", op);
  auto laneRowModulus = readI64Field(dict, "lane_row_modulus", op);
  auto laneColDivisor = readI64Field(dict, "lane_col_divisor", op);
  auto laneColStride = readI64Field(dict, "lane_col_stride", op);
  auto valuesPerLane = readI64Field(dict, "values_per_lane", op);
  auto ldmatrixTileCount = readI64Field(dict, "ldmatrix_tile_count", op);
  auto ldmatrixTranspose = readBoolField(dict, "ldmatrix_transpose", op);
  if (failed(role) || failed(instructionShape) || failed(repeatShape) ||
      failed(logicalShape) || failed(registerShape) || failed(registerOrder) ||
      failed(laneRowModulus) || failed(laneColDivisor) ||
      failed(laneColStride) || failed(valuesPerLane) ||
      failed(ldmatrixTileCount) || failed(ldmatrixTranspose)) {
    return failure();
  }
  spec.role = role->str();
  spec.instructionShape = parseI64Array(*instructionShape);
  spec.repeatShape = parseI64Array(*repeatShape);
  spec.logicalShape = parseI64Array(*logicalShape);
  spec.registerShape = parseI64Array(*registerShape);
  spec.registerOrder = parseI64Array(*registerOrder);
  spec.laneRowModulus = *laneRowModulus;
  spec.laneColDivisor = *laneColDivisor;
  spec.laneColStride = *laneColStride;
  spec.valuesPerLane = *valuesPerLane;
  spec.ldmatrixTileCount = *ldmatrixTileCount;
  spec.ldmatrixTranspose = *ldmatrixTranspose;
  return spec;
}

static LogicalResult validateFragmentSpec(const FragmentEncodingSpec &spec,
                                          Operation *op) {
  if (spec.role.empty())
    return op->emitError() << "fragment encoding spec must carry a role";
  if (spec.instructionShape.size() != 2 || spec.repeatShape.size() != 2 ||
      spec.logicalShape.size() != 2 || spec.registerShape.size() != 2) {
    return op->emitError()
           << "fragment encoding spec `" << spec.role
           << "` must carry rank-2 instruction/repeat/logical/register shapes";
  }
  if (spec.laneRowModulus <= 0 || spec.laneColDivisor <= 0 ||
      spec.laneColStride <= 0) {
    return op->emitError() << "fragment encoding spec `" << spec.role
                           << "` must carry positive lane mapping metadata";
  }
  return success();
}

static DictionaryAttr buildEncodingEntryAttr(Builder &builder,
                                             const EncodingEntry &entry) {
  NamedAttrList attrs;
  attrs.set("name", builder.getStringAttr(entry.name));
  attrs.set("encoding", entry.encodingAttr);
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<EncodingEntry> parseEncodingEntryAttr(DictionaryAttr dict,
                                                       Operation *op) {
  EncodingEntry entry;
  auto name = readStringField(dict, "name", op);
  auto encoding = dict.get("encoding");
  if (failed(name) || !encoding) {
    op->emitError() << "malformed encoding entry";
    return failure();
  }
  if (!isa<BlockedEncodingAttr, SharedEncodingAttr, DotOperandEncodingAttr,
           NVGPUMmaEncodingAttr, AccumulatorEncodingAttr>(encoding)) {
    op->emitError() << "unsupported tb encoding attr `" << encoding << "`";
    return failure();
  }
  entry.name = name->str();
  entry.encodingAttr = encoding;
  return entry;
}

static FailureOr<Attribute> getEncodingAttrImpl(const EncodingPlan &plan,
                                                int encodingId, Operation *op,
                                                StringRef role) {
  if (encodingId < 0 ||
      encodingId >= static_cast<int>(plan.encodings.size())) {
    op->emitError() << "invalid encoding id " << encodingId << " for `"
                    << role << "`";
    return failure();
  }
  const EncodingEntry &entry = plan.encodings[encodingId];
  if (!entry.encodingAttr) {
    op->emitError() << "missing encoding attr for entry `" << entry.name
                    << "`";
    return failure();
  }
  return entry.encodingAttr;
}

template <typename AttrT>
static FailureOr<AttrT> getTypedEncodingAttr(const EncodingPlan &plan,
                                             int encodingId, Operation *op,
                                             StringRef role,
                                             StringRef expectedKind) {
  auto attr = getEncodingAttrImpl(plan, encodingId, op, role);
  if (failed(attr))
    return failure();
  auto typed = dyn_cast<AttrT>(*attr);
  if (!typed) {
    op->emitError() << "encoding `" << role << "` must be " << expectedKind;
    return failure();
  }
  return typed;
}

struct SemanticLayoutSpec {
  SmallVector<int64_t, 2> order;
  bool zeroOffset = true;
  bool hasStaticStrides = true;
  bool unitStrideOnMinor = false;
  bool contiguous = false;
};

static FailureOr<SemanticLayoutSpec>
getSemanticLayoutSpec(MemDescType desc, Operation *op, StringRef role) {
  auto layout = dyn_cast<SemanticLayoutAttr>(desc.getEncoding());
  if (!layout) {
    op->emitError() << "semantic operand `" << role
                    << "` must carry #tb.semantic_layout";
    return failure();
  }
  if (layout.getOrder().size() != 2 || layout.getStrides().size() != 2) {
    op->emitError() << "semantic operand `" << role
                    << "` must carry rank-2 semantic order/stride metadata";
    return failure();
  }

  SemanticLayoutSpec spec;
  spec.order.assign(layout.getOrder().begin(), layout.getOrder().end());
  spec.zeroOffset = layout.getZeroOffset();
  spec.hasStaticStrides = layout.getHasStaticStrides();
  spec.unitStrideOnMinor = layout.getUnitStrideOnMinor();
  spec.contiguous = layout.getContiguous();
  return spec;
}

static FailureOr<std::pair<int64_t, int64_t>>
deriveWarpGrid(const KernelConfig &config, const TargetInfo &target,
               Operation *op) {
  int64_t instructionM = target.mmaInstrShape.size() >= 2 ? target.mmaInstrShape[0]
                                                          : 16;
  int64_t instructionN = target.mmaInstrShape.size() >= 2 ? target.mmaInstrShape[1]
                                                          : 8;
  int64_t bestWarpGridM = -1;
  int64_t bestWarpGridN = -1;
  int64_t bestScore = std::numeric_limits<int64_t>::max();
  for (int64_t warpGridM = 1; warpGridM <= config.numWarps; ++warpGridM) {
    if (config.numWarps % warpGridM != 0)
      continue;
    int64_t warpGridN = config.numWarps / warpGridM;
    if (config.blockM % warpGridM != 0 || config.blockN % warpGridN != 0)
      continue;
    int64_t warpTileM = config.blockM / warpGridM;
    int64_t warpTileN = config.blockN / warpGridN;
    if (warpTileM % instructionM != 0 || warpTileN % instructionN != 0)
      continue;
    int64_t score = std::abs(warpTileM - warpTileN);
    if (score < bestScore ||
        (score == bestScore &&
         ((config.blockM >= config.blockN && warpGridM > bestWarpGridM) ||
          (config.blockM < config.blockN && warpGridN > bestWarpGridN)))) {
      bestScore = score;
      bestWarpGridM = warpGridM;
      bestWarpGridN = warpGridN;
    }
  }
  if (bestWarpGridM <= 0 || bestWarpGridN <= 0) {
    op->emitError()
        << "failed to derive a legal warp grid from module num-warps, block "
           "shape and mma instruction shape";
    return failure();
  }
  return std::make_pair(bestWarpGridM, bestWarpGridN);
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

struct SharedEncodingHeuristic {
  SmallVector<int64_t, 2> order = {1, 0};
  int64_t perPhase = 1;
  int64_t maxPhase = 1;
  bool transposed = false;
  int64_t swizzlingByteWidth = 0;
};

static SharedEncodingHeuristic
deriveSharedEncodingHeuristic(const TargetInfo &target,
                              const SharedEncodingSpec &sharedSpec,
                              const FragmentEncodingSpec &fragmentSpec,
                              int64_t elementBytes) {
  SharedEncodingHeuristic heuristic;
  if (sharedSpec.logicalShape.size() != 2 || fragmentSpec.logicalShape.size() != 2 ||
      elementBytes <= 0) {
    return heuristic;
  }

  int64_t minorDim = heuristic.order.front();
  int64_t sharedMinor = sharedSpec.logicalShape[minorDim];
  int64_t fragmentMinor = fragmentSpec.logicalShape[minorDim];
  int64_t contiguousElems = std::min(sharedMinor, fragmentMinor);
  int64_t contiguousBytes = contiguousElems * elementBytes;
  if (contiguousBytes <= 0)
    return heuristic;

  int64_t swizzleBytes = 0;
  if (fragmentSpec.ldmatrixTileCount > 0 && elementBytes <= target.sharedBankBytes &&
      contiguousBytes >= 16) {
    swizzleBytes = 16;
  } else if (elementBytes == target.sharedBankBytes && contiguousBytes >= 32) {
    swizzleBytes = 32;
  }
  if (target.asyncCopyPreferredBytes > 0)
    swizzleBytes = std::min(swizzleBytes, target.asyncCopyPreferredBytes);
  if (swizzleBytes <= 0 || contiguousBytes % swizzleBytes != 0 ||
      swizzleBytes % elementBytes != 0) {
    return heuristic;
  }

  heuristic.swizzlingByteWidth = swizzleBytes;
  heuristic.perPhase = std::max<int64_t>(1, 128 / contiguousBytes);
  heuristic.maxPhase =
      std::max<int64_t>(1, contiguousBytes / heuristic.swizzlingByteWidth);
  heuristic.maxPhase = std::min<int64_t>(heuristic.maxPhase, 8);
  return heuristic;
}

static FailureOr<EncodingPlan> deriveEncodingPlanDirect(
    const KernelConfig &config, const TargetInfo &target,
    const MatmulSemantics &semantics, const ProgramMappingPlan &programMapping,
    Operation *op) {
  if (!target.supportsMmaSync) {
    op->emitError() << "target does not support mma.sync";
    return failure();
  }
  if (target.mmaInstrShape.size() != 3 || target.mmaInstrShape[0] <= 0 ||
      target.mmaInstrShape[1] <= 0 || target.mmaInstrShape[2] <= 0) {
    op->emitError() << "target must provide a valid rank-3 mma instruction shape";
    return failure();
  }

  int64_t instructionM = target.mmaInstrShape[0];
  int64_t instructionN = target.mmaInstrShape[1];
  int64_t instructionK = target.mmaInstrShape[2];
  if (config.blockM % instructionM != 0 || config.blockN % instructionN != 0 ||
      config.blockK % instructionK != 0) {
    op->emitError() << "block shape is incompatible with the target mma "
                       "instruction shape";
    return failure();
  }

  auto aSemanticDesc = dyn_cast<MemDescType>(semantics.aDescType);
  auto bSemanticDesc = dyn_cast<MemDescType>(semantics.bDescType);
  auto cSemanticDesc = dyn_cast<MemDescType>(semantics.cDescType);
  if (!aSemanticDesc || !bSemanticDesc || !cSemanticDesc) {
    op->emitError() << "semanticization must provide !tb.memdesc operands";
    return failure();
  }
  auto aLayout = getSemanticLayoutSpec(aSemanticDesc, op, "a");
  auto bLayout = getSemanticLayoutSpec(bSemanticDesc, op, "b");
  auto cLayout = getSemanticLayoutSpec(cSemanticDesc, op, "c");
  auto warpGrid = deriveWarpGrid(config, target, op);
  if (failed(aLayout) || failed(bLayout) || failed(cLayout) ||
      failed(warpGrid)) {
    return failure();
  }
  for (const auto &item :
       {std::make_tuple(StringRef("a"), *aLayout),
        std::make_tuple(StringRef("b"), *bLayout),
        std::make_tuple(StringRef("c"), *cLayout)}) {
    if (!std::get<1>(item).zeroOffset) {
      op->emitError() << "stage1 layout freeze currently requires zero-offset "
                         "semantic operand `"
                      << std::get<0>(item) << "`";
      return failure();
    }
    if (!std::get<1>(item).hasStaticStrides) {
      op->emitError() << "stage1 layout freeze currently requires static "
                         "semantic strides for operand `"
                      << std::get<0>(item) << "`";
      return failure();
    }
    if (!std::get<1>(item).unitStrideOnMinor) {
      op->emitError() << "stage1 layout freeze currently requires unit stride "
                         "on the minor dimension for operand `"
                      << std::get<0>(item) << "`";
      return failure();
    }
  }

  EncodingPlan plan;
  plan.contractModel = "strict_v2_layout_owner";
  int64_t warpGridM = warpGrid->first;
  int64_t warpGridN = warpGrid->second;
  if (config.blockM % warpGridM != 0 || config.blockN % warpGridN != 0) {
    op->emitError() << "block shape must be divisible by blocked encoding "
                       "warps_per_cta";
    return failure();
  }
  int64_t warpTileM = config.blockM / warpGridM;
  int64_t warpTileN = config.blockN / warpGridN;
  plan.aSharedSpec.logicalShape = {config.blockM, instructionK};
  plan.aSharedSpec.allocShape = plan.aSharedSpec.logicalShape;
  plan.bSharedSpec.logicalShape = {instructionK, config.blockN};
  plan.bSharedSpec.allocShape = plan.bSharedSpec.logicalShape;
  plan.fragmentA = {"a",
                    {instructionM, instructionK},
                    {warpTileM / instructionM, 1},
                    {warpTileM, instructionK},
                    {4, 2},
                    {0, 1},
                    /*laneRowModulus=*/instructionM,
                    /*laneColDivisor=*/instructionM,
                    /*laneColStride=*/instructionN,
                    8,
                    4,
                    /*ldmatrixTranspose=*/false};
  plan.fragmentB = {"b",
                    {instructionK, instructionN},
                    {1, warpTileN / instructionN},
                    {instructionK, warpTileN},
                    {2, 2},
                    {0, 1},
                    /*laneRowModulus=*/instructionM,
                    /*laneColDivisor=*/instructionM,
                    /*laneColStride=*/instructionN,
                    4,
                    4,
                    /*ldmatrixTranspose=*/true};
  plan.fragmentAcc = {"acc",
                      {instructionM, instructionN},
                      {warpTileM / instructionM, warpTileN / instructionN},
                      {warpTileM, warpTileN},
                      {2, 2},
                      {0, 1},
                      /*laneRowModulus=*/0,
                      /*laneColDivisor=*/0,
                      /*laneColStride=*/0,
                      4,
                      0,
                      /*ldmatrixTranspose=*/false};
  if (failed(validateSharedSpec(plan.aSharedSpec, op, "a_shared")) ||
      failed(validateSharedSpec(plan.bSharedSpec, op, "b_shared")) ||
      failed(validateFragmentSpec(plan.fragmentA, op)) ||
      failed(validateFragmentSpec(plan.fragmentB, op))) {
    return failure();
  }

  MLIRContext *ctx = op->getContext();
  auto cga = CGAEncodingAttr::get(ctx, programMapping.ctasPerCGA,
                                  programMapping.ctaSplitNum,
                                  programMapping.ctaOrder);
  auto mmaAttr = NVGPUMmaEncodingAttr::get(
      ctx, "mma_sync",
      /*versionMajor=*/2, /*versionMinor=*/0,
      ArrayRef<int64_t>{warpGridM, warpGridN},
      ArrayRef<int64_t>(target.mmaInstrShape), cga);

  SharedEncodingHeuristic aSharedHeuristic = deriveSharedEncodingHeuristic(
      target, plan.aSharedSpec, plan.fragmentA, getScalarByteWidth(config.aScalar));
  SharedEncodingHeuristic bSharedHeuristic = deriveSharedEncodingHeuristic(
      target, plan.bSharedSpec, plan.fragmentB, getScalarByteWidth(config.bScalar));

  auto makeBlockedGlobal = [&](StringRef name,
                               ArrayRef<int64_t> order) -> int {
    EncodingEntry entry;
    entry.name = name.str();
    entry.encodingAttr = BlockedEncodingAttr::get(
        ctx, ArrayRef<int64_t>{1, 1},
        ArrayRef<int64_t>{target.threadsPerWarp, 1},
        ArrayRef<int64_t>{warpGridM, warpGridN}, order, cga);
    int id = static_cast<int>(plan.encodings.size());
    plan.encodings.push_back(std::move(entry));
    return id;
  };

  auto makeShared = [&](StringRef name,
                        const SharedEncodingHeuristic &heuristic) -> int {
    EncodingEntry entry;
    entry.name = name.str();
    entry.encodingAttr = SharedEncodingAttr::get(
        ctx, ArrayRef<int64_t>(heuristic.order), heuristic.perPhase,
        heuristic.maxPhase, ArrayRef<int64_t>{}, ArrayRef<int64_t>{},
        heuristic.transposed, heuristic.swizzlingByteWidth, cga);
    int id = static_cast<int>(plan.encodings.size());
    plan.encodings.push_back(std::move(entry));
    return id;
  };

  auto makeDot = [&](StringRef name, int64_t operandIndex) -> int {
    EncodingEntry entry;
    entry.name = name.str();
    entry.encodingAttr = DotOperandEncodingAttr::get(ctx, operandIndex, mmaAttr,
                                                     /*kWidth=*/1);
    int id = static_cast<int>(plan.encodings.size());
    plan.encodings.push_back(std::move(entry));
    return id;
  };

  auto makeAcc = [&]() -> int {
    EncodingEntry entry;
    entry.name = "acc";
    entry.encodingAttr = AccumulatorEncodingAttr::get(
        ctx, mmaAttr, ArrayRef<int64_t>(plan.fragmentAcc.logicalShape),
        ArrayRef<int64_t>(plan.fragmentAcc.instructionShape),
        ArrayRef<int64_t>(plan.fragmentAcc.repeatShape),
        ArrayRef<int64_t>(plan.fragmentAcc.registerOrder));
    int id = static_cast<int>(plan.encodings.size());
    plan.encodings.push_back(std::move(entry));
    return id;
  };

  plan.aGlobal = makeBlockedGlobal("a_global", aLayout->order);
  plan.bGlobal = makeBlockedGlobal("b_global", bLayout->order);
  plan.aShared = makeShared("a_shared", aSharedHeuristic);
  plan.bShared = makeShared("b_shared", bSharedHeuristic);
  plan.aDot = makeDot("a_dot", 0);
  plan.bDot = makeDot("b_dot", 1);
  plan.acc = makeAcc();
  plan.cStore = makeBlockedGlobal("c_store", cLayout->order);
  return plan;
}

} // namespace

FailureOr<EncodingPlan> mlir::tb::deriveEncodingPlan(
    const KernelConfig &config, const TargetInfo &target,
    const MatmulSemantics &semantics, const ProgramMappingPlan &programMapping,
    Operation *op) {
  return deriveEncodingPlanDirect(config, target, semantics, programMapping, op);
}

DictionaryAttr mlir::tb::buildEncodingPlanAttr(Builder &builder,
                                               const EncodingPlan &plan) {
  NamedAttrList attrs;
  attrs.set("contract_model", builder.getStringAttr(plan.contractModel));
  attrs.set("a_shared_spec", buildSharedSpecAttr(builder, plan.aSharedSpec));
  attrs.set("b_shared_spec", buildSharedSpecAttr(builder, plan.bSharedSpec));
  attrs.set("fragment_a", buildFragmentAttr(builder, plan.fragmentA));
  attrs.set("fragment_b", buildFragmentAttr(builder, plan.fragmentB));
  attrs.set("fragment_acc", buildFragmentAttr(builder, plan.fragmentAcc));
  SmallVector<Attribute> entries;
  entries.reserve(plan.encodings.size());
  for (const EncodingEntry &entry : plan.encodings)
    entries.push_back(buildEncodingEntryAttr(builder, entry));
  attrs.set("encodings", builder.getArrayAttr(entries));
  attrs.set("a_global", builder.getI64IntegerAttr(plan.aGlobal));
  attrs.set("b_global", builder.getI64IntegerAttr(plan.bGlobal));
  attrs.set("a_shared", builder.getI64IntegerAttr(plan.aShared));
  attrs.set("b_shared", builder.getI64IntegerAttr(plan.bShared));
  attrs.set("a_dot", builder.getI64IntegerAttr(plan.aDot));
  attrs.set("b_dot", builder.getI64IntegerAttr(plan.bDot));
  attrs.set("acc", builder.getI64IntegerAttr(plan.acc));
  attrs.set("c_store", builder.getI64IntegerAttr(plan.cStore));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<EncodingPlan> mlir::tb::parseEncodingPlanAttr(Operation *op) {
  auto root = dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.encoding_plan"));
  if (!root) {
    op->emitError() << "missing `tb.encoding_plan` attribute";
    return failure();
  }

  EncodingPlan plan;
  auto contractModel = readStringField(root, "contract_model", op);
  auto aSharedSpecAttr = dyn_cast_or_null<DictionaryAttr>(root.get("a_shared_spec"));
  auto bSharedSpecAttr = dyn_cast_or_null<DictionaryAttr>(root.get("b_shared_spec"));
  auto fragmentA = dyn_cast_or_null<DictionaryAttr>(root.get("fragment_a"));
  auto fragmentB = dyn_cast_or_null<DictionaryAttr>(root.get("fragment_b"));
  auto fragmentAcc = dyn_cast_or_null<DictionaryAttr>(root.get("fragment_acc"));
  auto encodings = readArrayField(root, "encodings", op);
  auto aGlobal = readI64Field(root, "a_global", op);
  auto bGlobal = readI64Field(root, "b_global", op);
  auto aShared = readI64Field(root, "a_shared", op);
  auto bShared = readI64Field(root, "b_shared", op);
  auto aDot = readI64Field(root, "a_dot", op);
  auto bDot = readI64Field(root, "b_dot", op);
  auto acc = readI64Field(root, "acc", op);
  auto cStore = readI64Field(root, "c_store", op);
  if (failed(contractModel) || !aSharedSpecAttr ||
      !bSharedSpecAttr || !fragmentA || !fragmentB || !fragmentAcc ||
      failed(encodings) || failed(aGlobal) || failed(bGlobal) ||
      failed(aShared) || failed(bShared) || failed(aDot) || failed(bDot) ||
      failed(acc) || failed(cStore)) {
    op->emitError() << "malformed `tb.encoding_plan` attribute";
    return failure();
  }
  auto parsedASharedSpec = parseSharedSpecAttr(aSharedSpecAttr, op);
  auto parsedBSharedSpec = parseSharedSpecAttr(bSharedSpecAttr, op);
  auto parsedFragmentA = parseFragmentAttr(fragmentA, op);
  auto parsedFragmentB = parseFragmentAttr(fragmentB, op);
  auto parsedFragmentAcc = parseFragmentAttr(fragmentAcc, op);
  if (failed(parsedASharedSpec) || failed(parsedBSharedSpec) ||
      failed(parsedFragmentA) || failed(parsedFragmentB) ||
      failed(parsedFragmentAcc)) {
    return failure();
  }

  plan.contractModel = contractModel->str();
  plan.aSharedSpec = std::move(*parsedASharedSpec);
  plan.bSharedSpec = std::move(*parsedBSharedSpec);
  plan.fragmentA = std::move(*parsedFragmentA);
  plan.fragmentB = std::move(*parsedFragmentB);
  plan.fragmentAcc = std::move(*parsedFragmentAcc);
  if (failed(validateSharedSpec(plan.aSharedSpec, op, "a_shared")) ||
      failed(validateSharedSpec(plan.bSharedSpec, op, "b_shared")) ||
      failed(validateFragmentSpec(plan.fragmentA, op)) ||
      failed(validateFragmentSpec(plan.fragmentB, op)))
    return failure();
  plan.encodings.reserve(encodings->size());
  for (Attribute element : *encodings) {
    auto dict = dyn_cast<DictionaryAttr>(element);
    if (!dict) {
      op->emitError() << "`encodings` must contain dictionary entries";
      return failure();
    }
    auto entry = parseEncodingEntryAttr(dict, op);
    if (failed(entry))
      return failure();
    plan.encodings.push_back(std::move(*entry));
  }
  plan.aGlobal = static_cast<int>(*aGlobal);
  plan.bGlobal = static_cast<int>(*bGlobal);
  plan.aShared = static_cast<int>(*aShared);
  plan.bShared = static_cast<int>(*bShared);
  plan.aDot = static_cast<int>(*aDot);
  plan.bDot = static_cast<int>(*bDot);
  plan.acc = static_cast<int>(*acc);
  plan.cStore = static_cast<int>(*cStore);
  return plan;
}

FailureOr<Attribute> mlir::tb::getEncodingAttr(const EncodingPlan &plan,
                                               int encodingId, Operation *op,
                                               StringRef role) {
  return getEncodingAttrImpl(plan, encodingId, op, role);
}

FailureOr<BlockedEncodingAttr>
mlir::tb::getBlockedEncodingAttr(const EncodingPlan &plan, int encodingId,
                                 Operation *op, StringRef role) {
  return getTypedEncodingAttr<BlockedEncodingAttr>(plan, encodingId, op, role,
                                                   "a blocked encoding");
}

FailureOr<SharedEncodingAttr>
mlir::tb::getSharedEncodingAttr(const EncodingPlan &plan, int encodingId,
                                Operation *op, StringRef role) {
  return getTypedEncodingAttr<SharedEncodingAttr>(plan, encodingId, op, role,
                                                  "a shared encoding");
}

FailureOr<DotOperandEncodingAttr>
mlir::tb::getDotOperandEncodingAttr(const EncodingPlan &plan, int encodingId,
                                    Operation *op, StringRef role) {
  return getTypedEncodingAttr<DotOperandEncodingAttr>(
      plan, encodingId, op, role, "a dot operand encoding");
}

FailureOr<AccumulatorEncodingAttr>
mlir::tb::getAccumulatorEncodingAttr(const EncodingPlan &plan, int encodingId,
                                     Operation *op, StringRef role) {
  return getTypedEncodingAttr<AccumulatorEncodingAttr>(
      plan, encodingId, op, role, "an accumulator encoding");
}

FailureOr<const SharedEncodingSpec *>
mlir::tb::getSharedEncodingSpec(const EncodingPlan &plan, int encodingId,
                                Operation *op, StringRef role) {
  if (encodingId == plan.aShared)
    return &plan.aSharedSpec;
  if (encodingId == plan.bShared)
    return &plan.bSharedSpec;
  op->emitError() << "encoding `" << role
                  << "` does not reference a shared encoding summary";
  return failure();
}

FailureOr<llvm::SmallVector<int64_t, 2>>
mlir::tb::getMmaWarpsPerCTA(const EncodingPlan &plan, Operation *op) {
  auto accAttr = getAccumulatorEncodingAttr(plan, plan.acc, op, "acc");
  if (failed(accAttr))
    return failure();
  ArrayRef<int64_t> warpsPerCTA = accAttr->getParent().getWarpsPerCTA();
  if (warpsPerCTA.size() != 2 || llvm::any_of(warpsPerCTA, [](int64_t value) {
        return value <= 0;
      })) {
    op->emitError() << "accumulator mma encoding must carry positive rank-2 "
                       "warps_per_cta";
    return failure();
  }
  return llvm::SmallVector<int64_t, 2>(warpsPerCTA.begin(), warpsPerCTA.end());
}

FailureOr<llvm::SmallVector<int64_t, 2>>
mlir::tb::getAccumulatorTileShape(const EncodingPlan &plan, Operation *op) {
  auto accAttr = getAccumulatorEncodingAttr(plan, plan.acc, op, "acc");
  if (failed(accAttr))
    return failure();
  ArrayRef<int64_t> logicalShape = accAttr->getLogicalShape();
  if (logicalShape.size() != 2 ||
      llvm::any_of(logicalShape, [](int64_t value) { return value <= 0; })) {
    op->emitError() << "accumulator encoding must carry positive rank-2 "
                       "logical_shape";
    return failure();
  }
  return llvm::SmallVector<int64_t, 2>(logicalShape.begin(),
                                       logicalShape.end());
}
