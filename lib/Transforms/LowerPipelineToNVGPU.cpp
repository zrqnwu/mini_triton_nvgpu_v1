#include "tb/Analysis/KernelContract.h"
#include "tb/IR/TBOps.h"
#include "tb/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::tb;

namespace mlir::tb {
#define GEN_PASS_DEF_TBLOWERPIPELINETONVGPU
#include "tb/Transforms/Passes.h.inc"
} // namespace mlir::tb

namespace {

struct SharedPhysicalLayoutContract {
  int64_t minorDim = 1;
  int64_t minorExtent = 0;
  int64_t perPhase = 1;
  int64_t maxPhase = 1;
  int64_t swizzlingByteWidth = 0;
  int64_t elementBytes = 0;
};

struct LoweringContract {
  int64_t numWarps = 0;
  int64_t threadsPerWarp = 0;
  int64_t numKGroups = 0;
  int64_t workspaceTotalBytes = 0;
  ProgramMappingKind mappingKind = ProgramMappingKind::Tile;
  int64_t problemTilesM = 0;
  int64_t problemTilesN = 0;
  int64_t problemTilesK = 0;
  int64_t groupTileSpanM = 1;
  int64_t groupTileSpanN = 1;
  int64_t programsPerLaunchGroup = 1;
  int64_t totalPrograms = 1;
  int64_t splitK = 1;
  int64_t splitKGroupsPerTile = 0;
  int64_t splitKRemainderTiles = 0;
  ReductionMode reductionMode = ReductionMode::None;
  ReductionPlanKind reductionKind = ReductionPlanKind::None;
  bool persistentEnabled = false;
  int64_t persistentResidentPrograms = 1;
  int64_t accTilesM = 0;
  int64_t accTilesN = 0;
  int64_t bGroupCount = 0;
  int64_t bGroupTileSpan = 1;
  int64_t aSharedBacking = -1;
  int64_t bSharedBacking = -1;
  int64_t aWorkspaceOffsetBytes = 0;
  int64_t bWorkspaceOffsetBytes = 0;
  int64_t aSharedRows = 0;
  int64_t aSharedCols = 0;
  int64_t bSharedRows = 0;
  int64_t bSharedCols = 0;
  int64_t aSharedViewRows = 0;
  int64_t aSharedViewCols = 0;
  int64_t bSharedViewRows = 0;
  int64_t bSharedViewCols = 0;
  int64_t aAsyncCopyElems = 0;
  int64_t bAsyncCopyElems = 0;
  int64_t aFragRows = 0;
  int64_t aFragCols = 0;
  int64_t bSubFragRows = 0;
  int64_t bSubFragCols = 0;
  int64_t bGroupFragRows = 0;
  int64_t bGroupFragCols = 0;
  TargetLandingKind cLandingKind = TargetLandingKind::None;
  EpilogueReorderKind cReorderKind = EpilogueReorderKind::None;
  int64_t cInitWorkspaceOffsetBytes = 0;
  int64_t cStoreWorkspaceOffsetBytes = 0;
  int64_t cDirectPackRows = 0;
  int64_t cDirectPackCols = 0;
  int64_t cSharedTileRows = 0;
  int64_t cSharedTileCols = 0;
  int64_t cSharedPackSlots = 0;
  int64_t cGlobalVectorWidth = 0;
  int64_t cInitSharedStoreVectorWidth = 0;
  int64_t cInitSharedLoadVectorWidth = 0;
  int64_t cStoreSharedStoreVectorWidth = 0;
  int64_t cStoreSharedLoadVectorWidth = 0;
  int64_t cWorkspaceBarrierCount = 0;
  bool cLoadInitFromGlobal = true;
  bool cStoreViaAtomicAdd = false;
  bool cUseSharedPackForInit = false;
  bool cUseSharedPackForStore = false;
  bool cBarrierAfterInit = false;
  bool cBarrierBeforeStore = false;
  SharedPhysicalLayoutContract aSharedLayout;
  SharedPhysicalLayoutContract bSharedLayout;
  std::string cRequiredSyncKind;
};

struct ProgramWorkAssignment {
  Value tileM;
  Value tileN;
  Value splitKPart;
};

struct Position {
  int64_t stage = -1;
  int64_t cluster = -1;
  int64_t order = -1;
};

struct ExplicitPipelineCluster {
  ExpandedClusterKind kind = ExpandedClusterKind::AsyncIssue;
  int64_t ordinal = -1;
  int64_t stage = -1;
  int64_t cluster = -1;
  int64_t kGroup = -1;
  SmallVector<int64_t, 32> opIds;
  SmallVector<int64_t, 8> waitGroupIds;
  bool needsBarrier = false;
  std::string reason;
};

static bool isEarlierOrEqual(Position lhs, Position rhs) {
  if (lhs.stage != rhs.stage)
    return lhs.stage < rhs.stage;
  if (lhs.cluster != rhs.cluster)
    return lhs.cluster < rhs.cluster;
  return lhs.order <= rhs.order;
}

static bool isStrictlyEarlier(Position lhs, Position rhs) {
  if (lhs.stage != rhs.stage)
    return lhs.stage < rhs.stage;
  if (lhs.cluster != rhs.cluster)
    return lhs.cluster < rhs.cluster;
  return lhs.order < rhs.order;
}

struct SharedViewInfo {
  int64_t backing = -1;
  int64_t rowBase = 0;
  int64_t colBase = 0;
  int64_t rows = 0;
  int64_t cols = 0;
};

static FailureOr<SharedPhysicalLayoutContract>
buildSharedPhysicalLayoutContract(SharedEncodingAttr attr, int64_t rows,
                                  int64_t cols, int64_t elementBytes,
                                  Operation *op, StringRef role) {
  if (attr.getOrder().size() != 2) {
    op->emitError() << role << " shared encoding must stay rank-2";
    return failure();
  }
  int64_t minorDim = attr.getOrder().front();
  if (minorDim < 0 || minorDim > 1) {
    op->emitError() << role
                    << " shared encoding minor axis must be 0 or 1";
    return failure();
  }

  SharedPhysicalLayoutContract layout;
  layout.minorDim = minorDim;
  layout.minorExtent = minorDim == 0 ? rows : cols;
  layout.perPhase = std::max<int64_t>(attr.getPerPhase(), 1);
  layout.maxPhase = std::max<int64_t>(attr.getMaxPhase(), 1);
  layout.swizzlingByteWidth = attr.getSwizzlingByteWidth();
  layout.elementBytes = elementBytes;
  if (layout.minorExtent <= 0 || layout.elementBytes <= 0) {
    op->emitError() << role
                    << " shared layout contract must carry positive extent and "
                       "element width";
    return failure();
  }
  if (layout.swizzlingByteWidth > 0) {
    if (layout.swizzlingByteWidth % layout.elementBytes != 0) {
      op->emitError() << role << " swizzling byte width must align to the "
                                 "shared element width";
      return failure();
    }
    int64_t spanElems = layout.swizzlingByteWidth / layout.elementBytes;
    if (spanElems <= 0 || layout.minorExtent % spanElems != 0) {
      op->emitError() << role << " swizzle span must divide the shared minor "
                                 "extent exactly";
      return failure();
    }
  }
  return layout;
}

static Value createIndexConstant(OpBuilder &builder, Location loc,
                                 int64_t value) {
  return arith::ConstantIndexOp::create(builder, loc, value);
}

static Value createIndexAdd(OpBuilder &builder, Location loc, Value lhs,
                            Value rhs) {
  return arith::AddIOp::create(builder, loc, lhs, rhs);
}

static Value createIndexMul(OpBuilder &builder, Location loc, Value lhs,
                            Value rhs) {
  return arith::MulIOp::create(builder, loc, lhs, rhs);
}

static Value createIndexSub(OpBuilder &builder, Location loc, Value lhs,
                            Value rhs) {
  return arith::SubIOp::create(builder, loc, lhs, rhs);
}

static Value createIndexDivU(OpBuilder &builder, Location loc, Value lhs,
                             Value rhs) {
  return arith::DivUIOp::create(builder, loc, lhs, rhs);
}

static Value createIndexRemU(OpBuilder &builder, Location loc, Value lhs,
                             Value rhs) {
  return arith::RemUIOp::create(builder, loc, lhs, rhs);
}

static Value createIndexAddConst(OpBuilder &builder, Location loc, Value base,
                                 int64_t offset) {
  if (offset == 0)
    return base;
  return createIndexAdd(builder, loc, base, createIndexConstant(builder, loc, offset));
}

static Value createIndexCmpULT(OpBuilder &builder, Location loc, Value lhs,
                               Value rhs) {
  return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult, lhs, rhs);
}

static Value createIndexMinU(OpBuilder &builder, Location loc, Value lhs,
                             Value rhs) {
  Value lhsLtRhs = createIndexCmpULT(builder, loc, lhs, rhs);
  return arith::SelectOp::create(builder, loc, lhsLtRhs, lhs, rhs);
}

static FailureOr<Value> createTypedMemRefViewFromByteBuffer(
    OpBuilder &builder, Location loc, Value byteBuffer, MemRefType viewType,
    Value byteShift, Operation *sourceOp) {
  auto byteBufferType = dyn_cast<MemRefType>(byteBuffer.getType());
  if (!byteBufferType || byteBufferType.getRank() != 1 ||
      !byteBufferType.getElementType().isSignlessInteger(8) ||
      !byteBufferType.getLayout().isIdentity()) {
    sourceOp->emitError()
        << "typed shared alias expects a rank-1 identity-layout i8 byte buffer";
    return failure();
  }
  SmallVector<Value> dynamicSizes;
  for (int64_t size : viewType.getShape()) {
    if (ShapedType::isDynamic(size)) {
      sourceOp->emitError()
          << "typed shared alias currently expects static result shapes";
      return failure();
    }
  }
  return memref::ViewOp::create(builder, loc, viewType, byteBuffer, byteShift,
                                dynamicSizes)
      .getResult();
}

static FailureOr<ProgramWorkAssignment>
buildProgramWorkAssignment(OpBuilder &builder, Location loc, Value programId,
                           const LoweringContract &contract, Operation *op) {
  Value tilesM = createIndexConstant(builder, loc, contract.problemTilesM);
  switch (contract.mappingKind) {
  case ProgramMappingKind::Tile:
  case ProgramMappingKind::GroupedTile:
  case ProgramMappingKind::PersistentTile: {
    Value groupSpanM =
        createIndexConstant(builder, loc, contract.groupTileSpanM);
    Value programsPerLaunchGroup =
        createIndexConstant(builder, loc, contract.programsPerLaunchGroup);
    Value groupId =
        createIndexDivU(builder, loc, programId, programsPerLaunchGroup);
    Value firstTileM = createIndexMul(builder, loc, groupId, groupSpanM);
    Value remainingTilesM = createIndexSub(builder, loc, tilesM, firstTileM);
    Value groupSizeM =
        createIndexMinU(builder, loc, groupSpanM, remainingTilesM);
    Value inGroup =
        createIndexRemU(builder, loc, programId, programsPerLaunchGroup);
    Value tileM =
        createIndexAdd(builder, loc, firstTileM,
                       createIndexRemU(builder, loc, inGroup, groupSizeM));
    Value tileN = createIndexDivU(builder, loc, inGroup, groupSizeM);
    return ProgramWorkAssignment{
        tileM, tileN, createIndexConstant(builder, loc, 0)};
  }
  case ProgramMappingKind::SplitK: {
    Value splitK = createIndexConstant(builder, loc, contract.splitK);
    Value programsPerLaunchGroup =
        createIndexConstant(builder, loc, contract.programsPerLaunchGroup);
    Value groupId =
        createIndexDivU(builder, loc, programId, programsPerLaunchGroup);
    Value inGroup =
        createIndexRemU(builder, loc, programId, programsPerLaunchGroup);
    Value tileN = createIndexDivU(builder, loc, inGroup, splitK);
    Value splitKPart = createIndexRemU(builder, loc, inGroup, splitK);
    return ProgramWorkAssignment{groupId, tileN, splitKPart};
  }
  default:
    op->emitError() << "lowering currently only supports tile/grouped_tile/"
                       "split_k/persistent program mapping";
    return failure();
  }
}

static Value buildSplitKGroupActivePredicate(OpBuilder &builder, Location loc,
                                             Value splitKPart, int64_t kGroup,
                                             const LoweringContract &contract) {
  if (contract.splitK <= 1 || contract.reductionKind != ReductionPlanKind::SplitKAtomic)
    return Value();

  int64_t baseTiles = contract.problemTilesK / contract.splitK;
  int64_t extraTiles = contract.problemTilesK % contract.splitK;
  Value baseTilesValue = createIndexConstant(builder, loc, baseTiles);
  Value extraTilesValue = createIndexConstant(builder, loc, extraTiles);
  Value splitLtExtra =
      createIndexCmpULT(builder, loc, splitKPart, extraTilesValue);
  Value carriedExtra = createIndexMinU(builder, loc, splitKPart, extraTilesValue);
  Value tileStart = createIndexAdd(
      builder, loc, createIndexMul(builder, loc, splitKPart, baseTilesValue),
      carriedExtra);
  Value tileCount = createIndexAddConst(builder, loc, baseTilesValue, 0);
  tileCount = arith::SelectOp::create(
      builder, loc, splitLtExtra, createIndexAddConst(builder, loc, tileCount, 1),
      tileCount);

  Value groupsPerTile =
      createIndexConstant(builder, loc, contract.splitKGroupsPerTile);
  Value groupStart = createIndexMul(builder, loc, tileStart, groupsPerTile);
  Value groupEnd = createIndexMul(
      builder, loc, createIndexAdd(builder, loc, tileStart, tileCount),
      groupsPerTile);
  Value kGroupValue = createIndexConstant(builder, loc, kGroup);
  Value geStart = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::uge,
                                        kGroupValue, groupStart);
  Value ltEnd = createIndexCmpULT(builder, loc, kGroupValue, groupEnd);
  return arith::AndIOp::create(builder, loc, geStart, ltEnd);
}

static VectorType getVecType(Type elementType, ArrayRef<int64_t> shape) {
  return VectorType::get(shape, elementType);
}

static FailureOr<int64_t> getElementByteWidth(Type elementType, Operation *op,
                                              StringRef role) {
  if (!elementType.isIntOrFloat()) {
    op->emitError() << role << " must use integer or float element types";
    return failure();
  }
  int64_t bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth <= 0 || bitWidth % 8 != 0) {
    op->emitError() << role << " must have byte-addressable element types";
    return failure();
  }
  return bitWidth / 8;
}

static FailureOr<SmallVector<int64_t, 2>>
getRegisterShape(ArrayRef<int64_t> shape, Operation *op, StringRef role) {
  if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0) {
    op->emitError() << role << " must carry a positive rank-2 register shape";
    return failure();
  }
  return SmallVector<int64_t, 2>{shape[0], shape[1]};
}

static Value buildZeroVector(OpBuilder &builder, Location loc,
                             VectorType vectorType) {
  Attribute zeroAttr;
  Type elementType = vectorType.getElementType();
  if (elementType.isF16())
    zeroAttr = builder.getF16FloatAttr(0.0);
  else if (elementType.isF32())
    zeroAttr = builder.getF32FloatAttr(0.0);
  else
    llvm_unreachable("unexpected vector element type");
  return arith::ConstantOp::create(
      builder, loc, vectorType, DenseElementsAttr::get(vectorType, zeroAttr));
}

static SmallVector<int64_t, 8> copyI64Array(ArrayRef<int64_t> values) {
  return SmallVector<int64_t, 8>(values.begin(), values.end());
}

static ExplicitPipelineCluster buildExplicitCluster(AsyncIssueClusterOp op) {
  ExplicitPipelineCluster cluster;
  cluster.kind = ExpandedClusterKind::AsyncIssue;
  cluster.ordinal = op.getOrdinal();
  cluster.stage = op.getStage();
  cluster.cluster = op.getCluster();
  cluster.kGroup = op.getKGroup();
  cluster.opIds = copyI64Array(op.getOpIds());
  cluster.reason = op.getReason().str();
  return cluster;
}

static ExplicitPipelineCluster buildExplicitCluster(ConsumerWaitClusterOp op) {
  ExplicitPipelineCluster cluster;
  cluster.kind = ExpandedClusterKind::ConsumerWait;
  cluster.ordinal = op.getOrdinal();
  cluster.stage = op.getStage();
  cluster.cluster = op.getCluster();
  cluster.kGroup = op.getKGroup();
  cluster.opIds = copyI64Array(op.getOpIds());
  cluster.waitGroupIds = copyI64Array(op.getWaitGroupIds());
  cluster.needsBarrier = op.getNeedsBarrier();
  cluster.reason = op.getReason().str();
  return cluster;
}

static ExplicitPipelineCluster buildExplicitCluster(MmaComputeClusterOp op) {
  ExplicitPipelineCluster cluster;
  cluster.kind = ExpandedClusterKind::MmaCompute;
  cluster.ordinal = op.getOrdinal();
  cluster.stage = op.getStage();
  cluster.cluster = op.getCluster();
  cluster.kGroup = op.getKGroup();
  cluster.opIds = copyI64Array(op.getOpIds());
  cluster.reason = op.getReason().str();
  return cluster;
}

static FailureOr<SmallVector<ExplicitPipelineCluster, 8>>
collectExplicitPipelineClusters(PipelineMainlineOp op) {
  if (op.getBody().empty() || !llvm::hasSingleElement(op.getBody())) {
    op.emitError()
        << "pipeline_mainline must contain a single explicit pipeline body";
    return failure();
  }

  SmallVector<ExplicitPipelineCluster, 8> clusters;
  auto appendCluster = [&](Operation &nestedOp) -> LogicalResult {
    if (auto async = dyn_cast<AsyncIssueClusterOp>(nestedOp)) {
      clusters.push_back(buildExplicitCluster(async));
      return success();
    }
    if (auto wait = dyn_cast<ConsumerWaitClusterOp>(nestedOp)) {
      clusters.push_back(buildExplicitCluster(wait));
      return success();
    }
    if (auto mma = dyn_cast<MmaComputeClusterOp>(nestedOp)) {
      clusters.push_back(buildExplicitCluster(mma));
      return success();
    }
    op.emitError() << "unsupported op in explicit pipeline body: "
                   << nestedOp.getName().getStringRef();
    return failure();
  };

  for (Operation &nestedOp : op.getBody().front()) {
    if (failed(appendCluster(nestedOp)))
      return failure();
  }

  if (clusters.empty()) {
    op.emitError() << "pipeline_mainline must contain explicit pipeline clusters";
    return failure();
  }
  return clusters;
}

static int64_t findIterationCoord(ArrayRef<PipelineOp::IterationCoord> coords,
                                  StringRef axis) {
  auto it = llvm::find_if(coords, [&](const PipelineOp::IterationCoord &coord) {
    return coord.axis == axis;
  });
  return it == coords.end() ? -1 : it->value;
}

static FailureOr<std::pair<const DirectGlobalVectorPlan *,
                           const DirectGlobalVectorPlan *>>
getDirectEpiloguePlans(const EpiloguePlan &epilogue, Operation *op) {
  if (epilogue.initMode != AccumulatorInitMode::DirectGlobalVector ||
      epilogue.storeMode != AccumulatorStoreMode::DirectGlobalVector) {
    op->emitError() << "pipeline lowering expects direct_global_vector C path";
    return failure();
  }
  auto *init = std::get_if<DirectGlobalVectorPlan>(&epilogue.init);
  auto *store = std::get_if<DirectGlobalVectorPlan>(&epilogue.store);
  if (!init || !store) {
    op->emitError() << "missing direct global vector epilogue payload";
    return failure();
  }
  return std::make_pair(init, store);
}

static FailureOr<const TargetLandingPlan *>
getTargetLandingPlan(const EpiloguePlan &epilogue, Operation *op) {
  if (epilogue.targetLanding.kind !=
          TargetLandingKind::SharedPackThenGlobalVector &&
      epilogue.targetLanding.kind !=
          TargetLandingKind::RegisterPackGlobalVector) {
    op->emitError() << "pipeline lowering expects explicit "
                       "C target landing";
    return failure();
  }
  return &epilogue.targetLanding;
}

static LogicalResult validateDirectGlobalPlan(
    const DirectGlobalVectorPlan &plan, int64_t accumulatorFragments,
    int64_t fragmentVectorWidth, const LaneAccessPattern &fragmentLaneAccess,
    Operation *op, StringRef role) {
  if (plan.packs.empty() || plan.vectorWidth <= 0) {
    op->emitError() << role << " direct-global plan must carry packs";
    return failure();
  }
  if (plan.laneAccess.laneRowGroupSize <= 0 ||
      plan.laneAccess.laneColGroupSize <= 0 ||
      plan.laneAccess.laneColStride <= 0 ||
      plan.laneAccess.rowOffsets.empty()) {
    op->emitError() << role << " direct-global plan must carry explicit lane "
                       "access metadata";
    return failure();
  }
  if (plan.laneAccess.rowOffsets != fragmentLaneAccess.rowOffsets ||
      plan.laneAccess.laneRowGroupSize !=
          fragmentLaneAccess.laneRowGroupSize ||
      plan.laneAccess.laneColGroupSize !=
          fragmentLaneAccess.laneColGroupSize) {
    op->emitError() << role
                    << " direct-global plan must stay aligned with the "
                       "accumulator fragment row ownership";
    return failure();
  }
  if (fragmentVectorWidth <= 0 || plan.vectorWidth % fragmentVectorWidth != 0) {
    op->emitError() << role
                    << " direct-global plan vector width must be a multiple of "
                       "the accumulator fragment vector width";
    return failure();
  }

  int64_t fragmentsPerPack = plan.vectorWidth / fragmentVectorWidth;
  SmallVector<bool, 32> seenFragments(accumulatorFragments, false);
  for (const DirectGlobalVectorPlan::Pack &pack : plan.packs) {
    if (pack.rows <= 0 || pack.cols <= 0 || pack.vectorWidth != plan.vectorWidth) {
      op->emitError() << role
                      << " direct-global pack geometry is malformed";
      return failure();
    }
    if (static_cast<int64_t>(pack.fragmentIds.size()) != fragmentsPerPack) {
      op->emitError() << role
                      << " direct-global pack fragment count disagrees with its "
                         "vector width";
      return failure();
    }
    for (int64_t fragmentId : pack.fragmentIds) {
      if (fragmentId < 0 || fragmentId >= accumulatorFragments) {
        op->emitError() << role
                        << " direct-global pack references an invalid "
                           "accumulator fragment id";
        return failure();
      }
      if (seenFragments[fragmentId]) {
        op->emitError() << role
                        << " direct-global plan duplicates accumulator "
                           "fragment coverage";
        return failure();
      }
      seenFragments[fragmentId] = true;
    }
  }

  if (llvm::any_of(seenFragments, [](bool covered) { return !covered; })) {
    op->emitError() << role
                    << " direct-global plan must cover every accumulator "
                       "fragment exactly once";
    return failure();
  }
  return success();
}

static FailureOr<SharedViewInfo> resolveSharedView(const BufferModel &model,
                                                   int64_t viewId,
                                                   Operation *op) {
  auto viewIt = llvm::find_if(model.views, [&](const BufferView &view) {
    return view.id == viewId;
  });
  if (viewIt == model.views.end()) {
    op->emitError() << "missing shared view " << viewId;
    return failure();
  }
  auto backingIt = llvm::find_if(model.backings, [&](const BufferBacking &backing) {
    return backing.id == viewIt->backing;
  });
  auto backingMemory =
      backingIt == model.backings.end()
          ? FailureOr<MemorySpace>(failure())
          : getMemDescMemorySpace(backingIt->descType, op, "shared_view");
  if (backingIt == model.backings.end() || failed(backingMemory) ||
      *backingMemory != MemorySpace::Shared) {
    op->emitError() << "view " << viewId
                    << " does not reference a shared backing";
    return failure();
  }
  auto descType = getMemDescType(backingIt->descType, op, backingIt->debugName);
  if (failed(descType))
    return failure();
  ArrayRef<int64_t> allocShape = descType->getAllocShape();

  SharedViewInfo info;
  info.backing = backingIt->id;
  info.rows = viewIt->shape.size() >= 1 ? viewIt->shape[0] : 0;
  info.cols = viewIt->shape.size() >= 2 ? viewIt->shape[1] : 0;
  if (backingIt->stageIndexed) {
    if (allocShape.size() != 3 || viewIt->offsets.size() != 3) {
      op->emitError() << "stage-indexed shared view " << viewId
                      << " must carry rank-3 alloc/offset metadata";
      return failure();
    }
    info.rowBase = viewIt->offsets[0] * allocShape[1] + viewIt->offsets[1];
    info.colBase = viewIt->offsets[2];
  } else {
    if (!viewIt->offsets.empty())
      info.rowBase = viewIt->offsets[0];
    if (viewIt->offsets.size() > 1)
      info.colBase = viewIt->offsets[1];
  }
  return info;
}

static FailureOr<SmallVector<int64_t, 2>>
getFlattenedSharedAllocShape(const BufferModel &model, int64_t backingId,
                             Operation *op) {
  auto backingIt = llvm::find_if(model.backings, [&](const BufferBacking &backing) {
    return backing.id == backingId;
  });
  auto backingMemory =
      backingIt == model.backings.end()
          ? FailureOr<MemorySpace>(failure())
          : getMemDescMemorySpace(backingIt->descType, op, "shared_backing");
  if (backingIt == model.backings.end() || failed(backingMemory) ||
      *backingMemory != MemorySpace::Shared) {
    op->emitError() << "missing shared backing " << backingId
                    << " for workgroup allocation";
    return failure();
  }
  auto descType = getMemDescType(backingIt->descType, op, backingIt->debugName);
  if (failed(descType))
    return failure();
  ArrayRef<int64_t> allocShape = descType->getAllocShape();
  if (backingIt->stageIndexed) {
    if (allocShape.size() != 3) {
      op->emitError() << "stage-indexed shared backing " << backingId
                      << " must have rank-3 alloc shape";
      return failure();
    }
    return SmallVector<int64_t, 2>{
        allocShape[0] * allocShape[1], allocShape[2]};
  }
  if (allocShape.size() != 2) {
    op->emitError() << "shared backing " << backingId
                    << " must have rank-2 alloc shape";
    return failure();
  }
  return SmallVector<int64_t, 2>{allocShape[0], allocShape[1]};
}

static FailureOr<LoweringContract>
validateLoweringContract(const KernelConfig &spec, const TargetInfo &target,
                        const MatmulSemantics &semantics,
                        const ProgramMappingPlan &programMapping,
                        const ReductionPlan &reduction,
                        const PersistentWorkPlan &persistentWork,
                        const PipelineReady &pipelineReady,
                        const EncodingPlan &encodings,
                        const TransportPlan &transport,
                        const MatmulRewritePlan &rewrite,
                        const AccumulatorPlan &accumulator,
                        const EpiloguePlan &epilogue,
                        const EpilogueReorderPlan &epilogueReorder,
                        const SharedWorkspacePlan &sharedWorkspace,
                        const WarpDecompositionPlan &warpDecomposition,
                        const ResourceClosurePlan &resourceClosure,
                        const BufferModel &model,
                        ArrayRef<ExplicitPipelineCluster> explicitClusters,
                        const AsyncPlan &asyncPlan, Operation *op) {
  auto directPlans = getDirectEpiloguePlans(epilogue, op);
  auto targetLanding = getTargetLandingPlan(epilogue, op);
  if (failed(directPlans))
    return failure();
  if (failed(targetLanding))
    return failure();
  const DirectGlobalVectorPlan *initPlan = directPlans->first;
  const DirectGlobalVectorPlan *storePlan = directPlans->second;
  const TargetLandingPlan *landing = *targetLanding;
  if (initPlan->packs.empty() || storePlan->packs.empty() ||
      initPlan->packs.size() != storePlan->packs.size()) {
    op->emitError() << "lowering expects symmetric direct C init/store packs";
    return failure();
  }
  if (accumulator.laneAccess.rowOffsets.empty() ||
      accumulator.registersPerWarp <= 0) {
    op->emitError() << "lowering expects an explicit accumulator lane mapping";
    return failure();
  }
  if (warpDecomposition.warpTile.size() != 2 ||
      warpDecomposition.warpGrid.size() != 2 ||
      warpDecomposition.warps.size() != static_cast<size_t>(spec.numWarps)) {
    op->emitError()
        << "lowering expects explicit warp decomposition coverage for every warp";
    return failure();
  }
  if (resourceClosure.chosenLandingTradeoff.empty() ||
      resourceClosure.reason.empty()) {
    op->emitError() << "lowering expects an explicit resource closure plan";
    return failure();
  }
  if (resourceClosure.workspaceTotalBytes != sharedWorkspace.totalBytes ||
      resourceClosure.peakStaticSharedBytes != sharedWorkspace.peakBytes) {
    op->emitError()
        << "lowering expects resource closure to consume the shared workspace";
    return failure();
  }
  for (const DirectGlobalVectorPlan::Pack &pack : initPlan->packs) {
    if (pack.rows != landing->directPackRows ||
        pack.cols != landing->directPackCols) {
      op->emitError() << "direct init packs must match the target landing "
                         "direct-pack geometry";
      return failure();
    }
  }
  for (const DirectGlobalVectorPlan::Pack &pack : storePlan->packs) {
    if (pack.rows != landing->directPackRows ||
        pack.cols != landing->directPackCols) {
      op->emitError() << "direct store packs must match the target landing "
                         "direct-pack geometry";
      return failure();
    }
  }
  if (semantics.problemM != spec.problemM || semantics.problemN != spec.problemN ||
      semantics.problemK != spec.problemK || semantics.tileM != spec.blockM ||
      semantics.tileN != spec.blockN || semantics.tileK != spec.blockK ||
      semantics.exactTile != spec.exactTile) {
    op->emitError() << "tb.semantic_matmul disagrees with the kernel config";
    return failure();
  }
  if (pipelineReady.asyncGroups != static_cast<int64_t>(asyncPlan.groups.size())) {
    op->emitError()
        << "pipeline_ready async-group summary disagrees with tb.async_plan";
    return failure();
  }
  if (pipelineReady.requestedStages != spec.requestedStages) {
    op->emitError() << "pipeline_ready requested_stages disagrees with kernel "
                       "config";
    return failure();
  }
  bool validTileMapping =
      programMapping.mappingKind == ProgramMappingKind::Tile &&
      programMapping.launchOrder == ProgramLaunchOrder::RowMajor &&
      programMapping.swizzleKind == ProgramSwizzleKind::None &&
      programMapping.groupM == 1 &&
      programMapping.groupTileSpanM == 1 &&
      programMapping.groupTileSpanN == programMapping.problemTilesN &&
      programMapping.programsPerLaunchGroup == programMapping.problemTilesN;
  bool validGroupedMapping =
      programMapping.mappingKind == ProgramMappingKind::GroupedTile &&
      programMapping.launchOrder == ProgramLaunchOrder::GroupedM &&
      programMapping.swizzleKind == ProgramSwizzleKind::GroupedM &&
      programMapping.groupM > 1 &&
      programMapping.groupTileSpanM == programMapping.groupM &&
      programMapping.groupTileSpanN == programMapping.problemTilesN &&
      programMapping.programsPerLaunchGroup ==
          programMapping.groupTileSpanM * programMapping.groupTileSpanN;
  bool validSplitKMapping =
      programMapping.mappingKind == ProgramMappingKind::SplitK &&
      programMapping.launchOrder == ProgramLaunchOrder::RowMajor &&
      programMapping.swizzleKind == ProgramSwizzleKind::None &&
      programMapping.groupM == 1 &&
      programMapping.groupTileSpanM == 1 &&
      programMapping.groupTileSpanN ==
          programMapping.problemTilesN * programMapping.splitK &&
      programMapping.programsPerLaunchGroup == programMapping.groupTileSpanN &&
      programMapping.programsPerTile == programMapping.splitK;
  bool validPersistentMapping =
      programMapping.mappingKind == ProgramMappingKind::PersistentTile &&
      programMapping.launchOrder == ProgramLaunchOrder::Persistent &&
      programMapping.swizzleKind == ProgramSwizzleKind::None &&
      programMapping.groupM == 1 &&
      programMapping.groupTileSpanM == 1 &&
      programMapping.groupTileSpanN == programMapping.problemTilesN &&
      programMapping.programsPerLaunchGroup == programMapping.problemTilesN &&
      programMapping.programsPerTile == 1;
  if (programMapping.splitK > 1) {
    if (!validSplitKMapping || reduction.kind != ReductionPlanKind::SplitKAtomic ||
        reduction.finalExecutor != "global_atomic_accumulate") {
      op->emitError() << "stage1 lowering currently only supports explicit "
                         "split-k parallel atomic execution";
      return failure();
    }
  }
  if (programMapping.persistent) {
    if (!validPersistentMapping || !persistentWork.enabled ||
        !persistentWork.requiresOuterSerialLoop) {
      op->emitError() << "persistent lowering requires an explicit tile-stride "
                         "work plan";
      return failure();
    }
  }
  if ((!validTileMapping && !validGroupedMapping && !validSplitKMapping &&
       !validPersistentMapping) ||
      (programMapping.mappingKind != ProgramMappingKind::SplitK &&
       programMapping.programsPerTile != 1)) {
    op->emitError() << "strict lowering only accepts tile/grouped_tile/"
                       "split_k_parallel/persistent mainlines after mapping "
                       "cleanup";
    return failure();
  }
  if (programMapping.tileM != spec.blockM || programMapping.tileN != spec.blockN ||
      programMapping.tileK != spec.blockK) {
    op->emitError() << "program mapping tile shape disagrees with kernel config";
    return failure();
  }
  if (programMapping.problemTilesM <= 0 || programMapping.problemTilesN <= 0 ||
      programMapping.problemTilesK <= 0) {
    op->emitError() << "program mapping must carry positive problem tile counts";
    return failure();
  }
  int64_t expectedPrograms = programMapping.problemTilesM *
                             programMapping.problemTilesN *
                             std::max<int64_t>(programMapping.splitK, 1);
  if (programMapping.totalPrograms != expectedPrograms) {
    op->emitError() << "program mapping total_programs must equal the explicit "
                       "program coverage";
    return failure();
  }
  auto warpTileShape = getAccumulatorTileShape(encodings, op);
  auto mmaWarpsPerCTA = getMmaWarpsPerCTA(encodings, op);
  if (failed(warpTileShape) || failed(mmaWarpsPerCTA) ||
      encodings.aSharedSpec.logicalShape.size() != 2 ||
      encodings.bSharedSpec.logicalShape.size() != 2 ||
      encodings.fragmentA.instructionShape.size() < 2 ||
      encodings.fragmentB.instructionShape.size() < 2 ||
      encodings.fragmentAcc.repeatShape.size() < 2) {
    op->emitError() << "malformed encoding plan for lowering";
    return failure();
  }
  if ((*mmaWarpsPerCTA)[0] * (*mmaWarpsPerCTA)[1] != spec.numWarps) {
    op->emitError() << "mma warps_per_cta disagrees with kernel num_warps";
    return failure();
  }
  if (encodings.fragmentA.laneRowModulus <= 0 ||
      encodings.fragmentA.laneColDivisor <= 0 ||
      encodings.fragmentA.laneColStride <= 0 ||
      encodings.fragmentB.laneRowModulus <= 0 ||
      encodings.fragmentB.laneColDivisor <= 0 ||
      encodings.fragmentB.laneColStride <= 0) {
    op->emitError() << "operand fragment lane access must be explicit in the "
                       "encoding plan";
    return failure();
  }
  auto aRegisterShape =
      getRegisterShape(encodings.fragmentA.registerShape, op, "fragmentA");
  auto bRegisterShape =
      getRegisterShape(encodings.fragmentB.registerShape, op, "fragmentB");
  if (failed(aRegisterShape) || failed(bRegisterShape))
    return failure();
  if (encodings.fragmentA.ldmatrixTileCount <= 0 ||
      encodings.fragmentB.ldmatrixTileCount <= 0) {
    op->emitError() << "operand fragments must carry positive ldmatrix tile counts";
    return failure();
  }

  auto aSharedBacking =
      findUniqueBacking(model, BufferRole::OperandA, MemorySpace::Shared, op,
                        "lowering_shared_operand_a");
  auto bSharedBacking =
      findUniqueBacking(model, BufferRole::OperandB, MemorySpace::Shared, op,
                        "lowering_shared_operand_b");
  if (failed(aSharedBacking) || failed(bSharedBacking))
    return failure();
  auto aSharedShape =
      getFlattenedSharedAllocShape(model, (*aSharedBacking)->id, op);
  auto bSharedShape =
      getFlattenedSharedAllocShape(model, (*bSharedBacking)->id, op);
  if (failed(aSharedShape) || failed(bSharedShape))
    return failure();

  auto aSharedDesc =
      getMemDescType((*aSharedBacking)->descType, op, "shared_operand_a");
  auto bSharedDesc =
      getMemDescType((*bSharedBacking)->descType, op, "shared_operand_b");
  if (failed(aSharedDesc) || failed(bSharedDesc))
    return failure();
  auto aSharedEncoding = dyn_cast<SharedEncodingAttr>(aSharedDesc->getEncoding());
  auto bSharedEncoding = dyn_cast<SharedEncodingAttr>(bSharedDesc->getEncoding());
  if (!aSharedEncoding || !bSharedEncoding) {
    op->emitError() << "shared A/B backings must carry #tb.shared encodings";
    return failure();
  }
  if (!transport.operandA.asyncEligible || !transport.operandB.asyncEligible) {
    op->emitError() << "pipeline lowering expects async-eligible shared A/B";
    return failure();
  }
  if (transport.operandA.kind != "cp_async" ||
      transport.operandB.kind != "cp_async") {
    op->emitError() << "strict lowering only supports `cp_async` transport "
                       "kinds for shared A/B";
    return failure();
  }

  auto aWorkspaceSegment =
      findSharedWorkspaceSegment(sharedWorkspace,
                                 SharedWorkspaceSegmentKind::MainloopAStageBuffer,
                                 "mainloop_a_stage_buffer", op);
  auto bWorkspaceSegment =
      findSharedWorkspaceSegment(sharedWorkspace,
                                 SharedWorkspaceSegmentKind::MainloopBStageBuffer,
                                 "mainloop_b_stage_buffer", op);
  if (failed(aWorkspaceSegment) || failed(bWorkspaceSegment))
    return failure();

  int64_t instructionK = encodings.fragmentA.instructionShape.back();
  if (instructionK <= 0 || semantics.tileK % instructionK != 0) {
    op->emitError() << "block_k is incompatible with the fragment instruction K";
    return failure();
  }

  LoweringContract contract;
  contract.numWarps = spec.numWarps;
  contract.threadsPerWarp = target.threadsPerWarp;
  contract.numKGroups = (semantics.problemK + instructionK - 1) / instructionK;
  contract.workspaceTotalBytes = sharedWorkspace.totalBytes;
  contract.mappingKind = programMapping.mappingKind;
  contract.problemTilesM = programMapping.problemTilesM;
  contract.problemTilesN = programMapping.problemTilesN;
  contract.problemTilesK = programMapping.problemTilesK;
  contract.groupTileSpanM = programMapping.groupTileSpanM;
  contract.groupTileSpanN = programMapping.groupTileSpanN;
  contract.programsPerLaunchGroup = programMapping.programsPerLaunchGroup;
  contract.totalPrograms = programMapping.totalPrograms;
  contract.splitK = programMapping.splitK;
  contract.reductionMode = programMapping.reductionMode;
  contract.reductionKind = reduction.kind;
  contract.persistentEnabled = persistentWork.enabled;
  contract.persistentResidentPrograms = persistentWork.residentPrograms;
  contract.accTilesM = encodings.fragmentAcc.repeatShape.front();
  contract.accTilesN = encodings.fragmentAcc.repeatShape.back();
  contract.bGroupCount = rewrite.bGroupCount;
  contract.bGroupTileSpan = rewrite.bPath.consumerTileSpanN;
  contract.aSharedBacking = (*aSharedBacking)->id;
  contract.bSharedBacking = (*bSharedBacking)->id;
  contract.aWorkspaceOffsetBytes = (*aWorkspaceSegment)->byteOffset;
  contract.bWorkspaceOffsetBytes = (*bWorkspaceSegment)->byteOffset;
  contract.aSharedRows = encodings.aSharedSpec.logicalShape[0];
  contract.aSharedCols = encodings.aSharedSpec.logicalShape[1];
  contract.bSharedRows = encodings.bSharedSpec.logicalShape[0];
  contract.bSharedCols = encodings.bSharedSpec.logicalShape[1];
  contract.aSharedViewRows = (*aSharedShape)[0];
  contract.aSharedViewCols = (*aSharedShape)[1];
  contract.bSharedViewRows = (*bSharedShape)[0];
  contract.bSharedViewCols = (*bSharedShape)[1];
  contract.aFragRows = (*aRegisterShape)[0];
  contract.aFragCols = (*aRegisterShape)[1];
  contract.bSubFragRows = (*bRegisterShape)[0];
  contract.bSubFragCols = (*bRegisterShape)[1];
  contract.bGroupFragRows = contract.bSubFragRows * contract.bGroupTileSpan;
  contract.bGroupFragCols = contract.bSubFragCols;
  contract.cLandingKind = landing->kind;
  contract.cReorderKind = epilogueReorder.kind;
  contract.cDirectPackRows = landing->directPackRows;
  contract.cDirectPackCols = landing->directPackCols;
  contract.cSharedTileRows = epilogueReorder.sharedTileRows;
  contract.cSharedTileCols = epilogueReorder.sharedTileCols;
  contract.cSharedPackSlots = epilogueReorder.liveSlots;
  contract.cGlobalVectorWidth = landing->globalVectorWidth;
  contract.cInitSharedStoreVectorWidth =
      epilogueReorder.initSharedStoreVectorWidth;
  contract.cInitSharedLoadVectorWidth =
      epilogueReorder.initSharedLoadVectorWidth;
  contract.cStoreSharedStoreVectorWidth =
      epilogueReorder.storeSharedStoreVectorWidth;
  contract.cStoreSharedLoadVectorWidth =
      epilogueReorder.storeSharedLoadVectorWidth;
  contract.cLoadInitFromGlobal =
      reduction.kind == ReductionPlanKind::None;
  contract.cStoreViaAtomicAdd =
      reduction.kind == ReductionPlanKind::SplitKAtomic;
  contract.cUseSharedPackForInit =
      contract.cLoadInitFromGlobal && epilogueReorder.reorderNeededForInit;
  contract.cUseSharedPackForStore = epilogueReorder.reorderNeededForStore;
  contract.cWorkspaceBarrierCount = epilogueReorder.workspaceBarrierCount;
  contract.cBarrierAfterInit = contract.cUseSharedPackForInit;
  contract.cBarrierBeforeStore = epilogueReorder.reorderNeededForStore;
  contract.cRequiredSyncKind = epilogueReorder.workspaceSyncKind;
  contract.splitKGroupsPerTile = spec.blockK / instructionK;
  contract.splitKRemainderTiles = reduction.remainderKTiles;

  if (contract.numKGroups <= 0 || contract.accTilesM <= 0 ||
      contract.accTilesN <= 0 || contract.bGroupCount <= 0 ||
      contract.bGroupTileSpan <= 0 ||
      contract.groupTileSpanM <= 0 || contract.groupTileSpanN <= 0 ||
      contract.programsPerLaunchGroup <= 0 || contract.totalPrograms <= 0 ||
      contract.splitK <= 0 || contract.splitKGroupsPerTile <= 0 ||
      contract.bGroupCount * contract.bGroupTileSpan != contract.accTilesN) {
    op->emitError() << "invalid lowering contract geometry";
    return failure();
  }
  if (contract.threadsPerWarp <= 0) {
    op->emitError() << "target must define a positive threads_per_warp";
    return failure();
  }
  if (contract.aFragRows <= 0 || contract.aFragCols <= 0 ||
      contract.bSubFragRows <= 0 || contract.bSubFragCols <= 0) {
    op->emitError() << "operand fragment register shapes must be positive";
    return failure();
  }
  if (contract.workspaceTotalBytes <= 0 || contract.aSharedViewRows <= 0 ||
      contract.aSharedViewCols <= 0 || contract.bSharedViewRows <= 0 ||
      contract.bSharedViewCols <= 0) {
    op->emitError() << "shared workspace/view geometry must be explicit";
    return failure();
  }
  if (contract.cGlobalVectorWidth <= 0 || contract.cDirectPackRows <= 0 ||
      contract.cDirectPackCols <= 0 || contract.cRequiredSyncKind.empty()) {
    op->emitError() << "direct C target landing geometry must be explicit and "
                       "positive";
    return failure();
  }
  if (contract.cReorderKind == EpilogueReorderKind::CTASharedRowReorder) {
    auto initScratch = findSharedWorkspaceSegment(
        sharedWorkspace, SharedWorkspaceSegmentKind::EpilogueReorderScratch,
        "epilogue_init_reorder_scratch", op);
    auto storeScratch = findSharedWorkspaceSegment(
        sharedWorkspace, SharedWorkspaceSegmentKind::EpilogueReorderScratch,
        "epilogue_store_reorder_scratch", op);
    if (failed(initScratch) || failed(storeScratch))
      return failure();
    contract.cInitWorkspaceOffsetBytes = (*initScratch)->byteOffset;
    contract.cStoreWorkspaceOffsetBytes = (*storeScratch)->byteOffset;
    if (contract.cSharedTileRows <= 0 || contract.cSharedTileCols <= 0 ||
        contract.cSharedPackSlots <= 0 ||
        contract.cInitSharedStoreVectorWidth <= 0 ||
        contract.cInitSharedLoadVectorWidth <= 0 ||
        contract.cStoreSharedStoreVectorWidth <= 0 ||
        contract.cStoreSharedLoadVectorWidth <= 0 ||
        contract.cWorkspaceBarrierCount <= 0) {
      op->emitError()
          << "epilogue row reorder requires positive shared geometry";
      return failure();
    }
    if (contract.cSharedTileCols % contract.cGlobalVectorWidth != 0 ||
        contract.cSharedTileCols % contract.cInitSharedLoadVectorWidth != 0 ||
        contract.cSharedTileCols % contract.cStoreSharedStoreVectorWidth != 0) {
      op->emitError() << "epilogue reorder tile must stay aligned with "
                         "all shared/global vector widths";
      return failure();
    }
  } else if (contract.cReorderKind != EpilogueReorderKind::None) {
    op->emitError() << "unsupported epilogue reorder kind in lowering";
    return failure();
  }
  if (contract.cLandingKind != TargetLandingKind::RegisterPackGlobalVector) {
    op->emitError() << "unsupported C landing kind in lowering";
    return failure();
  }
  if (!StringRef(resourceClosure.selectedCLandingKind)
           .starts_with("register_direct_vector")) {
    op->emitError()
        << "resource closure must select register-direct final C landing";
    return failure();
  }
  if (resourceClosure.chosenLandingTradeoff == "register_direct_vector" &&
      contract.cLandingKind != TargetLandingKind::RegisterPackGlobalVector) {
    op->emitError()
        << "resource closure expects register-direct C landing, but epilogue "
           "target landing disagrees";
    return failure();
  }
  if (contract.aSharedCols != instructionK || contract.bSharedRows != instructionK) {
    op->emitError() << "shared operand slices do not match instruction K";
    return failure();
  }

  auto aElementBytes =
      getElementByteWidth(aSharedDesc->getElementType(), op, "shared A");
  auto bElementBytes =
      getElementByteWidth(bSharedDesc->getElementType(), op, "shared B");
  if (failed(aElementBytes) || failed(bElementBytes))
    return failure();
  auto aSharedLayout = buildSharedPhysicalLayoutContract(
      aSharedEncoding, contract.aSharedRows, contract.aSharedCols,
      *aElementBytes, op, "shared A");
  auto bSharedLayout = buildSharedPhysicalLayoutContract(
      bSharedEncoding, contract.bSharedRows, contract.bSharedCols,
      *bElementBytes, op, "shared B");
  if (failed(aSharedLayout) || failed(bSharedLayout))
    return failure();
  contract.aSharedLayout = *aSharedLayout;
  contract.bSharedLayout = *bSharedLayout;
  if (contract.aSharedViewRows * contract.aSharedViewCols * *aElementBytes !=
          (*aWorkspaceSegment)->byteSize ||
      contract.bSharedViewRows * contract.bSharedViewCols * *bElementBytes !=
          (*bWorkspaceSegment)->byteSize) {
    op->emitError() << "shared workspace byte segments disagree with the "
                       "flattened shared backing views";
    return failure();
  }

  int64_t aAsyncBytes = transport.operandA.asyncVectorBytes > 0
                            ? transport.operandA.asyncVectorBytes
                            : transport.operandA.vectorBytes;
  int64_t bAsyncBytes = transport.operandB.asyncVectorBytes > 0
                            ? transport.operandB.asyncVectorBytes
                            : transport.operandB.vectorBytes;
  if (aAsyncBytes <= 0 || aAsyncBytes % *aElementBytes != 0) {
    op->emitError()
        << "shared A async transport must define a legal vector width";
    return failure();
  }
  if (bAsyncBytes <= 0 || bAsyncBytes % *bElementBytes != 0) {
    op->emitError()
        << "shared B async transport must define a legal vector width";
    return failure();
  }
  contract.aAsyncCopyElems = aAsyncBytes / *aElementBytes;
  contract.bAsyncCopyElems = bAsyncBytes / *bElementBytes;
  int64_t numThreads = spec.numWarps * contract.threadsPerWarp;
  if ((contract.aSharedRows * contract.aSharedCols) %
              (numThreads * contract.aAsyncCopyElems) !=
          0) {
    op->emitError()
        << "shared A slice is incompatible with the exact async-copy thread "
           "decomposition";
    return failure();
  }
  if ((contract.bSharedRows * contract.bSharedCols) %
              (numThreads * contract.bAsyncCopyElems) !=
          0) {
    op->emitError()
        << "shared B slice is incompatible with the exact async-copy thread "
           "decomposition";
    return failure();
  }
  int64_t accumulatorFragments = static_cast<int64_t>(accumulator.packs.size());
  int64_t fragmentVectorWidth =
      accumulator.packs.empty() ? 0 : accumulator.packs.front().vectorWidth;
  if (failed(validateDirectGlobalPlan(*initPlan, accumulatorFragments,
                                      fragmentVectorWidth,
                                      accumulator.laneAccess, op, "init"))) {
    return failure();
  }
  if (failed(validateDirectGlobalPlan(*storePlan, accumulatorFragments,
                                      fragmentVectorWidth,
                                      accumulator.laneAccess, op, "store"))) {
    return failure();
  }
  if (asyncPlan.waits.empty()) {
    op->emitError() << "lowering expects explicit first-use waits for async loads";
    return failure();
  }
  if (explicitClusters.empty()) {
    op->emitError() << "lowering expects explicit pipeline clusters after cleanup";
    return failure();
  }

  DenseMap<int64_t, const PipelineOp *> opsById;
  DenseMap<int64_t, const ValueState *> valuesById;
  DenseMap<int64_t, const BufferView *> viewsById;
  DenseMap<int64_t, const BufferBacking *> backingsById;
  DenseMap<int64_t, Position> positionByOp;
  DenseMap<int64_t, const AsyncProducer *> cpAsyncProducerByValue;
  DenseMap<int64_t, const AsyncGroup *> groupById;
  DenseMap<int64_t, const WaitInfo *> waitByGroupId;
  DenseSet<int64_t> waitedGroups;
  DenseSet<int64_t> explicitOps;
  DenseSet<int64_t> explicitWaitedGroups;
  int64_t loadACount = 0;
  int64_t loadBCount = 0;
  int64_t localLoadACount = 0;
  int64_t localLoadBCount = 0;
  int64_t mmaCount = 0;
  int64_t maxScheduledStage = -1;

  for (const PipelineOp &pipelineOp : model.ops) {
    opsById[pipelineOp.id] = &pipelineOp;
    switch (pipelineOp.kind) {
    case BufferOpKind::LoadA:
      ++loadACount;
      break;
    case BufferOpKind::LoadB:
      ++loadBCount;
      break;
    case BufferOpKind::LocalLoadA:
      ++localLoadACount;
      break;
    case BufferOpKind::LocalLoadB:
      ++localLoadBCount;
      break;
    case BufferOpKind::Mma:
      ++mmaCount;
      break;
    default:
      break;
    }
  }
  for (const ValueState &value : model.values)
    valuesById[value.id] = &value;
  for (const BufferView &view : model.views)
    viewsById[view.id] = &view;
  for (const BufferBacking &backing : model.backings)
    backingsById[backing.id] = &backing;

  for (const auto &it : llvm::enumerate(explicitClusters)) {
    const ExplicitPipelineCluster &cluster = it.value();
    maxScheduledStage = std::max(maxScheduledStage, cluster.stage);
    if (cluster.ordinal != static_cast<int64_t>(it.index())) {
      op->emitError()
          << "explicit pipeline clusters must carry contiguous ordinals";
      return failure();
    }
    if (cluster.stage < 0 || cluster.cluster < 0 || cluster.kGroup < 0) {
      op->emitError()
          << "explicit pipeline clusters must carry non-negative metadata";
      return failure();
    }
    if (cluster.opIds.empty()) {
      op->emitError() << "explicit pipeline clusters must not be empty";
      return failure();
    }

    for (const auto &orderedOp : llvm::enumerate(cluster.opIds)) {
      int64_t opId = orderedOp.value();
      auto opIt = opsById.find(opId);
      if (opIt == opsById.end()) {
        op->emitError() << "explicit pipeline cluster references unknown op "
                        << opId;
        return failure();
      }
      if (!explicitOps.insert(opId).second) {
        op->emitError() << "explicit pipeline clusters duplicate op " << opId;
        return failure();
      }
      positionByOp[opId] = Position{cluster.stage, cluster.cluster,
                                    static_cast<int64_t>(orderedOp.index())};

      int64_t opKGroup = findIterationCoord(opIt->second->iterationCoords, "k_group");
      if (opKGroup != cluster.kGroup) {
        op->emitError() << "explicit pipeline cluster k_group " << cluster.kGroup
                        << " disagrees with op " << opId << " k_group "
                        << opKGroup;
        return failure();
      }

      switch (cluster.kind) {
      case ExpandedClusterKind::AsyncIssue:
        if (opIt->second->kind != BufferOpKind::LoadA &&
            opIt->second->kind != BufferOpKind::LoadB) {
          op->emitError()
              << "async_issue_cluster may only contain load_a/load_b ops";
          return failure();
        }
        break;
      case ExpandedClusterKind::ConsumerWait:
        if (opIt->second->kind != BufferOpKind::LocalLoadA &&
            opIt->second->kind != BufferOpKind::LocalLoadB) {
          op->emitError() << "consumer_wait_cluster may only contain "
                             "local_load_a/local_load_b ops";
          return failure();
        }
        break;
      case ExpandedClusterKind::MmaCompute:
        if (opIt->second->kind != BufferOpKind::Mma) {
          op->emitError()
              << "mma_compute_cluster may only contain mma ops";
          return failure();
        }
        break;
      }
    }
  }
  if (explicitOps.size() != model.ops.size()) {
    op->emitError()
        << "explicit pipeline clusters must cover every pipeline op exactly once";
    return failure();
  }
  if (pipelineReady.scheduledMaxStage != maxScheduledStage) {
    op->emitError() << "pipeline_ready stage summary disagrees with explicit "
                       "pipeline clusters";
    return failure();
  }

  for (const AsyncGroup &group : asyncPlan.groups) {
    if (!groupById.try_emplace(group.id, &group).second) {
      op->emitError() << "duplicate async group " << group.id;
      return failure();
    }
  }
  for (const AsyncProducer &producer : asyncPlan.producers) {
    if (producer.kind != AsyncProducerKind::CpAsync)
      continue;
    if (!producer.legal) {
      op->emitError() << "strict lowering does not accept illegal cp.async producer";
      return failure();
    }
    if (!cpAsyncProducerByValue.try_emplace(producer.valueId, &producer).second) {
      op->emitError() << "duplicate cp.async producer for value " << producer.valueId;
      return failure();
    }
    if (producer.srcOffsets.size() != 2) {
      op->emitError() << "cp.async producer for value " << producer.valueId
                      << " must carry rank-2 explicit src offsets";
      return failure();
    }
    auto valueIt = valuesById.find(producer.valueId);
    if (valueIt == valuesById.end() || producer.dstView != valueIt->second->ownerView) {
      op->emitError() << "cp.async producer must target the value owner view";
      return failure();
    }
    auto producerOpIt = opsById.find(producer.opId);
    if (producerOpIt == opsById.end() ||
        (producerOpIt->second->kind != BufferOpKind::LoadA &&
         producerOpIt->second->kind != BufferOpKind::LoadB)) {
      op->emitError() << "cp.async producer must be defined by load_a/load_b";
      return failure();
    }
    auto srcViewIt = viewsById.find(producer.srcView);
    auto dstViewIt = viewsById.find(producer.dstView);
    if (srcViewIt == viewsById.end() || dstViewIt == viewsById.end()) {
      op->emitError() << "cp.async producer references unknown src/dst view";
      return failure();
    }
    if (srcViewIt->second->kind != ViewKind::FullBuffer) {
      op->emitError() << "cp.async producer src_view must be a full global view";
      return failure();
    }
    auto srcBackingIt = backingsById.find(srcViewIt->second->backing);
    if (srcBackingIt == backingsById.end()) {
      op->emitError() << "cp.async producer src_view references unknown backing";
      return failure();
    }
    auto srcMemory =
        getMemDescMemorySpace(srcBackingIt->second->descType, op, "cp_async_src");
    if (failed(srcMemory))
      return failure();
    if (*srcMemory != MemorySpace::Global) {
      op->emitError() << "cp.async producer src_view must reference global memory";
      return failure();
    }
    BufferRole expectedSrcRole =
        producerOpIt->second->kind == BufferOpKind::LoadA ? BufferRole::OperandA
                                                          : BufferRole::OperandB;
    if (srcBackingIt->second->role != expectedSrcRole) {
      op->emitError() << "cp.async producer src_view role disagrees with its "
                         "load op kind";
      return failure();
    }
    if (srcViewIt->second->shape.size() < 2 || dstViewIt->second->shape.size() < 2) {
      op->emitError() << "cp.async producer src/dst views must be rank-2";
      return failure();
    }
    bool srcOverrun = producer.srcOffsets[0] < 0 || producer.srcOffsets[1] < 0 ||
                      producer.srcOffsets[0] + dstViewIt->second->shape[0] >
                          srcViewIt->second->shape[0] ||
                      producer.srcOffsets[1] + dstViewIt->second->shape[1] >
                          srcViewIt->second->shape[1];
    if (srcOverrun && !producer.predicated && !producer.zeroFill) {
      op->emitError() << "cp.async producer src offsets overrun the source view";
      return failure();
    }
    if (!groupById.count(producer.groupId)) {
      op->emitError() << "cp.async producer references missing group "
                      << producer.groupId;
      return failure();
    }
  }
  for (const WaitInfo &wait : asyncPlan.waits) {
    if (!waitByGroupId.try_emplace(wait.groupId, &wait).second) {
      op->emitError() << "duplicate wait metadata for async group "
                      << wait.groupId;
      return failure();
    }
    if (!groupById.count(wait.groupId)) {
      op->emitError() << "wait references unknown async group " << wait.groupId;
      return failure();
    }
    if (!waitedGroups.insert(wait.groupId).second) {
      op->emitError() << "duplicate wait for async group " << wait.groupId;
      return failure();
    }
    const AsyncGroup *group = groupById.lookup(wait.groupId);
    if (!group || group->producers.size() != 1) {
      op->emitError() << "strict lowering expects one producer per async group";
      return failure();
    }
    int64_t producerIndex = group->producers.front();
    if (producerIndex < 0 ||
        producerIndex >= static_cast<int64_t>(asyncPlan.producers.size())) {
      op->emitError() << "async group " << wait.groupId
                      << " references invalid producer index";
      return failure();
    }
    const AsyncProducer &producer = asyncPlan.producers[producerIndex];
    auto valueIt = valuesById.find(producer.valueId);
    auto usePosIt = positionByOp.find(wait.beforeOpId);
    auto defPosIt = positionByOp.find(producer.opId);
    if (valueIt == valuesById.end() || usePosIt == positionByOp.end() ||
        defPosIt == positionByOp.end()) {
      op->emitError() << "wait must anchor to a scheduled pipeline value";
      return failure();
    }
    if (!llvm::is_contained(valueIt->second->users, wait.beforeOpId)) {
      op->emitError() << "wait before_op " << wait.beforeOpId
                      << " is not a user of value " << producer.valueId;
      return failure();
    }
    Position required{wait.requiredStage, wait.requiredCluster, wait.requiredOrder};
    Position actualUse = usePosIt->second;
    if (required.stage != actualUse.stage || required.cluster != actualUse.cluster ||
        required.order != actualUse.order) {
      op->emitError() << "wait frontier for value " << producer.valueId
                      << " does not match the scheduled first-use position";
      return failure();
    }
    if (!isStrictlyEarlier(defPosIt->second, actualUse)) {
      op->emitError() << "waited value " << producer.valueId
                      << " must be defined before its first use";
      return failure();
    }
  }
  for (const PipelineOp &pipelineOp : model.ops) {
    if (pipelineOp.kind != BufferOpKind::LoadA &&
        pipelineOp.kind != BufferOpKind::LoadB) {
      continue;
    }
    if (pipelineOp.outputs.size() != 1 ||
        !cpAsyncProducerByValue.count(pipelineOp.outputs.front())) {
      op->emitError() << "every shared load must carry exactly one cp.async producer";
      return failure();
    }
  }
  for (const ReuseFence &fence : asyncPlan.reuseFences) {
    auto prevIt = valuesById.find(fence.retiringValueId);
    auto nextIt = valuesById.find(fence.acquiringValueId);
    auto atPosIt = positionByOp.find(fence.afterOpId);
    if (prevIt == valuesById.end() || nextIt == valuesById.end() ||
        atPosIt == positionByOp.end()) {
      op->emitError() << "reuse fence references unknown value/op";
      return failure();
    }
    if (prevIt->second->ownerView != fence.viewId ||
        nextIt->second->ownerView != fence.viewId ||
        fence.afterOpId != nextIt->second->definingOp) {
      op->emitError() << "reuse fence for view " << fence.viewId
                      << " is inconsistent with the buffer model";
      return failure();
    }
    Position requiredAfter{fence.requiredAfterStage, fence.requiredAfterCluster,
                           fence.requiredAfterOrder};
    bool sameOpCarry =
        llvm::is_contained(prevIt->second->users, fence.afterOpId);
    bool legalReuse = sameOpCarry ? isEarlierOrEqual(requiredAfter, atPosIt->second)
                                  : isStrictlyEarlier(requiredAfter,
                                                      atPosIt->second);
    if (!legalReuse) {
      op->emitError() << "reuse frontier for view " << fence.viewId
                      << " is later than the acquiring op";
      return failure();
    }
  }

  if (loadACount != contract.numKGroups || loadBCount != contract.numKGroups ||
      localLoadACount != contract.numKGroups * contract.accTilesM ||
      localLoadBCount != contract.numKGroups * contract.bGroupCount ||
      mmaCount != contract.numKGroups * contract.accTilesM * contract.accTilesN) {
    op->emitError() << "buffer model does not match the expected layout-driven "
                       "load/local-load/mma decomposition";
    return failure();
  }

  for (const ExplicitPipelineCluster &cluster : explicitClusters) {
    if (cluster.kind == ExpandedClusterKind::ConsumerWait &&
        cluster.waitGroupIds.empty()) {
      op->emitError()
          << "consumer_wait_cluster must carry wait ownership";
      return failure();
    }
    if (cluster.kind != ExpandedClusterKind::ConsumerWait &&
        !cluster.waitGroupIds.empty()) {
      op->emitError() << "only consumer_wait_cluster may carry wait "
                         "ownership";
      return failure();
    }
    if (cluster.kind == ExpandedClusterKind::ConsumerWait &&
        !cluster.needsBarrier) {
      op->emitError()
          << "consumer_wait_cluster must carry CTA barrier ownership";
      return failure();
    }
    if (cluster.kind != ExpandedClusterKind::ConsumerWait &&
        cluster.needsBarrier) {
      op->emitError()
          << "non-consumer explicit clusters must not request CTA barrier";
      return failure();
    }

    for (int64_t groupId : cluster.waitGroupIds) {
      if (!explicitWaitedGroups.insert(groupId).second) {
        op->emitError() << "explicit pipeline clusters duplicate waited group "
                        << groupId;
        return failure();
      }
      auto waitIt = waitByGroupId.find(groupId);
      if (waitIt == waitByGroupId.end()) {
        op->emitError()
            << "explicit pipeline clusters reference unknown wait group "
            << groupId;
        return failure();
      }
      const WaitInfo *wait = waitIt->second;
      if (wait->requiredStage != cluster.stage ||
          wait->requiredCluster != cluster.cluster) {
        op->emitError() << "explicit pipeline cluster does not match wait "
                           "frontier for group "
                        << groupId;
        return failure();
      }
      if (!cluster.needsBarrier && wait->needsBarrier) {
        op->emitError()
            << "explicit pipeline cluster dropped the required CTA barrier for "
               "group "
            << groupId;
        return failure();
      }
    }
  }

  if (explicitWaitedGroups.size() != asyncPlan.waits.size()) {
    op->emitError()
        << "explicit pipeline clusters must cover every async wait exactly once";
    return failure();
  }

  return contract;
}

static std::string makeKernelStem(Operation *op, int64_t ordinal) {
  StringRef parentName = "<unknown>";
  if (auto parent = op->getParentOfType<func::FuncOp>())
    parentName = parent.getName();
  return llvm::formatv("{0}_tb_kernel_{1}", parentName, ordinal).str();
}

static gpu::GPUModuleOp createGpuModule(OpBuilder &builder, ModuleOp module,
                                        StringRef name) {
  module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                  builder.getUnitAttr());
  builder.setInsertionPointToStart(module.getBody());
  return gpu::GPUModuleOp::create(builder, module.getLoc(), name);
}

static Value createAsyncGroup(OpBuilder &builder, Location loc,
                              ArrayRef<Value> tokens) {
  return nvgpu::DeviceAsyncCreateGroupOp::create(
             builder, loc,
             nvgpu::DeviceAsyncTokenType::get(builder.getContext()), tokens)
      .getResult();
}

static Value createIndexXor(OpBuilder &builder, Location loc, Value lhs,
                            Value rhs) {
  Type i64Type = builder.getI64Type();
  Value lhsI64 = arith::IndexCastUIOp::create(builder, loc, i64Type, lhs);
  Value rhsI64 = arith::IndexCastUIOp::create(builder, loc, i64Type, rhs);
  Value xorI64 = arith::XOrIOp::create(builder, loc, lhsI64, rhsI64);
  return arith::IndexCastUIOp::create(builder, loc, builder.getIndexType(),
                                      xorI64);
}

static std::pair<Value, Value>
applySharedPhysicalLayout(OpBuilder &builder, Location loc, Value row, Value col,
                          const SharedPhysicalLayoutContract &layout) {
  if (layout.swizzlingByteWidth <= 0 || layout.elementBytes <= 0 ||
      layout.maxPhase <= 1) {
    return {row, col};
  }

  int64_t spanElems = layout.swizzlingByteWidth / layout.elementBytes;
  if (spanElems <= 0)
    return {row, col};

  Value minor = layout.minorDim == 0 ? row : col;
  Value major = layout.minorDim == 0 ? col : row;
  Value span = createIndexConstant(builder, loc, spanElems);
  Value phase = createIndexRemU(
      builder, loc,
      createIndexDivU(builder, loc, major,
                      createIndexConstant(builder, loc,
                                          std::max<int64_t>(layout.perPhase, 1))),
      createIndexConstant(builder, loc, std::max<int64_t>(layout.maxPhase, 1)));
  Value chunk = createIndexDivU(builder, loc, minor, span);
  Value inChunk = createIndexRemU(builder, loc, minor, span);
  Value chunkCount = createIndexConstant(
      builder, loc, std::max<int64_t>(layout.minorExtent / spanElems, 1));
  Value swizzledChunk = createIndexRemU(
      builder, loc, createIndexXor(builder, loc, chunk, phase), chunkCount);
  Value swizzledMinor =
      createIndexAdd(builder, loc,
                     createIndexMul(builder, loc, swizzledChunk, span), inChunk);
  return layout.minorDim == 0 ? std::make_pair(swizzledMinor, col)
                              : std::make_pair(row, swizzledMinor);
}

static Value buildLaneRowBase(OpBuilder &builder, Location loc, Value laneId,
                              const LaneAccessPattern &laneAccess) {
  return createIndexDivU(builder, loc, laneId,
                         createIndexConstant(builder, loc,
                                             laneAccess.laneRowGroupSize));
}

static Value buildLaneColBase(OpBuilder &builder, Location loc, Value laneId,
                              const LaneAccessPattern &laneAccess) {
  Value laneGroup =
      createIndexRemU(builder, loc, laneId,
                      createIndexConstant(builder, loc,
                                          laneAccess.laneColGroupSize));
  return createIndexMul(builder, loc, laneGroup,
                        createIndexConstant(builder, loc,
                                            laneAccess.laneColStride));
}

static LogicalResult validateDirectGlobalVectorMemoryContract(
    const DirectGlobalVectorPlan &plan, Operation *op, StringRef role) {
  if (plan.vectorWidth <= 0) {
    op->emitError() << role << " must carry a positive vector width";
    return failure();
  }
  if (plan.laneAccess.laneColStride != plan.vectorWidth) {
    op->emitError()
        << role
        << " must use a chunk-aligned lane col stride equal to vector width";
    return failure();
  }
  for (const DirectGlobalVectorPlan::Pack &pack : plan.packs) {
    if (pack.vectorWidth != plan.vectorWidth || pack.colBase % plan.vectorWidth != 0) {
      op->emitError()
          << role
          << " pack geometry must stay aligned to the direct-global vector width";
      return failure();
    }
  }
  return success();
}

static Value emitGlobalVectorLoadRow(OpBuilder &builder, Location loc,
                                     Value cMemref, Value row, Value col,
                                     VectorType rowVectorType,
                                     const DirectGlobalVectorPlan &plan) {
  return EpilogueGlobalVectorLoadOp::create(
             builder, loc, rowVectorType, cMemref, row, col,
             builder.getI64IntegerAttr(plan.vectorWidth),
             builder.getBoolAttr(plan.boundaryAware),
             builder.getBoolAttr(plan.scalarTail))
      .getResult();
}

static void emitGlobalVectorStoreRow(OpBuilder &builder, Location loc,
                                     Value rowVector, Value cMemref, Value row,
                                     Value col,
                                     const DirectGlobalVectorPlan &plan) {
  EpilogueGlobalVectorStoreOp::create(
      builder, loc, rowVector, cMemref, row, col,
      builder.getI64IntegerAttr(plan.vectorWidth),
      builder.getBoolAttr(plan.boundaryAware),
      builder.getBoolAttr(plan.scalarTail));
}

static void emitSyncWarp(OpBuilder &builder, Location loc) {
  Value fullMask = arith::ConstantIntOp::create(builder, loc, -1, 32);
  NVVM::SyncWarpOp::create(builder, loc, fullMask);
}

static void storeDirectRowToSharedPack(
    OpBuilder &builder, Location loc, Value rowVector, Value sharedPack,
    Value laneId, Value warpId, Value slotId,
    const DirectGlobalVectorPlan &plan,
    int64_t rowOffset) {
  Value row = createIndexAddConst(builder, loc,
                                  buildLaneRowBase(builder, loc, laneId,
                                                   plan.laneAccess),
                                  rowOffset);
  Value col = buildLaneColBase(builder, loc, laneId, plan.laneAccess);
  vector::StoreOp::create(builder, loc, rowVector, sharedPack,
                          ValueRange{warpId, slotId, row, col});
}

static Value loadDirectRowFromSharedPack(
    OpBuilder &builder, Location loc, Value sharedPack, Value laneId,
    Value warpId, Value slotId, const DirectGlobalVectorPlan &plan,
    int64_t rowOffset,
    VectorType rowVectorType) {
  Value row = createIndexAddConst(builder, loc,
                                  buildLaneRowBase(builder, loc, laneId,
                                                   plan.laneAccess),
                                  rowOffset);
  Value col = buildLaneColBase(builder, loc, laneId, plan.laneAccess);
  return vector::LoadOp::create(builder, loc, rowVectorType, sharedPack,
                                ValueRange{warpId, slotId, row, col});
}

static Value loadAccumulatorFragmentFromSharedPack(
    OpBuilder &builder, Location loc, Value sharedPack, Value laneId,
    Value warpId, Value slotId, const AccumulatorPlan &accumulator,
    const AccumulatorPack &fragment,
    const DirectGlobalVectorPlan::Pack &pack,
    VectorType accumulatorFragmentType) {
  Value fragmentValue = buildZeroVector(builder, loc, accumulatorFragmentType);
  for (const auto &it : llvm::enumerate(accumulator.laneAccess.rowOffsets)) {
    Value row = createIndexAddConst(
        builder, loc,
        createIndexAddConst(builder, loc,
                            buildLaneRowBase(builder, loc, laneId,
                                             accumulator.laneAccess),
                            fragment.rowBase - pack.rowBase),
        it.value());
    Value col = createIndexAddConst(
        builder, loc,
        createIndexAddConst(builder, loc,
                            buildLaneColBase(builder, loc, laneId,
                                             accumulator.laneAccess),
                            fragment.colBase - pack.colBase),
        0);
    Value rowVector = vector::LoadOp::create(
        builder, loc,
        VectorType::get({accumulator.packs.front().vectorWidth},
                        accumulatorFragmentType.getElementType()),
        sharedPack, ValueRange{warpId, slotId, row, col});
    fragmentValue = vector::InsertOp::create(
        builder, loc, rowVector, fragmentValue,
        ArrayRef<int64_t>{static_cast<int64_t>(it.index())});
  }
  return fragmentValue;
}

static void storeAccumulatorFragmentToSharedPack(
    OpBuilder &builder, Location loc, Value fragmentValue, Value sharedPack,
    Value laneId, Value warpId, Value slotId,
    const AccumulatorPlan &accumulator,
    const AccumulatorPack &fragment,
    const DirectGlobalVectorPlan::Pack &pack) {
  for (const auto &it : llvm::enumerate(accumulator.laneAccess.rowOffsets)) {
    Value row = createIndexAddConst(
        builder, loc,
        createIndexAddConst(builder, loc,
                            buildLaneRowBase(builder, loc, laneId,
                                             accumulator.laneAccess),
                            fragment.rowBase - pack.rowBase),
        it.value());
    Value col = createIndexAddConst(
        builder, loc,
        createIndexAddConst(builder, loc,
                            buildLaneColBase(builder, loc, laneId,
                                             accumulator.laneAccess),
                            fragment.colBase - pack.colBase),
        0);
    Value rowVector = vector::ExtractOp::create(
        builder, loc, fragmentValue,
        ArrayRef<int64_t>{static_cast<int64_t>(it.index())});
    vector::StoreOp::create(builder, loc, rowVector, sharedPack,
                            ValueRange{warpId, slotId, row, col});
  }
}

static Value extractAccumulatorFragmentFromPack(
    OpBuilder &builder, Location loc, Value packValue, int64_t fragmentOrdinal,
    int64_t fragmentVectorWidth, VectorType accumulatorFragmentType) {
  return vector::ExtractStridedSliceOp::create(
      builder, loc, packValue,
      ArrayRef<int64_t>{0, fragmentOrdinal * fragmentVectorWidth},
      ArrayRef<int64_t>{accumulatorFragmentType.getShape()[0],
                        accumulatorFragmentType.getShape()[1]},
      ArrayRef<int64_t>{1, 1});
}

static Value assembleDirectPackFromFragments(
    OpBuilder &builder, Location loc, ArrayRef<Value> fragments,
    int64_t fragmentVectorWidth, VectorType directPackType) {
  Value packValue = buildZeroVector(builder, loc, directPackType);
  for (const auto &it : llvm::enumerate(fragments)) {
    packValue = vector::InsertStridedSliceOp::create(
        builder, loc, it.value(), packValue,
        ArrayRef<int64_t>{0, static_cast<int64_t>(it.index()) *
                                 fragmentVectorWidth},
        ArrayRef<int64_t>{1, 1});
  }
  return packValue;
}

static Value materializeDirectPackFromGlobalRows(
    OpBuilder &builder, Location loc, Value cMemref, Value rowBase,
    Value colBase, Value laneId, const DirectGlobalVectorPlan &plan,
    VectorType directPackType, VectorType rowVectorType) {
  Value packValue = buildZeroVector(builder, loc, directPackType);
  Value directLaneRowBase = buildLaneRowBase(builder, loc, laneId, plan.laneAccess);
  Value directLaneColBase = buildLaneColBase(builder, loc, laneId, plan.laneAccess);
  for (const auto &it : llvm::enumerate(plan.laneAccess.rowOffsets)) {
    Value row = createIndexAddConst(
        builder, loc, createIndexAdd(builder, loc, rowBase, directLaneRowBase),
        it.value());
    Value col = createIndexAdd(builder, loc, colBase, directLaneColBase);
    Value rowVector =
        emitGlobalVectorLoadRow(builder, loc, cMemref, row, col, rowVectorType,
                                plan);
    packValue = vector::InsertOp::create(
        builder, loc, rowVector, packValue,
        ArrayRef<int64_t>{static_cast<int64_t>(it.index())});
  }
  return packValue;
}

static void storeDirectPackToGlobalRows(OpBuilder &builder, Location loc,
                                        Value packValue, Value cMemref,
                                        Value rowBase, Value colBase,
                                        Value laneId,
                                        const DirectGlobalVectorPlan &plan) {
  Value directLaneRowBase = buildLaneRowBase(builder, loc, laneId, plan.laneAccess);
  Value directLaneColBase = buildLaneColBase(builder, loc, laneId, plan.laneAccess);
  for (const auto &it : llvm::enumerate(plan.laneAccess.rowOffsets)) {
    Value rowVector = vector::ExtractOp::create(
        builder, loc, packValue,
        ArrayRef<int64_t>{static_cast<int64_t>(it.index())});
    Value row = createIndexAddConst(
        builder, loc, createIndexAdd(builder, loc, rowBase, directLaneRowBase),
        it.value());
    Value col = createIndexAdd(builder, loc, colBase, directLaneColBase);
    emitGlobalVectorStoreRow(builder, loc, rowVector, cMemref, row, col, plan);
  }
}

static FailureOr<Value> getMemrefDimValue(OpBuilder &builder, Location loc,
                                          Value memrefValue, int64_t dim,
                                          StringRef role) {
  auto memrefType = dyn_cast<MemRefType>(memrefValue.getType());
  if (!memrefType || memrefType.getRank() != 2) {
    emitError(loc) << role << " requires a rank-2 memref operand";
    return failure();
  }
  if (!ShapedType::isDynamic(memrefType.getShape()[dim]))
    return createIndexConstant(builder, loc, memrefType.getShape()[dim]);
  return memref::DimOp::create(builder, loc, memrefValue, dim).getResult();
}

static void emitAtomicAddRowToGlobal(OpBuilder &builder, Location loc,
                                     Value rowVector, Value cMemref, Value row,
                                     Value colBase,
                                     const DirectGlobalVectorPlan &plan,
                                     Value problemRows, Value problemCols) {
  auto rowVectorType = dyn_cast<VectorType>(rowVector.getType());
  if (!rowVectorType || rowVectorType.getRank() != 1)
    llvm_unreachable("atomic epilogue row add expects a rank-1 vector");
  Value rowInBounds = plan.boundaryAware
                          ? createIndexCmpULT(builder, loc, row, problemRows)
                          : Value();
  for (int64_t i = 0, e = rowVectorType.getShape().front(); i < e; ++i) {
    Value laneValue = vector::ExtractOp::create(builder, loc, rowVector,
                                                ArrayRef<int64_t>{i});
    Value laneCol = createIndexAddConst(builder, loc, colBase, i);
    if (plan.boundaryAware) {
      Value colInBounds =
          createIndexCmpULT(builder, loc, laneCol, problemCols);
      Value inBounds = arith::AndIOp::create(builder, loc, rowInBounds,
                                             colInBounds);
      auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, inBounds,
                                    /*withElseRegion=*/false);
      OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
      memref::AtomicRMWOp::create(thenBuilder, loc, arith::AtomicRMWKind::addf,
                                  laneValue, cMemref, ValueRange{row, laneCol});
      scf::YieldOp::create(thenBuilder, loc);
      continue;
    }
    memref::AtomicRMWOp::create(builder, loc, arith::AtomicRMWKind::addf,
                                laneValue, cMemref, ValueRange{row, laneCol});
  }
}

static void storeDirectPackToGlobalRowsAtomicAdd(
    OpBuilder &builder, Location loc, Value packValue, Value cMemref,
    Value rowBase, Value colBase, Value laneId, const DirectGlobalVectorPlan &plan,
    Value problemRows, Value problemCols) {
  Value directLaneRowBase = buildLaneRowBase(builder, loc, laneId, plan.laneAccess);
  Value directLaneColBase = buildLaneColBase(builder, loc, laneId, plan.laneAccess);
  for (const auto &it : llvm::enumerate(plan.laneAccess.rowOffsets)) {
    Value rowVector = vector::ExtractOp::create(
        builder, loc, packValue,
        ArrayRef<int64_t>{static_cast<int64_t>(it.index())});
    Value row = createIndexAddConst(
        builder, loc, createIndexAdd(builder, loc, rowBase, directLaneRowBase),
        it.value());
    Value col = createIndexAdd(builder, loc, colBase, directLaneColBase);
    emitAtomicAddRowToGlobal(builder, loc, rowVector, cMemref, row, col, plan,
                             problemRows, problemCols);
  }
}

static Value buildOperandLaneRow(OpBuilder &builder, Location loc, Value laneId,
                                 const FragmentEncodingSpec &fragmentSpec) {
  return createIndexRemU(builder, loc, laneId,
                         createIndexConstant(builder, loc,
                                             fragmentSpec.laneRowModulus));
}

static Value buildOperandLaneCol(OpBuilder &builder, Location loc, Value laneId,
                                 const FragmentEncodingSpec &fragmentSpec) {
  Value laneGroup =
      createIndexDivU(builder, loc, laneId,
                      createIndexConstant(builder, loc,
                                          fragmentSpec.laneColDivisor));
  return createIndexMul(builder, loc, laneGroup,
                        createIndexConstant(builder, loc,
                                            fragmentSpec.laneColStride));
}

static Value loadOperandFragment(OpBuilder &builder, Location loc, Value shared,
                                 Value tileBaseRow, Value tileBaseCol,
                                 Value laneRow, Value laneCol,
                                 VectorType fragType,
                                 const FragmentEncodingSpec &fragmentSpec,
                                 const SharedPhysicalLayoutContract &layout) {
  Value row = createIndexAdd(builder, loc, tileBaseRow, laneRow);
  Value col = createIndexAdd(builder, loc, tileBaseCol, laneCol);
  std::pair<Value, Value> physical =
      applySharedPhysicalLayout(builder, loc, row, col, layout);
  return nvgpu::LdMatrixOp::create(builder, loc, fragType, shared,
                                   ValueRange{physical.first, physical.second},
                                   /*transpose=*/fragmentSpec.ldmatrixTranspose,
                                   /*numTiles=*/fragmentSpec.ldmatrixTileCount);
}

static void appendAsyncCopiesForSlice(OpBuilder &builder, Location loc,
                                      Value threadId, Value src, Value dst,
                                      ArrayRef<int64_t> tileShape, Value srcRowBase,
                                      Value srcColBase, int64_t dstRowBase,
                                      int64_t dstColBase, int64_t numThreads,
                                      int64_t elementsPerCopy, int64_t problemRows,
                                      int64_t problemCols, bool predicated,
                                      bool bypassL1,
                                      const SharedPhysicalLayoutContract &dstLayout,
                                      Value activePredicate,
                                      SmallVectorImpl<Value> &tokens) {
  int64_t rows = tileShape[0];
  int64_t cols = tileShape[1];
  int64_t copiesPerThread = (rows * cols) / (numThreads * elementsPerCopy);
  Value colsValue = createIndexConstant(builder, loc, cols);
  Value elementsPerCopyValue =
      createIndexConstant(builder, loc, elementsPerCopy);
  Value threadOffset =
      createIndexMul(builder, loc, threadId, elementsPerCopyValue);

  for (int64_t i = 0; i < copiesPerThread; ++i) {
    Value linear = threadOffset;
    if (i != 0) {
      linear = createIndexAddConst(builder, loc, linear,
                                   i * numThreads * elementsPerCopy);
    }
    Value row = createIndexDivU(builder, loc, linear, colsValue);
    Value col = createIndexRemU(builder, loc, linear, colsValue);
    Value srcRow = createIndexAdd(builder, loc, row, srcRowBase);
    Value srcCol = createIndexAdd(builder, loc, col, srcColBase);
    Value dstRow = createIndexAddConst(builder, loc, row, dstRowBase);
    Value dstCol = createIndexAddConst(builder, loc, col, dstColBase);
    std::pair<Value, Value> physicalDst =
        applySharedPhysicalLayout(builder, loc, dstRow, dstCol, dstLayout);
    Value srcElements = createIndexConstant(builder, loc, elementsPerCopy);
    if (predicated) {
      Value problemRowsValue = createIndexConstant(builder, loc, problemRows);
      Value problemColsValue = createIndexConstant(builder, loc, problemCols);
      Value rowInBounds = createIndexCmpULT(builder, loc, srcRow, problemRowsValue);
      Value colInBounds = createIndexCmpULT(builder, loc, srcCol, problemColsValue);
      Value remainingCols = arith::SelectOp::create(
          builder, loc, colInBounds,
          createIndexSub(builder, loc, problemColsValue, srcCol),
          createIndexConstant(builder, loc, 0));
      Value boundedCols = createIndexMinU(
          builder, loc, remainingCols,
          createIndexConstant(builder, loc, elementsPerCopy));
      srcElements = arith::SelectOp::create(
          builder, loc, rowInBounds, boundedCols,
          createIndexConstant(builder, loc, 0));
    }
    if (activePredicate) {
      srcElements = arith::SelectOp::create(
          builder, loc, activePredicate, srcElements,
          createIndexConstant(builder, loc, 0));
    }
    UnitAttr bypassL1Attr = bypassL1 ? builder.getUnitAttr() : UnitAttr();
    tokens.push_back(nvgpu::DeviceAsyncCopyOp::create(
                         builder, loc,
                         nvgpu::DeviceAsyncTokenType::get(builder.getContext()),
                         dst, ValueRange{physicalDst.first, physicalDst.second}, src,
                         ValueRange{srcRow, srcCol},
                         builder.getIndexAttr(elementsPerCopy),
                         /*srcElements=*/(predicated || activePredicate)
                             ? srcElements
                             : Value(),
                         /*bypassL1=*/bypassL1Attr)
                         .getResult());
  }
}

static LogicalResult emitKernelBody(OpBuilder &builder, gpu::GPUFuncOp gpuFunc,
                                    const KernelConfig &spec,
                                    const MatmulSemantics &semantics,
                                    const ProgramMappingPlan &programMapping,
                                    const EncodingPlan &encodings,
                                    const AccumulatorPlan &accumulator,
                                    const EpiloguePlan &epilogue,
                                    const BufferModel &model,
                                    ArrayRef<ExplicitPipelineCluster> explicitClusters,
                                    const AsyncPlan &asyncPlan,
                                    const LoweringContract &contract) {
  (void)programMapping;
  Location loc = gpuFunc.getLoc();
  Block &entry = gpuFunc.getBody().front();
  builder.setInsertionPointToStart(&entry);

  auto directPlans = getDirectEpiloguePlans(epilogue, gpuFunc.getOperation());
  if (failed(directPlans))
    return failure();
  const DirectGlobalVectorPlan *initPlan = directPlans->first;
  const DirectGlobalVectorPlan *storePlan = directPlans->second;
  if (failed(validateDirectGlobalVectorMemoryContract(
          *initPlan, gpuFunc.getOperation(), "init direct_global_vector")) ||
      failed(validateDirectGlobalVectorMemoryContract(
          *storePlan, gpuFunc.getOperation(), "store direct_global_vector"))) {
    return failure();
  }

  Value a = entry.getArgument(0);
  Value b = entry.getArgument(1);
  Value c = entry.getArgument(2);
  Type aElementType = cast<MemRefType>(a.getType()).getElementType();
  Type bElementType = cast<MemRefType>(b.getType()).getElementType();
  Type cElementType = cast<MemRefType>(c.getType()).getElementType();
  auto workgroupAttributions = gpuFunc.getWorkgroupAttributions();
  if (workgroupAttributions.empty()) {
    gpuFunc.emitError() << "lowered kernel must keep one CTA shared workspace";
    return failure();
  }
  Value workspace = workgroupAttributions[0];
  auto workspaceType = dyn_cast<MemRefType>(workspace.getType());
  if (!workspaceType || workspaceType.getRank() != 1 ||
      !workspaceType.getElementType().isSignlessInteger(8)) {
    gpuFunc.emitError() << "workgroup attribution must stay a rank-1 i8 workspace";
    return failure();
  }
  Attribute workspaceMemorySpace = workspaceType.getMemorySpace();
  MemRefType aSharedType = MemRefType::get(
      {contract.aSharedViewRows, contract.aSharedViewCols}, aElementType,
      AffineMap(), workspaceMemorySpace);
  MemRefType bSharedType = MemRefType::get(
      {contract.bSharedViewRows, contract.bSharedViewCols}, bElementType,
      AffineMap(), workspaceMemorySpace);
  auto aShared = createTypedMemRefViewFromByteBuffer(
      builder, loc, workspace, aSharedType,
      createIndexConstant(builder, loc, contract.aWorkspaceOffsetBytes),
      gpuFunc.getOperation());
  auto bShared = createTypedMemRefViewFromByteBuffer(
      builder, loc, workspace, bSharedType,
      createIndexConstant(builder, loc, contract.bWorkspaceOffsetBytes),
      gpuFunc.getOperation());
  if (failed(aShared) || failed(bShared))
    return failure();

  bool usesCSharedPack =
      contract.cUseSharedPackForInit || contract.cUseSharedPackForStore;
  Value cInitSharedPack;
  Value cStoreSharedPack;
  VectorType aFragType =
      getVecType(aElementType, {contract.aFragRows, contract.aFragCols});
  VectorType bFragGroupType =
      getVecType(bElementType,
                 {contract.bGroupFragRows, contract.bGroupFragCols});
  VectorType accFragType = getVecType(
      cElementType,
      {static_cast<int64_t>(accumulator.laneAccess.rowOffsets.size()),
       accumulator.packs.front().vectorWidth});
  VectorType cPackRowType =
      getVecType(cElementType, {initPlan->packs.front().vectorWidth});
  VectorType cDirectPackType =
      getVecType(cElementType,
                 {static_cast<int64_t>(initPlan->laneAccess.rowOffsets.size()),
                  initPlan->packs.front().vectorWidth});
  auto warpTileShape = getAccumulatorTileShape(encodings, gpuFunc.getOperation());
  auto mmaWarpsPerCTA = getMmaWarpsPerCTA(encodings, gpuFunc.getOperation());
  if (failed(warpTileShape) || failed(mmaWarpsPerCTA))
    return failure();
  if (usesCSharedPack) {
    if (contract.cSharedTileCols % contract.cGlobalVectorWidth != 0 ||
        contract.cSharedTileCols % accumulator.packs.front().vectorWidth != 0) {
      gpuFunc.emitError()
          << "C shared pack tile must stay divisible by both the direct-global "
             "and accumulator fragment vector widths";
      return failure();
    }
    MemRefType cSharedPackType = MemRefType::get(
        {contract.numWarps, contract.cSharedPackSlots, contract.cSharedTileRows,
         contract.cSharedTileCols},
        cElementType, AffineMap(), workspaceMemorySpace);
    if (contract.cUseSharedPackForInit) {
      auto initView = createTypedMemRefViewFromByteBuffer(
          builder, loc, workspace, cSharedPackType,
          createIndexConstant(builder, loc, contract.cInitWorkspaceOffsetBytes),
          gpuFunc.getOperation());
      if (failed(initView))
        return failure();
      cInitSharedPack = *initView;
    }
    if (contract.cUseSharedPackForStore) {
      auto storeView = createTypedMemRefViewFromByteBuffer(
          builder, loc, workspace, cSharedPackType,
          createIndexConstant(builder, loc, contract.cStoreWorkspaceOffsetBytes),
          gpuFunc.getOperation());
      if (failed(storeView))
        return failure();
      cStoreSharedPack = *storeView;
    }
  }
  Value aSharedValue = *aShared;
  Value bSharedValue = *bShared;

  Value threadsPerWarp =
      createIndexConstant(builder, loc, contract.threadsPerWarp);
  Value blockM = createIndexConstant(builder, loc, semantics.tileM);
  Value blockN = createIndexConstant(builder, loc, semantics.tileN);
  Value warpTileM = createIndexConstant(builder, loc, (*warpTileShape)[0]);
  Value warpTileN = createIndexConstant(builder, loc, (*warpTileShape)[1]);
  Value warpGridN = createIndexConstant(builder, loc, (*mmaWarpsPerCTA)[1]);

  Value threadId = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value laneId = gpu::LaneIdOp::create(builder, loc, /*upper_bound=*/nullptr);
  Value warpId = createIndexDivU(builder, loc, threadId, threadsPerWarp);
  Value warpM = createIndexDivU(builder, loc, warpId, warpGridN);
  Value warpN = createIndexRemU(builder, loc, warpId, warpGridN);
  Value warpBaseM = createIndexMul(builder, loc, warpM, warpTileM);
  Value warpBaseN = createIndexMul(builder, loc, warpN, warpTileN);

  Value laneRowA = buildOperandLaneRow(builder, loc, laneId, encodings.fragmentA);
  Value laneColA = buildOperandLaneCol(builder, loc, laneId, encodings.fragmentA);
  Value laneRowB = buildOperandLaneRow(builder, loc, laneId, encodings.fragmentB);
  Value laneColB = buildOperandLaneCol(builder, loc, laneId, encodings.fragmentB);

  DenseMap<int64_t, const PipelineOp *> opsById;
  DenseMap<int64_t, const ValueState *> valuesById;
  DenseMap<int64_t, const AsyncProducer *> cpAsyncProducerByOpId;
  for (const PipelineOp &info : model.ops)
    opsById[info.id] = &info;
  for (const ValueState &value : model.values)
    valuesById[value.id] = &value;
  for (const AsyncProducer &producer : asyncPlan.producers) {
    if (producer.kind == AsyncProducerKind::CpAsync)
      cpAsyncProducerByOpId[producer.opId] = &producer;
  }

  int64_t instructionM = encodings.fragmentA.instructionShape.front();
  int64_t instructionN = encodings.fragmentB.instructionShape.back();
  int64_t instructionK = encodings.fragmentA.instructionShape.back();
  int64_t bGroupSpan = instructionN * contract.bGroupTileSpan;
  SmallVector<int64_t, 3> mmaShape = {instructionM, instructionN, instructionK};
  auto problemRows = getMemrefDimValue(builder, loc, c, 0, "lowered epilogue");
  auto problemCols = getMemrefDimValue(builder, loc, c, 1, "lowered epilogue");
  if (failed(problemRows) || failed(problemCols))
    return failure();

  auto emitProgramInstance = [&](Value currentProgramId) -> LogicalResult {
    auto assignment = buildProgramWorkAssignment(builder, loc, currentProgramId,
                                                 contract, gpuFunc.getOperation());
    if (failed(assignment))
      return failure();
    Value ctaBaseM = createIndexMul(builder, loc, assignment->tileM, blockM);
    Value ctaBaseN = createIndexMul(builder, loc, assignment->tileN, blockN);
    Value splitKActivePart = assignment->splitKPart;

    SmallVector<Value, 32> accs(static_cast<size_t>(accumulator.packs.size()),
                                Value());
    for (Value &acc : accs)
      acc = buildZeroVector(builder, loc, accFragType);
    size_t packBatchSize = usesCSharedPack
                               ? static_cast<size_t>(contract.cSharedPackSlots)
                               : 0;
    if (contract.cLoadInitFromGlobal && contract.cUseSharedPackForInit) {
      for (size_t batchBegin = 0; batchBegin < initPlan->packs.size();
           batchBegin += packBatchSize) {
        size_t batchEnd =
            std::min(initPlan->packs.size(), batchBegin + packBatchSize);
        for (size_t packIdx = batchBegin; packIdx < batchEnd; ++packIdx) {
          const DirectGlobalVectorPlan::Pack &pack = initPlan->packs[packIdx];
          Value slotId = createIndexConstant(
              builder, loc, static_cast<int64_t>(packIdx - batchBegin));
          Value rowBase = createIndexAddConst(
              builder, loc, createIndexAdd(builder, loc, ctaBaseM, warpBaseM),
              pack.rowBase);
          Value colBase = createIndexAddConst(
              builder, loc, createIndexAdd(builder, loc, ctaBaseN, warpBaseN),
              pack.colBase);
          Value directLaneRowBase =
              buildLaneRowBase(builder, loc, laneId, initPlan->laneAccess);
          Value directLaneColBase =
              buildLaneColBase(builder, loc, laneId, initPlan->laneAccess);
          for (int64_t rowOffset : initPlan->laneAccess.rowOffsets) {
            Value row = createIndexAddConst(
                builder, loc,
                createIndexAdd(builder, loc, rowBase, directLaneRowBase),
                rowOffset);
            Value col = createIndexAdd(builder, loc, colBase, directLaneColBase);
            Value rowVector =
                emitGlobalVectorLoadRow(builder, loc, c, row, col, cPackRowType,
                                        *initPlan);
            storeDirectRowToSharedPack(builder, loc, rowVector, cInitSharedPack,
                                       laneId, warpId, slotId, *initPlan,
                                       rowOffset);
          }
        }
        emitSyncWarp(builder, loc);
        for (size_t packIdx = batchBegin; packIdx < batchEnd; ++packIdx) {
          const DirectGlobalVectorPlan::Pack &pack = initPlan->packs[packIdx];
          Value slotId = createIndexConstant(
              builder, loc, static_cast<int64_t>(packIdx - batchBegin));
          for (int64_t fragmentId : pack.fragmentIds) {
            accs[fragmentId] = loadAccumulatorFragmentFromSharedPack(
                builder, loc, cInitSharedPack, laneId, warpId, slotId,
                accumulator, accumulator.packs[fragmentId], pack, accFragType);
          }
        }
      }
      if (contract.cBarrierAfterInit)
        gpu::BarrierOp::create(builder, loc);
    } else if (contract.cLoadInitFromGlobal) {
      for (const DirectGlobalVectorPlan::Pack &pack : initPlan->packs) {
        Value rowBase = createIndexAddConst(
            builder, loc, createIndexAdd(builder, loc, ctaBaseM, warpBaseM),
            pack.rowBase);
        Value colBase = createIndexAddConst(
            builder, loc, createIndexAdd(builder, loc, ctaBaseN, warpBaseN),
            pack.colBase);
        Value packValue = materializeDirectPackFromGlobalRows(
            builder, loc, c, rowBase, colBase, laneId, *initPlan,
            cDirectPackType, cPackRowType);
        for (const auto &it : llvm::enumerate(pack.fragmentIds)) {
          accs[it.value()] = extractAccumulatorFragmentFromPack(
              builder, loc, packValue, static_cast<int64_t>(it.index()),
              accumulator.packs.front().vectorWidth, accFragType);
        }
      }
    }

    DenseMap<int64_t, Value> materializedValues;
    DenseMap<int64_t, Value> asyncGroupsById;

    auto readMaterializedValue =
        [&](int64_t valueId, StringRef role) -> FailureOr<Value> {
      auto it = materializedValues.find(valueId);
      if (it == materializedValues.end()) {
        gpuFunc.emitError() << role << " value " << valueId
                            << " has not been materialized yet";
        return failure();
      }
      return it->second;
    };

    auto emitClusterWaits =
        [&](const ExplicitPipelineCluster &cluster) -> LogicalResult {
      for (int64_t groupId : cluster.waitGroupIds) {
        auto tokenIt = asyncGroupsById.find(groupId);
        if (tokenIt == asyncGroupsById.end()) {
          gpuFunc.emitError() << "missing async group token for waited group "
                              << groupId;
          return failure();
        }
        nvgpu::DeviceAsyncWaitOp::create(builder, loc, tokenIt->second,
                                         IntegerAttr());
      }
      if (cluster.needsBarrier)
        gpu::BarrierOp::create(builder, loc);
      return success();
    };

    for (const ExplicitPipelineCluster &cluster : explicitClusters) {
      if (failed(emitClusterWaits(cluster)))
        return failure();
      Value splitKGroupActive = buildSplitKGroupActivePredicate(
          builder, loc, splitKActivePart, cluster.kGroup, contract);

      for (int64_t opId : cluster.opIds) {
        const PipelineOp *info = opsById.lookup(opId);
        if (!info) {
          gpuFunc.emitError() << "missing pipeline op for expanded cluster op "
                              << opId;
          return failure();
        }

        switch (info->kind) {
        case BufferOpKind::LoadA: {
          if (!info->inputs.empty() || info->outputs.size() != 1) {
            gpuFunc.emitError() << "load_a op " << info->id
                                << " must have zero inputs and exactly one output";
            return failure();
          }
          auto producerIt = cpAsyncProducerByOpId.find(info->id);
          if (producerIt == cpAsyncProducerByOpId.end()) {
            gpuFunc.emitError()
                << "missing cp.async producer contract for load_a op "
                << info->id;
            return failure();
          }
          auto dstInfo =
              resolveSharedView(model, producerIt->second->dstView,
                                gpuFunc.getOperation());
          if (failed(dstInfo))
            return failure();

          SmallVector<Value, 16> copyTokens;
          appendAsyncCopiesForSlice(
              builder, loc, threadId, a, aSharedValue,
              ArrayRef<int64_t>{dstInfo->rows, dstInfo->cols},
              createIndexAddConst(builder, loc, ctaBaseM,
                                  producerIt->second->srcOffsets[0]),
              createIndexConstant(builder, loc, producerIt->second->srcOffsets[1]),
              dstInfo->rowBase, dstInfo->colBase,
              spec.numWarps * contract.threadsPerWarp, contract.aAsyncCopyElems,
              semantics.problemM, semantics.problemK,
              producerIt->second->predicated || producerIt->second->zeroFill,
              producerIt->second->bypassL1, contract.aSharedLayout,
              splitKGroupActive, copyTokens);
          asyncGroupsById[producerIt->second->groupId] =
              createAsyncGroup(builder, loc, copyTokens);
          break;
        }
        case BufferOpKind::LoadB: {
          if (!info->inputs.empty() || info->outputs.size() != 1) {
            gpuFunc.emitError() << "load_b op " << info->id
                                << " must have zero inputs and exactly one output";
            return failure();
          }
          auto producerIt = cpAsyncProducerByOpId.find(info->id);
          if (producerIt == cpAsyncProducerByOpId.end()) {
            gpuFunc.emitError()
                << "missing cp.async producer contract for load_b op "
                << info->id;
            return failure();
          }
          auto dstInfo =
              resolveSharedView(model, producerIt->second->dstView,
                                gpuFunc.getOperation());
          if (failed(dstInfo))
            return failure();

          SmallVector<Value, 16> copyTokens;
          appendAsyncCopiesForSlice(
              builder, loc, threadId, b, bSharedValue,
              ArrayRef<int64_t>{dstInfo->rows, dstInfo->cols},
              createIndexConstant(builder, loc, producerIt->second->srcOffsets[0]),
              createIndexAddConst(builder, loc, ctaBaseN,
                                  producerIt->second->srcOffsets[1]),
              dstInfo->rowBase, dstInfo->colBase,
              spec.numWarps * contract.threadsPerWarp, contract.bAsyncCopyElems,
              semantics.problemK, semantics.problemN,
              producerIt->second->predicated || producerIt->second->zeroFill,
              producerIt->second->bypassL1, contract.bSharedLayout,
              splitKGroupActive, copyTokens);
          asyncGroupsById[producerIt->second->groupId] =
              createAsyncGroup(builder, loc, copyTokens);
          break;
        }
        case BufferOpKind::LocalLoadA: {
          if (info->inputs.size() != 1 || info->outputs.size() != 1) {
            gpuFunc.emitError() << "local_load_a op " << info->id
                                << " must have one input and one output";
            return failure();
          }
          auto srcValueIt = valuesById.find(info->inputs.front());
          if (srcValueIt == valuesById.end()) {
            gpuFunc.emitError() << "local_load_a op " << info->id
                                << " references unknown input value";
            return failure();
          }
          auto sharedView =
              resolveSharedView(model, srcValueIt->second->ownerView,
                                gpuFunc.getOperation());
          if (failed(sharedView))
            return failure();
          if (sharedView->backing != contract.aSharedBacking) {
            gpuFunc.emitError()
                << "local_load_a op " << info->id
                << " must read from the shared operand A backing";
            return failure();
          }
          int64_t mTile = findIterationCoord(info->iterationCoords, "m_tile");

          Value tileBaseM =
              createIndexAddConst(builder, loc, warpBaseM,
                                  sharedView->rowBase + mTile * instructionM);
          materializedValues[info->outputs.front()] =
              loadOperandFragment(builder, loc, aSharedValue, tileBaseM,
                                  createIndexConstant(builder, loc,
                                                      sharedView->colBase),
                                  laneRowA, laneColA, aFragType,
                                  encodings.fragmentA, contract.aSharedLayout);
          break;
        }
        case BufferOpKind::LocalLoadB: {
          if (info->inputs.size() != 1 || info->outputs.size() != 1) {
            gpuFunc.emitError() << "local_load_b op " << info->id
                                << " must have one input and one output";
            return failure();
          }
          auto srcValueIt = valuesById.find(info->inputs.front());
          if (srcValueIt == valuesById.end()) {
            gpuFunc.emitError() << "local_load_b op " << info->id
                                << " references unknown input value";
            return failure();
          }
          auto sharedView =
              resolveSharedView(model, srcValueIt->second->ownerView,
                                gpuFunc.getOperation());
          if (failed(sharedView))
            return failure();
          if (sharedView->backing != contract.bSharedBacking) {
            gpuFunc.emitError()
                << "local_load_b op " << info->id
                << " must read from the shared operand B backing";
            return failure();
          }
          int64_t nGroup = findIterationCoord(info->iterationCoords, "n_group");

          Value tileBaseN = createIndexAddConst(builder, loc, warpBaseN,
                                                sharedView->colBase +
                                                    nGroup * bGroupSpan);
          materializedValues[info->outputs.front()] = loadOperandFragment(
              builder, loc, bSharedValue,
              createIndexConstant(builder, loc, sharedView->rowBase), tileBaseN,
              laneRowB, laneColB, bFragGroupType, encodings.fragmentB,
              contract.bSharedLayout);
          break;
        }
        case BufferOpKind::Mma: {
          int64_t accIndex = findIterationCoord(info->iterationCoords, "acc_index");
          int64_t groupOffset =
              findIterationCoord(info->iterationCoords, "group_offset");
          if (info->inputs.size() < 2 || info->inputs.size() > 3 ||
              info->outputs.size() != 1 || accIndex < 0 ||
              accIndex >= static_cast<int64_t>(accs.size())) {
            gpuFunc.emitError() << "mma op " << info->id
                                << " is inconsistent with the lowering contract";
            return failure();
          }

          auto aFrag = readMaterializedValue(info->inputs[0], "A fragment");
          auto bGroup =
              readMaterializedValue(info->inputs[1], "B fragment group");
          if (failed(aFrag) || failed(bGroup))
            return failure();

          if (groupOffset < 0 || groupOffset >= contract.bGroupTileSpan) {
            gpuFunc.emitError()
                << "mma op " << info->id
                << " must reference a valid B fragment-group offset";
            return failure();
          }
          ArrayRef<int64_t> bGroupShape = bFragGroupType.getShape();
          Value bFrag = vector::ExtractStridedSliceOp::create(
              builder, loc, *bGroup,
              ArrayRef<int64_t>{groupOffset * contract.bSubFragRows, 0},
              ArrayRef<int64_t>{contract.bSubFragRows, bGroupShape[1]},
              ArrayRef<int64_t>{1, 1});

          Value acc = accs[accIndex];
          if (info->inputs.size() >= 3) {
            auto carriedAcc =
                readMaterializedValue(info->inputs[2], "accumulator");
            if (failed(carriedAcc))
              return failure();
            acc = *carriedAcc;
          }

          Value nextAcc = nvgpu::MmaSyncOp::create(builder, loc, *aFrag, bFrag,
                                                   acc, mmaShape);
          materializedValues[info->outputs.front()] = nextAcc;
          accs[accIndex] = nextAcc;
          break;
        }
        default:
          gpuFunc.emitError() << "unsupported pipeline op " << info->id;
          return failure();
        }
      }
    }

    if (contract.cUseSharedPackForStore) {
      if (contract.cBarrierBeforeStore)
        gpu::BarrierOp::create(builder, loc);
      for (size_t batchBegin = 0; batchBegin < storePlan->packs.size();
           batchBegin += packBatchSize) {
        size_t batchEnd =
            std::min(storePlan->packs.size(), batchBegin + packBatchSize);
        for (size_t packIdx = batchBegin; packIdx < batchEnd; ++packIdx) {
          const DirectGlobalVectorPlan::Pack &pack = storePlan->packs[packIdx];
          Value slotId = createIndexConstant(
              builder, loc, static_cast<int64_t>(packIdx - batchBegin));
          for (int64_t fragmentId : pack.fragmentIds) {
            storeAccumulatorFragmentToSharedPack(
                builder, loc, accs[fragmentId], cStoreSharedPack, laneId, warpId,
                slotId, accumulator, accumulator.packs[fragmentId], pack);
          }
        }
        emitSyncWarp(builder, loc);

        for (size_t packIdx = batchBegin; packIdx < batchEnd; ++packIdx) {
          const DirectGlobalVectorPlan::Pack &pack = storePlan->packs[packIdx];
          Value slotId = createIndexConstant(
              builder, loc, static_cast<int64_t>(packIdx - batchBegin));
          Value rowBase = createIndexAddConst(
              builder, loc, createIndexAdd(builder, loc, ctaBaseM, warpBaseM),
              pack.rowBase);
          Value colBase = createIndexAddConst(
              builder, loc, createIndexAdd(builder, loc, ctaBaseN, warpBaseN),
              pack.colBase);
          Value directLaneRowBase =
              buildLaneRowBase(builder, loc, laneId, storePlan->laneAccess);
          Value directLaneColBase =
              buildLaneColBase(builder, loc, laneId, storePlan->laneAccess);
          for (int64_t rowOffset : storePlan->laneAccess.rowOffsets) {
            Value rowVector = loadDirectRowFromSharedPack(
                builder, loc, cStoreSharedPack, laneId, warpId, slotId,
                *storePlan, rowOffset, cPackRowType);
            Value row = createIndexAddConst(
                builder, loc,
                createIndexAdd(builder, loc, rowBase, directLaneRowBase),
                rowOffset);
            Value col = createIndexAdd(builder, loc, colBase, directLaneColBase);
            if (contract.cStoreViaAtomicAdd) {
              emitAtomicAddRowToGlobal(builder, loc, rowVector, c, row, col,
                                       *storePlan, *problemRows, *problemCols);
            } else {
              emitGlobalVectorStoreRow(builder, loc, rowVector, c, row, col,
                                       *storePlan);
            }
          }
        }
      }
    } else {
      for (const DirectGlobalVectorPlan::Pack &pack : storePlan->packs) {
        SmallVector<Value, 4> fragmentValues;
        fragmentValues.reserve(pack.fragmentIds.size());
        for (int64_t fragmentId : pack.fragmentIds)
          fragmentValues.push_back(accs[fragmentId]);
        Value packValue = assembleDirectPackFromFragments(
            builder, loc, fragmentValues, accumulator.packs.front().vectorWidth,
            cDirectPackType);
        Value rowBase = createIndexAddConst(
            builder, loc, createIndexAdd(builder, loc, ctaBaseM, warpBaseM),
            pack.rowBase);
        Value colBase = createIndexAddConst(
            builder, loc, createIndexAdd(builder, loc, ctaBaseN, warpBaseN),
            pack.colBase);
        if (contract.cStoreViaAtomicAdd) {
          storeDirectPackToGlobalRowsAtomicAdd(builder, loc, packValue, c,
                                               rowBase, colBase, laneId,
                                               *storePlan, *problemRows,
                                               *problemCols);
        } else {
          storeDirectPackToGlobalRows(builder, loc, packValue, c, rowBase,
                                      colBase, laneId, *storePlan);
        }
      }
    }
    return success();
  };

  if (contract.persistentEnabled) {
    Value startProgram = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
    Value totalPrograms = createIndexConstant(builder, loc, contract.totalPrograms);
    Value loopStep =
        createIndexConstant(builder, loc, contract.persistentResidentPrograms);
    auto forOp = scf::ForOp::create(builder, loc, startProgram, totalPrograms,
                                    loopStep);
    builder.setInsertionPointToStart(forOp.getBody());
    if (failed(emitProgramInstance(forOp.getInductionVar())))
      return failure();
    gpu::BarrierOp::create(builder, loc);
    builder.setInsertionPointAfter(forOp);
  } else {
    Value programId = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
    if (failed(emitProgramInstance(programId)))
      return failure();
  }

  gpu::ReturnOp::create(builder, loc);
  return success();
}

class TBLowerPipelineToNVGPU
    : public mlir::tb::impl::TBLowerPipelineToNVGPUBase<
          TBLowerPipelineToNVGPU> {
public:
  using mlir::tb::impl::TBLowerPipelineToNVGPUBase<
      TBLowerPipelineToNVGPU>::TBLowerPipelineToNVGPUBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, gpu::GPUDialect,
                    memref::MemRefDialect, nvgpu::NVGPUDialect,
                    NVVM::NVVMDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    SmallVector<Operation *> toErase;
    int64_t kernelOrdinal = 0;
    bool hadFailure = false;

    module.walk([&](PipelineMainlineOp op) {
      auto contract = parseKernelContract(op.getOperation());
      if (failed(contract)) {
        hadFailure = true;
        return;
      }
      auto explicitClusters = collectExplicitPipelineClusters(op);
      if (failed(explicitClusters)) {
        hadFailure = true;
        return;
      }

      auto lowering =
          validateLoweringContract(contract->kernel, contract->target,
                                   contract->semantics,
                                   contract->programMapping,
                                   contract->reduction,
                                   contract->persistentWork,
                                   contract->pipelineReady,
                                   contract->encodings, contract->transport,
                                   contract->rewrite,
                                   contract->accumulator,
                                   contract->epilogue,
                                   contract->epilogueReorder,
                                   contract->sharedWorkspace,
                                   contract->warpDecomposition,
                                   contract->resourceClosure,
                                   contract->buffers,
                                   *explicitClusters, contract->async,
                                   op.getOperation());
      if (failed(lowering)) {
        hadFailure = true;
        return;
      }
      std::string stem = makeKernelStem(op.getOperation(), kernelOrdinal++);
      std::string moduleName = stem + "_module";
      std::string kernelName = stem;

      gpu::GPUModuleOp gpuModule = createGpuModule(builder, module, moduleName);

      auto workgroupAddressSpace = gpu::AddressSpaceAttr::get(
          builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
      auto sharedABacking =
          findUniqueBacking(contract->buffers, BufferRole::OperandA,
                            MemorySpace::Shared, op.getOperation(),
                            "kernel_shared_operand_a");
      auto sharedBBacking =
          findUniqueBacking(contract->buffers, BufferRole::OperandB,
                            MemorySpace::Shared, op.getOperation(),
                            "kernel_shared_operand_b");
      if (failed(sharedABacking) || failed(sharedBBacking)) {
        hadFailure = true;
        return;
      }
      if ((*sharedABacking)->id != lowering->aSharedBacking ||
          (*sharedBBacking)->id != lowering->bSharedBacking) {
        op.emitError() << "shared operand backing ids disagree with the "
                          "validated lowering contract";
        hadFailure = true;
        return;
      }
      auto aType = dyn_cast<MemRefType>(op.getA().getType());
      auto bType = dyn_cast<MemRefType>(op.getB().getType());
      auto cType = dyn_cast<MemRefType>(op.getC().getType());
      if (!aType || !bType || !cType) {
        op.emitError() << "pipeline_mainline operands must remain memrefs "
                          "through lowering";
        hadFailure = true;
        return;
      }
      MemRefType workspaceType = MemRefType::get(
          {lowering->workspaceTotalBytes}, builder.getI8Type(), AffineMap(),
          workgroupAddressSpace);
      SmallVector<Type, 1> workgroupTypes{workspaceType};

      builder.setInsertionPointToStart(gpuModule.getBody());
      auto gpuFunc = gpu::GPUFuncOp::create(
          builder, op.getLoc(), kernelName,
          builder.getFunctionType(TypeRange{aType, bType, cType}, TypeRange{}),
          TypeRange(workgroupTypes), TypeRange{});
      gpuFunc->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       builder.getUnitAttr());

      if (failed(emitKernelBody(builder, gpuFunc, contract->kernel,
                                contract->semantics,
                                contract->programMapping, contract->encodings,
                                contract->accumulator, contract->epilogue,
                                contract->buffers, *explicitClusters,
                                contract->async, *lowering))) {
        hadFailure = true;
        gpuModule.erase();
        return;
      }
      toErase.push_back(op.getOperation());
    });

    for (Operation *op : toErase)
      op->erase();

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace
