#include "tb/IR/TBOps.h"
#include "tb/Analysis/TargetInfo.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::tb;

namespace {

static LogicalResult verifyRankedMatrix(Value value, StringRef name,
                                        Operation *op) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type) {
    return op->emitOpError() << "expected `" << name << "` to be a memref";
  }
  if (type.getRank() != 2) {
    return op->emitOpError() << "expected `" << name << "` to be a rank-2 memref";
  }
  return success();
}

static bool knownDimMismatch(int64_t lhs, int64_t rhs) {
  return !ShapedType::isDynamic(lhs) && !ShapedType::isDynamic(rhs) && lhs != rhs;
}

template <typename KernelLikeOp>
static LogicalResult verifyKernelOperands(KernelLikeOp op) {
  if (failed(verifyRankedMatrix(op.getA(), "a", op.getOperation())) ||
      failed(verifyRankedMatrix(op.getB(), "b", op.getOperation())) ||
      failed(verifyRankedMatrix(op.getC(), "c", op.getOperation()))) {
    return failure();
  }

  auto aType = cast<MemRefType>(op.getA().getType());
  auto bType = cast<MemRefType>(op.getB().getType());
  auto cType = cast<MemRefType>(op.getC().getType());

  if (!aType.getElementType().isF16() || !bType.getElementType().isF16() ||
      !cType.getElementType().isF32()) {
    return op.emitOpError()
           << "expected fp16 x fp16 -> fp32 operand element types";
  }

  auto aShape = aType.getShape();
  auto bShape = bType.getShape();
  auto cShape = cType.getShape();
  if (knownDimMismatch(aShape[1], bShape[0]) || knownDimMismatch(aShape[0], cShape[0]) ||
      knownDimMismatch(bShape[1], cShape[1])) {
    return op.emitOpError()
           << "expected A:[M,K], B:[K,N], C:[M,N] operand shapes";
  }

  return success();
}

static LogicalResult verifyClusterMetadata(Operation *op, int64_t ordinal,
                                           int64_t stage, int64_t cluster,
                                           int64_t kGroup,
                                           ArrayRef<int64_t> opIds) {
  if (ordinal < 0 || stage < 0 || cluster < 0 || kGroup < 0)
    return op->emitOpError() << "cluster metadata must be non-negative";
  if (opIds.empty())
    return op->emitOpError() << "cluster must carry non-empty op_ids";
  return success();
}

static LogicalResult requireAttr(Operation *op, StringRef name) {
  if (!op->hasAttr(name))
    return op->emitOpError() << "requires `" << name << "` to be present";
  return success();
}

static LogicalResult rejectAttr(Operation *op, StringRef name) {
  if (op->hasAttr(name)) {
    return op->emitOpError()
           << "must not retain transient attr `" << name
           << "` after post-pipeline cleanup";
  }
  return success();
}

static LogicalResult requireModuleAttr(Operation *op, StringRef name) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return op->emitOpError() << "must live inside a module";
  if (!module->hasAttr(name))
    return op->emitOpError() << "requires module attr `" << name << "`";
  return success();
}

static LogicalResult verifyEpilogueVectorAccess(Value memrefValue, Type vectorTy,
                                                int64_t vectorWidth,
                                                Operation *op,
                                                StringRef role) {
  auto memrefType = dyn_cast<MemRefType>(memrefValue.getType());
  if (!memrefType || memrefType.getRank() != 2) {
    return op->emitOpError()
           << role << " requires a rank-2 memref operand";
  }
  auto vectorType = dyn_cast<VectorType>(vectorTy);
  if (!vectorType || vectorType.getRank() != 1) {
    return op->emitOpError()
           << role << " requires a rank-1 vector payload";
  }
  if (vectorWidth <= 0 || vectorType.getShape().front() != vectorWidth) {
    return op->emitOpError()
           << role << " vector_width must match the 1-D vector payload width";
  }
  if (memrefType.getElementType() != vectorType.getElementType()) {
    return op->emitOpError()
           << role << " memref/vector element types must match";
  }
  if (!memrefType.getElementType().isIntOrFloat()) {
    return op->emitOpError()
           << role << " only supports integer/float element types";
  }
  return success();
}

} // namespace

LogicalResult ConvertLayoutOp::verify() {
  auto srcType = dyn_cast<MemDescType>(getSource().getType());
  auto dstType = dyn_cast<MemDescType>(getResult().getType());
  if (!srcType || !dstType) {
    return emitOpError()
           << "requires !tb.memdesc source/result types for explicit layout "
              "ownership transfer";
  }
  if (srcType.getShape() != dstType.getShape()) {
    return emitOpError()
           << "source/result logical shapes must match for layout conversion";
  }
  if (srcType.getElementType() != dstType.getElementType()) {
    return emitOpError()
           << "source/result element types must match for layout conversion";
  }
  if (srcType.getMemorySpace() != dstType.getMemorySpace()) {
    return emitOpError()
           << "source/result memory spaces must match for layout conversion";
  }
  if (srcType.getMutableMemory() != dstType.getMutableMemory()) {
    return emitOpError() << "source/result mutability must match for layout "
                            "conversion";
  }
  if (!srcType.getEncoding() || !dstType.getEncoding()) {
    return emitOpError()
           << "layout conversion requires concrete source/result encodings";
  }
  if (srcType.getEncoding() == dstType.getEncoding() &&
      srcType.getAllocShape() == dstType.getAllocShape()) {
    return emitOpError()
           << "layout conversion must change encoding or alloc_shape";
  }
  return success();
}

OpFoldResult ConvertLayoutOp::fold(FoldAdaptor adaptor) {
  (void)adaptor;
  Value source = getSource();
  if (source.getType() == getResult().getType())
    return source;
  if (auto prev = source.getDefiningOp<ConvertLayoutOp>()) {
    if (prev.getSource().getType() == getResult().getType())
      return prev.getSource();
  }
  return {};
}

LogicalResult MatmulOp::verify() {
  if (failed(verifyKernelOperands(*this)))
    return failure();
  if (getBlockM() <= 0 || getBlockN() <= 0 || getBlockK() <= 0)
    return emitOpError() << "block sizes must be positive";
  if (getGroupM() <= 0)
    return emitOpError() << "group_m must be positive";
  return success();
}

LogicalResult PipelineMainlineOp::verify() {
  if (failed(verifyKernelOperands(*this)))
    return failure();
  for (StringRef attrName : {"tb.semantic_matmul",
                             "tb.program_mapping_plan",
                             "tb.reduction_plan",
                             "tb.persistent_work_plan",
                             "tb.encoding_plan", "tb.transport_plan",
                             "tb.accumulator_plan",
                             "tb.matmul_rewrite",
                             "tb.epilogue_plan", "tb.buffer_model",
                             "tb.async_plan", "tb.pipeline_ready"}) {
    if (failed(requireAttr(getOperation(), attrName)))
      return failure();
  }
  for (StringRef attrName : {"tb.target", "tb.num-warps",
                             "tb.threads-per-warp", "tb.num-ctas",
                             "tb.requested-stages"}) {
    if (failed(requireModuleAttr(getOperation(), attrName)))
      return failure();
  }
  for (StringRef attrName : {"tb.target_info", "tb.layout_plan",
                             "tb.c_register_plan", "tb.mainloop_graph",
                             "tb.schedule_plan", "tb.wait_plan",
                             "tb.loop_plan", "tb.latency_plan",
                             "tb.pipeline_plan", "tb.pipeline_expansion"}) {
    if (failed(rejectAttr(getOperation(), attrName)))
      return failure();
  }
  if (getBody().empty())
    return emitOpError() << "pipeline_mainline must carry a non-empty body";
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError() << "pipeline_mainline currently expects one body block";
  if (getBody().front().empty())
    return emitOpError() << "pipeline_mainline body must not be empty";
  int64_t expectedOrdinal = 0;
  int64_t previousStage = -1;
  int64_t previousCluster = -1;
  llvm::DenseSet<int64_t> seenKGroups;
  int64_t maxKGroup = -1;
  for (Operation &nestedOp : getBody().front()) {
    int64_t ordinal = -1;
    int64_t stage = -1;
    int64_t cluster = -1;
    int64_t kGroup = -1;
    if (!isa<AsyncIssueClusterOp, ConsumerWaitClusterOp, MmaComputeClusterOp>(
            nestedOp)) {
      return emitOpError()
             << "pipeline_mainline body may only contain explicit pipeline "
                "cluster ops";
    }
    if (auto async = dyn_cast<AsyncIssueClusterOp>(nestedOp)) {
      ordinal = async.getOrdinal();
      stage = async.getStage();
      cluster = async.getCluster();
      kGroup = async.getKGroup();
    } else if (auto wait = dyn_cast<ConsumerWaitClusterOp>(nestedOp)) {
      ordinal = wait.getOrdinal();
      stage = wait.getStage();
      cluster = wait.getCluster();
      kGroup = wait.getKGroup();
    } else if (auto mma = dyn_cast<MmaComputeClusterOp>(nestedOp)) {
      ordinal = mma.getOrdinal();
      stage = mma.getStage();
      cluster = mma.getCluster();
      kGroup = mma.getKGroup();
    }
    if (ordinal != expectedOrdinal) {
      return emitOpError()
             << "pipeline_mainline cluster ordinals must be contiguous from zero";
    }
    if (stage < previousStage ||
        (stage == previousStage && cluster < previousCluster)) {
      return emitOpError()
             << "pipeline_mainline clusters must stay ordered by stage/cluster";
    }
    ++expectedOrdinal;
    previousStage = stage;
    previousCluster = cluster;
    seenKGroups.insert(kGroup);
    maxKGroup = std::max(maxKGroup, kGroup);
  }
  for (int64_t kGroup = 0; kGroup <= maxKGroup; ++kGroup) {
    if (!seenKGroups.count(kGroup)) {
      return emitOpError()
             << "pipeline_mainline must carry contiguous explicit k_group "
                "coverage after cleanup";
    }
  }
  return success();
}

LogicalResult AsyncIssueClusterOp::verify() {
  return verifyClusterMetadata(getOperation(), getOrdinal(), getStage(),
                               getCluster(), getKGroup(), getOpIds());
}

LogicalResult ConsumerWaitClusterOp::verify() {
  if (failed(verifyClusterMetadata(getOperation(), getOrdinal(), getStage(),
                                   getCluster(), getKGroup(), getOpIds()))) {
    return failure();
  }
  if (getWaitGroupIds().empty()) {
    return emitOpError()
           << "consumer_wait_cluster must carry non-empty wait_group_ids";
  }
  if (!getNeedsBarrier()) {
    return emitOpError() << "consumer_wait_cluster must own the CTA barrier";
  }
  return success();
}

LogicalResult MmaComputeClusterOp::verify() {
  return verifyClusterMetadata(getOperation(), getOrdinal(), getStage(),
                               getCluster(), getKGroup(), getOpIds());
}

LogicalResult EpilogueGlobalVectorLoadOp::verify() {
  if (failed(verifyEpilogueVectorAccess(getSource(), getResult().getType(),
                                        getVectorWidth(), getOperation(),
                                        "load"))) {
    return failure();
  }
  if (getScalarTail() && !getBoundaryAware()) {
    return emitOpError()
           << "scalar_tail only makes sense on a boundary-aware load";
  }
  return success();
}

LogicalResult EpilogueGlobalVectorStoreOp::verify() {
  if (failed(verifyEpilogueVectorAccess(getDest(), getValue().getType(),
                                        getVectorWidth(), getOperation(),
                                        "store"))) {
    return failure();
  }
  if (getScalarTail() && !getBoundaryAware()) {
    return emitOpError()
           << "scalar_tail only makes sense on a boundary-aware store";
  }
  return success();
}

#define GET_OP_CLASSES
#include "tb/IR/TBOps.cpp.inc"
