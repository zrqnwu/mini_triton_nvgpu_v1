#include "tb/Analysis/KernelContract.h"

#include "tb/Analysis/KernelConfig.h"
#include "tb/IR/TBOps.h"

using namespace mlir;
using namespace mlir::tb;

FailureOr<KernelContract> mlir::tb::parseKernelContract(Operation *op) {
  auto pipelineMainline = dyn_cast<PipelineMainlineOp>(op);
  if (!pipelineMainline) {
    op->emitError()
        << "kernel contract can only be parsed from tb.pipeline_mainline";
    return failure();
  }

  auto kernel = getKernelConfig(pipelineMainline);
  auto semantics = parseMatmulSemanticsAttr(op);
  auto programMapping = parseProgramMappingPlanAttr(op);
  auto encodings = parseEncodingPlanAttr(op);
  auto transport = parseTransportPlanAttr(op);
  auto rewrite = parseMatmulRewritePlanAttr(op);
  auto accumulator = parseAccumulatorPlanAttr(op);
  auto epilogue = parseEpiloguePlanAttr(op);
  auto buffers = parseBufferModelAttr(op);
  auto pipelineReady = parsePipelineReadyAttr(op);
  auto async = parseAsyncPlanAttr(op);
  auto target = getTargetInfo(op);
  auto warpDecomposition = parseWarpDecompositionPlanAttr(op);
  auto resourceClosure = parseResourceClosurePlanAttr(op);
  if (failed(kernel) || failed(semantics) || failed(programMapping) ||
      failed(encodings) || failed(transport) || failed(rewrite) ||
      failed(accumulator) || failed(epilogue) || failed(buffers) ||
      failed(pipelineReady) || failed(async) || failed(target) ||
      failed(warpDecomposition) || failed(resourceClosure)) {
    return failure();
  }

  KernelContract contract;
  contract.kernel = *kernel;
  contract.semantics = std::move(*semantics);
  contract.target = *target;
  contract.programMapping = std::move(*programMapping);
  contract.encodings = std::move(*encodings);
  contract.transport = std::move(*transport);
  contract.rewrite = std::move(*rewrite);
  contract.buffers = std::move(*buffers);
  contract.pipelineReady = std::move(*pipelineReady);
  contract.async = std::move(*async);
  contract.accumulator = std::move(*accumulator);
  contract.epilogue = std::move(*epilogue);
  contract.warpDecomposition = std::move(*warpDecomposition);
  contract.resourceClosure = std::move(*resourceClosure);
  return contract;
}
