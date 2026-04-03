#ifndef MINI_TRITON_TB_ANALYSIS_KERNELCONTRACT_H
#define MINI_TRITON_TB_ANALYSIS_KERNELCONTRACT_H

#include "tb/Analysis/AccumulatorPlan.h"
#include "tb/Analysis/AsyncPlan.h"
#include "tb/Analysis/BufferModel.h"
#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/EpiloguePlan.h"
#include "tb/Analysis/KernelConfig.h"
#include "tb/Analysis/MatmulSemantics.h"
#include "tb/Analysis/MatmulRewritePlan.h"
#include "tb/Analysis/PipelineExpansion.h"
#include "tb/Analysis/PipelinePlan.h"
#include "tb/Analysis/PipelineReady.h"
#include "tb/Analysis/ProgramMappingPlan.h"
#include "tb/Analysis/ResourceClosurePlan.h"
#include "tb/Analysis/TargetInfo.h"
#include "tb/Analysis/TransportPlan.h"
#include "tb/Analysis/WarpDecompositionPlan.h"

namespace mlir {
class Operation;
}

namespace mlir::tb {

struct KernelContract {
  KernelConfig kernel;
  MatmulSemantics semantics;
  TargetInfo target;
  ProgramMappingPlan programMapping;
  EncodingPlan encodings;
  TransportPlan transport;
  MatmulRewritePlan rewrite;
  BufferModel buffers;
  PipelineReady pipelineReady;
  AsyncPlan async;
  AccumulatorPlan accumulator;
  EpiloguePlan epilogue;
  WarpDecompositionPlan warpDecomposition;
  ResourceClosurePlan resourceClosure;
};

FailureOr<KernelContract> parseKernelContract(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_KERNELCONTRACT_H
