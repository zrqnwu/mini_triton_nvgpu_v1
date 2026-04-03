#ifndef MINI_TRITON_TB_ANALYSIS_TRANSPORTPLAN_H
#define MINI_TRITON_TB_ANALYSIS_TRANSPORTPLAN_H

#include "tb/Analysis/EncodingPlan.h"
#include "tb/Analysis/TargetInfo.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include <string>

namespace mlir::tb {

struct TransportSpec {
  std::string role;
  int64_t dstEncoding = -1;
  std::string kind;
  int64_t vectorBytes = 0;
  int64_t asyncVectorBytes = 0;
  int64_t transactionBytes = 0;
  bool asyncEligible = false;
  bool bypassL1 = false;
  std::string cachePolicy;
};

struct TransportPlan {
  std::string contractModel;
  TransportSpec operandA;
  TransportSpec operandB;
};

FailureOr<TransportPlan> deriveTransportPlan(const TargetInfo &target,
                                             const EncodingPlan &encodings,
                                             Operation *op);
DictionaryAttr buildTransportPlanAttr(Builder &builder,
                                      const TransportPlan &plan);
FailureOr<TransportPlan> parseTransportPlanAttr(Operation *op);
FailureOr<const TransportSpec *>
getTransportSpecForDstEncoding(const TransportPlan &plan, int64_t dstEncoding,
                               Operation *op, StringRef role);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_TRANSPORTPLAN_H
