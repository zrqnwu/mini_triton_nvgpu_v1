#include "tb/Analysis/TransportPlan.h"

#include "tb/Analysis/BufferModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::tb;

namespace {

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

static LogicalResult validateTransportSpec(const TransportSpec &spec,
                                           Operation *op, StringRef role) {
  if (spec.role.empty())
    return op->emitError() << role << " transport spec must carry a role";
  if (spec.dstEncoding < 0)
    return op->emitError() << role << " transport spec requires a dst encoding";
  if (spec.kind.empty())
    return op->emitError() << role << " transport spec must carry a kind";
  if (spec.vectorBytes <= 0)
    return op->emitError() << role
                           << " transport vector bytes must be positive";
  if (spec.asyncEligible) {
    if (spec.kind == "sync_copy_fallback") {
      return op->emitError()
             << role << " async transport must not use sync_copy_fallback";
    }
    if (spec.asyncVectorBytes <= 0 || spec.transactionBytes <= 0) {
      return op->emitError() << role
                             << " async transport requires positive async/"
                                "transaction bytes";
    }
    if (spec.bypassL1 && spec.cachePolicy != "bypass_l1") {
      return op->emitError()
             << role << " bypass_l1 transport must carry `bypass_l1` policy";
    }
    if (!spec.bypassL1 && spec.cachePolicy == "bypass_l1") {
      return op->emitError()
             << role
             << " non-bypass transport must not claim `bypass_l1` policy";
    }
    if (spec.kind == "cp_async" &&
        spec.transactionBytes != spec.asyncVectorBytes) {
      return op->emitError()
             << role << " cp_async transaction bytes must match "
                        "async_vector_bytes";
    }
  } else {
    if (spec.kind != "sync_copy_fallback") {
      return op->emitError()
             << role
             << " non-async transport must use `sync_copy_fallback` kind";
    }
    if (spec.asyncVectorBytes != 0)
      return op->emitError() << role
                             << " non-async transport must not carry "
                                "async_vector_bytes";
    if (spec.transactionBytes != spec.vectorBytes) {
      return op->emitError() << role
                             << " non-async transport transaction bytes must "
                                "match vector bytes";
    }
  }
  return success();
}

static FailureOr<int64_t> getOperandElementBytes(Operation *op, StringRef role) {
  int operandIndex = -1;
  if (role == "a_global_to_shared")
    operandIndex = 0;
  else if (role == "b_global_to_shared")
    operandIndex = 1;
  else {
    op->emitError() << "unknown transport role `" << role << "`";
    return failure();
  }

  if (operandIndex >= op->getNumOperands()) {
    op->emitError() << "transport role `" << role
                    << "` is missing its source operand";
    return failure();
  }

  auto memrefType = dyn_cast<MemRefType>(op->getOperand(operandIndex).getType());
  if (!memrefType) {
    op->emitError() << "transport role `" << role
                    << "` expects a memref source operand";
    return failure();
  }

  Type elementType = memrefType.getElementType();
  if (!elementType.isIntOrFloat()) {
    op->emitError() << "transport role `" << role
                    << "` expects integer or float element types";
    return failure();
  }
  int64_t bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth <= 0 || bitWidth % 8 != 0) {
    op->emitError() << "transport role `" << role
                    << "` must have byte-addressable element types";
    return failure();
  }
  return bitWidth / 8;
}

static FailureOr<int64_t>
chooseAsyncVectorBytes(const TargetInfo &target, const EncodingPlan &encodings,
                       int64_t dstEncoding, int64_t elementBytes, Operation *op,
                       StringRef role) {
  auto sharedAttr = getSharedEncodingAttr(encodings, dstEncoding, op, role);
  auto sharedSpec = getSharedEncodingSpec(encodings, dstEncoding, op, role);
  if (failed(sharedAttr) || failed(sharedSpec))
    return failure();

  if ((*sharedSpec)->logicalShape.size() != 2 ||
      (*sharedAttr).getOrder().size() != 2) {
    op->emitError() << role
                    << " transport currently requires rank-2 shared layout truth";
    return failure();
  }

  int64_t minorDim = (*sharedAttr).getOrder().front();
  if (minorDim < 0 ||
      minorDim >= static_cast<int64_t>((*sharedSpec)->logicalShape.size())) {
    op->emitError() << role
                    << " shared encoding order must reference a valid minor axis";
    return failure();
  }

  int64_t contiguousBytes = (*sharedSpec)->logicalShape[minorDim] * elementBytes;
  if (contiguousBytes <= 0) {
    op->emitError() << role
                    << " shared logical shape must expose positive contiguous bytes";
    return failure();
  }

  int64_t maxBytes = target.asyncCopyMaxBytes > 0
                         ? target.asyncCopyMaxBytes
                         : target.asyncCopyPreferredBytes;
  if (maxBytes <= 0)
    maxBytes = contiguousBytes;
  maxBytes = std::min(maxBytes, contiguousBytes);
  if ((*sharedAttr).getSwizzlingByteWidth() > 0)
    maxBytes = std::min(maxBytes, (*sharedAttr).getSwizzlingByteWidth());

  int64_t minBytes =
      target.asyncCopyMinBytes > 0 ? target.asyncCopyMinBytes : elementBytes;
  minBytes = std::max(minBytes, elementBytes);
  if (maxBytes < minBytes)
    return int64_t{0};

  SmallVector<int64_t, 8> candidates;
  auto appendCandidate = [&](int64_t bytes) {
    if (bytes <= 0)
      return;
    if (bytes < minBytes || bytes > maxBytes)
      return;
    if (bytes % elementBytes != 0)
      return;
    if (contiguousBytes % bytes != 0)
      return;
    if (!llvm::is_contained(candidates, bytes))
      candidates.push_back(bytes);
  };

  appendCandidate(target.asyncCopyPreferredBytes);
  for (int64_t bytes = maxBytes; bytes >= minBytes; bytes /= 2) {
    appendCandidate(bytes);
    if (bytes == 1)
      break;
  }
  appendCandidate(minBytes);

  if (candidates.empty())
    return int64_t{0};
  return *llvm::max_element(candidates);
}

static DictionaryAttr buildTransportSpecAttr(Builder &builder,
                                             const TransportSpec &spec) {
  NamedAttrList attrs;
  attrs.set("role", builder.getStringAttr(spec.role));
  attrs.set("dst_encoding", builder.getI64IntegerAttr(spec.dstEncoding));
  attrs.set("kind", builder.getStringAttr(spec.kind));
  attrs.set("vector_bytes", builder.getI64IntegerAttr(spec.vectorBytes));
  attrs.set("async_vector_bytes",
            builder.getI64IntegerAttr(spec.asyncVectorBytes));
  attrs.set("transaction_bytes",
            builder.getI64IntegerAttr(spec.transactionBytes));
  attrs.set("async_eligible", builder.getBoolAttr(spec.asyncEligible));
  attrs.set("bypass_l1", builder.getBoolAttr(spec.bypassL1));
  attrs.set("cache_policy", builder.getStringAttr(spec.cachePolicy));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<TransportSpec> parseTransportSpecAttr(DictionaryAttr dict,
                                                       Operation *op) {
  TransportSpec spec;
  auto role = readStringField(dict, "role", op);
  auto dstEncoding = readI64Field(dict, "dst_encoding", op);
  auto kind = readStringField(dict, "kind", op);
  auto vectorBytes = readI64Field(dict, "vector_bytes", op);
  auto asyncVectorBytes = readI64Field(dict, "async_vector_bytes", op);
  auto transactionBytes = readI64Field(dict, "transaction_bytes", op);
  auto asyncEligible = readBoolField(dict, "async_eligible", op);
  auto bypassL1 = readBoolField(dict, "bypass_l1", op);
  auto cachePolicy = readStringField(dict, "cache_policy", op);
  if (failed(role) || failed(dstEncoding) || failed(kind) ||
      failed(vectorBytes) || failed(asyncVectorBytes) ||
      failed(transactionBytes) || failed(asyncEligible) ||
      failed(bypassL1) || failed(cachePolicy)) {
    return failure();
  }
  spec.role = role->str();
  spec.dstEncoding = *dstEncoding;
  spec.kind = kind->str();
  spec.vectorBytes = *vectorBytes;
  spec.asyncVectorBytes = *asyncVectorBytes;
  spec.transactionBytes = *transactionBytes;
  spec.asyncEligible = *asyncEligible;
  spec.bypassL1 = *bypassL1;
  spec.cachePolicy = cachePolicy->str();
  if (failed(validateTransportSpec(spec, op, spec.role)))
    return failure();
  return spec;
}

} // namespace

FailureOr<TransportPlan> mlir::tb::deriveTransportPlan(
    const TargetInfo &target, const EncodingPlan &encodings, Operation *op) {
  TransportPlan plan;
  plan.contractModel = "strict_v1_transport";
  auto buildSpec = [&](StringRef role, int64_t dstEncoding)
      -> FailureOr<TransportSpec> {
    auto elementBytes = getOperandElementBytes(op, role);
    if (failed(elementBytes))
      return failure();
    auto asyncVectorBytes = chooseAsyncVectorBytes(
        target, encodings, dstEncoding, *elementBytes, op, role);
    if (failed(asyncVectorBytes))
      return failure();

    int64_t vectorBytes = *asyncVectorBytes > 0 ? *asyncVectorBytes : *elementBytes;
    bool asyncEligible = target.supportsAsyncCopy && *asyncVectorBytes > 0;
    TransportSpec spec;
    spec.role = role.str();
    spec.dstEncoding = dstEncoding;
    spec.kind = asyncEligible ? target.preferredAsyncTransport
                              : "sync_copy_fallback";
    spec.vectorBytes = vectorBytes;
    spec.asyncVectorBytes = asyncEligible ? *asyncVectorBytes : 0;
    spec.transactionBytes = asyncEligible ? *asyncVectorBytes : vectorBytes;
    spec.asyncEligible = asyncEligible;
    spec.bypassL1 = asyncEligible;
    spec.cachePolicy = asyncEligible ? "bypass_l1" : "default";
    return spec;
  };

  auto operandA = buildSpec("a_global_to_shared", encodings.aShared);
  auto operandB = buildSpec("b_global_to_shared", encodings.bShared);
  if (failed(operandA) || failed(operandB))
    return failure();
  plan.operandA = std::move(*operandA);
  plan.operandB = std::move(*operandB);

  if (failed(validateTransportSpec(plan.operandA, op, "operandA")) ||
      failed(validateTransportSpec(plan.operandB, op, "operandB"))) {
    return failure();
  }
  return plan;
}

DictionaryAttr mlir::tb::buildTransportPlanAttr(Builder &builder,
                                                const TransportPlan &plan) {
  NamedAttrList attrs;
  attrs.set("contract_model", builder.getStringAttr(plan.contractModel));
  attrs.set("operand_a", buildTransportSpecAttr(builder, plan.operandA));
  attrs.set("operand_b", buildTransportSpecAttr(builder, plan.operandB));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<TransportPlan> mlir::tb::parseTransportPlanAttr(Operation *op) {
  auto root = dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.transport_plan"));
  if (!root) {
    op->emitError() << "missing `tb.transport_plan` attribute";
    return failure();
  }

  TransportPlan plan;
  auto contractModel = readStringField(root, "contract_model", op);
  auto operandA = dyn_cast_or_null<DictionaryAttr>(root.get("operand_a"));
  auto operandB = dyn_cast_or_null<DictionaryAttr>(root.get("operand_b"));
  if (failed(contractModel) || !operandA || !operandB) {
    op->emitError() << "malformed `tb.transport_plan` attribute";
    return failure();
  }

  auto parsedOperandA = parseTransportSpecAttr(operandA, op);
  auto parsedOperandB = parseTransportSpecAttr(operandB, op);
  if (failed(parsedOperandA) || failed(parsedOperandB))
    return failure();

  plan.contractModel = contractModel->str();
  plan.operandA = std::move(*parsedOperandA);
  plan.operandB = std::move(*parsedOperandB);
  return plan;
}

FailureOr<const TransportSpec *>
mlir::tb::getTransportSpecForDstEncoding(const TransportPlan &plan,
                                         int64_t dstEncoding, Operation *op,
                                         StringRef role) {
  if (dstEncoding == plan.operandA.dstEncoding)
    return &plan.operandA;
  if (dstEncoding == plan.operandB.dstEncoding)
    return &plan.operandB;
  op->emitError() << "transport role `" << role
                  << "` does not reference a known transport-owned dst encoding";
  return failure();
}
