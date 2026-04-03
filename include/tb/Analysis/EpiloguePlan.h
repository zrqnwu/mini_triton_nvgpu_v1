#ifndef MINI_TRITON_TB_ANALYSIS_EPILOGUEPLAN_H
#define MINI_TRITON_TB_ANALYSIS_EPILOGUEPLAN_H

#include "tb/Analysis/AccumulatorPlan.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <variant>

namespace mlir::tb {

enum class AccumulatorInitMode {
  Zero,
  DirectGlobalVector,
  SharedRelay,
};

enum class AccumulatorStoreMode {
  DirectGlobalVector,
  SharedRelay,
};

enum class TargetLandingKind {
  None,
  RegisterPackGlobalVector,
  SharedPackThenGlobalVector,
};

struct DirectGlobalVectorPlan {
  struct Pack {
    int64_t packId = 0;
    int64_t rowBase = 0;
    int64_t colBase = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t vectorWidth = 0;
    int64_t warpOwner = 0;
    llvm::SmallVector<int64_t, 4> fragmentIds;
  };

  std::string ownerScope = "per_warp_template";
  LaneAccessPattern laneAccess;
  llvm::SmallVector<Pack, 16> packs;
  int64_t vectorWidth = 0;
  bool boundaryAware = false;
  bool scalarTail = false;
};

struct SharedRelayPlan {
  int relayEncoding = -1;
  llvm::SmallVector<int64_t, 4> logicalShape;
  llvm::SmallVector<int64_t, 4> allocShape;
  llvm::SmallVector<AccumulatorPack, 16> packs;
};

enum class EpilogueExprKind {
  LoadBias,
  Add,
  Convert,
  Activation,
  Clamp,
};

struct EpilogueExprOp {
  EpilogueExprKind kind = EpilogueExprKind::Add;
  ScalarKind inputScalar = ScalarKind::F32;
  ScalarKind outputScalar = ScalarKind::F32;
  int64_t vectorWidth = 0;
  std::string aux;
};

struct TargetLandingPlan {
  TargetLandingKind kind = TargetLandingKind::None;
  int64_t globalVectorWidth = 0;
  int64_t globalAccessBytes = 0;
  int64_t producerFragmentRows = 0;
  int64_t producerFragmentCols = 0;
  int64_t directPackRows = 0;
  int64_t directPackCols = 0;
  int64_t warpBatchingGroup = 0;
  int64_t requiredSharedBytes = 0;
  int64_t expectedRegisterFootprint = 0;
  int64_t sharedTileRows = 0;
  int64_t sharedTileCols = 0;
  int64_t sharedPackSlots = 0;
  int64_t initSharedStoreVectorWidth = 0;
  int64_t initSharedLoadVectorWidth = 0;
  int64_t storeSharedStoreVectorWidth = 0;
  int64_t storeSharedLoadVectorWidth = 0;
  bool useSharedPackForInit = false;
  bool useSharedPackForStore = false;
  std::string requiredSyncKind = "none";
  std::string reason;
};

struct EpiloguePlan {
  AccumulatorInitMode initMode = AccumulatorInitMode::Zero;
  AccumulatorStoreMode storeMode = AccumulatorStoreMode::DirectGlobalVector;
  std::variant<std::monostate, DirectGlobalVectorPlan, SharedRelayPlan> init;
  std::variant<std::monostate, DirectGlobalVectorPlan, SharedRelayPlan> store;
  TargetLandingPlan targetLanding;
  // 中文标记：这里负责表达 fused epilogue 步骤，不再把这些语义塞进 lowering。
  llvm::SmallVector<EpilogueExprOp, 8> exprs;
};

FailureOr<EpiloguePlan> deriveEpiloguePlan(const KernelConfig &config,
                                           const TargetInfo &target,
                                           const EncodingPlan &encodings,
                                           const AccumulatorPlan &accumulator,
                                           Operation *op);
DictionaryAttr buildEpiloguePlanAttr(Builder &builder, const EpiloguePlan &plan);
FailureOr<EpiloguePlan> parseEpiloguePlanAttr(Operation *op);

} // namespace mlir::tb

#endif // MINI_TRITON_TB_ANALYSIS_EPILOGUEPLAN_H
