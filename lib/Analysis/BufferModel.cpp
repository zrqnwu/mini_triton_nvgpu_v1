#include "tb/Analysis/BufferModel.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

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

static StringRef stringifyMemorySpace(MemorySpace space) {
  switch (space) {
  case MemorySpace::Global:
    return "global";
  case MemorySpace::Shared:
    return "shared";
  case MemorySpace::Registers:
    return "registers";
  }
  llvm_unreachable("unknown memory space");
}

} // namespace

Type mlir::tb::getTypeForScalarKind(MLIRContext *context, ScalarKind kind) {
  Builder builder(context);
  switch (kind) {
  case ScalarKind::F16:
    return builder.getF16Type();
  case ScalarKind::F32:
    return builder.getF32Type();
  }
  llvm_unreachable("unknown scalar kind");
}

int64_t mlir::tb::getByteWidthForScalarKind(ScalarKind kind) {
  switch (kind) {
  case ScalarKind::F16:
    return 2;
  case ScalarKind::F32:
    return 4;
  }
  llvm_unreachable("unknown scalar kind");
}

StringRef mlir::tb::stringifyBufferOpKind(BufferOpKind kind) {
  switch (kind) {
  case BufferOpKind::LoadA:
    return "load_a";
  case BufferOpKind::LoadB:
    return "load_b";
  case BufferOpKind::LocalLoadA:
    return "local_load_a";
  case BufferOpKind::LocalLoadB:
    return "local_load_b";
  case BufferOpKind::Mma:
    return "mma";
  case BufferOpKind::AccumulatorInit:
    return "accumulator_init";
  case BufferOpKind::AccumulatorStore:
    return "accumulator_store";
  }
  llvm_unreachable("unknown buffer op kind");
}

StringRef
mlir::tb::stringifyPipelineOpSemanticClass(PipelineOpSemanticClass semanticClass) {
  switch (semanticClass) {
  case PipelineOpSemanticClass::AsyncProducer:
    return "async_producer";
  case PipelineOpSemanticClass::SharedConsumer:
    return "shared_consumer";
  case PipelineOpSemanticClass::TensorCoreCompute:
    return "tensor_core_compute";
  case PipelineOpSemanticClass::EpilogueInit:
    return "epilogue_init";
  case PipelineOpSemanticClass::EpilogueStore:
    return "epilogue_store";
  }
  llvm_unreachable("unknown pipeline semantic class");
}

PipelineOpSemanticClass
mlir::tb::getPipelineOpSemanticClass(BufferOpKind kind) {
  switch (kind) {
  case BufferOpKind::LoadA:
  case BufferOpKind::LoadB:
    return PipelineOpSemanticClass::AsyncProducer;
  case BufferOpKind::LocalLoadA:
  case BufferOpKind::LocalLoadB:
    return PipelineOpSemanticClass::SharedConsumer;
  case BufferOpKind::Mma:
    return PipelineOpSemanticClass::TensorCoreCompute;
  case BufferOpKind::AccumulatorInit:
    return PipelineOpSemanticClass::EpilogueInit;
  case BufferOpKind::AccumulatorStore:
    return PipelineOpSemanticClass::EpilogueStore;
  }
  llvm_unreachable("unknown buffer op kind");
}

bool mlir::tb::isMainloopPipelineSemanticClass(
    PipelineOpSemanticClass semanticClass) {
  switch (semanticClass) {
  case PipelineOpSemanticClass::AsyncProducer:
  case PipelineOpSemanticClass::SharedConsumer:
  case PipelineOpSemanticClass::TensorCoreCompute:
    return true;
  case PipelineOpSemanticClass::EpilogueInit:
  case PipelineOpSemanticClass::EpilogueStore:
    return false;
  }
  llvm_unreachable("unknown pipeline semantic class");
}

FailureOr<MemDescType> mlir::tb::getMemDescType(Type type, Operation *op,
                                                StringRef role) {
  auto desc = dyn_cast<MemDescType>(type);
  if (!desc) {
    op->emitError() << "expected !tb.memdesc for `" << role << "`";
    return failure();
  }
  return desc;
}

FailureOr<MemorySpace> mlir::tb::getMemDescMemorySpace(Type type, Operation *op,
                                                       StringRef role) {
  auto desc = getMemDescType(type, op, role);
  if (failed(desc))
    return failure();
  if (desc->isGlobalMemory())
    return MemorySpace::Global;
  if (desc->isSharedMemory())
    return MemorySpace::Shared;
  if (desc->isRegisterMemory())
    return MemorySpace::Registers;
  op->emitError() << "unsupported memory space `" << desc->getMemorySpace()
                  << "` for `" << role << "`";
  return failure();
}

FailureOr<const BufferBacking *>
mlir::tb::findUniqueBacking(const BufferModel &model, BufferRole role,
                            MemorySpace memorySpace, Operation *op,
                            StringRef contractName) {
  auto roleName = [&]() -> StringRef {
    switch (role) {
    case BufferRole::OperandA:
      return "operand_a";
    case BufferRole::OperandB:
      return "operand_b";
    case BufferRole::Output:
      return "output";
    case BufferRole::Accumulator:
      return "accumulator";
    case BufferRole::EpilogueRelay:
      return "epilogue_relay";
    case BufferRole::Descriptor:
      return "descriptor";
    case BufferRole::Generic:
      return "generic";
    }
    llvm_unreachable("unknown buffer role");
  }();
  const BufferBacking *match = nullptr;
  for (const BufferBacking &backing : model.backings) {
    auto backingSpace =
        getMemDescMemorySpace(backing.descType, op, contractName);
    if (failed(backingSpace))
      return failure();
    if (backing.role != role || *backingSpace != memorySpace)
      continue;
    if (match) {
      op->emitError() << "contract `" << contractName
                      << "` matched multiple backings with role `" << roleName
                      << "` in `"
                      << stringifyMemorySpace(memorySpace) << "` memory";
      return failure();
    }
    match = &backing;
  }
  if (!match) {
    op->emitError() << "contract `" << contractName
                    << "` could not find a backing with role `" << roleName
                    << "` in `"
                    << stringifyMemorySpace(memorySpace) << "` memory";
    return failure();
  }
  return match;
}

FailureOr<const BufferView *>
mlir::tb::findUniqueFullBufferView(const BufferModel &model, BufferRole role,
                                   MemorySpace memorySpace, Operation *op,
                                   StringRef contractName) {
  auto backing = findUniqueBacking(model, role, memorySpace, op, contractName);
  if (failed(backing))
    return failure();

  const BufferView *match = nullptr;
  for (const BufferView &view : model.views) {
    if (view.backing != (*backing)->id || view.kind != ViewKind::FullBuffer)
      continue;
    if (match) {
      op->emitError() << "contract `" << contractName
                      << "` matched multiple full-buffer views for backing "
                      << (*backing)->id;
      return failure();
    }
    match = &view;
  }
  if (!match) {
    op->emitError() << "contract `" << contractName
                    << "` could not find a full-buffer view for backing "
                    << (*backing)->id;
    return failure();
  }
  return match;
}

namespace {

static StringRef stringifyBufferRole(BufferRole role) {
  switch (role) {
  case BufferRole::OperandA:
    return "operand_a";
  case BufferRole::OperandB:
    return "operand_b";
  case BufferRole::Output:
    return "output";
  case BufferRole::Accumulator:
    return "accumulator";
  case BufferRole::EpilogueRelay:
    return "epilogue_relay";
  case BufferRole::Descriptor:
    return "descriptor";
  case BufferRole::Generic:
    return "generic";
  }
  llvm_unreachable("unknown buffer role");
}

static FailureOr<BufferRole> parseBufferRole(StringRef value, Operation *op) {
  if (value == "operand_a")
    return BufferRole::OperandA;
  if (value == "operand_b")
    return BufferRole::OperandB;
  if (value == "output")
    return BufferRole::Output;
  if (value == "accumulator")
    return BufferRole::Accumulator;
  if (value == "epilogue_relay")
    return BufferRole::EpilogueRelay;
  if (value == "descriptor")
    return BufferRole::Descriptor;
  if (value == "generic")
    return BufferRole::Generic;
  op->emitError() << "unknown buffer role `" << value << "`";
  return failure();
}

static StringRef stringifyViewKind(ViewKind kind) {
  switch (kind) {
  case ViewKind::FullBuffer:
    return "full_buffer";
  case ViewKind::StageSlice:
    return "stage_slice";
  case ViewKind::TileSlice:
    return "tile_slice";
  case ViewKind::Fragment:
    return "fragment";
  case ViewKind::Pack:
    return "pack";
  }
  llvm_unreachable("unknown view kind");
}

static FailureOr<ViewKind> parseViewKind(StringRef value, Operation *op) {
  if (value == "full_buffer")
    return ViewKind::FullBuffer;
  if (value == "stage_slice")
    return ViewKind::StageSlice;
  if (value == "tile_slice")
    return ViewKind::TileSlice;
  if (value == "fragment")
    return ViewKind::Fragment;
  if (value == "pack")
    return ViewKind::Pack;
  op->emitError() << "unknown view kind `" << value << "`";
  return failure();
}

static StringRef stringifyValueKind(BufferValueKind kind) {
  switch (kind) {
  case BufferValueKind::GlobalTile:
    return "global_tile";
  case BufferValueKind::SharedTile:
    return "shared_tile";
  case BufferValueKind::DotOperandFragment:
    return "dot_operand_fragment";
  case BufferValueKind::AccumulatorFragment:
    return "accumulator_fragment";
  case BufferValueKind::EpilogueFragment:
    return "epilogue_fragment";
  }
  llvm_unreachable("unknown value kind");
}

static FailureOr<BufferValueKind> parseValueKind(StringRef value,
                                                 Operation *op) {
  if (value == "global_tile")
    return BufferValueKind::GlobalTile;
  if (value == "shared_tile")
    return BufferValueKind::SharedTile;
  if (value == "dot_operand_fragment")
    return BufferValueKind::DotOperandFragment;
  if (value == "accumulator_fragment")
    return BufferValueKind::AccumulatorFragment;
  if (value == "epilogue_fragment")
    return BufferValueKind::EpilogueFragment;
  op->emitError() << "unknown value kind `" << value << "`";
  return failure();
}

static FailureOr<BufferOpKind> parseOpKind(StringRef value, Operation *op) {
  if (value == "load_a")
    return BufferOpKind::LoadA;
  if (value == "load_b")
    return BufferOpKind::LoadB;
  if (value == "local_load_a")
    return BufferOpKind::LocalLoadA;
  if (value == "local_load_b")
    return BufferOpKind::LocalLoadB;
  if (value == "mma")
    return BufferOpKind::Mma;
  if (value == "accumulator_init")
    return BufferOpKind::AccumulatorInit;
  if (value == "accumulator_store")
    return BufferOpKind::AccumulatorStore;
  op->emitError() << "unknown buffer op kind `" << value << "`";
  return failure();
}

template <typename RangeT, typename BuildFn>
static ArrayAttr buildDictArrayAttr(Builder &builder, const RangeT &values,
                                    BuildFn buildElement);

template <typename T, typename ParseFn>
static FailureOr<SmallVector<T, 0>> parseDictArrayAttr(Attribute attr,
                                                       StringRef name,
                                                       Operation *op,
                                                       ParseFn parseElement);

static DictionaryAttr buildIterationCoordAttr(
    Builder &builder, const PipelineOp::IterationCoord &coord) {
  NamedAttrList attrs;
  attrs.set("axis", builder.getStringAttr(coord.axis));
  attrs.set("value", builder.getI64IntegerAttr(coord.value));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<PipelineOp::IterationCoord>
parseIterationCoordAttr(DictionaryAttr dict, Operation *op) {
  PipelineOp::IterationCoord coord;
  auto axis = readStringField(dict, "axis", op);
  auto value = readI64Field(dict, "value", op);
  if (failed(axis) || failed(value))
    return failure();
  coord.axis = axis->str();
  coord.value = *value;
  return coord;
}

static DictionaryAttr buildBackingAttr(Builder &builder,
                                       const BufferBacking &backing) {
  NamedAttrList attrs;
  attrs.set("id", builder.getI64IntegerAttr(backing.id));
  attrs.set("role", builder.getStringAttr(stringifyBufferRole(backing.role)));
  attrs.set("desc_type", TypeAttr::get(backing.descType));
  attrs.set("depth", builder.getI64IntegerAttr(backing.depth));
  attrs.set("alias_group", builder.getI64IntegerAttr(backing.aliasGroup));
  attrs.set("stage_indexed", builder.getBoolAttr(backing.stageIndexed));
  attrs.set("debug_name", builder.getStringAttr(backing.debugName));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<BufferBacking> parseBackingAttr(DictionaryAttr dict,
                                                 Operation *op) {
  BufferBacking backing;
  auto id = readI64Field(dict, "id", op);
  auto role = readStringField(dict, "role", op);
  auto descType = dyn_cast_or_null<TypeAttr>(dict.get("desc_type"));
  auto depth = readI64Field(dict, "depth", op);
  auto aliasGroup = readI64Field(dict, "alias_group", op);
  auto stageIndexed = readBoolField(dict, "stage_indexed", op);
  auto debugName = readStringField(dict, "debug_name", op);
  if (failed(id) || failed(role) || !descType || failed(depth) ||
      failed(aliasGroup) || failed(stageIndexed) || failed(debugName)) {
    op->emitError() << "malformed backing entry";
    return failure();
  }
  auto parsedRole = parseBufferRole(*role, op);
  if (failed(parsedRole))
    return failure();
  if (failed(getMemDescType(descType.getValue(), op, *role)))
    return failure();
  backing.id = *id;
  backing.role = *parsedRole;
  backing.descType = descType.getValue();
  backing.depth = *depth;
  backing.aliasGroup = *aliasGroup;
  backing.stageIndexed = *stageIndexed;
  backing.debugName = debugName->str();
  return backing;
}

static DictionaryAttr buildViewAttr(Builder &builder, const BufferView &view) {
  NamedAttrList attrs;
  attrs.set("id", builder.getI64IntegerAttr(view.id));
  attrs.set("backing", builder.getI64IntegerAttr(view.backing));
  attrs.set("kind", builder.getStringAttr(stringifyViewKind(view.kind)));
  attrs.set("stage", builder.getI64IntegerAttr(view.stage));
  attrs.set("buffer_index", builder.getI64IntegerAttr(view.bufferIndex));
  attrs.set("encoding", builder.getI64IntegerAttr(view.encoding));
  attrs.set("offsets", buildI64ArrayAttr(builder, view.offsets));
  attrs.set("shape", buildI64ArrayAttr(builder, view.shape));
  attrs.set("debug_name", builder.getStringAttr(view.debugName));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<BufferView> parseViewAttr(DictionaryAttr dict, Operation *op) {
  BufferView view;
  auto id = readI64Field(dict, "id", op);
  auto backing = readI64Field(dict, "backing", op);
  auto kind = readStringField(dict, "kind", op);
  auto stage = readI64Field(dict, "stage", op);
  auto bufferIndex = readI64Field(dict, "buffer_index", op);
  auto encoding = readI64Field(dict, "encoding", op);
  auto offsets = readDenseI64ArrayField(dict, "offsets", op);
  auto shape = readDenseI64ArrayField(dict, "shape", op);
  auto debugName = readStringField(dict, "debug_name", op);
  if (failed(id) || failed(backing) || failed(kind) || failed(stage) ||
      failed(bufferIndex) || failed(encoding) || failed(offsets) ||
      failed(shape) || failed(debugName)) {
    return failure();
  }
  auto parsedKind = parseViewKind(*kind, op);
  if (failed(parsedKind))
    return failure();
  view.id = *id;
  view.backing = *backing;
  view.kind = *parsedKind;
  view.stage = *stage;
  view.bufferIndex = *bufferIndex;
  view.encoding = static_cast<int>(*encoding);
  view.offsets = parseI64Array(*offsets);
  view.shape = parseI64Array(*shape);
  view.debugName = debugName->str();
  return view;
}

static DictionaryAttr buildValueAttr(Builder &builder, const ValueState &value) {
  NamedAttrList attrs;
  attrs.set("id", builder.getI64IntegerAttr(value.id));
  attrs.set("kind", builder.getStringAttr(stringifyValueKind(value.kind)));
  attrs.set("defining_op", builder.getI64IntegerAttr(value.definingOp));
  attrs.set("users", buildI64ArrayAttr(builder, value.users));
  attrs.set("owner_view", builder.getI64IntegerAttr(value.ownerView));
  attrs.set("loop_distance", builder.getI64IntegerAttr(value.loopDistance));
  attrs.set("bundle", builder.getI64IntegerAttr(value.bundle));
  attrs.set("first_use_ordinal",
            builder.getI64IntegerAttr(value.firstUseOrdinal));
  attrs.set("last_use_ordinal",
            builder.getI64IntegerAttr(value.lastUseOrdinal));
  attrs.set("debug_name", builder.getStringAttr(value.debugName));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<ValueState> parseValueAttr(DictionaryAttr dict, Operation *op) {
  ValueState value;
  auto id = readI64Field(dict, "id", op);
  auto kind = readStringField(dict, "kind", op);
  auto definingOp = readI64Field(dict, "defining_op", op);
  auto users = readDenseI64ArrayField(dict, "users", op);
  auto ownerView = readI64Field(dict, "owner_view", op);
  auto loopDistance = readI64Field(dict, "loop_distance", op);
  auto bundle = readI64Field(dict, "bundle", op);
  auto firstUseOrdinal = readI64Field(dict, "first_use_ordinal", op);
  auto lastUseOrdinal = readI64Field(dict, "last_use_ordinal", op);
  auto debugName = readStringField(dict, "debug_name", op);
  if (failed(id) || failed(kind) || failed(definingOp) || failed(users) ||
      failed(ownerView) || failed(loopDistance) || failed(bundle) ||
      failed(firstUseOrdinal) || failed(lastUseOrdinal) || failed(debugName)) {
    return failure();
  }
  auto parsedKind = parseValueKind(*kind, op);
  if (failed(parsedKind))
    return failure();
  value.id = *id;
  value.kind = *parsedKind;
  value.definingOp = *definingOp;
  value.users = parseI64Array(*users);
  value.ownerView = *ownerView;
  value.loopDistance = *loopDistance;
  value.bundle = *bundle;
  value.firstUseOrdinal = *firstUseOrdinal;
  value.lastUseOrdinal = *lastUseOrdinal;
  value.debugName = debugName->str();
  return value;
}

static DictionaryAttr buildOpAttr(Builder &builder, const PipelineOp &opInfo) {
  NamedAttrList attrs;
  attrs.set("id", builder.getI64IntegerAttr(opInfo.id));
  attrs.set("kind", builder.getStringAttr(stringifyBufferOpKind(opInfo.kind)));
  attrs.set("inputs", buildI64ArrayAttr(builder, opInfo.inputs));
  attrs.set("outputs", buildI64ArrayAttr(builder, opInfo.outputs));
  attrs.set("iteration_coords",
            buildDictArrayAttr(builder, opInfo.iterationCoords,
                               buildIterationCoordAttr));
  return builder.getDictionaryAttr(attrs);
}

static FailureOr<PipelineOp> parseOpAttr(DictionaryAttr dict, Operation *op) {
  PipelineOp opInfo;
  auto id = readI64Field(dict, "id", op);
  auto kind = readStringField(dict, "kind", op);
  auto inputs = readDenseI64ArrayField(dict, "inputs", op);
  auto outputs = readDenseI64ArrayField(dict, "outputs", op);
  auto iterationCoords = parseDictArrayAttr<PipelineOp::IterationCoord>(
      dict.get("iteration_coords"), "iteration_coords", op,
      parseIterationCoordAttr);
  if (failed(id) || failed(kind) || failed(inputs) || failed(outputs) ||
      failed(iterationCoords)) {
    return failure();
  }
  auto parsedKind = parseOpKind(*kind, op);
  if (failed(parsedKind))
    return failure();
  opInfo.id = *id;
  opInfo.kind = *parsedKind;
  opInfo.inputs = parseI64Array(*inputs);
  opInfo.outputs = parseI64Array(*outputs);
  opInfo.iterationCoords = std::move(*iterationCoords);
  return opInfo;
}

template <typename RangeT, typename BuildFn>
static ArrayAttr buildDictArrayAttr(Builder &builder, const RangeT &values,
                                    BuildFn buildElement) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (const auto &value : values)
    attrs.push_back(buildElement(builder, value));
  return builder.getArrayAttr(attrs);
}

template <typename T, typename ParseFn>
static FailureOr<SmallVector<T, 0>> parseDictArrayAttr(Attribute attr,
                                                       StringRef name,
                                                       Operation *op,
                                                       ParseFn parseElement) {
  auto array = dyn_cast_or_null<ArrayAttr>(attr);
  if (!array) {
    op->emitError() << "missing array field `" << name << "`";
    return failure();
  }

  SmallVector<T, 0> values;
  values.reserve(array.size());
  for (Attribute element : array) {
    auto dict = dyn_cast<DictionaryAttr>(element);
    if (!dict) {
      op->emitError() << "field `" << name
                      << "` must contain only dictionary elements";
      return failure();
    }
    auto parsed = parseElement(dict, op);
    if (failed(parsed))
      return failure();
    values.push_back(std::move(*parsed));
  }
  return values;
}

static LogicalResult validateBufferModel(const BufferModel &model,
                                         Operation *op) {
  DenseSet<int64_t> backingIds;
  DenseSet<int64_t> viewIds;
  DenseSet<int64_t> opIds;
  DenseSet<int64_t> valueIds;

  for (const BufferBacking &backing : model.backings) {
    if (!backingIds.insert(backing.id).second)
      return op->emitError() << "duplicate backing id " << backing.id;
    if (failed(getMemDescType(backing.descType, op, backing.debugName)))
      return failure();
  }

  auto requireUniqueBacking = [&](BufferRole role,
                                  MemorySpace memorySpace) -> LogicalResult {
    int64_t matches = 0;
    for (const BufferBacking &backing : model.backings) {
      auto backingSpace =
          getMemDescMemorySpace(backing.descType, op, backing.debugName);
      if (failed(backingSpace))
        return failure();
      if (backing.role == role && *backingSpace == memorySpace)
        ++matches;
    }
    if (matches != 1) {
      return op->emitError()
             << "buffer model must contain exactly one backing with role `"
             << stringifyBufferRole(role) << "` in `"
             << stringifyMemorySpace(memorySpace) << "` memory, got "
             << matches;
    }
    return success();
  };

  if (failed(requireUniqueBacking(BufferRole::OperandA, MemorySpace::Global)) ||
      failed(requireUniqueBacking(BufferRole::OperandB, MemorySpace::Global)) ||
      failed(requireUniqueBacking(BufferRole::Output, MemorySpace::Global)) ||
      failed(requireUniqueBacking(BufferRole::OperandA, MemorySpace::Shared)) ||
      failed(requireUniqueBacking(BufferRole::OperandB, MemorySpace::Shared)))
    return failure();

  for (const BufferView &view : model.views) {
    if (!viewIds.insert(view.id).second)
      return op->emitError() << "duplicate view id " << view.id;
    if (!backingIds.count(view.backing))
      return op->emitError() << "view " << view.id
                             << " references missing backing " << view.backing;
  }

  for (const ValueState &value : model.values) {
    if (!valueIds.insert(value.id).second)
      return op->emitError() << "duplicate value id " << value.id;
    if (!viewIds.count(value.ownerView))
      return op->emitError() << "value " << value.id
                             << " references missing owner view "
                             << value.ownerView;

    auto ownerIt = llvm::find_if(model.views, [&](const BufferView &view) {
      return view.id == value.ownerView;
    });
    auto backingIt = llvm::find_if(model.backings, [&](const BufferBacking &backing) {
      return ownerIt != model.views.end() && backing.id == ownerIt->backing;
    });
    if (ownerIt != model.views.end() && backingIt != model.backings.end() &&
        backingIt->stageIndexed && ownerIt->kind == ViewKind::FullBuffer) {
      return op->emitError()
             << "stage-indexed value " << value.id
             << " must own a concrete stage slice view, not full buffer view "
             << ownerIt->id;
    }
  }

  for (const PipelineOp &opInfo : model.ops) {
    if (!opIds.insert(opInfo.id).second)
      return op->emitError() << "duplicate buffer op id " << opInfo.id;
    DenseSet<StringRef> axes;
    for (const auto &coord : opInfo.iterationCoords) {
      StringRef axis = coord.axis;
      if (!axes.insert(axis).second)
        return op->emitError() << "buffer op " << opInfo.id
                               << " contains duplicate iteration axis `" << axis
                               << "`";
    }
  }

  return success();
}

static FailureOr<std::pair<SmallVector<int64_t, 4>, SmallVector<int64_t, 4>>>
getEncodingShapes(const EncodingPlan &encodings, int encoding, StringRef role,
                  Operation *op) {
  SmallVector<int64_t, 4> logicalShape;
  SmallVector<int64_t, 4> allocShape;
  auto attr = getEncodingAttr(encodings, encoding, op, role);
  if (failed(attr))
    return failure();
  if (auto shared = dyn_cast<SharedEncodingAttr>(*attr)) {
    (void)shared;
    auto sharedSpec = getSharedEncodingSpec(encodings, encoding, op, role);
    if (failed(sharedSpec))
      return failure();
    logicalShape.assign((*sharedSpec)->logicalShape.begin(),
                        (*sharedSpec)->logicalShape.end());
    allocShape.assign((*sharedSpec)->allocShape.begin(),
                      (*sharedSpec)->allocShape.end());
  } else if (auto acc = dyn_cast<AccumulatorEncodingAttr>(*attr)) {
    logicalShape.assign(acc.getLogicalShape().begin(),
                        acc.getLogicalShape().end());
  } else if (isa<DotOperandEncodingAttr>(*attr)) {
    ArrayRef<int64_t> logical =
        role == "local_b"
            ? ArrayRef<int64_t>(encodings.fragmentB.logicalShape)
            : ArrayRef<int64_t>(encodings.fragmentA.logicalShape);
    logicalShape.assign(logical.begin(), logical.end());
  }

  if (logicalShape.empty()) {
    op->emitError() << "cannot derive logical shape for role `" << role
                    << "` from encoding " << encoding;
    return failure();
  }
  if (allocShape.empty())
    allocShape = logicalShape;
  return std::make_pair(std::move(logicalShape), std::move(allocShape));
}

static void addIterationCoord(
    SmallVectorImpl<PipelineOp::IterationCoord> &coords, StringRef axis,
    int64_t value) {
  if (value < 0)
    return;
  coords.push_back({axis.str(), value});
}

static int64_t findIterationCoord(ArrayRef<PipelineOp::IterationCoord> coords,
                                  StringRef axis) {
  auto it = llvm::find_if(coords, [&](const PipelineOp::IterationCoord &coord) {
    return coord.axis == axis;
  });
  return it == coords.end() ? -1 : it->value;
}

} // namespace

FailureOr<BufferModel> mlir::tb::deriveBufferModel(
    const KernelConfig &config, const EncodingPlan &encodings,
    const AccumulatorPlan &accumulator, const EpiloguePlan &epilogue,
    const MatmulRewritePlan &rewrite,
    Operation *op) {
  if (rewrite.instructionK <= 0) {
    op->emitError() << "buffer model requires a positive rewritten instruction K";
    return failure();
  }

  if (rewrite.accTilesM <= 0 || rewrite.accTilesN <= 0) {
    op->emitError() << "matmul rewrite must define a positive accumulator tile grid";
    return failure();
  }

  int64_t accTilesM = rewrite.accTilesM;
  int64_t accTilesN = rewrite.accTilesN;
  int64_t bTileSpanN = rewrite.bPath.consumerTileSpanN;
  if (accTilesM <= 0 || accTilesN <= 0 || bTileSpanN <= 0 ||
      rewrite.bGroupCount <= 0 ||
      rewrite.bGroupCount * bTileSpanN != accTilesN) {
    op->emitError() << "buffer model requires a layout-driven accumulator/B "
                       "fragment grid";
    return failure();
  }

  int64_t numKGroups = rewrite.kGroups;
  int64_t accFragments = rewrite.accumulatorFragments;
  if (static_cast<int64_t>(accumulator.packs.size()) != accFragments) {
    op->emitError() << "accumulator pack count does not match the warp "
                       "accumulator fragment grid";
    return failure();
  }
  if (epilogue.initMode != AccumulatorInitMode::DirectGlobalVector ||
      epilogue.storeMode != AccumulatorStoreMode::DirectGlobalVector) {
    op->emitError() << "V1 buffer model expects direct-global epilogue ownership";
    return failure();
  }
  auto *directInit = std::get_if<DirectGlobalVectorPlan>(&epilogue.init);
  auto *directStore = std::get_if<DirectGlobalVectorPlan>(&epilogue.store);
  if (!directInit || !directStore || directInit->packs.empty() ||
      directStore->packs.empty()) {
    op->emitError() << "buffer model requires direct-global epilogue payloads";
    return failure();
  }

  int64_t numBGroups = rewrite.bGroupCount;
  int64_t sharedASlot = 0;
  int64_t sharedBSlot = 1;
  int64_t localASlotBase = 2;
  int64_t localBSlotBase = localASlotBase + accTilesM;
  int64_t accSlotBase = localBSlotBase + numBGroups;
  int64_t numSlots = accSlotBase + accFragments;

  struct SlotDesc {
    int64_t id = -1;
    StringRef role;
    BufferRole bufferRole = BufferRole::Generic;
    MemorySpace memorySpace = MemorySpace::Registers;
    int encoding = -1;
    ScalarKind scalarKind = ScalarKind::F16;
  };

  SmallVector<SlotDesc, 32> slots;
  slots.push_back({sharedASlot, "shared_a", BufferRole::OperandA,
                   MemorySpace::Shared, encodings.aShared, config.aScalar});
  slots.push_back({sharedBSlot, "shared_b", BufferRole::OperandB,
                   MemorySpace::Shared, encodings.bShared, config.bScalar});
  for (int64_t mTile = 0; mTile < accTilesM; ++mTile)
    slots.push_back({localASlotBase + mTile, "local_a", BufferRole::OperandA,
                     MemorySpace::Registers, encodings.aDot, config.aScalar});
  for (int64_t nGroup = 0; nGroup < numBGroups; ++nGroup)
    slots.push_back({localBSlotBase + nGroup, "local_b", BufferRole::OperandB,
                     MemorySpace::Registers, encodings.bDot, config.bScalar});
  for (int64_t accIndex = 0; accIndex < accFragments; ++accIndex)
    slots.push_back({accSlotBase + accIndex, "acc", BufferRole::Accumulator,
                     MemorySpace::Registers, accumulator.encoding,
                     config.cScalar});

  BufferModel model;
  model.backings.reserve(slots.size() + 3);
  model.views.reserve(slots.size() + 2 * std::max<int64_t>(config.requestedStages, 1) +
                      3);
  model.values.reserve(numKGroups *
                       (2 + accTilesM + numBGroups + accFragments));
  model.ops.reserve(model.values.capacity());

  DenseMap<int64_t, SmallVector<int64_t, 4>> stageViewsByBacking;
  int64_t nextViewId = numSlots;
  int64_t nextBackingId = numSlots;
  for (const SlotDesc &slot : slots) {
    auto shapes = getEncodingShapes(encodings, slot.encoding, slot.role, op);
    if (failed(shapes))
      return failure();

    BufferBacking backing;
    backing.id = slot.id;
    backing.role = slot.bufferRole;
    backing.depth = 1;
    backing.aliasGroup = slot.id;
    backing.stageIndexed = false;
    backing.debugName = slot.role.str();

    SmallVector<int64_t, 4> sliceLogicalShape = std::move(shapes->first);
    SmallVector<int64_t, 4> sliceAllocShape = std::move(shapes->second);
    bool isSharedOperand =
        slot.memorySpace == MemorySpace::Shared &&
        (slot.bufferRole == BufferRole::OperandA ||
         slot.bufferRole == BufferRole::OperandB);
    backing.depth =
        isSharedOperand ? std::max<int64_t>(config.requestedStages, 1) : 1;
    backing.stageIndexed = isSharedOperand && backing.depth > 1;
    SmallVector<int64_t, 4> fullLogicalShape;
    SmallVector<int64_t, 4> fullAllocShape;
    if (backing.stageIndexed) {
      fullLogicalShape.push_back(backing.depth);
      fullLogicalShape.append(sliceLogicalShape.begin(), sliceLogicalShape.end());
      fullAllocShape.push_back(backing.depth);
      fullAllocShape.append(sliceAllocShape.begin(), sliceAllocShape.end());
    } else {
      fullLogicalShape = sliceLogicalShape;
      fullAllocShape = sliceAllocShape;
    }
    auto encodingAttr = getEncodingAttr(encodings, slot.encoding, op, slot.role);
    if (failed(encodingAttr))
      return failure();
    backing.descType = MemDescType::get(
        op->getContext(), ArrayRef<int64_t>(fullLogicalShape),
        getTypeForScalarKind(op->getContext(), slot.scalarKind), *encodingAttr,
        stringifyMemorySpace(slot.memorySpace), /*mutableMemory=*/true,
        ArrayRef<int64_t>(fullAllocShape));
    model.backings.push_back(backing);

    BufferView fullView;
    fullView.id = slot.id;
    fullView.backing = backing.id;
    fullView.kind = ViewKind::FullBuffer;
    fullView.encoding = slot.encoding;
    fullView.shape = fullLogicalShape;
    fullView.debugName = slot.role.str();
    model.views.push_back(fullView);

    if (!backing.stageIndexed)
      continue;

    auto &stageViews = stageViewsByBacking[backing.id];
    stageViews.reserve(backing.depth);
    for (int64_t stage = 0; stage < backing.depth; ++stage) {
      BufferView stageView;
      stageView.id = nextViewId++;
      stageView.backing = backing.id;
      stageView.kind = ViewKind::StageSlice;
      stageView.stage = stage;
      stageView.bufferIndex = stage;
      stageView.encoding = slot.encoding;
      stageView.offsets.push_back(stage);
      for (size_t i = 0; i < sliceAllocShape.size(); ++i)
        stageView.offsets.push_back(0);
      stageView.shape = sliceLogicalShape;
      stageView.debugName =
          (slot.role + "_stage" + std::to_string(stage)).str();
      stageViews.push_back(stageView.id);
      model.views.push_back(std::move(stageView));
    }
  }

  auto addGlobalBacking = [&](StringRef name, BufferRole role, int encoding,
                              ArrayRef<int64_t> shape,
                              ScalarKind scalarKind) -> LogicalResult {
    auto encodingAttr = getEncodingAttr(encodings, encoding, op, name);
    if (failed(encodingAttr))
      return failure();
    BufferBacking backing;
    backing.id = nextBackingId++;
    backing.role = role;
    backing.depth = 1;
    backing.aliasGroup = backing.id;
    backing.stageIndexed = false;
    backing.debugName = name.str();
    backing.descType = MemDescType::get(
        op->getContext(), shape, getTypeForScalarKind(op->getContext(), scalarKind),
        *encodingAttr, "global", /*mutableMemory=*/true, shape);
    model.backings.push_back(backing);

    BufferView view;
    view.id = nextViewId++;
    view.backing = backing.id;
    view.kind = ViewKind::FullBuffer;
    view.encoding = encoding;
    view.shape.assign(shape.begin(), shape.end());
    view.debugName = backing.debugName;
    model.views.push_back(std::move(view));
    return success();
  };

  if (failed(addGlobalBacking("global_a", BufferRole::OperandA, encodings.aGlobal,
                              ArrayRef<int64_t>{config.problemM, config.problemK},
                              config.aScalar))) {
    return failure();
  }
  if (failed(addGlobalBacking("global_b", BufferRole::OperandB, encodings.bGlobal,
                              ArrayRef<int64_t>{config.problemK, config.problemN},
                              config.bScalar))) {
    return failure();
  }
  if (failed(addGlobalBacking("global_c", BufferRole::Output, encodings.cStore,
                              ArrayRef<int64_t>{config.problemM, config.problemN},
                              config.cScalar))) {
    return failure();
  }

  auto getOwnerViewForSlot = [&](int64_t slotId,
                                 int64_t bundle) -> FailureOr<int64_t> {
    auto backingIt =
        llvm::find_if(model.backings, [&](const BufferBacking &backing) {
          return backing.id == slotId;
        });
    if (backingIt == model.backings.end()) {
      op->emitError() << "missing backing for slot " << slotId;
      return failure();
    }
    if (!backingIt->stageIndexed)
      return slotId;
    auto stageIt = stageViewsByBacking.find(slotId);
    if (stageIt == stageViewsByBacking.end() || stageIt->second.empty()) {
      op->emitError() << "missing stage views for multistage backing "
                      << slotId;
      return failure();
    }
    int64_t stage = bundle >= 0 ? bundle % backingIt->depth : 0;
    if (stage < 0 || stage >= static_cast<int64_t>(stageIt->second.size())) {
      op->emitError() << "invalid stage " << stage << " for backing "
                      << slotId;
      return failure();
    }
    return stageIt->second[stage];
  };

  auto addValue = [&](BufferValueKind kind, StringRef debugName,
                      int64_t definingOp, int64_t slotId,
                      int64_t bundle) -> FailureOr<int64_t> {
    auto ownerView = getOwnerViewForSlot(slotId, bundle);
    if (failed(ownerView))
      return failure();

    ValueState value;
    value.id = static_cast<int64_t>(model.values.size());
    value.kind = kind;
    value.definingOp = definingOp;
    value.ownerView = *ownerView;
    value.loopDistance = 0;
    value.bundle = bundle;
    value.firstUseOrdinal = -1;
    value.lastUseOrdinal = -1;
    value.debugName = debugName.str();
    model.values.push_back(std::move(value));
    return model.values.back().id;
  };

  auto appendUser = [&](int64_t valueId, int64_t userOpId) {
    ValueState &value = model.values[valueId];
    value.users.push_back(userOpId);
    if (value.firstUseOrdinal < 0)
      value.firstUseOrdinal = userOpId;
    value.lastUseOrdinal = userOpId;
  };

  auto appendOp = [&](BufferOpKind kind, ArrayRef<int64_t> inputs,
                      ArrayRef<int64_t> outputs, int64_t bundle, int64_t kGroup,
                      int64_t mTile, int64_t nTile, int64_t nGroup,
                      int64_t groupOffset,
                      int64_t accIndex) -> int64_t {
    PipelineOp opInfo;
    opInfo.id = static_cast<int64_t>(model.ops.size());
    opInfo.kind = kind;
    opInfo.inputs.assign(inputs.begin(), inputs.end());
    opInfo.outputs.assign(outputs.begin(), outputs.end());
    addIterationCoord(opInfo.iterationCoords, "bundle", bundle);
    addIterationCoord(opInfo.iterationCoords, "k_group", kGroup);
    addIterationCoord(opInfo.iterationCoords, "m_tile", mTile);
    addIterationCoord(opInfo.iterationCoords, "n_tile", nTile);
    addIterationCoord(opInfo.iterationCoords, "n_group", nGroup);
    addIterationCoord(opInfo.iterationCoords, "group_offset", groupOffset);
    addIterationCoord(opInfo.iterationCoords, "acc_index", accIndex);
    model.ops.push_back(std::move(opInfo));
    return model.ops.back().id;
  };

  SmallVector<int64_t, 32> previousAccValues(accFragments, -1);
  SmallVector<int64_t, 8> localAValues(accTilesM, -1);
  SmallVector<int64_t, 8> localBValues(numBGroups, -1);

  for (int64_t kGroup = 0; kGroup < numKGroups; ++kGroup) {
    int64_t loadABundle = kGroup;
    int64_t loadAOpId = appendOp(BufferOpKind::LoadA, {}, {}, loadABundle,
                                 kGroup, -1, -1, -1, -1, -1);
    auto aValueId = addValue(BufferValueKind::SharedTile, "a_shared", loadAOpId,
                             sharedASlot, loadABundle);
    if (failed(aValueId))
      return failure();
    model.ops[loadAOpId].outputs.push_back(*aValueId);

    int64_t loadBOpId = appendOp(BufferOpKind::LoadB, {}, {}, loadABundle,
                                 kGroup, -1, -1, -1, -1, -1);
    auto bValueId = addValue(BufferValueKind::SharedTile, "b_shared", loadBOpId,
                             sharedBSlot, loadABundle);
    if (failed(bValueId))
      return failure();
    model.ops[loadBOpId].outputs.push_back(*bValueId);

    for (int64_t mTile = 0; mTile < accTilesM; ++mTile) {
      int64_t bundle = kGroup * accTilesM + mTile;
      int64_t localLoadAOpId =
          appendOp(BufferOpKind::LocalLoadA, {}, {}, bundle, kGroup, mTile, -1,
                   -1, -1, -1);
      model.ops[localLoadAOpId].inputs.push_back(*aValueId);
      appendUser(*aValueId, localLoadAOpId);
      auto localAValueId = addValue(BufferValueKind::DotOperandFragment, "a_frag",
                                    localLoadAOpId, localASlotBase + mTile,
                                    bundle);
      if (failed(localAValueId))
        return failure();
      localAValues[mTile] = *localAValueId;
      model.ops[localLoadAOpId].outputs.push_back(*localAValueId);
    }

    for (int64_t nGroup = 0; nGroup < numBGroups; ++nGroup) {
      int64_t bundle = kGroup * numBGroups + nGroup;
      int64_t localLoadBOpId =
          appendOp(BufferOpKind::LocalLoadB, {}, {}, bundle, kGroup, -1, -1,
                   nGroup, -1, -1);
      model.ops[localLoadBOpId].inputs.push_back(*bValueId);
      appendUser(*bValueId, localLoadBOpId);
      auto localBValueId = addValue(BufferValueKind::DotOperandFragment,
                                    "b_frag_group", localLoadBOpId,
                                    localBSlotBase + nGroup, bundle);
      if (failed(localBValueId))
        return failure();
      localBValues[nGroup] = *localBValueId;
      model.ops[localLoadBOpId].outputs.push_back(*localBValueId);
    }

    for (int64_t mTile = 0; mTile < accTilesM; ++mTile) {
      for (int64_t nGroup = 0; nGroup < numBGroups; ++nGroup) {
        for (int64_t groupOffset = 0; groupOffset < bTileSpanN; ++groupOffset) {
          int64_t nTile = nGroup * bTileSpanN + groupOffset;
          int64_t accIndex = mTile * accTilesN + nTile;
          int64_t bundle = kGroup * accFragments + accIndex;
          int64_t mmaOpId =
              appendOp(BufferOpKind::Mma, {}, {}, bundle, kGroup, mTile, nTile,
                       nGroup, groupOffset, accIndex);
          model.ops[mmaOpId].inputs.push_back(localAValues[mTile]);
          model.ops[mmaOpId].inputs.push_back(localBValues[nGroup]);
          appendUser(localAValues[mTile], mmaOpId);
          appendUser(localBValues[nGroup], mmaOpId);
          if (previousAccValues[accIndex] >= 0) {
            model.ops[mmaOpId].inputs.push_back(previousAccValues[accIndex]);
            appendUser(previousAccValues[accIndex], mmaOpId);
          }

          auto accValueId = addValue(BufferValueKind::AccumulatorFragment, "acc",
                                     mmaOpId, accSlotBase + accIndex, accIndex);
          if (failed(accValueId))
            return failure();
          model.ops[mmaOpId].outputs.push_back(*accValueId);
          previousAccValues[accIndex] = *accValueId;
        }
      }
    }
  }

  DenseMap<int64_t, int64_t> kGroupByOp;
  kGroupByOp.reserve(model.ops.size());
  for (const PipelineOp &opInfo : model.ops) {
    int64_t kGroup = findIterationCoord(opInfo.iterationCoords, "k_group");
    if (kGroup < 0) {
      op->emitError() << "buffer model requires every pipeline op to carry a "
                         "`k_group` iteration coordinate";
      return failure();
    }
    kGroupByOp[opInfo.id] = kGroup;
  }
  for (ValueState &value : model.values) {
    if (value.definingOp < 0)
      continue;
    auto defIt = kGroupByOp.find(value.definingOp);
    if (defIt == kGroupByOp.end())
      continue;

    std::optional<int64_t> carriedDistance;
    for (int64_t userId : value.users) {
      auto userIt = kGroupByOp.find(userId);
      if (userIt == kGroupByOp.end())
        continue;
      int64_t distance = userIt->second - defIt->second;
      if (distance <= 0)
        continue;
      if (!carriedDistance) {
        carriedDistance = distance;
        continue;
      }
      if (*carriedDistance != distance) {
        op->emitError() << "stage1 buffer model only accepts one loop-carry "
                           "distance per value, but value "
                        << value.id << " reaches distances " << *carriedDistance
                        << " and " << distance;
        return failure();
      }
    }
    value.loopDistance = carriedDistance.value_or(0);
  }

  if (failed(validateBufferModel(model, op)))
    return failure();
  return model;
}

DictionaryAttr mlir::tb::buildBufferModelAttr(Builder &builder,
                                              const BufferModel &model) {
  NamedAttrList attrs;
  attrs.set("backings", buildDictArrayAttr(builder, model.backings, buildBackingAttr));
  attrs.set("views", buildDictArrayAttr(builder, model.views, buildViewAttr));
  attrs.set("values", buildDictArrayAttr(builder, model.values, buildValueAttr));
  attrs.set("ops", buildDictArrayAttr(builder, model.ops, buildOpAttr));
  return builder.getDictionaryAttr(attrs);
}

FailureOr<BufferModel> mlir::tb::parseBufferModelAttr(Operation *op) {
  auto root = dyn_cast_or_null<DictionaryAttr>(op->getAttr("tb.buffer_model"));
  if (!root) {
    op->emitError() << "missing `tb.buffer_model` attribute";
    return failure();
  }

  auto backings = parseDictArrayAttr<BufferBacking>(root.get("backings"),
                                                    "backings", op,
                                                    parseBackingAttr);
  auto views = parseDictArrayAttr<BufferView>(root.get("views"), "views", op,
                                              parseViewAttr);
  auto values = parseDictArrayAttr<ValueState>(root.get("values"), "values",
                                               op, parseValueAttr);
  auto ops = parseDictArrayAttr<PipelineOp>(root.get("ops"), "ops", op,
                                            parseOpAttr);
  if (failed(backings) || failed(views) || failed(values) || failed(ops)) {
    op->emitError() << "malformed `tb.buffer_model` attribute";
    return failure();
  }

  BufferModel model;
  model.backings = std::move(*backings);
  model.views = std::move(*views);
  model.values = std::move(*values);
  model.ops = std::move(*ops);
  if (failed(validateBufferModel(model, op))) {
    op->emitError() << "malformed `tb.buffer_model` attribute";
    return failure();
  }
  return model;
}
