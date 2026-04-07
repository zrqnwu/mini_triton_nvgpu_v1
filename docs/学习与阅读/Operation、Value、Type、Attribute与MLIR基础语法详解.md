# Operation、Value、Type、Attribute与MLIR基础语法详解

更新时间：2026-04-07

适用项目：

- `/home/zhangruiqi/mini_triton_nvgpu_v1`

适用对象：

- 已经有一点点 `MLIR` 和 `CUDA` 基础
- 想先把 `Operation / Value / Type / Attribute` 这四个概念彻底弄清楚
- 想结合当前项目的真实代码理解 MLIR 语法，不想只看抽象定义

---

## 一、先说结论

如果你现在只记一句话，可以先记这个：

- `Operation` 是“做事的节点”
- `Value` 是“数据流里的值”
- `Type` 是“值长什么样”
- `Attribute` 是“编译期静态元数据”

而 MLIR 这样设计的核心原因是：

- 行为
- 数据流
- 静态形状
- 编译期真相

这四类信息，本来就不是同一种东西。

如果把它们全混在一起，编译器会很快失控。  
你当前这个项目本质上也正是在利用这套拆分，把不同层次的真相放在不同地方。

---

## 二、先用一段真实 IR 当例子

先看你仓库里的示例：

文件：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/examples/smoke_64x64x32.mlir`

内容：

```mlir
module attributes {"tb.num-warps" = 1 : i64, "tb.requested-stages" = 2 : i64} {
  func.func @kernel(%A: memref<64x32xf16>, %B: memref<32x64xf16>,
                    %C: memref<64x64xf32>) {
    tb.matmul %A, %B, %C {block_m = 64 : i64, block_n = 64 : i64,
                          block_k = 32 : i64, exact_tile = true}
      : memref<64x32xf16>, memref<32x64xf16>, memref<64x64xf32>
    func.return
  }
}
```

在这段 IR 里：

- `module` 是一个 `Operation`
- `func.func` 是一个 `Operation`
- `tb.matmul` 是一个 `Operation`
- `%A`、`%B`、`%C` 是 `Value`
- `memref<64x32xf16>`、`memref<32x64xf16>`、`memref<64x64xf32>` 是 `Type`
- `"tb.num-warps" = 1 : i64`、`block_m = 64 : i64`、`exact_tile = true` 是 `Attribute`

你以后读任何 MLIR，都可以先按这四类去拆。

---

## 三、什么是 Operation

## 1. 最直观理解

`Operation` 就是 IR 里的“节点”。

它表示：

- 做了一件事
- 有自己的输入
- 可能有输出
- 可能带静态元数据
- 可能带嵌套区域

也就是说，MLIR 里真正的基本单位不是“语句”，而是 `Operation`。

## 2. 一个 Operation 通常会带什么

从概念上，一个 op 大致有这些部分：

- 名字
- operands
- results
- attributes
- regions
- location

并且还可能有：

- verifier
- folder
- traits

## 3. 结合你项目看真实 op

你项目里最典型的 op 定义在：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td`

### 例子 1：`tb.matmul`

定义位置：

- `TBOps.td` 第 32 行附近

它的特点：

- 有 3 个输入：
  - `a`
  - `b`
  - `c`
- 有多个静态 attrs：
  - `block_m`
  - `block_n`
  - `block_k`
  - `group_m`
  - `exact_tile`
- 没有 result

这说明：

- **不是所有 op 都必须有输出 value**
- 有些 op 更像“高层语义声明”

`tb.matmul` 就是在说：

- 这里有一个高层 matmul 语义节点
- 先拥有问题规模和 tile 语义
- 后面的 layout / transport / pipeline 真相由 pass 再推出来

### 例子 2：`tb.convert_layout`

定义位置：

- `TBOps.td` 第 7 行附近

它的特点：

- 有 1 个输入 `source`
- 有 1 个输出 `result`

这说明：

- 它是典型的“消费一个 value，产出一个新 value”的 op

这个 op 在你项目里非常关键，因为它在表达：

- layout conversion 是显式行为
- 不能让后面的 pass 从 side-car 结构里偷偷猜出“布局已经变了”

### 例子 3：`tb.pipeline_mainline`

定义位置：

- `TBOps.td` 第 60 行附近

它的特点：

- 有 3 个输入：`a / b / c`
- 没有 result
- 有一个 region：`body`

这说明：

- op 不只是“单条指令”
- 还可以是一个“承载结构化主线”的语义容器

这正是为什么 `tb.pipeline_mainline` 能作为：

- post-cleanup
- pre-lowering

的显式执行 owner。

## 4. 你读项目时怎么看 Operation

以后看到一个 op，先问：

1. 它做的是什么事？
2. 它的输入是哪些 operands？
3. 它有没有结果 value？
4. 它的 attrs 在表达什么静态真相？
5. 它有没有 region，是否在承载结构化语义？

只要先问这 5 个问题，op 就不会看乱。

---

## 四、什么是 Value

## 1. 最直观理解

`Value` 是 IR 里的“值”。

它表示：

- 某个 op 的结果
- 或者 block/function 的参数

MLIR 和 LLVM 一样，采用的是 SSA 风格，所以：

- 一个 value 只有一个定义点
- 后面所有使用都引用它

## 2. 结合你项目里的真实例子

还是看示例：

```mlir
func.func @kernel(%A: memref<64x32xf16>, %B: memref<32x64xf16>,
                  %C: memref<64x64xf32>) {
```

这里：

- `%A` 是一个 value
- `%B` 是一个 value
- `%C` 是一个 value

它们的定义点不是某个 op 结果，而是函数参数。

### 再看有 result 的 op

例如你项目里：

- `tb.convert_layout`
- `tb.epilogue_global_vector_load`

它们都有 result，所以也会产出新的 value。

比如 [tb.epilogue_global_vector_load](/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td) 的 `result`，就是一个新 value。

## 3. Value 和 Attribute 的根本区别

这是最重要的一点。

`Value`：

- 参与 SSA 数据流
- 代表运行时会流动的数据

`Attribute`：

- 不参与 SSA 数据流
- 代表编译期静态信息

所以：

- `%A` 是 value
- `block_m = 64 : i64` 不是 value，是 attr

## 4. Value 为什么重要

因为编译器必须清楚知道：

- 数据从哪里来
- 到哪里去
- 哪些 op 使用了它

没有这一层，你后面的：

- def-use 分析
- rewrite
- canonicalization
- lowering

都会变得很难做。

---

## 五、什么是 Type

## 1. 最直观理解

`Type` 是“值长什么样”的描述。

它主要表达：

- 这是哪一类值
- 形状是什么
- 元素类型是什么

## 2. 结合你项目里的真实例子

示例里：

- `%A` 的 type 是 `memref<64x32xf16>`
- `%B` 的 type 是 `memref<32x64xf16>`
- `%C` 的 type 是 `memref<64x64xf32>`

这些都不是 attr，而是 type。

## 3. 你项目里最常见的 type

### `memref<64x32xf16>`

含义：

- container 是 `memref`
- shape 是 `64 x 32`
- 元素类型是 `f16`

### `memref<64x64xf32>`

含义：

- 二维 memref
- 元素类型是 `f32`

### `index`

含义：

- MLIR 的索引类型
- 常用于地址计算、loop index、memref 索引

在你项目里，像：

- `tb.epilogue_global_vector_load`
- `tb.epilogue_global_vector_store`

就会用到 `Index:$row` 和 `Index:$col`

### `vector<4xf32>`

含义：

- 长度为 4 的 `f32` 向量

在你项目里，late epilogue vector IO 就和这种 vector type 强相关。

## 4. Type 和 Attribute 的区别

再强调一次：

- `memref<64x32xf16>` 是 type
- `64 : i64` 不是 type，是一个 attribute 里的整数字面量

最简单判断方法：

- 如果它描述 value 的形状和元素种类，通常是 type
- 如果它是静态配置/合同，通常是 attr

## 5. 为什么 MLIR 不把所有真相都塞进 Type

这是很重要的设计观念。

在你项目里，很多关键真相并没有直接塞进 type，例如：

- shared swizzle
- operand fragment lane 布局
- target landing kind
- async group / wait frontier

原因是：

- type 适合描述“值本身的基本静态形状”
- 但不适合承载所有中高层优化语义

否则 type 会膨胀得非常难维护。

所以 MLIR 常常是：

- 基础形状放 type
- 更复杂的语义放 attr / op / contract

---

## 六、什么是 Attribute

## 1. 最直观理解

`Attribute` 是编译期静态元数据。

它不是 value，不参与运行时数据流。

它通常用来表达：

- 静态配置
- 编译期真相
- pass 构造出的合同

## 2. 结合你项目里的真实例子

### 例子 1：`tb.matmul` 上的 attr

定义位置：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td`

对应字段：

- `block_m`
- `block_n`
- `block_k`
- `group_m`
- `exact_tile`

打印出来就是：

```mlir
tb.matmul %A, %B, %C {block_m = 64 : i64, block_n = 64 : i64,
                      block_k = 32 : i64, exact_tile = true}
```

这些都是 `Attribute`。

为什么不用 value？

因为它们不是运行时数据流，而是：

- tile 请求
- 编译期语义标志

### 例子 2：module attrs

示例文件最外层：

```mlir
module attributes {"tb.num-warps" = 1 : i64, "tb.requested-stages" = 2 : i64}
```

这里的：

- `"tb.num-warps"`
- `"tb.requested-stages"`

也是 attribute，不过是 module-level attr。

它们表示：

- 整个 module / kernel 的执行上下文

### 例子 3：pass 生成的新 attr

看：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/SemanticizeMatmul.cpp`

关键代码：

```cpp
op->setAttr("tb.semantic_matmul",
            buildMatmulSemanticsAttr(builder, *semantics));
```

这说明：

- `tb.semantic_matmul` 是一个 op attr
- 它不是输入时就有的
- 而是 pass 分析出来以后挂上的显式合同

再看：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AttachTargetInfo.cpp`

这里会把 target truth 挂到 module attr 上。

这说明：

- attr 不只是写死在源码里的字面量
- 也可以是 pass 逐步构造出来的编译期真相

## 3. 你项目里常见的 attribute 类型

从 `TBOps.td` 里能看到：

- `I64Attr`
- `BoolAttr`
- `StrAttr`
- `DenseI64ArrayAttr`

它们常用于表达：

- 整数静态参数
- 布尔开关
- 说明字符串 / reason
- ID 列表、cluster 覆盖列表

## 4. Attribute 为什么在你项目里特别关键

因为你这个项目最重要的设计之一就是：

- 把很多原本可能藏在 lowering 或 side-car 里的真相
- 提前显式挂成 attr / contract

所以你读项目时，会频繁看到：

- pass 先分析
- 然后 `setAttr(...)`
- 后面 pass 再读这个 attr

这正是“owner truth”思路的体现。

---

## 七、MLIR 相关基础语法怎么读

这一部分是你现在最该熟的。

## 1. module 语法

```mlir
module attributes {"tb.num-warps" = 1 : i64, "tb.requested-stages" = 2 : i64} {
  ...
}
```

含义：

- `module` 本身是一个 operation
- `attributes {...}` 是它的 attr 字典
- 花括号内部是它的 region/body

## 2. func.func 语法

```mlir
func.func @kernel(%A: memref<64x32xf16>, %B: memref<32x64xf16>,
                  %C: memref<64x64xf32>) {
  ...
}
```

含义：

- `func.func` 是 function op
- `@kernel` 是 symbol 名
- `%A / %B / %C` 是 function 参数，也就是 values
- 后面的 `memref<...>` 是这些 values 的 type

## 3. 通用 op 语法

一个常见 op 大致可以想成：

```mlir
%0 = dialect.op %arg0, %arg1 {attr = value} : type0, type1 -> result_type
```

但实际打印格式经常会被自定义 `assemblyFormat` 改写。

所以你看到的 op 表面格式可能和 generic form 不一样。

## 4. `%名字` 语法

例如：

- `%A`
- `%B`
- `%0`

它们都是 SSA values。

## 5. `@名字` 语法

例如：

- `@kernel`

通常表示 symbol 名，比如函数名。

## 6. type 语法

你项目里最常见的：

- `i64`
- `f16`
- `f32`
- `index`
- `memref<64x32xf16>`
- `vector<4xf32>`

## 7. attribute 语法

最常见形式：

- `64 : i64`
- `true`
- `"some_reason"`
- `{block_m = 64 : i64, exact_tile = true}`
- `{"tb.num-warps" = 1 : i64}`

## 8. region 语法

像：

- `module { ... }`
- `func.func { ... }`
- `tb.pipeline_mainline { ... }`

这些花括号内部都是 region。

这说明这些 op 不是扁平节点，而是能承载结构化 body。

## 9. `attr-dict`

在 ODS 的 `assemblyFormat` 里经常能看到：

- `attr-dict`

它的作用是：

- 把没有单独展开打印的 attrs 统一打印成 `{...}`

所以你看到：

```mlir
tb.matmul %A, %B, %C {block_m = 64 : i64, ...}
```

本质上就是 `attr-dict` 打印出来的。

---

## 八、再往深一点：为什么 MLIR 要把这四个概念分开

因为它们本来就是不同类别的信息。

## 1. 如果没有 Operation

你很难表达：

- 高层语义节点
- 显式执行边界
- 带 region 的结构化主线

## 2. 如果没有 Value

你很难清楚表达：

- 数据从哪里来
- 数据流到哪里去
- 哪些 op 使用了某个结果

## 3. 如果没有 Type

你很难表达：

- 值到底是 memref 还是 vector
- 形状和元素类型是什么

## 4. 如果没有 Attribute

你会被迫把大量静态信息塞进：

- value
- type
- 或者 lowering 私有逻辑

最终整个编译器会很乱。

所以 MLIR 把这四类信息拆开，是为了：

- 行为和数据流分离
- 值的基本静态信息和更高层合同分离
- 让 pass 更容易做结构清楚的逐层收口

而你当前这个项目，本质上就是这种设计思想的直接实践。

---

## 九、在你这个项目里，这四个概念分别扮演什么角色

可以直接记成这张表：

| 概念 | 在你项目里的角色 | 例子 |
|---|---|---|
| `Operation` | 表达语义节点和执行边界 | `tb.matmul`、`tb.pipeline_mainline` |
| `Value` | 表达真实数据流 | `%A`、`%B`、`%C`、layout convert 的结果 |
| `Type` | 表达值的静态形状与元素种类 | `memref<64x32xf16>`、`vector<4xf32>` |
| `Attribute` | 表达编译期静态真相和 pass 合同 | `block_m`、`exact_tile`、`tb.semantic_matmul` |

---

## 十、你现在最容易混淆的几个地方

## 1. `Value` 和 `Attribute`

- `%A` 是 value
- `block_m = 64 : i64` 是 attr

## 2. `Type` 和 `Attribute`

- `memref<64x32xf16>` 是 type
- `64 : i64` 是 attr 里的整数值

## 3. `Operation` 和 `Value`

- `tb.matmul` 是 operation
- `%A` 是 value

## 4. ODS 里的 `arguments` 不全是 operand

这点很关键。

在 `TBOps.td` 里，`let arguments = (ins ...)` 里既可能有：

- operand/value-like 输入
- 也可能有 attr

区分方法：

- `AnyMemRef`、`Index` 这种通常是 operands
- `I64Attr`、`BoolAttr`、`StrAttr` 这种是 attrs

例如 `tb.epilogue_global_vector_load`：

- `source`、`row`、`col` 是 operands
- `vector_width`、`boundary_aware` 是 attrs

---

## 十一、你现在最该做的 3 个练习

## 练习 1：拆解示例 IR

打开：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/examples/smoke_64x64x32.mlir`

把里面每个东西都标一遍：

- 哪些是 operation
- 哪些是 value
- 哪些是 type
- 哪些是 attribute

## 练习 2：专门读 3 个 op

打开：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/include/tb/IR/TBOps.td`

重点读：

- `tb.matmul`
- `tb.pipeline_mainline`
- `tb.epilogue_global_vector_load`

把它们的：

- operands
- results
- attrs
- regions

分别列出来。

## 练习 3：看两个 pass 怎么挂 attr

打开：

- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/AttachTargetInfo.cpp`
- `/home/zhangruiqi/mini_triton_nvgpu_v1/lib/Transforms/SemanticizeMatmul.cpp`

重点看：

- module attr 是怎么挂上的
- op attr 是怎么挂上的

把这两个 pass 各自写一句总结：

- “这个 pass 构造了什么静态真相，并把它挂在什么地方”

---

## 十二、一句话总结

对你现在这个项目来说，先把这四件事彻底弄清楚非常重要：

- `Operation` 是节点
- `Value` 是数据流
- `Type` 是值长什么样
- `Attribute` 是编译期静态真相

只要这四个概念稳了，你再去读：

- `tb.matmul`
- `EncodingPlan`
- `EpiloguePlan`
- `tb.pipeline_mainline`
- 各层 pass

就不会只是“看代码”，而是会开始真正理解这个项目为什么这样设计。
