(chap_data_layout)=
# 数据布局及其表示法

:::{admonition} 概览
:class: overview

- *数据布局*将张量的逻辑索引映射到物理位置，它决定合并、bank 冲突，以及引擎是否能读取块。
- 本书用一种表示法书写布局：`S[(shape) : (strides)]`，带命名轴（`@laneid`、`@TLane`……）和用于广播或复制数据的复制项 `R[...]`。
- Swizzle 是地址的 XOR 重映射，消除共享内存 bank 冲突。
:::

相同的数字，以不同的物理排列写入内存，在同一 GPU 上可以运行相差一个数量级。

原因是张量的逻辑索引没有说明其字节实际位于何处。硬件对该放置高度敏感：它决定 32 个 lane 的加载是合并为一次事务还是分散为 32 次，它们的地址是落在不同的内存 bank 中还是冲突并串行化，甚至块是否匹配张量核心能读取的字节排列。

机器学习程序通常按逻辑形状描述张量。**数据布局**添加了缺失的物理部分：它说明逻辑索引为 `(i, j, …)` 的元素存在于何处，无论是在内存中、寄存器中，还是在其他硬件存储中。

本章介绍现代 GPU 编程中出现的主要布局。为了使讨论易于管理，我们开发一种紧凑的**表示法**来描述它们在机器学习系统遇到的各种情况中的表现。我们以**swizzling**结束，这是一种使行式和列式访问块同时高效的机制。

## 形状-步长模型

在到达 GPU 特定布局之前，值得从最简单的开始，因为本章中的其他一切都是建立在它之上的。在其核心，布局只是两件事：一个**形状**和一组匹配的**步长**。我们将这对写为 `S[(shape) : (strides)]`，要找到逻辑索引的位置，我们将该索引与步长进行点积。例如，行主序 4×4 矩阵看起来像这样：

```text
S[(4, 4) : (4, 1)]        addr(i, j) = i·4 + j·1
```

这不过是经典的形状/步长模型，紧凑地书写（CuTe 表示法的行主序简化），后续一切都是从它构建的。

事实上，你几乎肯定已经使用过这个模型。任何编写过 PyTorch 或 NumPy 的人都用过，因为这些库中的张量*正是*一个形状加上平坦存储缓冲区上的步长：

```python
import torch
t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← 恰好是 S[(3, 4) : (4, 1)]
```

一旦你以这种方式看待张量，就很清楚为什么这么多"重塑"操作从不触及数据。它们只是重写步长并返回同一存储上的**视图**，最清晰的例子是转置或排列：

```python
tt = t.permute(1, 0)               # 或 t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← 步长交换，没有数据移动
tt.data_ptr() == t.data_ptr()      # True，相同字节
```

这里 `t.permute(1, 0)` 是*同一*内存上的 `S[(4, 3) : (1, 4)]`：转置纯粹是步长的改变，没有移动一个字节。连续张量上的 `reshape` 或 `view` 故事相同：旧存储上的新形状和新步长。（NumPy 行为相同；唯一的区别是它的 `.strides` 以字节而非元素计数。）

这正是布局在 GPU 上的工作方式，本章其余部分实际上是一个想法的一系列变体：块的映射（无论进入内存，还是通过我们即将引入的命名轴进入 lane 和寄存器）是固定缓冲区上的步长规则，因此重新排列块通常是*布局*的改变而不是复制。我们应该注意这种推理的边界。零拷贝故事在单线性地址空间上的逻辑视图上干净地成立；在 GPU 上，它仅在新视图与现有字节和所有权排列兼容时适用。一旦你改变哪个线程或寄存器拥有元素，或改变 SMEM swizzle，你通常需要真正的数据移动：加载、存储、洗牌、`ldmatrix`、转置。

## 块布局

到目前为止，我们描述了整个张量的布局。然而，GPU 内核很少一次操作整个矩阵；它们处理较小的块，这些块由硬件的不同部分加载、转换和计算。好消息是分块不需要新东西。它仍然只是一个布局，只是现在用更多维度书写。将 8×8 矩阵切割成 2×4 块，我们得到 4D 布局，坐标为 `(tile_row, row_in_tile, tile_col, col_in_tile)`，步长选择使每个块保持连续：

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

逻辑 `(i, j)` 首先变为 `(i//2, i%2, j//4, j%4)`，然后通过步长运行。值得注意的是，表示法在没有任何特殊"块"概念的情况下表达了分块：它与之前相同的形状-步长模型，索引只是被拆分为外部和内部坐标。

下面的交互可视化显示逻辑矩阵索引如何分解为块坐标，然后映射到物理地址。

```{raw} html
<iframe src="../demo/tiled_layout.html" title="块布局：交互式地址计算" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互演示：点击单元格查看其分块索引和地址。*

## 命名轴

到目前为止，`S[...]` 中的每个步长都命名了线性内存中的偏移，我们将地址视为那里的位置。然而在 GPU 上，数据可以存在于多个位置：除了内存，块可以分布在 warp lane、线程寄存器，或 TMEM lane 和列之间。为了统一描述所有这些，我们用**命名轴**扩展表示法。想法是让每个步长系数携带一个轴标签，说明它通过哪个空间移动：`@m` 用于普通内存，`@laneid` 用于 warp lane，`@reg` 用于寄存器，`@warpid` 用于 warp，`@TLane`/`@TCol` 用于 TMEM 坐标。有了标签，单个布局不仅可以描述数据在内存中的位置，还可以描述它如何分布在操作它的硬件资源上。

一旦内存标签变得显式，内存中的行主序 8×16 块简单地是

```text
S[(8, 16) : (16@m, 1@m)]
```

当布局描述*跨线程分布*的数据而非内存中的数据时，标签开始发挥作用。取 `S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]`：它不是指向线性内存，而是将行和列映射到 lane ID 和每 lane 寄存器。这里 `laneid` 意味着 warp 内的 warp lane 索引，`thread_index % warp_size`。这正是你在 {ref}`chap_layout_generations` 中将遇到的张量核心寄存器片段。

下面的交互可视化显示布局如何跨 warp lane 和每 lane 寄存器分布张量元素，而不是将它们放在线性内存中。

```{raw} html
<iframe src="../demo/thread_register.html" title="通过命名轴的线程 + 寄存器布局" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互演示：`@laneid` 和 `@reg` 上的布局；点击单元格查看哪个 lane/寄存器持有它。*

## 分布式布局

命名轴如此有用的原因是它们让我们能够跨系统多个级别统一描述放置，包括*跨整个设备*的放置。我们刚刚将它们用于单个 GPU 内的 lane 和寄存器，但同样的思路向外延伸：`@gpuid_x` 和 `@gpuid_y` 等轴可以说明数据在 GPU 网格中的位置，借助它们，表示法捕获了分布式训练和推理中出现的分片模式。轴尚未捕获的是*复制*，即复制到多个位置的数据，因此我们添加表示法 `R[n : stride]`，其中 `R` 标记复制维度。例如，`R[2 : 1@gpuid_x]` 描述沿 `@gpuid_x` 轴的复制。将两者结合，单个表达式可以同时将张量分片到 2×2 GPU 网格并沿一个轴复制：

```text
S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)] + R[2 : 1@gpuid_x]
```

下面的演示在小型 GPU 网格上显示该组合分区和复制模式。点击任何单元格查看哪个设备持有它，观察 `@gpuid_x` 复制如何在配对设备上放置相同的副本；按钮在完全分片、分片 + 副本和分片 + 偏移布局之间切换。

```{raw} html
<iframe src="../demo/tile_distributed.html" title="跨 GPU 网格的分布式布局" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互演示：跨 2×2 GPU 网格分布的布局；点击单元格查看哪个设备持有它。*

### 内核内复制模式：TMEM 中的缩放因子

我们刚刚为 GPU 网格引入的复制维度 `R[...]` 不仅关于多个设备。同样的构造也被证明描述了完全发生在单个内核内部的事情：硬件*跨 lane 广播*的数据。Blackwell 的分块缩放 MMA（{ref}`chap_layout_generations`）就是一个好例子。其缩放因子存在于 TMEM 中，其中 128 行缩放向量仅存储在 **32 个 TMEM lane** 中，逻辑行 `r` 到 TMEM lane `r % 32`，`r // 32` 沿列运行。这 32 个存储的 TMEM lane 然后**沿 TMEM `TLane` 轴复制**，从 32 个到 128 个 TMEM lane，使读取 warpgroup 中的四个 warp 中的每一个都在自己的 32 lane TMEM 窗口中找到副本。这是一个 `warpx4` 广播，我们用复制维度书写它。读取本身由这些 warp 的线程执行：

```text
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]
```

这给出四个步长为 32 TMEM lane 的副本：TMEM lane `l`、`l+32`、`l+64` 和 `l+96` 都持有相同的缩放。如前所述，复制维度不携带新数据；它只是说"相同的值，位于四个 TMEM lane 位置中"，就像刚才 `@gpuid_x` 跨 GPU 网格广播一行一样。

下面的交互演示同时显示两个步骤：紧凑打包到 32 个 TMEM lane，然后 `warpx4` 广播到 128 个读取 lane。

```{raw} html
<iframe src="../demo/sf_tmem.html" title="TMEM 中的缩放因子：打包和 warpx4 复制" loading="lazy"
        style="width:100%; min-width:1040px; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互演示：点击缩放因子 `SFA[m, sf]`；它在 lane `m mod 32`、列 `(m // 32)·4 + sf` 处打包到 TMEM，然后 `warpx4` 沿 `TLane` 轴广播到四个 lane 副本（`l`、`l+32`、`l+64`、`l+96`），每个 warp 的 32 lane 窗口一个。*

每列内的字节打包（`scale_vec` 1X/2X/4X 模式）和 `cta_group::2` 拆分在 {ref}`chap_layout_generations` 中介绍。

已经了解 CuTe 的读者可以将本章中的表示法视为其行主序变体，用显式硬件命名轴和专用复制结构扩展。

## Swizzle 布局

本章最后的布局为了解决一个特定的硬件问题。GPU 上的共享内存组织为内存 bank，当不同 lane 落在不同 bank 上时，访问运行最快。当几个 lane 而不是落在*同一* bank 内的不同地址时，硬件别无选择只能串行化它们，我们付出 **bank 冲突**的代价。

在张量程序中这很难避免，因为内存不是以纯线性顺序访问的。处理矩阵时，我们经常需要读取同一块的行切片和列切片，这产生了真正的张力：对行式访问高效的布局往往对列式访问产生 bank 冲突，而有利于列的布局损害行。**Swizzling**是旨在打破这种张力的技术。

swizzle 背后的想法是排列地址映射，通常通过将列索引与行进行 XOR，使*行和列*访问最终分布在不同 bank 上。它提供的无冲突保证是特定的：它适用于匹配的元素宽度、swizzle 模式和访问模式（引擎描述符期望的模式），而不适用于任意元素宽度或对齐。

下面的第一个交互演示使这一点具体化。点击列索引并观察每个元素落在哪个 bank 中：在左侧的普通行主序列块中，一列将所有八个元素漏斗到单个 bank 中，因此读取串行化为八个周期；在右侧的 XOR-swizzled 布局中，同一列分布在八个不同的 bank 中，单周期读取。

```{raw} html
<iframe src="../demo/swizzle_8x8.html" title="8×8 XOR swizzle" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互演示：8×8 块，在普通行主序中列有 bank 冲突，XOR swizzle 后无冲突。*

小小的 8×8 示例捕获了核心思想，但实际 GPU 内存有比该玩具图片暗示的多得多的 bank。为了使 swizzle 在全规模下工作，我们不将整个块视为一个整体对象。相反，我们将内存切割成小段，并在每段内应用 swizzle 模式。实践中最常见的情况是 `SWIZZLE_128B`，围绕 128 字节段组织，使相同的行/列重映射技巧自然地适应 32 bank 内存系统。

下面的交互演示显示一个具体的硬件 swizzle `SWIZZLE_128B`，因此在我们跨格式泛化之前，重复的逐段模式是可见的。

```{raw} html
<iframe src="../demo/swizzle_128B.html" title="SWIZZLE_128B 布局" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互演示：128 字节段内的 `SWIZZLE_128B` 模式；逐步读取周期以查看 `physical_sector = logical_sector XOR row` 将每列分布在不同 bank 上。*

同样的思路超越这个 128 字节情况。为了简化可视化，我们现在将使用单个色块来引用一个段，而不是绘制单个 bank。通常，硬件定义一个小的重复**原子**，在其上应用排列，不同的 swizzle 模式选择不同的原子大小。`SWIZZLE_128B` 使用 8 × 128 B 原子，`SWIZZLE_64B` 使用 8 × 64 B 原子，`SWIZZLE_32B` 使用 8 × 32 B 原子；然后整个块由正在使用的原子分块。

最后的交互演示让你在这些格式之间切换（包括 16 B 交织模式），选择数据类型，并悬停任何单元格以直接检查一个原子内的元素排列，这是推理加载/存储指令期望哪个 swizzle 的正确详细程度。

```{raw} html
<iframe src="../demo/swizzle_atom_general.html" title="每种格式的 swizzle 原子布局 (128B/64B/32B)" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互演示：选择 swizzle 格式（和数据类型）以查看其原子形状（8 × N B）；悬停单元格以查看其元素如何被排列。*

你应该选择哪种模式？经验法则是优先选择块能填充的*最大*原子。N 字节原子需要块的连续维度至少 N 字节，且是其倍数，因此 `SWIZZLE_128B` 仅在行跨越至少 128 字节或 64 个 `float16` 元素时适用。当它适合时，它是默认选择，因为其 8 × 128 B 原子覆盖完整的 128 字节 bank 行，因此一次将一列分散到所有 32 个 bank，在 fp16 中一次提供 8 行和 8 列的无冲突访问。当问题形状强制连续维度较小时，块不再能填充 128 B 原子，你降级到 `SWIZZLE_64B` 或 `SWIZZLE_32B`，行仍能覆盖的最大原子。

你永远不会手动计算这些排列地址，值得精确说明 swizzle 与 `S[...]` 表示法的关系：它*不是*该仿射映射的一部分。它是组合在其上的独立非仿射层。`S[...]` 布局将元素放在线性内存（`@m`）地址上，然后 swizzle 排列该地址，在 TIRx 布局 API 中写为 `ComposeLayout(swizzle, tile)`（{ref}`chap_tirx_layout_api`）。你的工作只是为接触块的每个操作选择一个一致的模式，让组合布局完成其余工作。

相同的组合布局也是硬件填充的，这就是 swizzling 和分块结合的地方。TMA 描述符是多维的，因此单个三维框可以描述块的原子分块和每个原子内的 swizzle；一次 TMA 加载然后逐原子铺设块并在写入共享内存时对其进行 swizzle（{ref}`chap_tma`），无需单独的 swizzle 传递。*哪种* swizzle 每个引擎要求是特定于代的，这是下一章的主题。
