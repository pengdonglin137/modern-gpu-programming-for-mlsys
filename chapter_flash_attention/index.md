(chap_flash_attention)=
# Flash Attention 4

:::{admonition} 概览
:class: overview

- 注意力运行两个 MMA，中间夹着 softmax，因此不能像 GEMM 那样重复一个 MMA。
- 内核组合第一部分的硬件原语（TMA、`tcgen05`、TMEM、屏障）和第三部分的 GEMM 技术，加上 warp 角色、在线 softmax 重缩放、因果掩码和 GQA。
:::

注意力是决定 transformer 能否运行的内核，也是我们到目前为止构建的所有内容最终必须协同工作的地方。我们为 GEMM 组装的每个部分都延续到这里：TMA 块移动、`tcgen05` MMA、TMEM、warpgroup 寄存器块和显式屏障。

挑战在于注意力不是一个 MMA 的重复。它是两个 MMA，中间夹着真正的工作：在线 softmax、因果掩码以及保持前后块在共同缩放中的重缩放。

中间阶段是新困难所在。普通矩阵乘法只是累加到累加器；注意力必须在新的键和值流式传入时重新访问和重缩放它已经计算的结果。softmax 工作本身也在两个张量核心 MMA 之间在 CUDA 核心上运行，因此指数和行级归约直接位于关键路径上。

这就是为什么注意力优化的大部分实际上是 softmax 优化：重新表述 `exp`，并将 softmax 与 MMA 重叠而不是在它上面停顿。

我们在本章的目标不是从头重新推导 Flash Attention。我们将只保留足够的算法使内核可读，然后将注意力放在真正新的部分：该算法如何转化为 TIRx。

最清晰的方式是跟踪一个块流经内核的过程。`Q`、`K` 和 `V` 作为输入块进入，从 GMEM 加载到 SMEM。分数 MMA 将 `Q` 和 `K` 乘入 TMEM 中的分数块 `S`。Softmax 将 `S` 转换为分子块 `P`，值 MMA 组合 `P` 和 `V` 以更新输出累加器 `O`。

到目前为止，这看起来像两个矩阵乘法粘在一起，但有一个 GEMM 从未必须处理的转折：每当运行的 softmax 最大值改变时，到目前为止累加的 `O` 突然处于错误的缩放中。在下一个值 MMA 安全地累加到它之前，必须重新缩放它。下面的章节首先跟踪这条路径，然后才展示 TIRx 如何将每个阶段交给 warpgroup 并将阶段连接在一起。

## 算法形状

在我们可以将块放入内存之前，我们需要这些块服务的算法。对于一个查询块，Flash Attention 计算：

$$O = \text{softmax}(QK^{\top} / \sqrt{d})V$$

字面上阅读，该公式说要形成完整分数矩阵 `S = QKᵀ`，对其做 softmax，然后乘以 `V`。这是我们不能使用的方法，因为完整的 `S` 巨大。在 seq=4096 时，每个头大约有 1600 万个元素，fp32 中约 64 MB，比 SMEM 或单个 128×512 TMEM 区域大几个数量级。芯片上根本没有地方放置它。Flash Attention 的答案是根本不具体化 `S`。相反，它以块流式传输 `K/V`，并携带三个每行运行状态，总结到目前为止看到的所有内容：

- `row_max`：到目前为止看到的最大分数。
- `row_sum`：softmax 的运行分母。
- `O`：运行输出累加器。

流式更新使这些状态在新块到达时保持正确。微妙之处在于，每次我们处理一个块时，运行最大值可能上升，一旦上升，我们在旧最大值下计算的所有内容现在处于错误的缩放中。因此在添加新贡献之前，我们首先将旧状态拉回到新的缩放中：

```text
S = Q_block @ K_block.T
m_new = max(row_max, rowmax(S))
scale = exp((row_max - m_new) / sqrt(d))
P = exp((S - m_new) / sqrt(d))
row_sum = row_sum * scale + rowsum(P)
O = O * scale + P @ V_block
row_max = m_new
```

单个 `scale` 因子在这里承担双重职责：它重新缩放运行分母和运行输出，使前后块的贡献最终以共同缩放衡量。

上面的伪代码使用自然 `exp` 和显式 `/sqrt(d)` 书写，因为这样最容易阅读，但内核采取更便宜的路线。它将 `1/sqrt(d)` 和 `log2(e)` 折叠为一个常数 `scale_log2 = log2(e)/sqrt(d)`，并使用硬件 `exp2` 在原始分数上评估每个指数，使用恒等式 `exp(x/sqrt(d)) = exp2(x · scale_log2)`。动机很简单：在这种硬件上 `exp2` 比自然 `exp` 更快。

有一点值得在我们继续之前确定：这里的 `P` *不是*最终的归一化注意力矩阵。它只是当前 K/V 块的 softmax 分子。归一化被故意推迟，只有在最后一个块之后，内核才写入 `O / row_sum`。

对于 TIRx，知道算法计算什么只是故事的一半。另一半是*每个块存在于何处*，因为那是决定布局和屏障代码的因素。`S`、`P` 和 `O` 都是块值，每个都有一个位置：

- `S` 是分数块。分数 MMA 将其写入 TMEM。
- `P` 是 softmax 分子块。Softmax 从 TMEM 读取 `S` 到寄存器，计算 `P = exp((S - m_new) / sqrt(d))`，并将 `P` 写回 TMEM。
- `O` 是输出累加器块。值 MMA 从 TMEM 读取 `P`，从 SMEM 读取 `V`，然后累加到 TMEM 中的 `O`。

我们之前标记的重缩放也是一个块操作，不是标量簿记：当 `row_max` 改变时，旧 `O` 从 TMEM 读取，在寄存器中相乘，然后写回 TMEM，然后下一个值 MMA 累加到其中。每个后续章节遵循相同的结构：块放置、硬件路径，以及证明下一个消费者可以运行的屏障。

## 块原语图

有了运行状态及其位置，我们可以将算法布局为具体的块移动序列。对于一个 K/V 块，内核从上到下走这条块路径：

```text
Q, K, V 在 GMEM 中
  -> Q, K, V 在 SMEM 中        通过 TMA 加载
  -> S 在 TMEM 中              通过分数 MMA: QK^T
  -> P 在 TMEM 中              通过 softmax 分子: TMEM -> RF -> TMEM
  -> O 在 TMEM 中              通过值 MMA: P V
  -> O 在 GMEM 中              通过归一化、SMEM 暂存和 TMA 存储
```

与 GEMM 的区别归结为一行。GEMM 是一个 MMA 链的重复；FA4 有两个 MMA 阶段，softmax 夹在链的中间。几乎所有后续内容都是那个额外阶段的后果。

如果我们将短路径扩展为显式的生产者-消费者边，我们得到完整的图：

| 阶段 | 块移动或计算 | TIRx 原语 | 硬件路径 |
|------|------------|----------|---------|
| 加载 Q/K/V | GMEM 块 -> SMEM 块 | `Tx.copy_async(..., dispatch="tma")` | TMA 加载 |
| 分数 MMA | SMEM 中的 Q 和 K -> TMEM 中的分数块 `S` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` |
| Softmax 读取 | TMEM 中的 `S` -> warpgroup 寄存器块 | `Tx.wg.copy_async(reg, tmem)` | `tcgen05.ld` |
| Softmax 写入 | 寄存器中的分子块 `P` -> fp16 TMEM 视图 | `Tx.copy_async(tmem_as_f16, reg)` | TMEM 存储，后跟 `tcgen05.wait.st()` |
| 值 MMA | TMEM 中的 `P` 和 SMEM 中的 V -> TMEM 中的输出累加器 `O` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | 带 TMEM 操作数的 `tcgen05.mma` |
| 修正 | TMEM 中的 `O` -> 寄存器 -> TMEM 中的 `O` | TMEM 回读、寄存器乘法、TMEM 存储 | `tcgen05.ld` / TMEM 存储 |
| 尾声 | TMEM 中的最终 `O` -> 寄存器 -> SMEM -> GMEM | TMEM 回读、`Tx.copy`、TMA 存储 | `tcgen05.ld` + TMA 存储 |

新行是 softmax 和修正。两者都添加 TMEM -> 寄存器 -> TMEM 流量，两者都在分数 MMA 和值 MMA 之间创建额外的交接。

**试一试**：让它只跟踪上面的短路径。对于每条箭头，命名生产者阶段、消费者阶段、源块、目标块和硬件路径。然后问哪些箭头在 GEMM 章节中不存在。

## Warp 角色和作用域

数据路径确定后，自然的下一个问题是谁实际运行每个阶段。这里的每个 CTA 有 4 个 warpgroup，总共 512 个线程，它们不是按接触的数据分配，而是按 warpgroup 做的*工作类型*分配：

- WG3 驱动硬件引擎：TMA 加载、MMA 和 TMA 存储。
- WG0、WG1 和 WG2 执行引擎调用之间发生的寄存器密集型数学：softmax、修正和尾声。

确切的角色表是：

| 拥有者 | 角色 | 做什么 |
|--------|------|--------|
| WG3, warp 1 | TMA 加载 | 从 GMEM 加载 Q、K 和 V 块到 SMEM |
| WG3, warp 0 | MMA | 发出分数 MMA 和值 MMA |
| WG3, warp 2 | TMA 存储 | 将最终 O 块从 SMEM 存储到 GMEM |
| WG0 | Q 阶段 0 的 Softmax | 从 TMEM 读取 S，计算 P，将 P 写入 TMEM |
| WG1 | Q 阶段 1 的 Softmax | 第二个 Q 流水线阶段的相同工作 |
| WG2 | 修正和尾声 | 重缩放 TMEM 中的 O，归一化，暂存输出 |

很容易将"两个 Q 阶段"误读为两个注意力头，但它们不是。它们只是 Q 流水线中的两个槽，WG0 拥有一个，WG1 拥有另一个，这样两个 Q 块可以同时在飞行中。这就是 softmax 工作出现两次的原因，一次在 WG0 上，一次在 WG1 上。

代码用符号坐标选出这些角色：

```python
wg_id = T.warpgroup_id([4])
warp_id = T.warp_id_in_wg([4])
```

阅读内核时，首先找到角色分支。它告诉你哪个团队拥有嵌套在其中的每个块原语。

- WG3 warp 1 启动 TMA 加载命令。一个选举 lane 发出复制，TMA 引擎移动块。
- WG3 warp 0 发出 `tcgen05.mma` 指令。
- WG0 和 WG1 在完整 warpgroup 作用域下运行 softmax。
- WG2 在完整 warpgroup 作用域下运行修正和尾声工作。

一个不对称最终塑造了整个屏障图：*每个* MMA，分数和值，都只从 WG3 warp 0 发出。WG0 和 WG1 从不发出 MMA。它们只消费分数块，运行 softmax，并将 `P` 写回 TMEM。

这种分离正是 softmax 需要屏障的原因。`s_ready` 将分数块从 MMA warp 携带到 softmax；`p_o_rescale` 携带 `P` 和一个对值 MMA 安全的 `O` 槽，要么已经重缩放，要么因为不需要重缩放而被释放。我们将在本章剩余部分不断回归这两个名称。

## 阅读片段

本章中的片段摘自 [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py)，因此它们不可避免地引用内核中我们未重现的部分定义的名称。自描述的（`wg_id`、`warp_id`、`BLK_M`/`BLK_N`、`HEAD_DIM`、`kv_stage`、`SMEM_PIPE_DEPTH_*`/`TMEM_PIPE_DEPTH` 深度、`should_accumulate` 和 `CTA_GROUP`（这里为 1））我们在下面首次需要时介绍。其余的在这里的表中获得一行说明，这样当片段将不熟悉的名称放在你面前时，你有地方查看：

| 名称 | 含义 |
|------|------|
| `q_stage`、`i_q` | Q 流水线阶段，0 或 1，即哪个 Q 块槽（`SMEM_PIPE_DEPTH_Q = 2`）。在 WG0/WG1 softmax 内，warpgroup 自己的 `wg_id`（0 或 1）*就是*相同的阶段索引，所以 `S_region[q_stage]`、`P_region[wg_id]` 和 `O_region[i_q]` 都选择相同的 Q 阶段 |
| `MMA_N` | TMEM 列中的分数/输出块宽度（128） |
| `MMA_K` | `P`/`V` 列中的 MMA 内部 K 步长（16）；`K_SPLIT = 6 * MMA_K = 96` |
| `K_SPLIT` | 值 MMA 调度的拆分点（见*两个 MMA 阶段*）；第一个值 MMA 覆盖列 `0:K_SPLIT`（`6 * MMA_K = 96`） |
| `should_rescale` | WG2 每行标志：旧 `O` 是否需要在下一个值 MMA 之前重缩放（通过 `any_sync` 在 warpgroup 内归约） |
| `rescale_threshold` | 跳过小行最大值变化的阈值；当前内核使用 `8.0`，跳过的重缩放将 `acc_scale` 设置为恰好 `1.0` |
| `scale_log2` | log2 单位的 softmax 缩放，`log2(e)/√d`，所以 `P = exp2((S - m) · scale_log2)` |
| `acc_scale` | softmax 通过 SMEM 信箱传递给 WG2 的每行重缩放因子 |
| `chunk_start`/`chunk_end`、`p_start`/`p_end` | 正在读取/写入的 32 宽 softmax 块的列范围 |

## 两个 MMA 阶段

对于每个流式 K/V 块，Flash Attention 运行两个 MMA 阶段，softmax 桥接它们：

```text
Q, K -> 分数 MMA -> S
S    -> softmax  -> P
P, V -> 值 MMA   -> O
```

将此视为三个生产者排成一行的流水线。第一个 MMA 产生注意力分数 `S`，softmax 将 `S` 转换为分子 `P`，第二个 MMA 消费 `P` 以更新输出累加器 `O`。`row_sum` 的归一化被推迟到尾声，一旦每个 K/V 块都发表了意见。

下面的每个块操作都获得我们用于 GEMM 步骤的相同**作用域/布局/调度**卡片，外加一行**交接**，命名将块传递给下一个角色的屏障。

计算代码从不以原始 TMEM 列数说话。相反，内核将其单个 TMEM 分配切割为每阶段视图（`S_region`、`P_region`、`O_region`），并按流水线阶段索引它们（`S_region[q_stage]`、`O_region[i_q]`、`P_region[i_q, 0:K_SPLIT]`）。这些视图在 [TMEM 布局和重用](#tmem-layout-and-reuse) 部分用 `T.TMEMStages` 定义；现在将每个区域视为同一物理 TMEM 的命名切片就足够了。

### 分数 MMA

两个阶段中的第一个是分数 MMA，打开每次 K/V 迭代的矩阵乘法。它计算：

$$S = Q_{\text{block}}K_{\text{block}}^{\top}$$

并将 `128×128` 分数块写入 TMEM：

```python
Tx.warp.gemm_async(
    S_region[q_stage],
    Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
    K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
if T.ptx.elect_sync():
    s_ready.arrive(q_stage)
```

我们可以问 GEMM 章节问每个块操作的四个相同问题：谁运行它，块存在于何处，如何调度，以及如何交接：

> **块原语读数：分数 MMA**
> - 作用域：WG3 warp 0 发出它；一个选举 lane 到达 `s_ready`。
> - 布局：SMEM 中的 Q、K → TMEM 中的 `S`（`S_region[q_stage]`）。
> - 调度：`tcgen05`。
> - 交接：`s_ready`（→ softmax）。

到达 `s_ready` 的单个选举线程就是整个交接。它宣布该分数块已完成，softmax warpgroup 现在可以自由读取它。

### MMA 之间的 Softmax

两个 MMA 之间是 softmax，将分数块 `S` 转换为分子块 `P` 的阶段。它的读数卡片是：

> **块原语读数：Softmax**
> - 作用域：WG0（Q 阶段 0）/ WG1（Q 阶段 1），完整 warpgroup。
> - 布局：TMEM 中的 `S` → 寄存器 → fp16 TMEM 中的 `P`（`P_region[wg_id]`）。
> - 调度：`tcgen05.ld` 读取，TMEM 存储写入；它们之间的寄存器中级数学。
> - 交接：等待 `s_ready`；到达 `p_o_rescale`（前 96 列）和 `p_ready_2`（最后 32 列）。

这个阶段是完全没有 GEMM 对应物的。WG0/WG1 等待分数块到达 `s_ready`，然后从 TMEM 中逐个寄存器大小的块读取它：

```python
Tx.copy_async(
    s_chunk[:, chunk_start : chunk_end],
    S_region[wg_id, chunk_start : chunk_end],
)
```

这是 warpgroup 作用域下的 TMEM 到寄存器块读取。现在分数位于寄存器中，softmax warpgroup 按顺序做三件事：

1. 计算行最大值和行和，
2. 计算 softmax 分子块 `P`，
3. 将 `P` 以 fp16 写回 TMEM。

最后一步看起来像：

```python
Tx.copy_async(
    P_region[wg_id, p_start : p_end],
    p_chunk[:, p_start : p_end],
)
```

为什么要将 `P` 写回 TMEM，当我们刚在寄存器中计算完它？因为值 MMA 需要 `P` 作为*块操作数*，MMA 不能将分散的每线程标量寄存器作为矩阵读取。这个内核中 MMA 可读的 `P` 形式是 `P_region`，fp16 TMEM 别名 `tmem_as_f16` 上的视图。所以写回不是冗余移动；它是将 `P` 放入下一个 MMA 实际能消费的唯一形状。

### 值 MMA

第二个阶段，也是关闭每次 K/V 迭代的阶段，是值 MMA。它计算：

$$O = O + P_{\text{block}}V_{\text{block}}$$

到这个 MMA 运行时，`O` 已经为当前 K/V 块放入正确的状态，在第一个块上初始化，在后续块上重缩放，所以 MMA 只需累加。它与 GEMM 的区别在于操作数存在于何处：A 操作数是 TMEM 中的 `P`，B 操作数是 SMEM 中的 `V`，累加器 `O` 也在 TMEM 中：

```python
# First sub-MMA: columns 0:K_SPLIT (the first 96 of P / rows of V).
Tx.warp.gemm_async(
    O_region[i_q],
    P_region[i_q, 0:K_SPLIT],
    V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
    transB=True,
    accum=should_accumulate,
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
# The second sub-MMA (same form, accum=True, gated on p_ready_2) covers the
# remaining columns K_SPLIT:BLK_N.
```

> **Tile-primitive readout: Value MMA**
> - Scope: WG3 warp 0.
> - Layout: `P` in TMEM + V in SMEM → `O` in TMEM (`O_region[i_q]`).
> - Dispatch: `tcgen05` with a TMEM operand.
> - Handoff: waits `p_o_rescale`, `p_ready_2`, `kv_load.full`; arrives `o_ready` (→ epilogue).

这种操作数放置是两个 MMA 之间的硬件区别：

- 分数 MMA 从 SMEM 读取两个操作数：Q 和 K。
- 值 MMA 从 TMEM 读取一个操作数 `P`。
- 值 MMA 从 SMEM 读取另一个操作数 V。
- 结果累加到 TMEM 中的 `O`。

`accum=should_accumulate` 标志实现了算法中的"初始化或累加"选择：它在查询块的第一个 K/V 块上为假，在之后的每个块上为真。

你可能还注意到值 MMA 不是一次性运行，而是拆分为 `96 + 32` 调度：

1. Softmax 以四个 32 列块写入 `P`。
2. 一旦前三个块就绪，值 MMA 就开始处理 `P` 的前 96 列和 `V` 的匹配行。
3. 最后 32 列等待 `p_ready_2`。
4. 第二个 MMA 消费该最终块并完成块。

拆分的原因是保持张量核心忙碌。将值 MMA 作为单条指令运行，整个阶段将停顿，直到四个 32 列 `P` 块都被指数化和存储。通过立即对前三个块发起攻击，内核将最后一个块的 `exp` 和 TMEM 写入与已经在飞行中的 96 宽 MMA 重叠，将原本的空闲时间变为有用工作。

(tmem-layout-and-reuse)=
## TMEM 布局和重用

`S`、`P` 和 `O` 都必须共享一个 `128×512` TMEM 分配，它们被打包到其中的方式正是屏障和布局在这个内核中不可分离的原因：

下图直接显示了该打包：分数槽、分子槽和输出槽都共享一个 TMEM 分配，所以屏障协议是使重用合法的因素。

![TMEM 布局](../img/tmem_layout_v3.png)

该图读作一组块槽：

- 分数槽持有 `S = QK^T`。
- 分子槽持有 softmax 指数化步骤后的 `P` 块。
- 输出槽持有 fp32 `O` 累加器。

这些不是独立缓冲区。它们是*同一*分配的区域，共享不是风格选择而是被迫的。Q 流水线深度为 2 时，两个 `S` 槽（2 × MMA_N = 256 列）和两个 `O` 槽（2 × MMA_N = 256 列）已经占满了所有 512 个 fp32 列。没有剩余给 `P`，所以 `P` 别无选择，只能通过更窄的 fp16 视图别名相同的字节。这安全的唯一原因是每个区域在其先前消费者完成后才被重用，而那个时序正是屏障保证的。所以在 FA4 中，屏障不仅仅是调度；它们是使布局合法的首要因素。

别名技巧通过 `T.TMEMPool` 设置。内核为分数和输出累加器获取一个 fp32 视图（`tmem`），然后将池基倒回 0，并在*相同*物理字节上获取第二个 fp16 视图（`tmem_as_f16`）：

```python
tmem_pool = T.TMEMPool(pool, total_cols=N_COLS_TMEM, cta_group=CTA_GROUP, tmem_addr=tmem_addr)
tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
tmem_pool.move_base_to(0)
tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
tmem_pool.commit()
```

Because fp16 elements are half as wide, the fp16 view exposes twice as many indexable columns over those same bytes, and that is precisely the space `P` lives in, space the fp32 layout had no room for. With both views in hand, the kernel carves the `S`, `P`, and `O` slots out as staged regions with `T.TMEMStages`, which lets the compute code index by pipeline stage rather than by raw columns:

```python
S_region = T.TMEMStages(tmem,        col_start=0,                       width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
O_region = T.TMEMStages(tmem,        col_start=MMA_N * SMEM_PIPE_DEPTH_Q, width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
P_region = T.TMEMStages(tmem_as_f16, col_start=MMA_N,                   width=BLK_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N * 2)
```

The `* 2` in `P_region`'s stride is the one place the aliasing visibly leaks into the code. `S_region` and `O_region` are measured in fp32 `tmem` columns, while `P_region` is measured in fp16 `tmem_as_f16` columns, which are half as wide, so stage-to-stage movement needs the doubled stride to land on the same physical bytes. Once the regions are defined, though, the compute code stays clean: it writes `S_region[q_stage]`, reads `S_region[wg_id, ...]`, writes `P_region[wg_id, ...]`, and accumulates into `O_region[i_q]`, never once touching a raw column index.

**Try with your agent**: Ask it to explain the fp32 (`tmem`) and fp16 (`tmem_as_f16`) views in this FA4 kernel. Which physical TMEM regions hold `S`, `P`, and `O`, and why does `P_region`'s stride use `MMA_N * 2`? Save the reuse question for the next section: after the barrier table, check which consumers must finish before each region can be reused.

## How Barriers Connect the Roles

This is the hardest part of the kernel, so it pays to come at it gradually. Start with the handful of barriers that move data along the main compute path, and treat everything else as bookkeeping you can look up later. The data-ready handoffs are:

| Handoff | Meaning |
|---------|---------|
| TMA load -> score/value MMA | Q, K, or V has arrived in SMEM and can feed MMA |
| score MMA -> softmax | `S` is ready in TMEM |
| softmax/correction -> value MMA | `P` is ready in TMEM, and `O` is safe for accumulation |
| value MMA -> epilogue | final `O` is ready in TMEM |
| epilogue -> TMA store | `O_smem` is ready to store |

Everything not in that list is pipeline bookkeeping: barriers that release an SMEM, TMEM, or staging buffer so that another role may reuse it. The useful thing is that every barrier, whether it carries data or only bookkeeping, reads the same way, as a tile handoff. You ask who produced data, who consumes it, and which buffer becomes free once they are both done.

The next figure collapses those handoffs into the exact readiness gates for the two MMA phases:
what the score MMA waits on, and what the value MMA must wait on before it can accumulate.

![Flash Attention 4 MMA Input Gates](../img/flash_attention_main_handoff.png)

Read this diagram as a set of correctness gates rather than a schedule. It answers "what must be true before this MMA may fire," and says nothing about timing. The score MMA waits for Q and K in SMEM, then produces `S`. The value MMA waits on three things at once: V in SMEM, the `P` tile from softmax, and an `O` slot that WG2 has either released or rescaled. The softmax-to-value gate is split for the reason we already met: the value MMA may begin once the first 96 columns of `P` are in place, and `p_ready_2` releases the final 32.

There is one handoff that does not fit the tile-readiness mold: the softmax-to-correction edge. Rather than passing a tile, softmax passes a single scalar (`acc_scale` during the K/V loop, or the final `row_sum` in the epilogue) through a one-slot SMEM mailbox to WG2. Since that slot is reused on every iteration, a `full`/`empty` barrier pair has to guard it:

The figure below zooms in on that mailbox handshake, which is why this one barrier pair should be
read as a scalar producer-consumer channel rather than as a tile-ready gate.

![Flash Attention 4 Softmax Scale-Slot Handshake](../img/flash_attention_softmax_correction.png)

Read `softmax_corr.full` and `softmax_corr.empty` as a producer-consumer pair:

1. Softmax waits for `softmax_corr.empty` before reusing the scale/sum slot.
2. Softmax writes `acc_scale` or final `row_sum` into that slot.
3. Softmax arrives on `softmax_corr.full`.
4. WG2 waits on `softmax_corr.full`, then reads the slot.
5. WG2 arrives on `softmax_corr.empty`.
6. The softmax warpgroup may reuse the slot in the next phase.

It is worth being careful about what `softmax_corr.empty` does and does not mean. It signals only that WG2 has consumed the scale/sum slot. It says nothing about whether `P` is ready, and it is emphatically *not* the gate that lets the value MMA start. That gate is `p_o_rescale`, which fires when the first 96 columns of `P` are written and the `O` slot is safe to accumulate into. Confusing the two is a classic source of wrong-result bugs.

With the main path in hand, the full barrier list serves as a reference:

| Barrier | Producer -> consumer | What becomes safe |
|---------|----------------------|-------------------|
| `q_load.full` | TMA load -> score MMA | Q SMEM tile can feed MMA |
| `q_load.empty` | all score MMAs for this Q stage -> TMA load | Q SMEM stage can be reused for the next task |
| `kv_load.full` | TMA load -> score/value MMA | K or V SMEM tile can feed MMA |
| `kv_load.empty` | score/value MMA -> TMA load | K/V SMEM stage can be reused |
| `s_ready` | score MMA -> softmax | S TMEM tile can be read |
| `p_o_rescale` | softmax + WG2 -> value MMA | first 96 columns of P are in TMEM, and the O slot is safe for value MMA |
| `p_ready_2` | softmax -> value MMA | final quarter of P is in TMEM |
| `o_ready` | value MMA -> epilogue | final O accumulator is ready |
| `softmax_corr.full` | softmax -> WG2 | `acc_scale` or final `row_sum` is ready in the SMEM mailbox |
| `softmax_corr.empty` | WG2 -> softmax | the same SMEM mailbox slot can be reused after WG2 reads it |
| `corr_epi.full` | epilogue -> TMA store | O_smem is ready to store |
| `corr_epi.empty` | TMA store -> epilogue | O_smem stage can be reused |

Just as in GEMM, you can predict a barrier's type from who produces the signal:

- TMA loads use `TMABar`, because the TMA engine byte-counts its own completion.
- MMA completion uses `TCGen05Bar`, because `tcgen05.commit` signals the completion group.
- Pure thread-to-thread handoffs use `MBarrier`, where the participating threads arrive explicitly.

The split softmax-to-value handoff rewards a closer look. It uses two gates:

- `p_o_rescale` lets the value MMA start once the first 96 columns of `P` are written and the `O` tile is safe to accumulate into.
- `p_ready_2` releases the last 32 columns of `P`, matching the `96 + 32` value-MMA schedule from the previous section.

The first K/V block is the easy case. WG2 pre-arrives `p_o_rescale`, because there is no old `O` tile to rescale yet.

Later blocks have to be more careful. WG2 arrives at `p_o_rescale` only after it has either skipped an unnecessary rescale or finished rescaling the old `O`. The skip test is deliberately conservative: softmax computes the log2-scaled delta `(m_old - m_new) * scale_log2`; if that value is still above `-rescale_threshold`, the new max has not moved far enough to justify rescaling, so the kernel keeps the old max and sets `acc_scale` to exactly 1.0. Only a larger max jump takes the `exp2` path and asks WG2 to rescale `O`.

WG2 then reduces `should_rescale` across the warpgroup with `any_sync`. If no row needs the update, it leaves `O` alone. That skip matters because rescaling `O` is a full TMEM -> RF -> TMEM read-modify-write over the whole accumulator, pure wasted work when the threshold logic has already kept `acc_scale` at 1.0.

Notice that all the new barriers cluster in one place. `s_ready`, `p_o_rescale`, `p_ready_2`, and the softmax/correction pair are all barriers around softmax. They exist for a single reason: the score MMA and value MMA are no longer adjacent. Register math, TMEM rewrites, and output rescaling now sit between them, and every one of those steps needs a handoff of its own.

**Try with your agent**: Ask it to trace one K/V block through `s_ready`, `p_o_rescale`, `p_ready_2`, and `o_ready`. For each barrier, ask who waits, who arrives, what tile becomes safe to read, and what storage can be reused afterward.

## Pipelining Structure

The barriers told us what must be *ready* before a role consumes a tile. What they did not tell us is what actually runs *concurrently*, and that is the question we turn to now. The two really are different: a correctness gate can be satisfied long before, or long after, the producer happens to run.

There is no single pipeline depth here, because different tile streams move at different rates. The kernel therefore keeps a separate ring for each:

- Q pipeline depth 2: one CTA works on two Q stages. WG0 handles one stage, and WG1 handles the other.
- KV pipeline depth 3: K and V blocks stream through the inner loop while the same Q stages are reused.
- TMEM pipeline depth 2: each Q stage has its own S/P/O TMEM slots, and those slots are reused after the matching barriers fire.

The figure below switches from correctness gates to a timeline view, showing which roles can be
active at roughly the same time once those separate rings are in flight.

![Flash Attention 4 Pipeline Structure](../img/flash_attention_pipeline_v2.png)

Read this as a timeline rather than a barrier graph. It shows which roles are active at roughly the same moment, whereas the earlier barrier-flow figure is where you go to check the exact producer-consumer waits. Between them, the two figures answer the two different questions we raised at the start of this section.

Each row matches one of the code's role branches:

- WG3 warp 1 issues TMA loads.
- WG3 warp 0 issues both score MMA and value MMA.
- WG0 and WG1 run softmax for the two Q stages.
- WG2 releases or rescales `O`, then later normalizes the final output.
- WG3 warp 2 issues the TMA store.

Following the figure from left to right traces one representative pipeline wave. The load warp begins with `Q0`, `K[n-1]`, `Q1`, `V[n-1]`, and then keeps streaming lower-index K/V blocks. The MMA warp issues the first score MMAs to produce `S0` and `S1`, and WG0/WG1 turn those into `P0` and `P1`.

It is important that the MMA warp does *not* run all the score MMAs and then all the value MMAs. Once both Q stages are primed, it interleaves the two kinds: a value MMA for the current `V` block, then a score MMA for the next `K` block, and so on:

```text
score Q0*K[n-1]
score Q1*K[n-1]
value P0*V[n-1]
score Q0*K[n-2]
value P1*V[n-1]
score Q1*K[n-2]
value P0*V[n-2]
...
```

This interleaving is the reason the score, softmax, correction, and value rows all overlap in the figure instead of running in tidy succession.

The WG2 row is labelled `release / rescale`, and the two halves correspond to the two cases we have seen. On the first K/V block there is no old `O` yet, so WG2 only takes part in the handoff that lets the value MMA proceed; on later blocks it may rescale the old `O` before the value MMA accumulates into it. Normalization and the TMA store happen exactly once, after the final K/V block of the attention task.

No single GEMM-style pipeline could describe FA4, because Q, K/V, and TMEM slots all advance on independent schedules. TIRx keeps those schedules explicit, as separate tile buffers, `PipelineState` cursors, and barrier phases, rather than hiding the kernel behind one monolithic primitive. The cost is more moving parts, but the benefit is that the complexity stays visible and inspectable.

## Rescaling and Writeback

The rescale is mandatory, not an optimization we could drop. Online softmax can raise the per-row maximum with each new score tile, and whenever it does, the `O` accumulated from earlier blocks was scaled by the *old* maximum. That makes each earlier term too large by a factor of `exp(m_new - m_old)`. Skip the correction and those blocks are over-weighted, and the final output is simply wrong. The fix is a TMEM → registers → TMEM tile operation:

$$O_{\text{old}} \leftarrow O_{\text{old}} \cdot e^{(m_{\text{old}} - m_{\text{new}}) / \sqrt{d}}$$

The work is split across two roles. Softmax computes the per-row scale and drops it in the SMEM mailbox; WG2 waits on `softmax_corr.full`, reads the current `O` out of TMEM, multiplies by that scale, and writes `O` back:

```python
RESCALE_TILE = T.meta_var(16)
o_row = T.wg_reg_tile(RESCALE_TILE)
Tx.copy_async(o_row, O_region[i_q, d_start : d_start + RESCALE_TILE])
Tx.mul(o_row, o_row, acc_scale)
Tx.copy_async(O_region[i_q, d_start : d_start + RESCALE_TILE], o_row)
T.ptx.tcgen05.wait.st()
```

It is worth stressing that this is a full TMEM → registers → TMEM tile operation over the whole `O` accumulator, not a bit of scalar bookkeeping, and it carries the same readout card as every other stage:

> **Tile-primitive readout: Correction (rescale)**
> - Scope: WG2, full warpgroup.
> - Layout: `O` in TMEM → registers → `O` in TMEM (`O_region[i_q]`).
> - Dispatch: `tcgen05.ld` to read, TMEM store to write; register multiply between them.
> - Handoff: waits `softmax_corr.full`; arrives `p_o_rescale` (→ value MMA) and `softmax_corr.empty` (→ softmax).

Tracing the synchronization from end to end:

1. Softmax writes the scale value to SMEM.
2. WG2 waits on `softmax_corr.full`.
3. WG2 rescales `O` in TMEM.
4. WG2 arrives on `p_o_rescale`.
5. WG3's value MMA can now consume `P` and accumulate into the rescaled `O` tile.

The loop closes when `softmax_corr.empty` releases the SMEM slot after WG2 has read it, which frees softmax to reuse the mailbox on the next iteration.

Once the K/V loop ends, WG2 switches from correction to epilogue. It waits for the final `row_sum` and `o_ready`, reads the final `O` from TMEM, multiplies by `1 / row_sum` (the normalization we deferred at the very start), casts to fp16, and writes `O_smem`. WG3's TMA store warp then carries `O_smem` back to GMEM.

One limitation is worth flagging for anyone who plans to extend this kernel. It computes the forward output only, whereas a training forward pass would normally also store the log-sum-exp (LSE) the backward pass needs. Adding that comes with a scaling detail to keep in mind: this kernel keeps `row_max` as the maximum of the *raw*, unscaled `QK^T` scores, while `row_sum` accumulates `exp((S - row_max) / sqrt(d))`. So the `1/\sqrt{d}` factor has to be reapplied to `row_max` when forming the natural-log LSE:

$$\mathrm{LSE}_i = \log(\mathrm{row\_sum}_i) + \mathrm{row\_max}_i / \sqrt{d}$$

This implementation is forward-output only and does not write LSE.

## 因果掩码

因果注意力添加了一个约束（查询只能关注其自身位置或之前的位置），内核以两种互补方式满足它，一种便宜，一种精确。

便宜的方式是完全跳过工作。许多 K/V 块完全位于对角线之上，对给定 Q 块没有贡献，所以 `get_n_block_max(...)` 计算该块可能需要的最后一个块，循环根本不加载或计算其余部分。

精确的方式处理跨越对角线的块，其中一些列有效，一些无效。这些块仍然运行分数 MMA，但 softmax 在指数化之前掩码无效列。对于每行，它从行的查询位置和块偏移推导列限制，保留该限制处及以下的列，并将之后的每列在寄存器中设置为 `-inf`，这样这些列对行最大值或 `exp2` 分子都没有贡献。

实现不是逐元素分支，而是用 `mask_r2p(...)` 应用限制，将其转换为整个 32 宽分数块上的位掩码，一次性掩码该块。完全位于对角线下方的块保留所有列，根本不需要掩码。

从块原语的角度看，因果模式根本不重写数据路径。它只是削减 K/V 迭代次数，并在分数 MMA 和 `P` 写回之间将掩码步骤插入寄存器驻留的 softmax 中。

## GQA 支持

分组查询注意力让多个查询头共享单个 K/V 头。这节省了内存带宽，但它提出了一个打包问题：我们如何在仍然为许多查询头提供数据的同时只保留一个 K/V 块？内核的答案是一次处理一整个查询头组，针对一个调度的 `kv_head_idx`：

```python
GQA_RATIO = num_qo_heads // num_kv_heads
SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
```

The trick is to reinterpret the 128 Q-tile rows. For `GQA_RATIO=4` they no longer stand for 128 sequence positions; they stand for 32 sequence positions times 4 query heads, packed together so that all four heads ride the same K/V tile. The row decoding is:

```text
seq_pos = row // GQA_RATIO
q_head  = row % GQA_RATIO
```

The Q load expresses this packing with a 3D view. The source is the natural `Q[batch, seq, qo_head, dim]` layout, while the destination is the very same SMEM tile the score MMA will later read as a flat `128 x HEAD_DIM` operand. The view is what reconciles the two, and it does so without any copying:

```python
Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
Tx.copy_async(
    Q_smem_3d[i_q, :, :, :],
    Q[batch_idx,
      m_start : m_start + SEQ_Q_PER_TILE,
      kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
      :],
    **tma_copy_q,
)
```

K and V are never expanded in memory, and that is the whole point of GQA: the single K/V tile for `kv_head_idx` is reused by all `GQA_RATIO` query heads packed into the Q rows. The output side mirrors the input, with a matching 3D view storing the packed rows back to `O[batch, seq, qo_head, dim]` after the epilogue.

The consequence is that GQA lives entirely at the Q-load and O-store boundaries. Inside the compute path the score MMA still sees a plain `128 x HEAD_DIM` Q tile, and the rest of the tile-primitive graph is untouched.

## Tile Scheduling

The scheduler's job is to map each CTA to a `(batch, kv_head, m_block)` attention task, and the right strategy depends on whether the masking makes those tasks equal in cost:

- Non-causal mode uses `FlashAttentionLinearScheduler`. Every task does the same amount of work, so a fixed CTA pool advancing by `num_ctas` is all it takes to spread them evenly.
- Causal mode uses `FlashAttentionLPTScheduler`, because causal masking makes the work wildly uneven: a Q block near the start attends to roughly one K/V block, while one near the end attends to all of them. A naive split would leave some CTAs finishing long after others, so the longest-processing-time scheduler front-loads the heavy blocks to even out finish times, while still keeping nearby batch/head tasks together for L2 locality.

For all their differences, the two schedulers expose an identical loop interface:

```python
while scheduler.valid():
    m_block_idx = scheduler.m_block_idx
    batch_idx = scheduler.batch_idx
    kv_head_idx = scheduler.head_idx
    # process one Q block against its K/V block range
    scheduler.next_tile()
```

The only behavioral difference lies in what `next_tile()` does: in non-causal mode it advances the CTA to another task, whereas in causal mode it ends the loop after the current one. Either way this is purely a scheduling decision: it chooses *which* attention tile the CTA owns, never how that tile is computed. Inside the loop the same local primitives run regardless: TMA load, score MMA, softmax, value MMA, correction, TMA store.

## Compile and Verify

Everything above has been excerpts, so to put it all together and actually run the kernel we import the real thing from `tirx-kernels`, compile it, and check it against a torch reference. The complete kernel, with every piece this chapter walked through assembled into one file, is [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py) in the `tirx-kernels` repository. Two things differ from the GEMM verify cell: Flash Attention has a richer entry point (`get_flash_attention4_kernel`), and it takes an extra `profiler_buf` argument for its built-in profiler. This is the one cell to run for the whole chapter:

```python
import torch
import torch.nn.functional as F
import tvm
from tirx_kernels.attention.flash_attention4 import (
    get_flash_attention4_kernel, PROFILER_BUFFER_SIZE)

B, S, Hq, Hkv, D = 1, 1024, 32, 8, 128   # GQA: 32 query heads share 8 KV heads
Q = torch.randn(B, S, Hq, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
O = torch.empty(B, S, Hq, D, dtype=torch.float16, device="cuda")
prof = torch.zeros(PROFILER_BUFFER_SIZE, dtype=torch.uint64, device="cuda")

kernel = get_flash_attention4_kernel(B, S, S, Hq, Hkv, D, is_causal=False)
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
ex.mod(Q, K, V, O, prof)   # ex.mod takes torch tensors directly, like every other chapter
torch.cuda.synchronize()

# torch reference; enable_gqa lets the 32 query heads share the 8 KV heads
qt, kt, vt = (x.transpose(1, 2).float() for x in (Q, K, V))
ref = F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=True).transpose(1, 2).half()
torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)
print(f"FA4: B={B} S={S} Hq={Hq} Hkv={Hkv} D={D}, non-causal -> PASS")
```

**Expected output**: `... -> PASS`. The kernel accumulates the online softmax in fp32, yet several distinct approximations still separate its result from a high-precision reference. There is the fp16 storage and rounding of the inputs and operands; the `exp2`-based softmax reformulation (the `scale_log2 = log2(e)/√d` reframing of every exponential); the online-softmax reordering and per-row rescaling, which sums the blocks in a running scale rather than all at once; and finally the fp16 cast of `O` on writeback. The `rtol`/`atol` chosen here, the same tolerance the source kernel's own test uses, is sized to cover all of these together against the torch reference, not fp16 rounding on its own. So if you ever see a genuine failure here, not just a borderline near-miss, read it as a signpost pointing back at the softmax path: a dropped `s_ready` / `p_o_rescale` / `p_ready_2` wait, or a `row_max` / `row_sum` update that the rescale step failed to apply. Those are exactly the handoffs this chapter spent its barriers on.

## Differences from GEMM

The table below compares FA4 with GEMM along the axes that changed:

| Aspect | GEMM | Flash Attention 4 |
|--------|------|-------------------|
| MMA phases | one repeated MMA | score MMA and value MMA |
| Work between MMAs | none beyond pipeline handoffs | online softmax, masking, and O rescaling |
| Running state | accumulator only | row max, row sum, O accumulator |
| Main intermediate | accumulator TMEM tile | S, P, and O TMEM tile regions |
| Warp roles | TMA producer, MMA consumer, writeback | TMA load, MMA, softmax, correction, TMA store |
| Barriers | mostly load/compute/writeback handoffs | additional score/softmax/value/correction handoffs |
| Scheduling unit | output matrix tile | attention task: `(batch, kv_head, m_block)` |

Every one of these differences traces back to the structural change we opened the chapter with: a second MMA, with softmax wedged between the two. The underlying TIRx contracts, on the other hand, never changed at all:

- the tile primitive says what tile moves or computes,
- the surrounding scope says which threads cooperate,
- the layout says where the tile lives,
- the barrier says when the next role may consume it.

So FA4 is harder than GEMM not because it relies on different hardware, but because there are simply more tile values and more handoffs between them.

## Exercises

1. Compared with GEMM, what new tile handoff appears between the two MMA phases in FA4? Name the producer, the TMEM tile, and the consumer.
2. Why does softmax write the numerator tile `P` back to TMEM instead of keeping it only in registers for the value MMA?
3. Pick `p_o_rescale` or `p_ready_2`. What exactly does the barrier prove, and what could go wrong if the value MMA skipped that wait?

**Try with your agent**: Pick one unannotated tile primitive, such as an epilogue `Tx.copy_async`, the fp32 -> fp16 `Tx.cast`, or the second `gemm_pv` sub-MMA. Ask for its scope / layout / dispatch / handoff card, then check the answer against the source guards, allocations, and waits.
