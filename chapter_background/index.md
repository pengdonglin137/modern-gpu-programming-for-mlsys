(chap_background)=
# GPU 执行模型

:::{admonition} 概览
:class: overview

- 内核在线程层次结构（线程 → warp → warpgroup → CTA → 集群 → 网格）上运行，跨越不同的内存空间（寄存器、SMEM、GMEM、TMEM）。
- 计算分为 CUDA 核心和张量核心；TMA 等专用引擎搬运为它们提供数据的数据。
- 内核是一个流水线，通过这些内存空间暂存数据，并在独立的计算和数据搬运引擎之间交接工作；反复出现的目标是让这些引擎同时保持忙碌。
:::

要编写快速的 GPU 程序，理解硬件本身以及代码如何在该硬件上运行非常重要。本章概述 GPU 执行模型：执行工作的线程层次结构、保存和移动数据的内存空间，以及执行繁重工作的计算和数据搬运引擎。我们首先逐一介绍这些组件，然后将它们组合到 GEMM 流水线中，以清楚地展示数据和执行如何流经硬件。本书后面的几乎所有优化都是以某种方式在这些相同组件之间安排工作。

现代 GPU 还包含许多专用硬件单元。为了先给一个初步印象，下面的交互演示在我们深入每个部分之前，展示了 Blackwell 流式多处理器内部的主要元素。你可以点击每个部分查看其详细信息。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/sm_architecture.html" title="Blackwell SM 架构" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互演示：Blackwell SM，显示其 warp/warpgroup、共享内存、张量内存，以及张量核心和 TMA 引擎。*

## 执行层次结构

我们从执行工作的线程开始。GPU 不会将其数千个线程作为一个平坦的池呈现。相反，它将它们分组为嵌套层次结构，这样做是因为协作同时发生在几个不同的尺度上。每一层的存在都是为了使其中一个尺度上的协作变得廉价。下图显示了 Blackwell 上的层次结构；你可以点击每一层来高亮它。

```{raw} html
<iframe src="../demo/thread_hierarchy.html" title="Blackwell 线程层次结构" loading="lazy"
        style="width:100%; min-width:900px; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互演示：点击层级：线程 → warp → warpgroup → CTA → 集群 → 网格。*

- **线程**：标量执行单元。每个线程有自己的程序计数器和自己的寄存器，通过其 warp 内的 lane ID 标识。
- **Warp**：32 个线程以 SIMT（*单指令多线程*）方式执行。warp 的 lane 一起发出相同指令，但每个 lane 保持自己的寄存器并可以单独屏蔽，这使单个 warp 的 lane 可以遵循不同分支。
- **Warpgroup**：四个连续的 warp，即 128 个线程。Hopper 引入 warpgroup 作为发出 warpgroup 级 MMA（`wgmma`）的单元，在 Blackwell 上它承担第二个角色：它是张量内存访问的协作单元，128 个线程一起将 TMEM 块移入或移出寄存器。
- **CTA**（*协作线程数组*，CUDA 也称为线程块）：硬件调度的基本单元。CTA 在单个 SM 上运行，并在其内部拥有私有的共享内存分配。多个 CTA 可以同时驻留在同一 SM 上，当它们这样做时，它们在它们之间分配该 SM 的共享内存容量。
- **集群**：一组协作的 CTA，可以位于不同的 SM 上。集群中的 CTA 可以相互同步，可以读写彼此的共享内存，这种能力称为分布式共享内存。

这些层级值得深入研究，因为与早期架构不同，Blackwell 的关键操作**并非都由同一线程组发出**。TMA 复制由单个线程启动，然后由硬件执行。TMEM 到寄存器的加载是 warpgroup 分布式的：四个 warp 协作，每个移动其自己的 TMEM 块切片。`tcgen05` MMA 由一个选举线程提交，而集群 MMA 跨两个 CTA。因此每个操作都有自己的自然粒度，运行它的线程集就是我们所说的该操作的**作用域**，这是本书反复回归的三个设计要素（作用域、布局和调度）中的第一个。

## 内存空间

该层次结构中的线程只有在数据到达它们时才快，因此我们接下来转向数据存在于何处。没有一种内存同时具有大容量和高速度；物理学强制在容量和速度之间做出权衡。因此 GPU 提供多种内存而不是一种，每种都在不同的点上做出这种权衡，内核通过在它们之间移动数据来工作。每个空间有自己的容量、自己的延迟，以及谁可以访问它的自己的规则。

| 内存 | 拥有者 | 角色 | 备注 |
|------|--------|------|------|
| **全局内存 (GMEM)** | 设备级 | 持久张量存储 | 大容量 HBM，所有 SM 共享 |
| **共享内存 (SMEM)** | 每 CTA（一个 SM） | 块暂存 | 低延迟暂存器；B200 上最多 228 KB/SM |
| **张量内存 (TMEM)** | 按 CTA | MMA 累加器存储 | Blackwell 新增；由 `tcgen05` 使用 |
| **寄存器文件 (RF)** | 每线程 | 标量和每线程块片段 | 快速；保存尾声/临时值 |

按顺序读取，这些空间描述了一条路径。本书中几乎每个内核的数据路径是 **GMEM → SMEM →（计算）→ 寄存器 → SMEM → GMEM**，对于张量核心内核，TMEM 位于该路径的中间，在数学运算运行时保存累加器。

在四者中，**张量内存 (TMEM)** 是唯一在 Blackwell 之前硬件上没有类似物的，其完整细节留到 {ref}`chap_tensor_cores`。但现在值得理解其动机。早期 GPU 将大型 MMA 累加器保存在寄存器中，它们在那里争夺稀缺资源。Blackwell  instead 将 `tcgen05` 累加器输出写入 TMEM，这是一个 CTA 作用域的 2D 暂存器，每个 CTA 128 个 lane 乘最多 512 个 32 位列（数组物理上位于 SM 上）。然后内核必须在尾声代码之前显式地将 TMEM 读回寄存器。这个额外步骤不是免费的，其两个后果将在全书中反复出现。第一个是 TMEM 读取是**显式且 warpgroup 分布式的**，由 warpgroup 的四个 warp 协作执行。第二个是 TMEM 不像寄存器，必须**显式分配和释放**。

### 跨集群的分布式共享内存

集群是层次结构中唯一一个成员可以跨越多个 SM 的层级，这种覆盖范围赋予了其他层级缺乏的内存能力。CTA 在一个 SM 上运行并使用该 SM 的共享内存工作，但单个 CTA 的 SMEM 预算是有限的，大型块通常需要比单个块能提供的更多的操作数存储或更多重用。Hopper 的答案是**线程块集群**：一组比独立块更紧密协作的 CTA，它们可以一起同步并读写彼此的共享内存，这种能力称为**分布式共享内存 (DSMEM)**。Blackwell 保留集群并添加了动态调度（{ref}`chap_clc`）和 2-CTA 协作 MMA。

DSMEM 允许 CTA 直接寻址和访问对等 CTA 的共享内存。线程可以命名对等体 SMEM 中的位置，并批量将块直接从自己的 SMEM 复制到对等体的 SMEM 中，在字节落地后发出完成屏障（{ref}`chap_async_barriers`）。第三部分的 2-CTA 集群 GEMM 正是建立在这种机制之上，使用它在两个 CTA 之间共享操作数块，而无需通过全局内存路由。

下图显示了 CTA 集群使能的额外 DSMEM 跳跃；点击一块以查看每个 CTA 拥有什么以及跨 CTA 读取发生在哪里。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/cta_cluster.html" title="共享分布式共享内存的 2-CTA 集群" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互演示：2-CTA 集群，每个 CTA 拥有 A 的一半和 B 的一半，跨集群读取对方的 B（DSMEM），该对产生 256×256 输出块。*

## 计算：CUDA 核心和张量核心

线程和它们移动的数据必须在算术单元相遇，SM 提供两种不同的数学引擎而不是一种。两者之间的分工塑造了几乎每个内核的编写方式，它们扮演互补的角色。

- **CUDA 核心**是通用 SIMT ALU。它们运行处理索引计算、逐元素数学、归约和控制流的标量和向量指令，这些是围绕大型矩阵工作的粘合逻辑。
- **张量核心**是执行密集矩阵乘加的专用功能单元，以*块*粒度计算 $D = AB + C$。

这种分工重要的原因是张量核心提供比 CUDA 核心多得多的算术吞吐量，大约 10 倍或更多 FLOP/s，因此密集线性代数（GEMM、卷积和注意力）只有在张量核心上运行时才能达到峰值性能。因此获得性能在很大程度上是保持这些张量核心供给充足的问题。从一代 GPU 到下一代变化的是*如何*编程张量核心以及它们的结果*在哪里*停留。Hopper 引入了异步 warpgroup MMA（`wgmma.mma_async`）；Blackwell 第五代张量核心 `tcgen05` 将其累加器放在张量内存中而不是寄存器中，我们用 {ref}`chap_tensor_cores` 专门介绍它。

集群以两种方式扩展这些引擎，这两种方式在 GEMM 章节中反复出现。**2-CTA 协作 MMA** 让两个 CTA 各自贡献其 SMEM 操作数到单个更大的张量核心 MMA 块中。**TMA 多播**让数据搬运引擎的一次加载将相同的 GMEM 块同时传递给多个 CTA，消除否则单独加载会产生的冗余全局流量。两者都建立在前面介绍的分布式共享内存之上。

## GEMM 数据流水线

到目前为止，我们已经单独介绍了硬件单元。为了了解它们如何协同工作，我们可以使用典型的通用矩阵乘法（GEMM）流水线作为示例。下面的交互演示显示了三阶段 GEMM 块流水线中涉及的单元；点击 `tma load` 等操作以高亮其跨硬件单元的数据路径。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/pipeline_arch.html" title="Blackwell GEMM 数据流水线" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互演示：Blackwell 上的加载 → MMA → 尾声流水线；点击操作以跟踪其跨硬件单元的数据路径。*

单个 GEMM 块流经三个阶段。

1. **加载。** TMA 复制（{ref}`chap_tma`）将 A 或 B 操作数块从 GMEM 流式传输到 SMEM。一个线程发出复制命令，预先记录预期到达的字节数。随着字节落地，TMA 引擎报告其进度，只有当所有预期字节都已交付时，完成屏障才会翻转。
2. **计算。** `tcgen05` MMA（{ref}`chap_tensor_cores`）从 SMEM 中读取操作数块，并将乘积累加到 TMEM 块中。一个选举线程发出它，当数学运算完成时，它会发出屏障信号。
3. **尾声。** Warpgroup 将 TMEM 累加器读回寄存器，将结果转换为输出数据类型，并将其存储到 GMEM，通常通过暂存到 SMEM 并发出 TMA 存储。

这样写出来，三个阶段看起来是严格顺序的，但慢内核和快内核之间的全部区别在于**重叠**。朴素内核确实按顺序运行这些步骤（加载、等待、计算、等待、存储），因此在等待前一个引擎时让每个引擎空闲。快速内核 instead 流水线化它们：当张量核心在块 `k` 上计算时，TMA 引擎已经在获取块 `k+1`，尾声代码忙于排空块 `k-1`，因此所有三个引擎同时保持忙碌。让三个异步引擎安全地相互交接工作正是屏障和阶段模型（{ref}`chap_async_barriers`）的工作，第三部分的 GEMM 阶梯建立在其之上。

## 接下来读什么

现在我们已经看到了高层次的图景，我们可以转向更深入地探讨主要机制的章节：

- {ref}`chap_tensor_cores` 详细解释 `tcgen05` 计算和张量内存。
- {ref}`chap_tma` 涵盖基于 TMA 的异步数据搬运。
- {ref}`chap_async_barriers` 介绍协调这些引擎的 mbarrier 和阶段模型。
