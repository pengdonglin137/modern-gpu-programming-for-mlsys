(chap_tma)=
# 异步数据搬运：TMA

:::{admonition} 概览
:class: overview

- TMA 是用于全局内存和共享内存之间异步块复制的硬件引擎。一个线程发出复制命令，引擎搬运字节。
- TMA 复制由张量映射描述符描述。描述符告诉引擎全局张量形状、步长、块坐标和共享内存 swizzle 模式。
- 在加载路径上，TMA 可以在写入共享内存时对块进行 swizzle，使块直接落入张量核心期望的布局。
- TMA 加载通过带有字节计数跟踪的 `mbarrier` 完成。TMA 存储使用提交组和等待组。
:::

张量核心只有在有数据可供消费时才有帮助。在 GEMM 或注意力内核中，一旦流水线填满，数学运算可能是计算瓶颈（{ref}`chap_performance`），但流水线只有在下一个操作数块及时到达时才能保持填满。

搬运块的旧方法是让线程自己复制。每个线程计算地址，从全局内存发出加载，并将值存入共享内存。这可行，但它将 warp 指令花在地址计算和复制簿记上，而不是计算上。它还使复制路径在应该为张量核心提供数据的相同 warp 的指令流中可见。

张量内存加速器（TMA）将此工作转移到硬件复制引擎中。一个线程发出块复制命令。然后复制引擎异步地在全局内存和共享内存之间搬运矩形块。在引擎搬运字节时，CTA 的其余部分可以继续其他工作。

TMA 还处理部分布局问题。张量核心不仅需要共享内存中的正确值。它需要它们在正确的共享内存布局中。在加载路径上，TMA 可以在写入块时应用共享内存 swizzle。这使块直接落入后续 MMA 期望的布局。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tma_intro.html" title="TMA：张量内存加速器" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互演示：TMA 将块从全局内存复制到共享内存。切换 swizzle 模式并悬停源单元格以查看其在共享内存中的落点。*

## 一个线程发出，硬件搬运块

TMA 复制从一个发出线程开始。该线程不会遍历块中的所有元素。它给硬件一个复制描述，然后 TMA 引擎执行传输。

主要输入是张量映射描述符。描述符描述全局张量以及应如何从中读取块。它记录张量形状、步长、元素大小、块形状和 swizzle 模式等信息。发出线程还提供块应落入的共享内存地址。

指令发出后，复制异步运行。发出线程可以继续。CTA 中的其他线程也可以继续。传输现在由 TMA 引擎负责，而不是普通加载和存储指令的循环。

这给了内核两种不同的方式来表达相同的逻辑操作"复制这个块"。

一种路径是线程复制。线程协作从全局内存加载并存入共享内存。这给了内核对每次访问的直接控制，但它消耗线程指令和寄存器用于地址计算。

另一种路径是 TMA 复制。一个线程发出传输命令，硬件复制引擎执行矩形复制。这是大型规则块的自然路径，特别是张量核心内核使用的操作数块。

这两种路径有不同的同步规则和不同的性能行为。选择哪种是调度决策。布局告诉内核它想要什么内存排列。作用域告诉它哪些线程或 CTA 参与。调度决定复制是由普通线程代码实现还是由 TMA 实现。

## Swizzled 布局

仅仅搬运块是不够的。块还必须以张量核心可以高效读取的布局放置在共享内存中。

这就是 TMA swizzle 的用途。当 TMA 将块写入共享内存时，它可以排列共享内存地址模式。全局内存块仍然是逻辑矩形，但共享内存中的目标布局可以是 swizzled 的。

swizzle 模式是 TMA 描述符的一部分。一旦描述符设置好，发出线程不必手动应用 swizzle。引擎在字节落入共享内存时应用它。

重要的要求是一致性。TMA 描述符、共享内存块布局和后续 MMA 指令必须都描述相同的布局（{ref}`chap_data_layout`）。如果 TMA 用一种 swizzle 写入块，但 MMA 像用另一种 swizzle 那样读取它，硬件仍然会精确地执行它被要求做的事情。字节只是会为计算错误地排列。

这就是布局表示法不仅仅是簿记工具的地方。DSL 使用的布局必须与 TMA 描述符和张量核心指令使用的硬件布局匹配。例如，如果内核说操作数块存储在 128 字节 swizzled 布局中，TMA 描述符必须使用匹配的 swizzle 模式，MMA 调度必须期望相同的共享内存排列。上面的演示让你在无 swizzle 和 128 字节 swizzle 之间切换；悬停源元素以查看应用 swizzle 后的落点。

理解 swizzle 的一个有用方式是 TMA 没有改变逻辑块。它改变了逻辑元素在共享内存中的物理落点。后续 MMA 仍然消耗相同的逻辑 A 或 B 块。swizzle 只决定该块如何跨共享内存 bank 排列。

## 用于分块和 Swizzle 的 3D TMA

普通 TMA 复制搬运平坦的 2D 块，但张量核心想要的共享内存布局通常*分块*为 swizzle 原子（{ref}`chap_data_layout` 中的 8 × 128 字节原子）。TMA 通过额外的描述符维度处理这一点。**3D TMA** 将共享内存框描述为 `(group, row, col)`，其中 group 维度遍历原子，内两个维度在单个原子内寻址。单次 3D 复制既逐原子铺设块（分块），又在每个原子内应用 swizzle，因此数据到达时已经处于 MMA 期望的布局，无需单独的分块或 swizzle 传递。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo/tma_3d.html" title="使用 3D TMA 进行分块和 swizzle" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互演示：3D TMA 复制，地址为 (group, row, col)，分块到 swizzled 共享内存中。*

选择 swizzle *格式*与此分块相关。更宽的 swizzle 将一列分散到更多 bank，因此 128 字节 swizzle 在适合时是默认选择，但 N 字节原子需要块的连续维度来填充它。由于形状约束而较小的块因此不能使用 128 字节 swizzle，必须降级到 64 字节或 32 字节：经验法则是选择块能填充的最大 swizzle（{ref}`chap_data_layout`）。下面的演示直接显示了约束：16 × 16 块上的 128 字节 swizzle 只有在块被分成匹配原子的 16 × 8 组时才变得无冲突。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo/tiling_constraint.html" title="Swizzle 施加分块约束" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
<script>
(function () {
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    document.querySelectorAll('iframe.demo-tma3d').forEach(function (f) {
      if (e.source === f.contentWindow) f.style.height = d.height + 'px';
    });
  });
})();
</script>
```
*交互演示：16 × 16 块上的 128 字节 swizzle，分成 16 × 8 组后变得无冲突。*

## 完成：加载

复制是异步的，因此仅仅发出它还不够。消费者不能仅因 TMA 指令已发出就读取共享内存块。只有在引擎完成写入字节后，块才可安全读取。

对于 TMA 加载，完成信号是 `mbarrier`（{ref}`chap_async_barriers`）。

通常的序列是：

1. 初始化或重用流水线阶段的 `mbarrier`；
2. 告诉屏障 TMA 传输预期写入多少字节；
3. 发出 TMA 加载；
4. 让 TMA 引擎在字节到达时更新屏障；
5. 让消费者在读取共享内存块之前等待屏障阶段。

字节计数通过以下操作设置：

```text
mbarrier.arrive.expect_tx(bytes)
```

这做两项工作。它记录预期传输大小，同时也执行发出线程对屏障的到达。屏障不会仅因此调用发生就完成。它仍然等待 TMA 引擎报告预期字节已到达。

随着传输进行，引擎对屏障执行 complete-tx 更新。只有当两个条件都满足时屏障阶段才会翻转：到达计数满足，且待处理字节计数达到零。

然后消费者等待该屏障。一旦等待完成预期阶段，共享内存块就准备就绪。此时 MMA 路径可以安全地读取它。

![TMA 加载同步流程](../img/tma_sync_flow.png)

这与其他异步生产者-消费者交接使用的屏障模型相同。生产者是 TMA 引擎。消费者是 MMA 路径或任何其他读取共享内存块的代码。屏障是它们之间的显式交接。

## 完成：存储

TMA 存储以相反方向搬运数据，从共享内存到全局内存。它们也是异步的，但完成机制不同。

TMA 加载通常为同一内核内的消费者提供数据。MMA 路径需要知道共享内存块何时就绪。这就是加载路径使用 `mbarrier` 的原因。

TMA 存储通常将最终数据写入全局内存。通常没有立即的内核内消费者等待存储结果。内核需要知道的主要事情是什么时候可以安全地重用共享内存缓冲区或完成存储序列。

为此，TMA 存储使用提交组和等待组。内核发出一个或多个存储，提交组，稍后等待组排空。等待完成后，从内核的角度看，该组中的存储已完成，存储使用的共享内存区域可以安全重用。

所以规则很简单：

```text
TMA load:  通过带有字节计数跟踪的 mbarrier 等待
TMA store: 通过提交组和等待组等待
```

两种机制在不同的交接点服务于相同的目的。加载需要使共享内存块对后续消费者可见。存储需要确保出站传输完成，然后内核才能重用源存储或依赖存储已排空。

## 为什么 TMA 对流水线很重要

TMA 在作为流水线的一部分时最有用。内核可以在张量核心计算当前块时为未来块发出加载。加载在后台运行。计算在前台运行。当未来块变为当前块时，屏障连接两者。

典型的 GEMM 循环重复使用这种结构。共享内存的一个阶段保存 MMA 当前消耗的块。另一个阶段正在被 TMA 填充。随着循环推进，角色轮换。在 MMA 读取一个阶段之前，它等待该阶段的加载屏障。在 TMA 覆写一个阶段之前，内核确保之前的消费者已完成使用它。

这就是为什么 TMA 和 `mbarrier` 通常在 Blackwell 和 Hopper 风格的内核中一起出现。TMA 给内核一个异步复制引擎。屏障给内核一个精确的方式来知道复制的字节何时就绪。
