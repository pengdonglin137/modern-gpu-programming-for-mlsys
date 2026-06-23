(chap_gemm_basics)=
# 构建分块 GEMM

:::{admonition} 概览
:class: overview

- 从 TIRx 块原语构建正确的分块 GEMM，从单个输出块开始。
- 第 1 步是单块 GEMM，第 2 步添加 K 循环累加，第 3 步跨 CTA 空间分块处理完整矩阵。
- 正确性优先；性能是后面两章的任务。
:::

GEMM 是本书构建围绕的工作负载。它位于线性层、注意力投影和卷积之下，这些占据了 GPU 的大部分时间，因此正确 GEMM 和快速 GEMM 之间的差距就是让大部分芯片空闲和饱和它之间的差距。

这个差距太大，无法一步跨越。饱和内核让你同时调试内存搬运、累加、分块和张量核心调度，没有可信赖的比较对象。更安全的路径是从产生正确答案的最小内核开始，然后每次做一个决策来增长它。

本章编写第一个正确的分块 GEMM。前面的章节抽象地介绍了 TIRx 作用域/布局/调度模型；这里我们将其应用于真实内核。我们从一个 128×128 输出块开始，将其增长为处理完整大小矩阵的内核，添加 K 维度累加，然后跨多个 CTA 空间分块。

这是三章中的第一章，端到端地走完单条 GEMM 优化路径。本章我们构建一个正确的分块内核并停在那里。下一章（{ref}`chap_gemm_async`）用 TMA 替换线程复制，并通过流水线将数据搬运与计算重叠，{ref}`chap_gemm_advanced` 进一步使用 warp 特化和 CTA 集群。每章建立在前一章之上，因此内核累积特性而不是重新开始。

将每一步视为对单个契约的编辑会有所帮助，该契约有三个项：哪个**作用域**运行操作，操作数块使用哪个**布局**，以及哪条**调度**路径执行它。大多数步骤有一个主要变化，因此我们用一个小卡片打开它们，命名该变化并指出使重用安全所需的任何同步细节。第 1 步建立路径其余部分编辑的基线。

## GEMM

GEMM 是位于线性层、注意力投影和许多卷积实现之下的密集矩阵乘法，这就是为什么快速 GEMM 内核几乎在你看到的每个地方都有回报。本教程中的示例使用 $D = A B^{\top}$：

- $A$ 形状为 $M \times K$。
- $B$ 形状为 $N \times K$。
- $D$ 形状为 $M \times N$。
- $D[m,n] = \sum_k A[m,k] \cdot B[n,k]$。

转置不是我们选择执行的额外操作；它来自数据的存储方式。示例将 $B$ 保持为长度 $K$ 的 $N$ 行，这是线性层权重通常的布局，因此沿 $K$ 收缩自然读取 $B^{\top}$ 而无需任何重排。

在本教程中，我们通过 TFLOPS 吞吐量来衡量内核，将每次乘加的两个浮点操作与墙钟时间计算：

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

### GEMM 数据路径

本教程中的每个优化都归结为数据存在于何处以及如何移动，因此在编写任何代码之前值得映射出来。核心上，Blackwell GEMM 内核围绕两个活动组织：在内存之间移动块，以及对它们进行计算。下图跟踪一个块从输入到输出过程中经过的每个内存：

![*内存数据流*](../img/memory_dataflow.png)

上图显示了每个后续优化编辑但从不替换的基线路径。
从左到右读：操作数块首先从 GMEM 移动到 SMEM；`tcgen05.mma` 然后消费 SMEM 操作数并将累加器写入 TMEM；最后尾声代码在将结果存储到 GMEM 之前将 TMEM 读回寄存器。记住这个链，因为下面每一步都改变其中一个跳跃的*方式*；它从不改变跳跃本身。

## 优化路径

上面的简单数据路径足以获得正确答案，但它让大部分硬件空闲。本教程的其余部分通过一次添加一个 Blackwell 特性来缩小这个差距，每个特性通过 TIRx 块原语表达。我们将遵循的路径依次访问这些特性：

- **TMA 异步搬运**通过 Blackwell 的硬件复制路径移动 GMEM <-> SMEM 块，屏障跟踪完成。
- **软件流水线**使用多个 SMEM 阶段，使下一个 K 块的数据搬运可以与当前块的张量核心计算重叠。
- **持久调度**保持固定数量的 CTA 池，每个通过块调度器处理多个输出块，而不是为每个块启动一个 CTA。
- **Warp 特化**将生产者、MMA 消费者和写回角色分配到不同的 warpgroup。
- **CTA 集群**让两个 CTA 协作处理单个更大的 Blackwell MMA 块。
- **多消费者执行**使用多个消费者 warpgroup 同时计算块的不同部分，提高计算密度。

---

(chap_single_tile)=
## 第 1 步：顺序单块 GEMM

仍然使用完整硬件路径的最简单 GEMM 是计算单个输出块的那个。所以这是我们开始的地方。第 1 步计算一个 K = 64 的 128×128 输出块，足够小以至于不需要循环，数据路径的每个部分恰好出现一次。没有重复，我们可以在必须推理循环之前单独查看每个跳跃。

> **这一步建立的：基线**
> - 作用域：单个 warpgroup 的 128 个线程按顺序走完整个路径，一个阶段接一个阶段。
> - 布局：A 和 B 块存在于 SMEM 中，累加器在 TMEM 中，结果通过寄存器暂存输出。
> - 调度：同步 `Tx.copy` 执行加载，`tcgen05` 运行 MMA。

### 单块数据流

基线契约确定后，下一件要确定的事情是一个块通过它的顺序。这个第一个内核恰好走一次核心 GEMM 数据路径，与数据流图中相同的 GMEM -> SMEM -> TMEM -> 寄存器 -> GMEM 链，没有循环包裹它。它分配工作内存，加载操作数，计算乘积，写回结果，然后清理自己：

1. **分配**：SMEM（池分配器）、TMEM（`tcgen05.alloc`）、mbarrier
2. **加载**：所有 128 个线程协作将 A 和 B 块从 GMEM 复制到 SMEM（同步 `Tx.copy`）
3. **计算**：单个选举线程发出 `Tx.gemm_async` + `tcgen05.commit`；所有线程等待 mbarrier
4. **写回**：Warpgroup 读取 TMEM → 寄存器；每个线程将 fp32 转换为 fp16 并写入 GMEM
5. **释放**：TMEM 释放

### 第一个内核的四个部分

完整内核只有几十行，但分部分更容易理解。我们将分四部分阅读它（内存分配、同步加载、MMA 调度和写回），然后将它们组装成一个内核。沿途出现的 API 名称是第二部分（{ref}`chap_tirx_primer`、{ref}`chap_tirx_layout_api`）介绍的 TIRx 块原语词汇。

**内存分配。** 内核首先为操作数分配共享内存，以及 TMEM 地址和 mbarrier 的槽位：

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM address (4 bytes)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier (8 bytes)
pool.move_base_to(1024)                           # Skip to offset 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

两个细节值得停下来关注。`pool.move_base_to(1024)` 将 Asmem 和 Bsmem 推到偏移 1024，为上面的小元数据保留低地址，使大型操作数块落在干净的边界上。`layout=A_layout` 向 `tma_shared_layout` 请求 TMA 和 `tcgen05.mma` 都能直接读取的 swizzled SMEM 放置，正是第二部分描述的布局即契约义务。

**同步加载。** 缓冲区就位后，操作数仍然需要到达 SMEM。在这个第一个版本中，我们让 CTA 自己的线程执行复制：

```python
Tx.cta.copy(Asmem[:, :], A[:, :])
Tx.cta.copy(Bsmem[:, :], B[:, :])
T.cuda.cta_sync()
```

因为这里只有一个块（M=N=128, K=64），复制整个 A 和 B 就是完整的加载。`Tx.cta.copy(...)` 让 CTA 协作执行该复制，每个线程负责自己的数据切片。随后的 `T.cuda.cta_sync()` 承担双重职责：它等待每个线程完成并发布它们的共享内存写入，这样当 MMA 稍后读取 `Asmem` 和 `Bsmem` 时，它看到完整的块而不是半填充的缓冲区。这个线程驱动的复制也是我们将替换的第一件事；下一章（{ref}`chap_gemm_async`）将其换成 TMA。

**MMA 调度。** 操作数现在位于 SMEM 中，我们可以发出 MMA，我们从单个选举线程执行：

```python
if warp_id == 0:
    if T.ptx.elect_sync():
        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                      accum=False, dispatch="tcgen05", cta_group=1)
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

两个嵌套守卫分两步缩小发出者。外层 `if warp_id == 0` 只保留 warpgroup 的 warp 0，内层 `if T.ptx.elect_sync():` 然后在该 warp 中选举一个活跃 lane。它们一起恰好留下一个线程来运行 `Tx.gemm_async` 和 `tcgen05.commit`。

It is worth being clear about what that single thread does and does not mean, because the natural reading is misleading. A single issuing thread does *not* imply a single-threaded multiply. The computation is still a full tile-level MMA: the hardware performs the cooperative multiply for the tile described by the SMEM operand layouts and the TMEM accumulator layout. The key is that `Tx.gemm_async` is one *tile operation*, not one hardware instruction. The K = 64 tile is wider than the hardware MMA K-atom (`MMA_K = 16`), so this one tile op lowers to a short sequence of raw `tcgen05.mma` instructions stepped along K, and the warpgroup drives each of them cooperatively. The reason only one thread issues the tile op is that each underlying `tcgen05.mma` is itself a *single-instruction* cooperative op: one launch drives that K-atom of the tile MMA. If all 128 threads issued the sequence, the same work would simply be launched 128 times over. Finally, the `accum=False` flag tells the MMA to overwrite the TMEM destination rather than add into it, which is what we want here, since there is no prior partial sum to extend.

**Writeback.** The product now sits in TMEM, but the caller wants it back in GMEM as fp16. The epilogue therefore has to bring the result down through registers and cast it along the way:

```python
Dreg = T.alloc_local((BLK_N,), acc_type)        # per-thread fp32 register row
Dreg_f16 = T.alloc_local((BLK_N,), d_type)      # same row, cast to fp16
Dreg_wg = Dreg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
Tx.cast(Dreg_f16[:], Dreg[:])
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

The MMA leaves a 128 x 128 fp32 accumulator tile in TMEM. The fp32 is deliberate: GEMM sums many products along K, and keeping the running sum in higher precision holds down the rounding error that would otherwise accumulate. But `D` is fp16, so the values cannot go straight out. They first land in registers, are narrowed to fp16 there, and only then reach GMEM.

The two register buffers play distinct roles. `Dreg` is a per-thread buffer of `BLK_N` elements, while `Dreg_wg` is a warpgroup-wide *view* of those same registers under a chosen layout:

```python
TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])
```

This layout maps the tile's first dimension onto the warpgroup's threads: thread 0 owns row 0, thread 1 owns row 1, and so on through row 127. The second dimension stays inside each thread's own register buffer, so a single thread holds all the columns of its one row. With 128 threads in the warpgroup and 128 rows in the tile, the 128 x 128 output divides neatly into one row per thread.

Reading the accumulator out under that view is precisely what `Tx.wg.copy_async(Dreg_wg, tmem)` does, and it lowers to the Blackwell TMEM load path, `tcgen05.ld`. Because that load is asynchronous, `T.ptx.tcgen05.wait.ld()` has to complete before any thread touches `Dreg`; otherwise a thread would read registers the load has not yet filled.

Once the wait returns, each thread's private `Dreg[:]` holds the fp32 values for its one logical output row. The thread narrows those to fp16 in `Dreg_f16`, works out which global row it is responsible for,

```python
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
```

and writes `D[m_thr, n_st:n_st + BLK_N]`. The rows partition cleanly across the four warps: warp 0 writes rows 0-31, warp 1 writes rows 32-63, warp 2 writes rows 64-95, and warp 3 writes rows 96-127.

### Complete Kernel

Now we stitch the four pieces back together into one runnable kernel (M=N=128, K=64). The imports come first:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

The kernel is wrapped in the same `hgemm_vX(M, N, K)` style that the later steps use. Step 1 runs with `M=N=128, K=64`, so the launch contains exactly one output tile:

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K document the underlying hardware MMA tile; they are not
    # passed to gemm_async (which derives the MMA shape from the operand and
    # accumulator tiles), so the later steps omit them.
    MMA_M, MMA_N, MMA_K = 128, 128, 16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # Step 1 is a single-tile kernel: M = BLK_M and N = BLK_N, so the grid
        # is 1x1. Starting with a 1x1 grid keeps the per-CTA tile offsets
        # (m_st, n_st) trivially zero; Steps 3+ generalise this to larger M / N.
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])      # single warpgroup, so wg_id is always 0 (unused below)
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()
    
        # --- Barrier + TMEM init (warp 0 only) ---
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
    
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
    
        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )
    
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_mma: T.int32 = 0
    
        # --- Load: all threads copy global -> shared (synchronous).
        # With M=BLK_M and N=BLK_N the slices below cover the full matrices;
        # the slice form is kept so the diff to Step 3 (multi-tile) is minimal.
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()
    
        # --- Compute: single elected thread issues MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
    
        # --- Writeback: TMEM -> RF -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
    
        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

Every GEMM step that follows compiles, runs, and checks itself in the same way, so we spell that scaffolding out in full just once, here, and from then on show only the kernel. To run a later step, drop in its `hgemm_vX` and the matching problem size in place of the ones below. One caveat is worth remembering: compile a single step per fresh Python session and restart before trying another, since the examples reuse inner names and the compiler holds per-session state.

```python
import torch

target = tvm.target.Target("cuda")
device = torch.device('cuda')  # gpu(0)

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

# ex.mod(...) takes torch tensors directly, the same call form used in every chapter.
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
# Relative tolerance, like the warp-specialization and Flash Attention cells:
# output magnitude grows with K, so a fixed absolute bound would fail at larger K.
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")

# Optional timing for larger kernels.
ITERS = 10
for _ in range(3):
    ex.mod(A_tensor, B_tensor, D_tensor)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(ITERS):
    ex.mod(A_tensor, B_tensor, D_tensor)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / ITERS
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

Steps 1 through 3 run at deliberately small sizes (128×128 here, 256³ in Step 3) to keep these first walkthroughs simple to follow. The cross-step *End-to-End Result* table at the end of {ref}`chap_gemm_advanced` takes the opposite approach: it measures every step, including this Step 1 algorithm, at a single M=N=K=4096 size, so that its speedup ratios are directly comparable.

### Limits of the Single-Tile Kernel

This kernel is correct, which was the whole point of Step 1, but it is correct only in a very narrow setting. Four limitations are baked in on purpose, and the rest of the optimization path lifts them one at a time:

- It handles only a single K tile, so it cannot contract over a large K.
- It handles only a single output tile, so M and N are pinned to 128.
- It uses synchronous GMEM -> SMEM copies rather than TMA.
- It does not overlap data movement with compute, so the two never run at once.

---

(chap_k_loop)=
## Step 2: K-Loop Accumulation

The first limit to remove is the smallest one. Step 1 handles only a single 64-wide K tile, yet real matrices contract over far more than that. In Step 2 we keep the single output tile but let K span many 64-wide chunks.

The idea is straightforward: repeat the load -> MMA -> wait sequence once per chunk, and let each MMA accumulate into the same TMEM slot. The real work, it turns out, is in the synchronization. Reusing one mbarrier across iterations introduces this chapter's first genuine correctness hazard. If the code tracks the wrong phase, a wait can return *before* its MMA has actually finished, silently corrupting the result. The mechanics below show exactly how that goes wrong, and how to avoid it.

> **What this step changes: Layout reuse**
> - Scope: unchanged, still a single warpgroup.
> - Layout/reuse: the same SMEM tile pair and TMEM accumulator slot are reused across the K-loop. No new storage is allocated; the operand tiles stream through one fixed pair of buffers, and the accumulator state stays in one TMEM slot.
> - Synchronization: the reused MMA barrier must advance through the right phase on every K chunk, or a later wait can observe an earlier completion.
> - Dispatch: unchanged.

### K-Loop Mechanics

Step 1 contracted over a single 64-wide K tile; here we keep its single output tile but let K run as long as the matrices demand. To cover a K larger than 64, we walk K in chunks of `BLK_K=64`. Each iteration loads the next A and B K-slice into SMEM and issues `Tx.gemm_async`. The `accum` flag is what stitches these chunks together into one dot product: on the first chunk, `accum=False` initializes the TMEM accumulator, and on every later chunk, `accum=True` adds that chunk's product into the running sum already sitting in TMEM.

Synchronization is where the care is needed. We reuse a single mbarrier for every MMA completion, and reusing it safely comes down to tracking which barrier phase we are waiting on. An mbarrier carries a 1-bit phase, either 0 or 1, and it flips to the other value each time the expected arrival lands. The subtle part is the wait condition itself: `try_wait(bar, phase)` blocks until the barrier's internal phase *differs* from the `phase` argument. So the argument we pass has to name the phase we expect to leave behind, not the one we are waiting to reach:

| K iteration | Local `phase_mma` before wait | What `try_wait` waits for | Local update after wait |
|---|---:|---|---:|
| 0 | 0 | barrier flips to 1 | `phase_mma = 1` |
| 1 | 1 | barrier flips to 0 | `phase_mma = 0` |
| 2 | 0 | barrier flips to 1 | `phase_mma = 1` |

The single line `phase_mma ^= 1` is what keeps that table honest. Drop it, and the second iteration still calls `try_wait(bar, 0)`, but the barrier already flipped to phase 1 after the first MMA, so the wait sees a mismatch and returns immediately, before the second MMA has finished. The kernel then reads a half-computed accumulator and reports a wrong answer with no error at all. This is a bug that compiles and runs perfectly, which is exactly why the phase flip is worth this much attention.

### Complete Kernel

The full kernel below is simply Step 1 with the K-loop and the phase flip folded in. The imports are the same as before:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

It is wrapped in `hgemm_v2(M, N, K)`. The grid is still `[1, 1]`, since we are still computing a single output tile; all that has grown is its K extent:

```python
def hgemm_v2(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])  # still one output tile (M=N=128)
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # === K-loop: iterate over K in chunks of BLK_K ===
        for i in T.serial(K_TILES):   # serial device loop (keeps the full-K A/B parameters correctly shaped)
            # Load the i-th K chunk
            Tx.cta.copy(Asmem[:, :], A[:, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[:, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            # MMA: accum=False for first tile, True for rest
            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            # Wait for MMA, then flip phase
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # === Writeback (same as Step 1) ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

---

(chap_spatial_tiling)=
## Step 3: Spatial Tiling (Multi-CTA)

The K-loop took care of the contraction dimension, but M and N are still pinned to a single 128 x 128 tile. A real output is far larger than one tile, so the last piece of the basic kernel is to cover M and N with many tiles at once. Step 3 launches a 2D grid of CTAs, one per output tile, and lets the GPU compute all the tiles in parallel. The example uses M=N=K=256, which gives a 2x2 grid of tiles, just enough to make the indexing non-trivial without burying it.

> **What this step changes: Scope**
> - Scope: a 2D grid of CTAs, with each CTA owning one 128 x 128 output tile.
> - Layout: unchanged; within a CTA, this is the same SMEM/TMEM/register path as Step 2.
> - Dispatch: unchanged.

### Grid Mapping

The grid shape follows directly from the tiling: with one CTA per 128 x 128 output tile, we need `[M // BLK_M, N // BLK_N]` CTAs in total. The only genuinely new work compared to Step 2 is teaching each CTA which slice of the matrices is *its* slice to compute.

CTA `(bx, by)` owns this output region:

```text
D[bx * BLK_M : (bx + 1) * BLK_M,
  by * BLK_N : (by + 1) * BLK_N]
```

and to produce it, the CTA's K-loop repeatedly loads the matching K-slices of its own row band of A and column band of B:

```text
A[bx * BLK_M : (bx + 1) * BLK_M, k : k + BLK_K]
B[by * BLK_N : (by + 1) * BLK_N, k : k + BLK_K]
```

The indexing follows straight from the `D = A @ B.T` convention: `bx` selects rows of A and D, while `by` selects rows of B, which become the columns of D once the transpose is applied.

One tile per CTA is the simplest mapping that works, but it is also wasteful. Every CTA in a row reloads the same A tiles from GMEM, and every CTA in a column reloads the same B tiles, so nothing reuses the data that neighboring CTAs have already pulled in. We will leave that waste in place for now; persistent scheduling (Step 6 in {ref}`chap_gemm_async`) comes back to it and keeps those shared operands hot in L2.

**Try with your agent**: With `M=N=K=256`, `BLK_M=BLK_N=128`, and `BLK_K=64`, ask it to trace CTA `(1, 0)` and CTA `(0, 1)`. For each CTA, list `m_st`, `n_st`, the A and B slices loaded for each K iteration, and the D region written. Which B rows become D columns because the kernel computes `D = A @ B.T`?

### Complete Kernel

The kernel is once again Step 2, this time with just two changes: the grid shape and the per-CTA offsets. The inner K-loop and the writeback are untouched. The imports are the same:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

The grid becomes `[M // BLK_M, N // BLK_N]` rather than `[1, 1]`, and the loads and stores are now offset by the CTA's own `m_st` and `n_st`:

```python
def hgemm_v3(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 2D grid: one CTA per 128x128 output tile
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0

        # Per-CTA tile offsets
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # K-loop with offset A and B slices
        for i in T.serial(K_TILES):   # serial device loop (keeps the full-K A/B parameters correctly shaped)
            Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # Writeback to the correct output tile
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st:n_st+BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## Exercises

1. In Steps 1-3, `Tx.copy` moves A and B tiles into SMEM before MMA. Why does the kernel need `T.cuda.cta_sync()` before `Tx.gemm_async` reads those SMEM tiles?
2. In Step 2, what happens if `phase_mma ^= 1` is removed from the K-loop? Does the kernel wait for every MMA, or can a later wait pass too early?
3. For M=N=4096 with BLK_M=BLK_N=128, how many CTAs are launched in Step 3? Which operand tiles are logically reused across neighboring CTAs, and does Step 3 exploit that reuse?
