(chap_gemm_async)=
# 使用 TMA 流水线化 GEMM

:::{admonition} 概览
:class: overview

- 基本 GEMM 在两者可以同时运行时浪费时间轮流（复制一个块、计算、复制下一个）。
- 第 4 步切换到 TMA 异步加载，第 5 步双缓冲 SMEM 并预取（PIPE_DEPTH=2）；完整的加载/计算重叠通过第 7 步的 warp 特化实现，第 6 步用块调度器使内核持久化。
- 目标是在张量核心处理当前块时加载下一个块。
:::

张量核心是芯片上最昂贵的单元，前一章的正确分块 GEMM 让它们在大部分时钟周期中空闲。内核轮流：线程将块复制到共享内存，张量核心处理它，线程复制下一个块，张量核心等待。每个阶段在它前面的阶段上停顿，即使加载下一个块和在当前块上计算使用完全独立的硬件，可以同时运行。缩小这个差距不需要新的数据路径；块、布局和数学已经正确。必须改变的是工作*何时*发生以及*由谁*调度。本章保持块数据路径与之前完全相同，直接攻击空闲。

我们通过三个增量步骤达到目标，知道目的地会有所帮助。在第 4 步中，我们将批量 GMEM <-> SMEM 传输交给 TMA，让专用复制硬件移动块而不是线程。在第 5 步中，我们添加两阶段软件流水线，给下一个 K 块一个着陆的地方，同时当前块仍在被乘。在第 6 步中，我们将启动重塑为由块调度器驱动的持久化内核，它摊薄了每块设置，并让我们选择保持操作数热的块顺序。在整个过程中，SMEM、TMEM 和寄存器布局与前一章保持完全相同。唯一真正新的想法是硬件单元之间的异步交接：让一个引擎超前于另一个而不是齐步走。

(chap_tma_async)=
## 第 4 步：TMA 异步加载

我们的第一步是将复制本身从关键路径上移除。想想 CTA 在第 1-3 步中做了什么：它的每个线程计算地址并发出加载指令，唯一的目的只是将块运送到 SMEM 中。这是花在管道而非数学上的指令带宽。第 4 步用 TMA 替换同步 `Tx.copy`，一个线程发出一个命令，TMA 引擎自己执行整个块传输。从这里开始，示例以完整的 M=N=K=4096 大小运行，而不是第 1-3 步的小大小，它们的端到端计时出现在 {ref}`chap_gemm_advanced` 末尾的*端到端结果*表中。

> **这一步改变的：调度**
> - 作用域：不变，一个 warpgroup。
> - 布局：不变，相同的 SMEM/TMEM/寄存器块。
> - 调度：GMEM → SMEM 加载从同步 `Tx.copy` 移到 TMA 引擎。

### TMA 发出模式

第 4 步的一个改变是用 TMA 加载替换同步块复制，所以值得仔细看看该加载如何发出。对源代码的编辑只有几行，但这些行背后的执行模型在本质上不同。同步 `Tx.copy` 是 CTA 线程自己用它们自己的指令做的工作；TMA 复制是一个线程发出的命令，之后 TMA 硬件执行所有移动。值得将两者并排查看。

**之前（第 3 步）**：所有 128 个线程参与复制，然后 `cta_sync` 使共享内存写入可见：
```python
Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])   # all 128 threads
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])
T.cuda.cta_sync()
```

**After (Step 4)**: one thread issues the TMA load, and the mbarrier tracks when the hardware transfer is complete:
```python
tid = warp_id * 32 + lane_id                 # 0..127 within the warpgroup
if tid == 0:  # exactly one thread starts TMA
    Tx.copy_async(Asmem, A[...], dispatch="tma")
    Tx.copy_async(Bsmem, B[...], dispatch="tma")
    T.ptx.mbarrier.arrive.expect_tx(tma_bar, byte_count)  # bytes expected from TMA
T.ptx.mbarrier.try_wait(tma_bar, phase)                  # wait before MMA reads SMEM
```

Notice that the load is gated on `tid == 0`, not on `elect_sync()`, and the distinction matters more than it looks. `elect.sync` elects one active lane *per warp*, and a warpgroup has four warps, so `elect_sync()` would actually let four threads enter the load protocol. The trouble is that the protocol announces the expected byte count to the mbarrier, and it must announce it exactly once; four announcements would corrupt the count and the wait would never release correctly. Picking precisely one thread by its warpgroup-wide id is the clean way to avoid that.

It is important to be honest about where the speedup comes from. Step 4 still waits after every TMA load, so we are not yet overlapping the load with the compute; that is the work of Step 5. The win here comes purely from the change of data-movement path:

- `Tx.copy` uses CTA threads to compute addresses and issue load/store instructions.
- TMA uses one issued command to start a hardware tile transfer. Address generation, coalescing, and swizzling are described by the TMA descriptor and carried out by the TMA engine.

So even though Step 4 still blocks on each load, it ends up faster anyway. TMA absorbs the bulk transfer, which frees the CTA threads from spending instruction bandwidth shuffling tiles around, and that saving alone is enough to move the needle.

### TMA Load and Store Synchronization

We have seen how a TMA copy is issued; the other half of the story is knowing when it has finished. Switching to TMA changes two things at once: who starts a copy, and how the code knows when it finished. The first is obvious from the code; the second is easy to overlook, and getting it wrong gives you a silent correctness bug rather than a crash. With `Tx.cta.copy`, the CTA threads do the copy together and a following `cta_sync()` is enough to know it is done. With TMA, one selected thread issues `Tx.copy_async(..., dispatch="tma")`, the engine performs the transfer on its own schedule, and it signals completion through an mbarrier.

This is exactly why `cta_sync()` is no longer sufficient. `cta_sync()` waits only for the CTA's own threads and orders only their shared-memory writes; it knows nothing about an in-flight TMA transfer, so it happily returns while the tile is still arriving. The fix is to make completion explicit: for a TMA load, the selected thread first tells the mbarrier how many bytes to expect, and the CTA then waits on *that* mbarrier before any MMA touches the SMEM tile. The figure below traces that handshake end to end.

![TMA Async Load: Synchronization Flow](../img/tma_sync_flow.png)

The figure above isolates the load-side handshake: one selected thread launches TMA, the mbarrier
counts the expected bytes, and MMA waits on the release before reading SMEM. Where it says
"Elected Thread" it means the selected thread that starts TMA, which in our code is the `tid == 0`
thread, not an `elect_sync()` lane.

Putting the load path together, then: the selected thread issues both `copy_async` calls and follows them with `arrive.expect_tx(total_bytes)`, where the byte count is precisely how much data the mbarrier should hold out for. Once the engine has moved that many bytes, the matching `mbarrier.try_wait(phase)` releases, and only then is the SMEM tile safe to feed to MMA.

The store side travels over the same hardware but waits in a different way, so it pays to keep the two protocols clearly apart in your head: loads track completion with mbarriers and byte counts, while stores track it with commit groups and wait groups. After the threads write their fp16 results into `Dsmem` and synchronize, one selected thread starts `Tx.copy_async(D[...], Dsmem, dispatch="tma")`, and then `cp_async.bulk.commit_group()` followed by `cp_async.bulk.wait_group(0)` block until the store has drained. That wait is not optional: `Dsmem` cannot be reused for the next tile until the previous store is gone.

**Try with your agent**: Trace the Step 4 load and store synchronization for one K tile. Identify which thread starts each TMA command, which mbarrier or commit group tracks completion, which wait protects MMA reads of `Asmem` and `Bsmem`, and which wait protects reuse of `Dsmem`. Why would `elect_sync()` be the wrong thread selection for the TMA load protocol here?

### Complete Kernel

The complete kernel folds the TMA load and store into the Step 3 structure, leaving the rest of that structure untouched. The imports are the same as before:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

It is wrapped in `hgemm_v4(M, N, K)`, a pattern we follow throughout: the wrapper keeps the shape-dependent constants and layouts right next to the kernel that uses them.

```python
def hgemm_v4(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    F16_SIZE = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM allocation (now includes Dsmem for TMA store) ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((1,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()
    
        # --- Barrier + TMEM init ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
        if warp_id == 0:
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
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0
    
        # --- Inline helpers ---
        @T.inline
        def tma_load(k_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([0])
            })
            Tx.copy_async(Asmem[:, :],
                          A[m_st : m_st + BLK_M, k_st : k_st + BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[:, :],
                          B[n_st : n_st + BLK_N, k_st : k_st + BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([0]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
            )
    
        @T.inline
        def mma(accum):
            Tx.gemm_async(
                tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                accum=accum, dispatch="tcgen05", cta_group=1
            )
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        # --- K-loop with TMA async ---
        tid = T.meta_var(warp_id * 32 + lane_id)
        for k in range(K_TILES):
            k_st = T.meta_var(k * BLK_K)
    
            # Single thread issues TMA load
            if tid == 0:
                tma_load(k_st)
    
            # Wait for TMA to finish; the mbarrier release carries SMEM
            # visibility to the subsequent MMA, so no extra fence is needed.
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)
    
            # Single thread issues MMA
            if tid == 0:
                mma(accum=k != 0)
    
            # Wait for MMA to finish
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_tma ^= 1
            phase_mma ^= 1
    
        # --- TMA Store Writeback ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
    
        # Read TMEM -> registers (async; wait.ld then cta_sync to ensure read completes)
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        # Cast fp32 -> fp16
        Tx.cast(Dreg_f16[:], Dreg[:])
        # Write registers -> Dsmem, flush, then sync
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        # TMA store: Dsmem -> GMEM. One selected thread starts the store and drains the
        # store group before Dsmem is reused.
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)
    
        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

### TMA Configuration in the Kernel

Almost everything in that kernel is carried over from Step 3. Only five configuration points actually carry the TMA semantics, and it is worth knowing each by name:

- **TMA config**: `{"dispatch": "tma", "cta_group": 1, "mbar": tma_bar.ptr_to([0])}` tells `Tx.copy_async` to use TMA and to report load completion through `tma_bar`.

- **Byte count**: `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` is the number of bytes loaded by the two fp16 operand tiles. `arrive.expect_tx(...)` gives this count to the mbarrier.

- **mbarrier initialization**: `init(tma_bar.ptr_to([0]), 1)` creates the completion barrier used by the TMA load.

- **`@T.inline`**: `tma_load(...)` and `mma(...)` are helper functions. They are expanded into the kernel body at compile time and can use variables from the surrounding kernel.

- **TMA store synchronization**: The epilogue first writes fp16 rows into `Dsmem`. `fence.proxy_async` and `warpgroup_sync` make those thread-written SMEM values ready for the TMA store path. The store then uses `commit_group()` and `wait_group(0)` to wait for the SMEM-to-GMEM transfer to finish.

At this point we have the right pieces but the wrong rhythm. Step 4 still finishes each load before starting the matching MMA, so the load and the multiply never actually run at the same time; the two engines we worked so hard to separate still take turns. The next step leaves the TMA load and store path exactly as it is and instead rearranges the schedule, so that loading one K tile can proceed while compute runs on another.

(chap_software_pipeline)=
## 第 5 步：软件流水线（PIPE_DEPTH=2）

为什么第 4 步不能将加载与计算重叠，当两个引擎明显独立时？障碍原来是存储。只有一个 SMEM 块对时，下一个加载无处可去：它不能在当前 MMA 完成读取该对之前开始，因为提前开始会覆盖仍在使用的数据。第 5 步通过双缓冲共享内存消除该存储冲突。单 warpgroup 循环在发出下一个 TMA 加载之前仍然等待每个 MMA，但它现在有Distinct的阶段可以预取和重用。我们仍然在完整的 M=N=K=4096 大小。

> **这一步改变的：布局**
> - 作用域：不变，一个 warpgroup。
> - 布局：单个 SMEM 块对变为 `PIPE_DEPTH` 阶段环形缓冲区。
> - 调度：不变，TMA 加载和 `tcgen05` MMA；这一步添加预取和阶段重用，而完整的加载/计算重叠在第 7 步到达。

### 流水线演练

使用 `PIPE_DEPTH=2`，内核分配两个 SMEM 阶段，为加载路径和 MMA 路径提供独立的工作槽位。

将下图视为两阶段缓冲区旨在启用的流水线结构，而不是这个单 warpgroup 内核的精确执行跟踪。第 5 步构建环形缓冲区并预取后续阶段，但主循环在发出下一个 TMA 加载之前仍然等待当前 MMA。完整的加载/计算重叠在第 7 步到达，当 warp 特化给 TMA 和 MMA 独立角色时。

![*流水线 PIPE_DEPTH=2，目标调度；这个单 warpgroup 步骤只预取，完整重叠通过第 7 步的 warp 特化到达*](../img/pipe_depth2.png)

一旦预热，循环在两个阶段之间交替。两个 TMA 加载预先填充两个阶段；之后，循环等待当前阶段，在其上运行 MMA，等待该 MMA 完成读取阶段，然后为 `k + PIPE_DEPTH` 启动加载到刚变为可重用的阶段。这还不是并发 TMA/MMA 调度，但它建立了第 7 步将跨生产者和消费者角色拆分的环形缓冲区结构。

具体来说，代码与第 4 步在四个地方不同：

1. `Asmem` 和 `Bsmem` 获得前导 `PIPE_DEPTH` 维度，所以每个阶段有自己的 SMEM 存储。
2. `tma_bar` 变为每阶段一个 mbarrier 的数组。
3. 在主 K 循环之前，内核预取前两个阶段。
4. K 循环使用 `stage = k % PIPE_DEPTH`：等待当前阶段，在其上运行 MMA，然后为 `k + PIPE_DEPTH` 重用该阶段。

### 流水线机制

**1. 预取**：在主循环运行之前，我们加载前 `PIPE_DEPTH` 个阶段，这样循环总是在第一次迭代时找到等待它的数据：
```python
for s in range(min(PIPE_DEPTH, K_TILES)):
    tma_load(s, s * BLK_K)
```

**2. Main loop**: for each K tile we wait for its stage to be ready, run MMA on it, and then immediately put that now-free stage back to work by launching the load for the tile `PIPE_DEPTH` ahead:
```python
stage = k % PIPE_DEPTH
wait(tma_bar[stage], phase_tma)
mma(stage, accum)
wait(mma_bar[0], phase_mma)
phase_mma ^= 1
tma_load(stage, next_k * BLK_K)
```

**3. Phase management**: this is the part that trips people up, but the rule is simpler than it first appears. The phase-flip rule for each barrier follows directly from how many slots that barrier has, which is why the two barriers flip on different cadences. The MMA accumulator lives in one TMEM slot, so `mma_bar` is a single barrier (`mma_bar.ptr_to([0])`) that every iteration revisits, and a barrier you revisit every iteration must have its phase flipped every iteration. The TMA barriers tell a different story: they form a `PIPE_DEPTH`-element array with one barrier per stage, and any given stage's barrier only comes back around once per trip through the ring. So `phase_tma` flips only when the stage index wraps back to 0:
```python
if stage == PIPE_DEPTH - 1:
    phase_tma ^= 1
```

**Try with your agent**: With `PIPE_DEPTH=2` and `K_TILES=5`, ask it to trace the main loop. For each `k`, list `stage`, the `phase_tma` and `phase_mma` values passed to the waits, and whether a new prefetch is issued. Where exactly does `phase_tma` flip, and why is there no prefetch for the last two iterations?

### Complete Kernel

The complete kernel keeps the Step 4 TMA load and store path verbatim, then wraps it in the staged buffers and phase logic we just described. The imports are unchanged:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

It is wrapped in `hgemm_v5(M, N, K)`. The `PIPE_DEPTH=2` constant sets the number of pipeline stages (two of them here, which is exactly double buffering):

```python
PIPE_DEPTH = 2

def hgemm_v5(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    # Double-buffered layouts: first dimension is pipeline stage
    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        # Double-buffered TMA barriers (one per stage), single MMA barrier
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # Initialize barriers: PIPE_DEPTH for TMA, 1 for MMA
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                for s in range(PIPE_DEPTH):
                    T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0

        @T.inline
        def tma_load(stage, k_offset):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        tid = T.meta_var(warp_id * 32 + lane_id)

        # === Prefetch: load first PIPE_DEPTH stages ===
        if tid == 0:
            for s in range(min(PIPE_DEPTH, K_TILES)):
                tma_load(s, s * BLK_K)

        # === Main loop ===
        for k in range(K_TILES):
            stage = k % PIPE_DEPTH

            # Wait for TMA to finish loading this stage
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)

            # MMA on this stage's data
            if tid == 0:
                mma(stage, accum=(k != 0))

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

            # Issue next prefetch load (k + PIPE_DEPTH)
            next_k = k + PIPE_DEPTH
            if next_k < K_TILES:
                if tid == 0:
                    tma_load(stage, next_k * BLK_K)

            # TMA phase flips when stage wraps around
            if stage == PIPE_DEPTH - 1:
                phase_tma ^= 1

        # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.cast(Dreg_f16[:], Dreg[:])
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

(chap_persistent_kernel)=
## Step 6: Persistent Kernel + Tile Scheduler

Everything up to now has optimized the work inside a single tile. Step 6 changes the scale of the question and optimizes across tiles.

Step 5 launches one CTA per 128 x 128 output tile. For a 4096 x 4096 output, that means 1024 separate CTAs, each paying its own setup cost and then vanishing the moment its tile is done.

Step 6 launches a fixed pool of CTAs instead, then asks each CTA to process many tiles in turn. This buys us two things: setup work is amortized across several tiles, and tile assignment moves inside the kernel, where the scheduler can choose an order that reuses operands. We remain at the full M=N=K=4096 size.

> **What this step changes: Scope**
> - Scope: a fixed pool of persistent CTAs, each looping over many output tiles via the scheduler.
> - Layout: unchanged, the same per-tile SMEM/TMEM/register path.
> - Dispatch: unchanged.

### Persistent Scheduling

The defining idea of a persistent kernel is that it sizes its grid to the hardware rather than to the problem. It launches `SM_COUNT` CTAs, roughly one per SM, no matter how many output tiles there happen to be, with the aim of keeping each SM continuously occupied. We say "roughly" deliberately: exact 1:1 residency is not guaranteed, since it depends on occupancy and on how the hardware chooses to schedule CTAs.

On the B200 we are targeting here, `SM_COUNT=148`. Each of those 148 CTAs loops over the tiles handed to it by `ClusterPersistentScheduler2D`.

The first payoff is amortization. TMEM allocation, barrier initialization, and scheduler state now happen once per CTA and are reused across the roughly 7 tiles that CTA handles, rather than being repeated 1024 times across throwaway CTAs.

The second payoff comes from the order the scheduler picks. Setting `l2_group_size=8` groups nearby tiles together, so tiles sharing a row band reuse the same A row-tiles, and tiles sharing a column band reuse the same B tiles. Running those tiles back-to-back keeps the operands hot in L2 instead of re-fetching them from HBM. This is exactly the reuse that Step 3 left on the table.

```python
bx = T.cta_id([SM_COUNT])  # 1D grid, one CTA per SM

tile_scheduler = ClusterPersistentScheduler2D(
    "ts",
    num_m_tiles=M // BLK_M,
    num_n_tiles=N // BLK_N,
    l2_group_size=8,       # Group 8 nearby tiles together
    num_clusters=SM_COUNT
)
tile_scheduler.init(bx)
```

Looping over tiles brings one correctness consequence that is easy to miss. Each tile runs its own fresh K-loop, which means its barrier phases have to start from a known state. In Step 5 a CTA handled exactly one tile, so initializing `phase_tma` and `phase_mma` a single time was perfectly fine. In Step 6 those initializers must move *inside* the `while tile_scheduler.valid()` loop, so that each tile begins with phase state matched to its own TMA and MMA work, rather than inheriting whatever the previous tile happened to leave behind:

```python
while tile_scheduler.valid():
    phase_tma: T.int32 = 0
    phase_mma: T.int32 = 0
    ...
```

### Complete Kernel

Structurally, the kernel is nothing more than Step 5's pipeline wrapped in a tile-level outer loop. The only new dependency is the scheduler itself, which we import alongside the rest:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D
```

The grid dimension is now simply `SM_COUNT` rather than `(M//BLK_M, N//BLK_N)`, and a `ClusterPersistentScheduler2D` takes over the job of handing each CTA its tiles:

```python
SM_COUNT = 148  # Number of SMs on NVIDIA B200 GPU
PIPE_DEPTH = 2

def hgemm_v6(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 1D grid: one CTA per SM (not a 2D grid anymore!)
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation (same as Step 5) ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # --- Barrier + TMEM init (same as Step 5) ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            for s in range(PIPE_DEPTH):
                T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        # Tile scheduler: assigns tiles to CTAs in L2-friendly order
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // BLK_M,
            num_n_tiles=N // BLK_N,
            l2_group_size=8,
            num_clusters=SM_COUNT
        )
        tile_scheduler.init(bx)

        tid = T.meta_var(warp_id * 32 + lane_id)

        @T.inline
        def tma_load(stage, k_offset, m_st, n_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # === Outer loop: iterate over tiles ===
        while tile_scheduler.valid():
            # Get current tile position from scheduler
            m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

            # === Inner loop: same pipeline as Step 5 ===
            phase_tma: T.int32 = 0
            phase_mma: T.int32 = 0

            # Prefetch first PIPE_DEPTH stages
            if tid == 0:
                for s in range(min(PIPE_DEPTH, K_TILES)):
                    tma_load(s, s * BLK_K, m_st, n_st)

            # Main K-loop
            for k in range(K_TILES):
                stage = k % PIPE_DEPTH
                T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)
                if tid == 0:
                    mma(stage, accum=(k != 0))
                T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma ^= 1
                next_k = k + PIPE_DEPTH
                if next_k < K_TILES:
                    if tid == 0:
                        tma_load(stage, next_k * BLK_K, m_st, n_st)
                if stage == PIPE_DEPTH - 1:
                    phase_tma ^= 1

            # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
            Dreg = T.alloc_local((BLK_N,), acc_type)
            Dreg_f16 = T.alloc_local((BLK_N,), d_type)
            Dreg_wg = Dreg.view(128, BLK_N,
                                layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
            Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
            T.ptx.tcgen05.wait.ld()
            T.cuda.cta_sync()
            Tx.cast(Dreg_f16[:], Dreg[:])
            Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.warpgroup_sync(10)
            if tid == 0:
                Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                              Dsmem[:, :], dispatch="tma")
                T.ptx.cp_async.bulk.commit_group()
                T.ptx.cp_async.bulk.wait_group(0)
            T.cuda.warpgroup_sync(10)

            T.cuda.cta_sync()
            tile_scheduler.next_tile()  # Move to next tile

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## Exercises

1. In Step 4, `arrive.expect_tx` uses `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` bytes. What does the mbarrier wait for if this byte count is too small or too large?
2. In Step 5, why does each SMEM stage need its own TMA barrier instead of sharing one `tma_bar` for both stages?
3. In Step 6, a 4096 x 4096 output with `BLK_M=BLK_N=128` has how many output tiles? With `SM_COUNT=148`, how many tiles does each persistent CTA process on average?
