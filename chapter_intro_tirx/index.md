(chap_tirx_primer)=
# TIRx 简介

:::{admonition} 概览
:class: overview

- TIRx 是一个 Python DSL，用于在 IR 层面编写 GPU 内核：你直接命名硬件，但通过结构化 IR。
- 每个块操作由三个设计元素控制：*作用域*（哪些线程）、*布局*（块存在于何处）和*调度*（哪条硬件路径）。
- 一个可运行的单 MMA GEMM 展示了所有三个；本书其余部分是这些设计元素的大规模应用。
:::

:::{admonition} 运行示例
:class: note

这些示例需要 Blackwell GPU（`sm_100a`，如 B200）。TIRx 编译器作为 Apache TVM wheel 的 `tvm.tirx` 模块发布；与 CUDA 版本的 PyTorch 一起安装：

```bash
pip install apache-tvm==0.25.0
```

确认它可以通过 `python -c "import tvm, tvm.tirx; print(tvm.__version__)"` 导入。相同的设置运行本书中的每个可运行示例。
:::

第一部分解释了硬件是什么。要让它计算任何东西，我们需要一种编程方式。

我们可以编写原始 CUDA 或 PTX，许多快速内核正是那样编写的。问题是真正决定内核行为的决策在那里很难看到：哪些线程运行操作，每个数据块存在于何处，以及哪条硬件路径执行它。这些选择隐藏在内部函数参数、地址计算和约定中。

TIRx（Tensor IR neXt）是一个 Python DSL，将这三个决策提升到显式位置：**作用域**（哪些线程运行操作）、**布局**（操作数块存在于何处）和**调度**（哪条硬件路径执行它）。它仍然直接命名硬件概念，包括线程、共享内存和张量内存、屏障和 `tcgen05` MMA。区别是这些选择现在是可以被编译器降低、检查和调度的结构化 IR。

我们不是抽象地介绍这些概念，而是从一个完整的内核开始：一个最小的单 MMA GEMM。我们先让它运行起来，然后逐行回顾它，看看作用域、布局和调度如何各自塑造它以及内核如何编译。内核依赖的张量布局模型在 {ref}`chap_tirx_layout_api` 中独立开发，完整的语言特性集在 {ref}`chap_language_reference` 中；这里我们专注于这一个内核和三个设计元素。

## 第一个内核：单 MMA GEMM

我们承诺的内核是一个最小的 GEMM，精简到仍然使用张量核心的最小版本。它计算 K = 64 时 `D = A B^T` 的单个 128 x 128 输出块。整个计算从头到尾表示为一个 `Tx.gemm_async` 块操作。（那个块操作不映射到单个硬件指令：因为硬件 MMA K-原子是 16，K=64 块降低为沿 K 步进的短序列 `tcgen05.mma` 指令。DSL 的关键正是我们编写块，而不是序列。）围绕该操作，内核执行常规任务：分配共享内存（SMEM）和张量内存（TMEM），将 A 和 B 从全局复制到共享内存，将块 MMA 发送到 TMEM 累加器，通过寄存器将累加器读回，并存储结果。尽管很小，这个内核是 {ref}`chap_gemm_basics` 中我们攀登的 GEMM 阶梯的第 1 步，它在那里带着完整讲解返回。

每个 TIRx 内核都从相同的几个导入开始，所以值得先看一次：

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

我们将内核包装在一个小构建器 `hgemm_v1(M, N, K)` 中，它接受问题形状并返回 `PrimFunc`。对于我们选择的形状 `M=N=128, K=64`，启动恰好包含一个输出块，这使第一个版本足够简单，可以一次读完：

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K 记录底层硬件 MMA 块；它们不传递给
    # gemm_async（它从操作数和累加器块推导 MMA 形状），
    # 所以后续步骤省略它们。
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
        # 第 1 步是单块内核：M = BLK_M 且 N = BLK_N，所以网格
        # 是 1x1。从 1x1 网格开始使每 CTA 块偏移
        # (m_st, n_st) 平凡地为零；第 3 步+ 将此推广到更大的 M / N。
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])      # 单 warpgroup，所以 wg_id 总是 0（下面未使用）
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM 分配 ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()
    
        # --- 屏障 + TMEM 初始化（仅 warp 0） ---
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
    
        # --- 加载：所有线程复制全局 -> 共享（同步）。 ---
        # 当 M=BLK_M 且 N=BLK_N 时，下面的切片覆盖完整矩阵；
        # 保留切片形式使与第 3 步（多块）的差异最小。
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()
    
        # --- 计算：单个选举线程发出 MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
    
        # --- 写回：TMEM -> RF -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
    
        # --- 释放 TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

在阅读内核之前，让我们确保它能工作。我们编译它并将其输出与 torch 参考进行比较。我们不必指定确切的架构：架构（例如 `sm_100a`）从设备自动检测，所以目标 `"cuda"` 就足够了，`tir_pipeline="tirx"` 是选择 TIRx 降低流水线的选项。编译后，`ex.mod(...)` 直接接受 torch 张量，无需手动转换。

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

# ex.mod(...) 直接接受 torch 张量，每章使用相同的调用形式。
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")
```

## 作用域、布局、调度

现在内核运行了，我们可以回顾它并问它的行实际上决定了什么。从这个角度看，整个内核是一组沿三个设计元素的选择。其中每个操作都回答相同的三个问题：*谁*运行它，*哪里*它的数据存在，以及*如何*执行它，这三个答案正是作用域、布局和调度。本节接下来逐一介绍设计元素；下面的交互演示让你看到每个设计元素控制哪些行。

```{raw} html
<iframe src="../demo/tirx_dispatch.html" title="TIRx：作用域、布局、调度" loading="lazy"
        style="width:100%; min-width:960px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互演示：点击作用域 / 布局 / 调度以聚焦每个设计元素控制的内核行。*

使用演示时，注意三个问题：

- **作用域：谁运行操作？** `Tx.cta.copy(...)` 是 CTA 作用域的，所以所有 128 个线程帮助 GMEM → SMEM 复制。`Tx.gemm_async(...)` 由一个选举线程发出一次，因为每个降低的 `tcgen05.mma` 指令已经是一个协作 MMA 启动。`Tx.wg.copy_async(...)` 是 warpgroup 作用域的，所以 warpgroup 的 128 个线程逐行拆分 TMEM 回读。
- **布局：每个块存在于何处？** A 和 B 使用 `tcgen05.mma` 期望的 swizzled SMEM 布局。累加器在 TMEM 中，使用 `TLane`/`TCol` 布局。寄存器回读视图将行映射到 `tid_in_wg`，所以每个 warpgroup 线程拥有一个行片段。
- **调度：哪条硬件路径执行它？** `Tx.gemm_async(..., dispatch="tcgen05", ...)` 选择 Blackwell 张量核心路径。复制操作也有调度选择：这个第一个内核使用普通线程复制，后续 GEMM 步骤将这些复制替换为 TMA，而不改变周围的作用域或布局。

**试一试**：从第一个内核中选三行：一个复制、一个 MMA 和一个 TMEM 回读。让 agent 按作用域、布局和调度标记每行，然后检查答案是否与代码中的守卫、缓冲区布局和 `dispatch=` 参数匹配。

## 编译如何工作

我们已经在上面编译了内核来测试它；现在我们更仔细地看看那一步做了什么。方法很短：将 `PrimFunc` 包装在 `IRModule` 中并交给 `tvm.compile(mod, target=..., tir_pipeline="tirx")`。这运行 TIRx 降低流水线并返回你可以直接调用的 `Executable`。

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

值得至少大致了解 `tir_pipeline="tirx"` 启动了什么。流水线的核心传递 `LowerTIRx` 解析每个块原语与其作用域/布局/调度契约：这是我们刚才讨论的三个设计元素实际兑现为指令的地方。之后，常规的主机/设备分离和最终化步骤产生可启动的模块。如果你愿意，你也可以在 `with target:` 块内编译，让内核获取周围的目标上下文。

这个流程的一个好特性是没有什么对你隐藏：结果可以在两个级别检查。你可以用 `.show()` 或 `.script()` 阅读 IR 本身，也可以直接从编译后的模块阅读编译器最终发出的 CUDA C。

```python
kernel.show()                          # 美化打印 TIRx（TVMScript）
print(kernel.script())                 # ... 相同，作为字符串

# 生成的 CUDA C 源码，来自编译后的 Executable：
print(ex.mod.imports[0].inspect_source())
```

这只是一个概要。完整的降低故事，涵盖所有传递、块原语调度如何解析，以及主机/设备分离如何完成，请参见 {ref}`chap_arch`。

## 接下来去哪里

一个内核足以认识作用域、布局和调度，并看到它们编译和运行。三个设计元素中的每一个，以及内核本身，都打开一个章节将其进一步推进：

- {ref}`chap_tirx_layout_api`：张量布局模型（`TileLayout`、命名轴、swizzle），上面的操作数和累加器放置就是建立在此之上。如果布局设计元素感觉是三个中最神秘的，从这里开始。
- {ref}`chap_language_reference`：完整的语言特性集，涵盖解析器工具、数据类型、缓冲区和内存、控制流和线程同步，当你需要完整词汇表而不仅仅是导览时。
- {ref}`chap_gemm_basics`：这个内核作为 GEMM 优化路径的第 1 步，通过 K 循环累加、空间分块、TMA 和 warp 特化构建。如果你想看相同的三个设计元素扩展到真实内核，这是自然的下一站。
