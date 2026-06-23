# 面向机器学习系统的现代 GPU 编程

机器学习系统是现代 AI 工作负载的核心。在这些系统中，性能往往取决于少数几个 GPU 内核的质量。注意力内核、LLM 预填充和解码内核、低精度分块缩放 GEMM、融合 MoE 层以及其他大型融合内核，都直接决定了训练和推理的端到端速度。

然而，要让这些内核运行得更快，仅靠一堆优化技巧是不够的。现代 GPU 不再是旧设计的简单变体。最新的架构引入了更丰富的内存空间、新的访问模式，以及越来越专用的执行单元。要高效地编程这些硬件，我们既需要对硬件有清晰的心智模型，也需要理解高性能内核的构建方式。本书正是为了培养这两方面的能力。

本书遵循一个简单的递进路径：先理解 GPU 硬件，再学习我们将使用的编程模型，最后一步步构建最先进的内核。我们的主要目标是 Blackwell 架构，主要示例是快速矩阵乘法（GEMM）和 FlashAttention。在此过程中，我们还将研究 GPU 优化的核心要素：数据布局、异步数据搬运和异步协调。

本材料源自卡内基梅隆大学的[机器学习系统](https://mlsyscourse.org/)课程系列。为了让概念更易学习和运行，本书使用 **TIRx** Python DSL 逐步构建真实的 GPU 内核示例。TIRx 紧贴硬件，使我们能够在学习可运行代码的同时，理解底层控制细节。

## 本书结构

- **第一部分，理解 GPU。** 本部分介绍 GPU 的整体架构、编写快速内核的通用方法，以及数据布局、异步内存操作和协调等关键概念。它为全书奠定了硬件直觉基础。
- **第二部分，TIRx 概览。** 本部分介绍 TIRx 的核心要素，它们是本书代码示例的基础。
- **第三部分，GEMM：从分块到 SOTA。** 完整的分块 GEMM 优化指南，涵盖 TMA 流水线、持久调度、warp 特化和 2-CTA 集群。
- **第四部分，Flash Attention 4。** 基于第三部分技术构建的完整注意力内核：两个 MMA 之间夹 softmax、在线 softmax 重缩放、因果掩码和 GQA。
- **参考。** TIRx 语言参考和编译器内部实现。

```{toctree}
:caption: 第一部分，理解 GPU
:maxdepth: 1

chapter_background/index
chapter_performance/index
chapter_data_layout/index
chapter_layout_generations/index
chapter_tma/index
chapter_tensor_cores/index
chapter_tmem/index
chapter_async_barriers/index
chapter_clc/index
```

```{toctree}
:caption: 第二部分，TIRx 概览
:maxdepth: 1

chapter_intro_tirx/index
chapter_tirx_layout_api/index
```

```{toctree}
:caption: "第三部分，GEMM：从分块到 SOTA"
:maxdepth: 2

chapter_gemm_basics/index
chapter_gemm_async/index
chapter_gemm_advanced/index
```

```{toctree}
:caption: 第四部分，Flash Attention 4
:maxdepth: 2

chapter_flash_attention/index
```

```{toctree}
:caption: 参考
:maxdepth: 1

appendix/index
appendix/debugging_warp_specialized
tirx_guide/arch/index
tirx_guide/language_reference/index
```
