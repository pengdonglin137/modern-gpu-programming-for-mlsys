# 面向机器学习系统的现代 GPU 编程

本书以递进方式教授现代 GPU 内核编程：**理解 GPU 硬件 → 学习编程方法 → 编写最先进的内核。** 它以 Blackwell 架构 GPU 为主题——涵盖其内存层次结构和张量内存、张量核心和异步数据搬运引擎、warp 组和集群。教学载体是 **TIRx**（Tensor IR neXt），一个在 IR 层面编写 GPU 内核的 Python DSL。

📖 **在线阅读：<https://mlc.ai/modern-gpu-programming-for-mlsys/>**

## 内容概览

- **第一部分——理解 GPU。** 执行模型和内存模型、性能模型（屋顶线、重叠）、数据布局深入讲解、内存和计算引擎（TMA、张量内存、张量核心）、异步协调，以及高级调度（CLC）。
- **第二部分——使用 TIRx 编程 GPU。** 通过一个可运行的单 MMA GEMM 介绍 TIRx——作用域、布局和调度，以及编译原理——还包括张量布局模型（`TileLayout`、命名轴、swizzle）。
- **第三部分——GEMM：从分块到 SOTA。** 通过 TMA 流水线、持久调度、warp 特化和 2-CTA 集群逐步构建分块 GEMM。
- **第四部分——Flash Attention 4。** 基于第三部分技术构建的完整注意力内核：两个 MMA 之间夹 softmax、在线 softmax 重缩放、因果掩码和 GQA。
- **参考。** TIRx 语言参考和编译器内部实现。

## 本地构建

本书是一个 [Sphinx](https://www.sphinx-doc.org/) 站点（Markdown/MyST + reStructuredText）：

```bash
pip install -r requirements-docs.txt
sphinx-build -b html . _build/html
```

### 预览

```bash
python -m http.server -d _build/html 8000
```

打开 <http://localhost:8000>。如果在远程机器上运行，需要端口转发——`ssh -L 8000:localhost:8000 user@your-server`——然后在本地打开 URL。（VS Code Remote SSH 会自动转发。）

## 运行内核（需要 Blackwell GPU）

本书中的内核针对 Blackwell（`sm_100a`）架构，因此运行它们需要 Blackwell GPU（如 B200）、TIRx 编译器和 CUDA 版本的 PyTorch。

**1. 安装 TIRx 编译器。** 它作为 Apache TVM wheel 的 `tvm.tirx` 模块发布：

```bash
pip install apache-tvm==0.25.0
```

验证安装：

```bash
python -c "import tvm, tvm.tirx; print(tvm.__version__)"
```

**2. 安装 PyTorch**（使用与 GPU 匹配的 CUDA 构建版本，用于示例输入和参考检查）——参见 <https://pytorch.org>。

**3.（可选）参考内核。** 完整的 GEMM 和 Flash Attention 4 内核位于配套的 `tirx-kernels` 包中（从检出目录运行 `pip install -e .`）；运行示例：`python -m tirx_kernels.test --kernel fp16_bf16_gemm`。

TIRx 通过 Python 源码检查来解析内核源代码，因此示例应放在文件或 notebook 单元格中，而不是 `python -c` 命令中。

## 部署

每次推送到 `main` 分支都会由 GitHub Actions（`.github/workflows/build_deploy.yaml`）自动构建并发布到 <https://mlc.ai/modern-gpu-programming-for-mlsys/>。
