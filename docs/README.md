# OpenPI 仓库代码导读

本文面向第一次阅读 OpenPI 源码的读者，说明仓库各部分的职责，并给出一条从推理入门、逐步深入模型和训练代码的阅读路线。这里介绍的是当前仓库的代码组织；模型和依赖支持情况以根目录 [`README.md`](../README.md) 及各示例文档为准。

## 这个仓库做什么

OpenPI 是机器人视觉语言动作（VLA）模型及其训练、推理代码。模型接收相机图像、机器人状态和语言指令，输出一段机器人动作（action chunk）。仓库主要包含 π₀、π₀-FAST、π₀.₅ 的实现，以及若干机器人平台的数据适配和运行示例。

从一次推理看，主要数据流是：

```text
机器人观测与指令
  → 平台输入适配与数据变换
  → Policy 整理输入并调用模型
  → 模型预测一段动作
  → 输出变换还原平台动作格式
  → 本地客户端或 WebSocket 服务把动作交回机器人
```

## 目录与主要职责

| 路径 | 主要功能 | 阅读入口 |
| --- | --- | --- |
| `src/openpi/models/` | JAX 版本的模型结构、视觉与语言骨干、动作预测头、tokenizer 和模型配置。 | [`pi0.py`](../src/openpi/models/pi0.py)、[`pi0_fast.py`](../src/openpi/models/pi0_fast.py)、[`model.py`](../src/openpi/models/model.py) |
| `src/openpi/models_pytorch/` | PyTorch 版本的 π₀ / π₀.₅ 模型、预处理代码，以及为项目适配 Transformers 的替换实现。 | [`pi0_pytorch.py`](../src/openpi/models_pytorch/pi0_pytorch.py)、[`preprocessing_pytorch.py`](../src/openpi/models_pytorch/preprocessing_pytorch.py) |
| `src/openpi/policies/` | 策略抽象、模型与检查点装配、平台观测和动作格式适配。 | [`policy.py`](../src/openpi/policies/policy.py)、[`policy_config.py`](../src/openpi/policies/policy_config.py) |
| `src/openpi/training/` | 训练配置、数据读取、优化器、设备分片、检查点保存恢复和权重加载。 | [`config.py`](../src/openpi/training/config.py)、[`data_loader.py`](../src/openpi/training/data_loader.py) |
| `src/openpi/training/misc/` | 特定实验或基准的训练配置，例如 RoboArena 相关配置。 | `roboarena_config.py` |
| `src/openpi/shared/` | 多处共用的图像处理、归一化、模型下载、数组类型和 JAX 工具。 | [`normalize.py`](../src/openpi/shared/normalize.py)、[`image_tools.py`](../src/openpi/shared/image_tools.py) |
| `src/openpi/transforms.py` | 训练和推理共用的数据变换：重排字段、注入指令、tokenize、归一化及反归一化。 | [`transforms.py`](../src/openpi/transforms.py) |
| `src/openpi/serving/` | WebSocket 策略服务端；加载好的策略在这里接收请求并返回动作。 | [`websocket_policy_server.py`](../src/openpi/serving/websocket_policy_server.py) |
| `packages/openpi-client/` | 可单独安装的客户端包，包括 WebSocket 客户端、消息序列化、图像工具和运行时接口。 | [`websocket_client_policy.py`](../packages/openpi-client/src/openpi_client/websocket_client_policy.py) |
| `scripts/` | 命令行入口，包括训练、推理服务启动和数据归一化统计计算。 | [`serve_policy.py`](../scripts/serve_policy.py)、[`train.py`](../scripts/train.py)、[`compute_norm_stats.py`](../scripts/compute_norm_stats.py) |
| `examples/` | 特定机器人、仿真环境和数据集的运行/训练示例，包括 ALOHA、DROID、LIBERO、UR5 和简单客户端。 | [`examples/droid/README.md`](../examples/droid/README.md)、[`examples/libero/README.md`](../examples/libero/README.md) |
| `third_party/` | 仓库集成的第三方代码或子模块，例如 ALOHA 相关代码。 | 阅读具体示例需要时再看 |
| `docs/` | 安装、Docker、远程推理、归一化统计等主题文档；本文也在此目录。 | [`remote_inference.md`](remote_inference.md)、[`norm_stats.md`](norm_stats.md) |

仓库根目录的 `pyproject.toml` 定义 Python 包、依赖、开发工具和工作区；`uv.lock` 锁定依赖版本。各目录中的 `*_test.py` 是对应模块的测试文件，可用来补充理解输入输出和边界行为。

## 推荐阅读顺序

### 第一阶段：先看懂一个推理请求

建议选择 DROID 或 LIBERO 作为贯穿例子。若只想读代码、暂时不运行模型，不需要先下载检查点或连接机器人。

1. **读示例说明**：从 [`examples/droid/README.md`](../examples/droid/README.md) 了解示例所需的机器人、相机和策略服务结构；或读 [`examples/libero/README.md`](../examples/libero/README.md) 了解仿真评估流程。
2. **读平台适配**：看 [`droid_policy.py`](../src/openpi/policies/droid_policy.py) 或 [`libero_policy.py`](../src/openpi/policies/libero_policy.py)，记录原始观测字段如何映射成模型字段，以及模型动作如何转成平台动作。
3. **读策略执行**：看 [`policy.py`](../src/openpi/policies/policy.py)，跟踪 `infer()` 中输入变换、模型调用和输出变换。
4. **读策略装配**：看 [`policy_config.py`](../src/openpi/policies/policy_config.py)，了解训练配置、模型权重、归一化统计和 transforms 如何组合成可调用的策略。
5. **读服务入口和通信**：从 [`serve_policy.py`](../scripts/serve_policy.py) 跟到 [`websocket_policy_server.py`](../src/openpi/serving/websocket_policy_server.py)，再看客户端的 [`websocket_client_policy.py`](../packages/openpi-client/src/openpi_client/websocket_client_policy.py)。这样能补齐“机器人如何把观测发给模型、又如何收到动作”的链路。

读完这一阶段，应能回答：输入有哪些字段？谁负责图像/状态预处理？模型返回什么？动作如何变成机器人能执行的格式？

### 第二阶段：理解模型结构

OpenPI 有 JAX 和 PyTorch 两套模型实现。它们共享策略输入输出约定，但模型代码和推理实现不同。建议先选一套读通，再按需要对照另一套。

#### JAX 部分

1. 先读 [`model.py`](../src/openpi/models/model.py)：理解通用的 Observation、Actions、模型接口和公共预处理。
2. 再读 [`pi0.py`](../src/openpi/models/pi0.py)：关注图像与文本如何编码、状态和动作如何进入网络，以及动作预测的主要路径。
3. 根据兴趣选择 [`pi0_fast.py`](../src/openpi/models/pi0_fast.py)（自回归 / FAST 路线），并读 [`pi0_config.py`](../src/openpi/models/pi0_config.py) 了解模型变体与结构配置。
4. 遇到视觉或骨干网络细节时，再深入 `gemma.py`、`gemma_fast.py`、`siglip.py`、`vit.py` 和 `tokenizer.py`。

#### PyTorch 部分

1. 先读 [`model.py`](../src/openpi/models/model.py)：了解两套实现共享的 Observation、Actions 和 `BaseModelConfig`，重点看 `load_pytorch()` 如何加载 PyTorch 模型。
2. 读 [`preprocessing_pytorch.py`](../src/openpi/models_pytorch/preprocessing_pytorch.py)：了解 PyTorch 模型的图像预处理。
3. 读 [`gemma_pytorch.py`](../src/openpi/models_pytorch/gemma_pytorch.py)：了解 PyTorch 模型使用的视觉语言骨干组件。
4. 再读 [`pi0_pytorch.py`](../src/openpi/models_pytorch/pi0_pytorch.py)：跟踪完整模型的输入处理、前向计算和 `sample_actions()` 动作生成过程。
5. 回看 [`policy.py`](../src/openpi/policies/policy.py)：关注 PyTorch 分支如何把 NumPy 输入转成 Torch 张量、调用模型并将结果转回 NumPy。
6. 最后串起 [`policy_config.py`](../src/openpi/policies/policy_config.py)：了解策略如何检测 PyTorch 检查点、加载模型并组装推理流程。

当前仓库的 PyTorch 实现支持 π₀ 和 π₀.₅，不支持 π₀-FAST。`models_pytorch/transformers_replace/` 中是项目对 Transformers 的适配代码，建议了解 PyTorch 主流程后再读。

阅读模型代码时先追踪张量的语义和形状，再深入注意力层、采样过程等数学细节。可以把问题记成：每个输入张量代表什么？在哪里合并？模型输出的动作维度和时间长度由什么配置决定？

### 第三阶段：理解数据和训练

1. 先看 [`config.py`](../src/openpi/training/config.py)：训练配置把模型、数据、权重初始化、超参数和输出目录组合起来，也能找到各任务的配置定义。
2. 再看 [`data_loader.py`](../src/openpi/training/data_loader.py)：理解数据集样本怎样被读取、变换、批处理并送到模型。
3. 对照 [`transforms.py`](../src/openpi/transforms.py) 和平台策略适配，确认训练数据与推理输入使用一致的数据约定。
4. 看 [`scripts/compute_norm_stats.py`](../scripts/compute_norm_stats.py) 和 [`normalize.py`](../src/openpi/shared/normalize.py)，理解动作/状态归一化统计如何计算和应用。
5. 看 [`scripts/train.py`](../scripts/train.py)，再根据需要追踪 `optimizer.py`、`utils.py`、`sharding.py`、`weight_loaders.py` 和 `checkpoints.py`。
6. 读 DROID 全量数据训练时，继续看 [`droid_rlds_dataset.py`](../src/openpi/training/droid_rlds_dataset.py) 以及 [`examples/droid/README_train.md`](../examples/droid/README_train.md)；小型自定义数据则可从各示例的数据转换脚本入手。

### 第四阶段：按兴趣扩展

- **要接入新机器人**：从最接近的平台策略文件开始，定义输入/输出 transforms，再补充训练数据配置。
- **要部署远程推理**：读 [`remote_inference.md`](remote_inference.md)、服务端、客户端和具体平台 `main.py`。
- **要改数据处理**：读数据转换脚本、`data_loader.py` 和 `transforms.py`，再看相应配置。
- **要研究训练效率**：深入优化器、分片、数据加载和检查点代码。
- **要确认接口行为**：阅读对应模块旁边的 `*_test.py`。

## 建议的阅读方法

不要一开始试图读完所有模型和训练实现。每读一个模块，先写下三件事：

1. 它接收什么输入？
2. 它产生什么输出？
3. 它对数据做了什么转换，下一步由谁接手？

沿着函数调用逐步跳转，优先读入口函数和被调用的核心函数；遇到不影响当前链路的底层细节时，先记下文件位置，后续再回看。这样可以先建立完整的数据流，再按需要深入模型数学、设备并行或数据工程。

## 常用命令入口

在仓库根目录中，可以通过 `uv run` 使用项目虚拟环境运行脚本：

```bash
# 查看可用训练配置（具体参数以脚本帮助为准）
uv run scripts/train.py --help

# 启动检查点策略服务的命令形式
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=/path/to/checkpoint

# 查看单个模块的测试文件，了解预期行为
rg --files src/openpi | rg '(_test\\.py|test_)'
```

真实模型推理通常还需要匹配的检查点、依赖和 GPU 环境；只做源码阅读时，可以先从上述入口和类型/数据变换开始，无需运行完整服务。
