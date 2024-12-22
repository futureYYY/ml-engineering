# README - 中文翻译

## 资源

### 有用的汇编

- @StellaAthena 创建了 [Common LLM 设置电子表格](https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit#gid=0)，当你即将开始新的LLM训练时，它会是一个非常有用的资源——因为它告诉你已知的LLM训练数量。

- 几年前我开始收集有关模型是在哪种数据类型下训练的信息：[模型预训练精度数据库（FP16、FP32、BF16）](https://discuss.huggingface.co/t/model-pre-training-precision-database-fp16-fp32-bf16/5671)——它只包含少数模型，但如果你正在研究数据类型，它仍然很有用。我曾利用这些信息尝试编写一个 [模型预训练数据类型自动检测工具](https://github.com/stas00/ml-ways/blob/master/numbers/detect-model-pretrained-in-bf16-fp16-fp32.ipynb)，并有一个相关的 [float16与bfloat16数值特性比较](https://github.com/stas00/ml-ways/blob/master/numbers/bfloat16-vs-float16-study.ipynb)。

### 公开的训练LLM/VLM日志

LLM/VLM训练的日志和编年史是学习处理训练不稳定性和选择良好超参数的最佳来源之一。

如果您知道列表中没有的公开LLM/VLM训练日志，请告知我或通过PR添加。谢谢！

按年份分组，顺序无关紧要。

#### 2021

- BigScience 预BLOOM 108B 训练实验（2021）：
[编年史](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide/chronicles.md) |
[完整的规格和讨论](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide)
（备份：
[1](https://github.com/stas00/bigscience-backup/blob/master/train/tr8-104B-wide/chronicles.md) |
[2](https://github.com/stas00/bigscience-backup/blob/master/train/tr8-104B-wide))

#### 2022

- BigScience BLOOM-176B（2022）：
[前传编年史](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles-prequel.md) |
[编年史](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md) |
[完整的规格和讨论](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/)
（备份：
[1](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles-prequel.md) |
[2](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles.md) |
[3](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/))

- Meta OPT-175B（2022）：
[日志簿](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles) | [视频](https://www.youtube.com/watch?v=p9IxoSkvZ-M) （备份：[1](https://github.com/stas00/metaseq-backup/tree/main/projects/OPT/chronicles)）

- THUDM GLM-130B（2022）：[英文日志簿](https://github.com/THUDM/GLM-130B/blob/main/logs/main-log-en.md) | [中文版本](https://github.com/THUDM/GLM-130B/blob/main/logs/main-log.md) （备份：[1](https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log-en.md) | [2](https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log.md)）

#### 2023

- HuggingFace IDEFICS-80B 多模态（Flamingo 复现）（2023）：[学习日志](https://github.com/huggingface/m4-logs/blob/master/memos/README.md) | [训练编年史](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md) （备份：[1](https://github.com/stas00/m4-logs-backup/blob/master/memos/README.md) | [2](https://github.com/stas00/m4-logs-backup/blob/master/tr-190-80b/chronicles.md)）

- BloombergGPT 50B LLM - [《BloombergGPT：用于金融的大规模语言模型》](https://arxiv.org/abs/2303.17564) 的C部分

#### 2024

- [MegaScale：将大规模语言模型训练扩展到超过10,000个GPU](https://arxiv.org/abs/2402.15627) - 该论文涵盖了各种训练问题及其解决方案——尽管模型是专有的，但仍具有教学和实用价值。

- Imbue 的 [从裸金属到70B模型：基础设施设置和脚本](https://imbue.com/research/70b-infrastructure/) 是一篇非常详细的技术文章，涵盖了他们在训练一个专有70B参数模型时遇到的各种训练相关问题。

### 硬件设置日志

- Imbue 发布了一个详细的日志，记录了他们如何搭建一个512节点的IB-fat-tree集群并使其运行：[从裸金属到70B模型：基础设施设置和脚本](https://imbue.com/research/70b-infrastructure/)，他们还在过程中开源了创建的 [集群工具](https://github.com/imbue-ai/cluster-health)。