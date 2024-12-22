# README - 中文翻译

## 模型并行

## 并行概述

在现代机器学习中，各种并行方法用于：

1. 克服GPU内存限制。例如：
   - 训练非常大的模型——例如，t5-11b的模型参数就有45GB
   - 训练非常长的序列——例如，
2. 显著加快训练速度——将原本需要一年才能完成的训练缩短到几小时

我们将首先详细讨论各种一维并行技术及其优缺点，然后看看如何将它们组合成二维和三维并行，以实现更快的训练并支持更大的模型。还将介绍一些其他强大的替代方法。

虽然主要概念可能适用于任何其他框架，但本文重点放在基于PyTorch的实现上。

主要有两种方法可以实现比加速器内存更大的模型的训练和推理：
1. 三维并行（3D）——非常网络高效，但可能对建模代码有较大侵入性，并且需要大量工作才能正确运行
2. 零冗余优化器（ZeRO）——不太网络高效，但对建模代码几乎不需要更改并且很容易实现。

## 可扩展性概念

以下是本文稍后将详细介绍的主要概念的简要描述。

1. [数据并行](#数据并行)（DP）——相同的设置被多次复制，每个设置处理一部分数据。处理是并行进行的，并且在每个训练步骤结束时所有设置同步。
2. [张量并行](#张量并行)（TP）——每个张量被分割成多个块，因此不是整个张量都驻留在一个GPU上，而是每个张量碎片都在其指定的GPU上。在处理过程中，每个碎片分别在不同的GPU上并行处理，结果在步骤结束时同步。这可以称为水平并行，因为分割发生在水平层面上。
3. [管道并行](#管道并行)（PP）——模型在多个GPU之间垂直（层级别）拆分，这样只有模型的一个或几个层被放置在一个GPU上。每个GPU并行处理管道的不同阶段，并处理批处理的一小部分。
4. [零冗余优化器](#零数据并行)（ZeRO）——也执行类似于TP的张量分片，除了整个张量会在前向或反向计算时重新构造，因此不需要修改模型。它还支持各种卸载技术来补偿有限的GPU内存。分片DDP是ZeRO的基础概念的另一个名称，被各种ZeRO实现使用。
5. [序列并行](#序列并行)——训练长输入序列需要大量的GPU内存。这种技术将单个序列的处理分布在多个GPU上。
6. [专家并行](#专家并行)——混合专家（MoE）可以分区，使得每个专家都有一个专用的GPU（或多个）。

本文档的引言部分可能是我找到的最佳解释常见并行技术的文章之一：[广度优先管道并行](https://arxiv.org/abs/2211.05953)。

## 数据并行

### DDP

大多数用户只需两个GPU就可以通过`DataParallel` (DP) 和 `DistributedDataParallel` (DDP) 享受增加的训练速度，这几乎是不费吹灰之力。这是Pytorch内置的功能。

详情请参阅[DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

### ZeRO 数据并行

ZeRO驱动的数据并行（ZeRO-DP）如以下来自这篇[博客文章](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)的图所示：
![DeepSpeed-Image-1](images/parallelism-zero.png)

这可能很难理解，但实际上这个概念相当简单。这只是普通的`DataParallel` (DP)，不同之处在于，每个GPU只存储模型参数、梯度和优化器状态的一部分。然后在运行时，当给定层只需要完整的层参数时，所有GPU会同步，将缺失的部分传递给彼此——就是这样。

考虑一个具有3层的简单模型，每层有3个参数：
```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```
层La的权重为a0, a1和a2。

如果我们有3个GPU，分片DDP（= Zero-DP）将模型按如下方式划分到3个GPU上：

```
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

在某种意义上，这与张量并行类似，如果你想象典型的DNN图。垂直切片是指将整个层组放置在不同的GPU上。但这只是开始。

现在，这些GPU中的每一个都会像DP一样得到一个常规的小批量：
```
x0 => GPU0
x1 => GPU1
x2 => GPU2
```

输入未作修改——它们认为自己会被普通模型处理。

首先，输入进入层La。

让我们只关注GPU0：x0需要a0, a1, a2参数来进行前向路径，但GPU0只有a0——它从GPU1获得a1，从GPU2获得a2，将模型的所有部分组合在一起。

同时，GPU1获取小批量x1，它只有a1，但需要a0和a2参数，所以它从GPU0和GPU2获得这些参数。

同样的事情发生在GPU2上，它接收输入x2。它从GPU0和GPU1获得a0和a1，并用它的a2重构完整的张量。

所有3个GPU都重建了完整的张量并进行了前向计算。

一旦计算完成，不再需要的数据就会被丢弃——它只在计算过程中使用。通过预取可以高效地重构。

整个过程再重复一遍，先进行层Lb的前向计算，然后再进行Lc的前向计算，最后反向计算Lc -> Lb -> La。

对我来说，这听起来像是一个高效的团体背包重量分配策略：

1. 人A携带帐篷
2. 人B携带炉子
3. 人C携带斧头

现在每天晚上他们都分享他们拥有的东西，并从其他人那里获取他们没有的东西，第二天早上他们打包他们分配的装备继续前进。这就是分片DDP / Zero DP。

将其与简单的策略相比，每个人必须携带自己的帐篷、炉子和斧头，这将效率低得多。这是Pytorch中的DataParallel (DP和DDP)。

在阅读有关这一主题的文献时，您可能会遇到以下同义词：分片、分区。

如果您仔细观察ZeRO如何分区模型权重——它看起来与稍后讨论的张量并行非常相似。这是因为它分片/分区每一层的权重，而不同于接下来讨论的垂直模型并行。

ZeRO-DP阶段1+2+3的实现：
- [DeepSpeed](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch](https://pytorch.org/docs/stable/fsdp.html)（最初在[FairScale](https://github.com/facebookresearch/fairscale/)中实现，后来被上游集成到PyTorch核心）
- [torchtitan](https://github.com/pytorch/torchtitan)

Deepspeed ZeRO集成：
- [HF Trainer集成](https://huggingface.co/docs/transformers/main_classes/deepspeed)
- [Accelerate](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html)
- [Determined.AI](https://docs.determined.ai/latest/model-dev-guide/api-guides/apis-howto/deepspeed/_index.html)

FSDP集成：
- [HF Trainer集成](https://huggingface.co/docs/transformers/main/en/fsdp)
- [Accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)
- [torchtitan](https://github.com/pytorch/torchtitan)

重要论文：

Deepspeed和ZeRO总体：
- [ZeRO：优化内存以训练万亿参数模型](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload：使十亿规模模型训练民主化](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity：突破GPU内存墙以进行极端规模深度学习](https://arxiv.org/abs/2104.07857)
- [ZeRO++：巨型模型训练的极高效集体通信](https://arxiv.org/abs/2306.10209)
- [DeepSpeed Ulysses：系统优化以支持极端长序列Transformer模型的训练](https://arxiv.org/abs/2309.14509)
- [AMSP：减少ZeRO的通信开销以提高LLM训练效率](https://arxiv.org/abs/2311.00257)

PyTorch：
- [PyTorch FSDP：完全分片数据并行扩展的经验](https://arxiv.org/abs/2304.11277)

主要的DeepSpeed ZeRO资源：
- [项目的GitHub](https://github.com/microsoft/deepspeed)
- [使用文档](https://www.deepspeed.ai/getting-started/)
- [API文档](https://deepspeed.readthedocs.io/en/latest/index.html)
- [博客文章](https://www.microsoft.com/en-us/research/search/?q=deepspeed)

#### 克服巨大的全局批量大小问题

如果您使用1024个加速器，那么每个加速器上的分片将很小，而微批次大小（MBS）将有很多可用内存，假设您可以容纳MBS=32，最终得到GBS=32k——这很可能不是您想要的。

因此，您要么需要部署[张量并行](#张量并行)，这实现起来并不容易，或者通常更简单的是部署[序列并行](#序列并行)。我还没有实际尝试过，但据我所知，对于：

- 使用Deepspeed ZeRO时，使用[Deepspeed-Ulysses](#deepspeed-ulysses-sp)
- 使用FSDP时，使用[Paged Ring Attention](https://github.com/lucidrains/ring-attention-pytorch)（[论文](https://arxiv.org/abs/2402.08268)）

请注意，这可能不如[张量并行](#张量并行)高效——但到目前为止我还不知道实际的额外开销。

#### ZeRO 多副本

默认情况下，ZeRO 使用所有GPU来创建一个单一的模型副本——即模型分布在所有GPU上。这导致了各种限制，例如：

1. 全局批量大小不灵活——它总是总GPU数乘以微批次大小的函数——这在大型集群中可能导致一个巨大的全局批量大小，这可能不利于有效的收敛。诚然，可以通过使用很小的微批次大小来控制全局批量大小，但这会导致每个GPU上的矩阵变小，从而计算效率降低。
2. 由于较慢的节点间网络定义了整体通信速度，因此无法充分利用更快的节点内网络。

[ZeRO++]解决了第二个限制，通过引入ZeRO的层次化权重分区（hpZ）。在这种方法中，不是将整个模型权重分布在所有GPU上，而是每个模型副本仅限于一个节点。这将增加总节点数的内存使用，但现在将两次`all_gather`调用用于收集分片组件的操作是在更快的节点内连接上进行的。仅将`reduce_scatter`操作用于聚合和重新分布梯度是在较慢的节点间网络上进行的。

第一个限制并没有真正解决，因为总的全局批量大小保持不变，但由于每个副本更高效，并且额外的内存压力可能限制每个GPU上的可能微批次大小，这应该会改善系统的吞吐量。

PyTorch FSDP 在[shardingStrategy.HYBRID_SHARD](https://pytorch.org/docs/stable/fsdp.html) 中实现了此功能。

论文：

- [ZeRO++：巨型模型训练的极高效集体通信](https://arxiv.org/abs/2306.10209)
- [PyTorch FSDP：完全分片数据并行扩展的经验](https://arxiv.org/abs/2304.11277)


#### ZeRO 的变体

提出修改ZeRO协议的已发表论文：

- [MiCS：在公共云上训练巨型模型的近线性扩展](https://arxiv.org/abs/2205.00119)（2022年）
- [AMSP：通过高级模型状态分区超级扩展LLM训练](https://arxiv.org/abs/2311.00257)（2023年）