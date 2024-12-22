# README - 中文翻译

# 软件调优以获得最佳性能

模型训练得越快，模型完成训练所需的时间就越短，这不仅对抢先发表研究结果很重要，还可能节省大量资金。

一般来说，最大化吞吐量主要是通过运行多个实验并测量结果，然后选择其中表现更优的参数。

在某些情况下，您的建模团队可能会要求您选择一些超参数，这些参数虽然会损害吞吐量，但从整体上有利于模型的成功。

## 术语和概念

- HFU：硬件浮点运算利用率
- MFU：模型浮点运算利用率

### MACs、FLOP、FLOPS 和 FLOP/s

本节旨在澄清常见的性能指标定义及其相互关系。

**MAC 与 FLOP**：

- 1 次 FLOP（浮点运算）可以是加法、减法、乘法或除法操作之一。

- 1 次 MAC（乘积累加）操作是一个乘法后跟一个加法，即：`a * b + c`

因此，1 次 MAC = 2 次 FLOP。现代硬件通常可以在单个时钟周期内执行 1 次 MAC。

请注意，要计算与 FLOP 相关的 MAC 数量，逆逻辑也适用，即 MAC = 0.5 FLOP——我们刚刚说 1 MAC = 2 FLOP，但这确实符合逻辑——观察：100 FLOP = 50 MAC——因为在每个 MAC 中有 2 次 FLOP。

此外，虽然 1 次 MAC = 2 次 FLOP，但反过来不一定成立。也就是说，2 次 FLOP 不一定等于 1 次 MAC。例如，如果您重复执行 `.5*.6` 100 次，那么总共会有 100 次 FLOP，在这种情况下，它将等于 100 次 MAC，因为这里只执行了 MAC 的乘法部分。

**FLOP 与 FLOPS 与 FLOP/s**

- 1 次 FLOP（浮点运算）可以是任何浮点加法、减法、乘法或除法运算。

- 1 次 FLOPS（每秒浮点运算）是在 1 秒内执行了多少次浮点运算——参见[FLOPS](https://en.wikipedia.org/wiki/FLOPS)

此外，您还会看到以下缩写：GFLOPS = 千兆浮点运算，TFLOPS = 太浮点运算，等等，因为这样更容易快速理解 150TFLOPS 而不是 150000000000000FLOPS。

当 FLOPS 用于书面表达时存在歧义——有时人们用它来表示操作的总量，而在其他时候它指的是每秒的操作次数。后者是最常见的用法，这也是本书中使用的定义。

在科学写作中，FLOP/s 经常被用来明确告诉读者这是每秒的操作次数。尽管这种方法在转换为变量名时仍然会变成 `flops`，因为非法字符被移除了。

在某些地方，您也可能看到 FLOPs，这也可能是两者中的任何一个，因为它太容易混淆大小写的 `s`。

如果定义不明确，请尝试搜索上下文，这有助于推断出所指的内容：

- 如果这是一个数学方程，并且有时间的除法，则知道它是每秒的操作次数。
- 如果讨论速度或性能，通常指的是每秒的操作次数。
- 如果讨论完成某项任务所需的计算量，则指的是总的操作次数。

### TFLOPS 作为性能指标

在开始优化训练设置的性能之前，您需要一个可以用来判断吞吐量是否有所改善的指标。您可以测量每次迭代所需的时间，或者每次迭代的次数，或者其他类似的计时方式，但有一个更有用的指标，即测量 TFLOPS。

测量 TFLOPS 更优越，因为没有它，您就不知道是否接近能达到的最佳性能。这个度量给出了您离硬件制造商报告的峰值性能有多远的信息。

在这个部分，我将以 BLOOM 的训练为例进行说明。我们使用了 80GB A100 NVIDIA GPU，并在混合 bf16 精度下进行了训练。所以让我们看看[A100 规格](https://www.nvidia.com/en-us/data-center/a100/)，它告诉我们：

```
BFLOAT16 Tensor Core 312 TFLOPS
```

因此我们现在知道，如果我们只运行大型 bf16 矩阵的 `matmul`，而不进行设备之间的复制，我们应该能够达到大约 312 TFLOPS 的最大值。

但实际上，由于磁盘 I/O、通信以及将数据从 GPU 内存传输到其计算单元的开销，以及我们不能全部使用 bf16 并且有时需要在 fp32（或 tf32）中进行计算，我们实际期望得到的数值会低得多。现实中的值因加速器而异，但在 2022 年，对于复杂的 384 GPU 训练设置，超过 50%（155 TFLOPS）的可持续吞吐量是非常出色的。

注释：在 2023 年，闪存注意力等技术的发明将这一标准提高到了 50% 以上。

当我们刚开始调优时，我们的吞吐量低于 100 TFLOPS，几周后我们在启动训练时达到了 150 TFLOPS。

需要注意的是，我们知道我们无法进一步提升很多，我们也知道没有更多的理由去继续优化。

因此，对于准备大规模模型训练的一般规则是：询问在给定的加速器上，多节点设置中指定精度下的最高 TFLOPS 是多少——并优化直到接近该值。一旦达到该值，停止优化并开始训练。

注释：对于 80GB A100 在 2022 年，那值为 155，在 2023 年已经提升到约 180 TFLOPS。

注释：计算 TFLOPS 时，重要的是记住，如果启用了梯度检查点，数学计算会有所不同，因为启用时会使用更多的计算，必须将其考虑进去。通常成本是额外的前向路径，但最近找到了更好的方法来节省一些重新计算。

对于解码器变换模型，以下是一个估计公式，略微低估了真实的 TFLOPS：

TFLOPS：`model_size_in_B * 4 * 2 * seqlen * global_batch_size / (time_in_sec_per_interation * total_gpus * 1e3)`

激活/梯度检查点时使用 4，否则为 3。对于 100B+ 模型，几乎总是启用激活检查点。

因此 `3*2` 常被称为“模型 FLOP”，`4*2` - “硬件 FLOP”，对应于 MFU 和 HFU（每秒模型和硬件 FLOPS 除以加速器的理论峰值 FLOPS）

```
perl -le '$ng=64; $ms=52; $gbs=1024; $sp=127; $seqlen=2048; print $ms*4*2*$seqlen*$gbs / ( $sp * $ng * 1e3)'
```
(ng = 总 GPU 数，ms = 模型大小（B），gbs = 全局批量大小，sp = 每次迭代的时间（秒）)

以下是使用 `bash` 环境变量的相同公式，并将其分解为 `MBS*DP*GAS`（在这种情况下，GAS 对应于 `pp_chunks`，即管道中的块数，但通常 GAS 只是指梯度累积步骤）
```
echo "($MSIZE*4*2*SEQLEN*$MICRO_BATCH_SIZE*$DP_SIZE*$GAS)/($THROUGHPUT*$NNODES*4*1000)" | bc -l
```

精确公式见《使用 Megatron-LM 在 GPU 集群上高效训练大规模语言模型》论文第 5.1 节方程 3。您可以看到代码 [在这里](https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/251)。

注释：仅用于推理时：`24Bsh^2 + 4Bs^2h` 每层的浮点运算。

#### 自动计算 FLOP

直到最近，我们还需要依靠手动计算 FLOP，如上一节所述——许多这些公式中存在错误，而且许多模型根据各种配置设置的行为不同。因此正确地获取 FLOP 计数（尤其是在多种不同模型架构之间）可能很棘手。但是别担心，PyTorch 团队开发了一种自动测量 FLOP 的方法。

```python
from torch.utils.flop_counter import FlopCounterMode

flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
with flop_counter:
    model(**input).sum().backward()
total_flops =  flop_counter.get_total_flops()
```

Voila，您的 FLOP 已经为您计算好了！

在我的代码中，我在第二次迭代时运行它（因为第一次迭代很可能有一些额外的计算是一次性的）。您不需要再次运行它，您可以缓存其值（除非由于某种原因迭代不一致）。

所以剩下的就是测量每次特定迭代所需的时间并将 FLOP 除以时间（以秒为单位）和 `1e12` 得到性能 TFLOPS。

```python
tflops = total_flops / time / 1e12
```

这将在每次迭代中给出略有不同的值。

### MFU 与 HFU

模型浮点运算利用率 (MFU) 和硬件浮点运算利用率 (HFU) 估计了在模型的 `forward` 和 `backward` 过程中硬件的利用程度（包括任何同步网络开销和可能的数据加载器 I/O）。

```
MFU = 实现的 FLOPS / 理论上的 FLOPS
HFU = 实际实现的 FLOPS / 理论上的 FLOPS
```

HFU 衡量实际 FLOPS。例如，[梯度检查点/激活重计算] (#梯度检查点) 技术会在第二次重复 `forward` 通道的一部分，因此实际上使用了更多的 FLOPS。而 MFU 忽略了实现细节，只计算了计算的理论需求，因此准确性较低。

阅读《减少大型变压器模型中的激活重计算》（[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)）这篇论文可以更好地了解这些概念。

`理论上的 FLOPS` 是您在官方加速器规格中看到的。您可以在 [这里](../../compute/accelerator#tflops-comparison-table) 找到这些值的表格。所以我们以 H100 为例。它的 BF16 理论 TFLOPS 为 989 TFLOPS。

现在，假设您测量了实际训练循环的性能，结果为 400 TFLOPS 作为实际实现的 FLOPS。那么您的 MFU 就是：
```
HFU = 400/989 = 0.40%
```

如果没有使用激活重计算功能（不重复 `forward`），则 HFU 和 MFU 将相同。如果使用了它，您的计算会导致较少的 FLOPS，从而降低 FLOPS 并使 MFU 低于 HFU。

例如，[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 发布了 A100-80GB 的以下统计数据：

| 模型大小 | 模型 FLOP 利用率 | 硬件 FLOP 利用率 |
| :---: | :---: | :---: |
| 22B   | 41.5% | 43.7% |
| 175B  | 51.4% | 52.8% |
| 530B  | 56.0% | 57.0% |
| 1T    | 56.3% | 57.0% |

如您所见，由于 Megatron-LM 在这些训练中使用了激活重计算，MFU < HFU。

更近的 H100+A100 的 MFU/HFU 数字已发布 [这里](https://github.com/mosaicml/llm-foundry/tree/main/scripts/train/benchmarking#mfu-and-hfu)。

现在，每当您看到发布的 MFU 或 HFU 数字时，您必须小心比较这些数字，直到您知道使用了相同的方法来计算 FLOPS。由于 `HFU=实际实现的 FLOPS/理论上的 FLOPS` 和 `理论上的 FLOPS` 是固定的，唯一的变量是 `实际实现的 FLOPS`，并且大多数情况下是基于参数形状估算的值，因此存在多种计算方法，有些稍微不准确，有些非常不准确。编译器也可能影响有效的 FLOPS，通过优化某些操作。而且您不知道迭代时间是如何测量的。

回想一下 `TFLOPS = FLOPS / 迭代持续时间`。因此，为了进行公平比较，有两个主要问题需要问：
1. 总使用的浮点运算是否以相同的方式计算？
2. 时间组件是否是连续测量的，包括 `DataLoader` 和日志记录，还是仅 `fwd`+`bwd` 部分。

如果有任一或两个都不匹配，则无法进行公平比较。

不幸的是，大多数情况下，论文和博客文章只是报告了 MFU 数字而没有链接到计算方法。

但是，如果您在比较自己的结果与其他竞争结果时遇到困难，请记住上述测量误差。这些误差不会提高底线吞吐量，因此只要您始终使用所选的方法来计算 TFLOPS，您将立即看到应用程序性能的改善或恶化，因为相对数值对您来说最重要。


## 如何提高速度并节省内存

您拥有的 GPU 内存越多，批处理大小（BS）就越大，GPU 在执行计算时就越高效，您完成任务的速度也就越快，因为您将能够更快地处理数据。

当然，本节对于当您即使使用 BS=1 也会出现 GPU OOM 时非常重要，并且您不想租用/购买更多硬件。

以下是有助于提高速度或节省内存的功能概述：

| 方法                   | 速度  | 内存 |
| :----------------------  | :----  | :----- |
| 梯度累积                | 是     | 是    |
| 梯度检查点              | 否*    | 是    |
| 混合精度训练            | 是     | 否    |
| 批量大小                | 是     | 是    |
| 优化器选择              | 是     | 是    |
| 数据加载器              | 是     | 否    |
| DeepSpeed Zero          | 否     | 是    |
| 闪存注意力             | 是     | 是    |

\* 梯度检查点对于给定的批处理大小会减慢速度，但由于释放了大量的内存，允许更大的 BS，实际上提高了整体速度。

### 模型操作的解剖学

变压器架构包括三组主要操作，按计算强度分组如下。

1. **张量收缩**

    线性层和多头注意力组件都执行批量矩阵乘法。这些操作是训练变压器最计算密集的部分。

2. **统计归一化**

    Softmax 和层归一化的计算强度小于张量收缩，涉及一次或多次 **降维操作**，结果通过映射应用。

3. **元素操作**

    这些是剩余的操作：**偏置、dropout、激活和残差连接**。这些是计算强度最低的操作。

当分析性能瓶颈时，这些知识可能会有所帮助。

此总结来源于 [Data Movement Is All You Need: A Case Study on Optimizing Transformers 2020](https://arxiv.org/abs/2007.00072)


### 模型内存使用的解剖学

我们已经看到，训练模型使用的内存比仅仅将模型放在 GPU 上要多得多。这是因为训练期间有许多组件使用 GPU 内存。GPU 内存中的组件如下：

1. 模型权重
2. 优化器状态
3. 梯度
4. 为梯度计算保存的前向激活
5. 临时缓冲区
6. 功能特定内存

在混合精度中使用 AdamW 训练的典型模型每模型参数需要 18 字节的内存加上激活内存和临时内存。

让我们来看看细节。

**模型权重：**

- 4 字节 * 参数数量用于 fp32 训练
- 6 字节 * 参数数量用于混合精度训练（在内存中保持模型为 fp32 和一个 fp16/bf16）

**优化器状态：**

- 8 字节 * 参数数量用于正常的 AdamW（维护两个状态）
- 4 字节 * 参数数量用于在 bf16 下运行的 AdamW。详见 [这项工作](https://github.com/huggingface/transformers/pull/21312) 使用 `AnyPrecisionAdamW`。
- 4 字节 * 参数数量用于像带有动量的 SGD 或 LION、Adafactor（以及其他）这样的优化器（仅维护一个状态）（Adafactor 使用一些附加内存而不是 4 字节）
- 2 字节 * 参数数量用于 8 位 AdamW 优化器，如 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

**梯度**

- 4 字节 * 参数数量用于 fp32 精度，以及在某些框架中使用混合半精度训练。
- 2 字节 * 参数数量用于非混合半精度，以及在某些框架中使用混合半精度训练。

**前向激活**

- 大小取决于许多因素，关键因素包括序列长度、隐藏大小和批量大小。

前向和后向函数传递和返回的输入和输出，以及为梯度计算保存的前向激活。

**临时内存**

此外，还有所有类型的临时变量，这些变量在计算完成后会被释放，但在计算过程中它们可能会占用额外的内存并可能导致 OOM。因此，在编写代码时，战略性地考虑这些临时变量，并在不再需要时显式释放它们是很重要的。

**功能特定内存**

然后您的软件可能有特殊的内存需求。例如，当使用束搜索生成文本时，软件需要维护多个输入和输出的副本。


对于 **推理**，数学计算非常相似，只是少了优化器状态和梯度。而对于模型权重，只有参数数量的一个倍数：

- 混合精度（4+2）6 字节
- fp32 4 字节
- 半精度 2 字节
- 量化 int8 精度 1 字节

另一个非常有用的资源是 [Transformer Math 101](https://blog.eleuther.ai/transformer-math/)，它带您了解内存需求和其他要求。

[EAI cookbook](https://github.com/EleutherAI/cookbook) 包含一组 [计算脚本](https://github.com/EleutherAI/cookbook/tree/main/calc)，可以根据您的配置和设置输出给定训练或推理计算运行的理论内存开销。

还有一个非常方便的 [GPU VRAM 估算器](https://vram.asmirnov.xyz/) 来自 Alexander Smirnov，以及 [如何工作的笔记](https://asmirnov.xyz/vram)。