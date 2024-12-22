# training-loss-patterns - 中文翻译

理解训练损失模式

训练损失图类似于心电图——有好的、坏的和你应该担心的。经过对许多训练损失轨迹的研究，人们逐渐发展出一种直觉来解释训练过程中的各种损失行为以及如何应对这些情况。

在此部分标题中的“理解”一词含义丰富，因为很多时候我们并不真正了解为什么会出现某些类型的尖峰。这里的“理解”主要指识别各种模式。然后我们通常会有一些技术来克服不良模式，并成功地完成训练。

因此，你会在这里看到一个训练损失模式的画廊，有时会有真实的解释，但更多的是基于教育性猜测来推测发生了什么。

请注意，由于这些图表来自不同来源，时间跨度也不同，因此它们看起来可能差异很大。

## 好的、坏的和意想不到的

让我们看看一些好的、坏的和不寻常的模式。

### 非常失败的训练

在开始BLOOM-176B训练之前，我们进行了多次实验，使用的是[104B模型](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide)。我们未能找出避免早期发散的方法。

![](images/pre-bloom-104B-en-fail.png)

如你所见，进行了许多尝试，应用了许多技术（参见[编年史](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide/chronicles.md)）。我们认为两个主要障碍是使用fp16和数据中有很多垃圾。对于BLOOM-176B，我们切换到bf16，使用了更干净的数据，并添加了一个嵌入层归一化，这使得所有事情都变得不同。

### 几乎完美的训练

![](images/bloom-176B-success.png)

[BLOOM-176B](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml) 训练的损失轨迹几乎完美，只有一个尖峰，在200步内恢复了。

你可以检查[TB](https://huggingface.co/bigscience/tr11-176B-logs/tensorboard)来放大并查看其他图表。

这确实是一个近乎完美的训练。付出了大量的努力才达到这个结果。

### 理解的瞬间

最近我在进行性能测试时，在8个A100节点上用 llama-2-7b 进行了从头开始训练，使用了全局批量大小为8。（使用HF Transformers [Llama](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama) 实现，配合Deepspeed ZeRO-3 DP）。

![](images/llama-7b-grokking-no-zoom.png)

可以看到，在仅仅480个样本之后，损失迅速从4下降到2.5。我的同事Gautam Mittal称之为“理解”的瞬间。在短短几步之内，模型突然能够更好地预测被掩码的标记。

通常情况下，使用更大的批量大小不会看到如此显著的改进。

如果我们放大观察，大约需要60个每次迭代8个样本的步骤：

![](images/llama-7b-grokking.png)

## 主要类型的损失尖峰

总的来说，有三种类型的损失尖峰：

1. 快速恢复的尖峰
2. 缓慢恢复的尖峰
3. 没有完全恢复的尖峰

尖峰通常是因为数据集中的某个不良区域，可能是由于数据混合不当或数据未清理干净导致的。

虽然你可能会怀疑前一批数据是触发因素，但如果你研究这批数据的内容，很可能会发现没有什么异常——问题通常是在许多步骤之前就开始发展，然后突然发生。然而，研究这批数据也可能不容易，因为当全局批量大小和序列长度都非常大时，这批数据可能相当于一本书的大小。

### 快速恢复的尖峰

损失尖峰经常发生，只要它们能快速反弹回原来的位置，训练通常会继续进行，好像什么也没发生过：

这里有一个[13B预BLOOM训练实验](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr1-13B-base)的例子：

![](images/pre-bloom-tr1-13B-glitch-1-2.png)

如你所见，有许多尖峰，有些幅度非常大，但它们都已经迅速恢复了。

### 缓慢恢复的尖峰

这里有一个从[IDEFICS-80B](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md) 训练中提取的缓慢恢复的尖峰：

![](images/idefics-80b-tr-190-01-spike-recover-2023-05-30.png)

### 没有完全恢复的尖峰

这个[104B模型尝试](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide)出现了一个尖峰，开始恢复但没有完全恢复，而是开始发散。

![](images/pre-bloom-tr8-104B-glitch-1.png)

这里还有一个来自[IDEFICS-80B](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md) 训练的例子：

![](images/idefics-80b-tr-190-01-spike-2023-05-27.png)

### 非尖峰发散

这里有一些没有通过尖峰而直接发散的例子

![](images/pre-bloom-tr8-104B-glitch-5.png)

还有一些更多的例子：

![](images/pre-bloom-tr8-104B-glitch-7-10.png)

如你所见，每次重启都会取得一些进展，然后模型就会发散。

这些都来自[104B模型尝试](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide)。

### 多个数据集的尖峰

在[IDEFICS-80B](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md) 训练期间，我们使用了两种不同类型的数据集混合在一起：

![](images/idefics-80b-tr-190-01-losses-2023-06-04.png)

图例：cm4（高），平均值（中）和pmd（低）

你可以看到，有时两个数据集的损失同时出现尖峰，而在其他时候只有其中一个数据集的损失会尖峰。

在这个模型学习两种不同的数据分布时，可以看到它报告的损失和尖峰行为在两个数据分布上都不相同。pmd数据集的损失比cm4数据集更容易处理。

## 与恢复相关的尖峰

由于硬件故障或需要回滚到较早的检查点以应对发散情况而进行的训练恢复几乎是不可避免的。如果你的训练软件不能完美地恢复，使模型感觉不到曾有过恢复，那么可能会遇到各种问题。

恢复中最复杂的挑战之一是恢复各种随机数生成器（RNG），回到以前恢复的DataLoader索引位置，以及处理复杂DataLoader的各种其他需求，这些需求特定于你的设置。

### 数据采样器相关问题

在[IDEFICS-80B](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md) 训练期间，我们使用了一个非常复杂的DataLoader，当DataLoader在恢复时出现了图像到文本比例的波动，因此每次恢复时都会出现一个小尖峰，然后恢复：

![](images/idefics-80b-tr-190-01-image2text.png)

你可以看到损失和比率图的相关性。由于我们不得不恢复大约十几次，所以看到了很多这样的尖峰。

### 重复数据的影响

我正在训练Llama2的一个变体，看到了一个非常不寻常的尖峰，它既没有发散也没有恢复，而是切换到了一个新的更高的损失水平：

![](images/ptl-repeat-data-p1.png)

我回滚到奇怪行为发生之前的状态并重新启动。损失训练在相同的损失水平上进行了一段时间，然后再次尖峰并转移到更高的损失。

![](images/ptl-repeat-data-p2.png)

我以前从未见过这种类型的发散。我思考了一段时间，然后决定从大局出发来看待这个问题。

截至本文撰写时，[Wandb](https://wandb.ai/) 如果执行了回滚操作，则无法正确绘制恢复后的数据，即它会忽略所有新的数据直到旧数据的步数被覆盖。这迫使我们在每次回滚后都开始一个新的Wandb绘图，以便显示新数据。如果你需要查看整个图表，就必须拼接它们，包括不再有效的死数据点。所以我拼接了这些数据并看到了这个拼图：

![](images/ptl-repeat-data-p3.png)

在前两次运行中实际上没有真正的尖峰。损失从未上升。在两次回滚中，由于存在完全重复的数据，损失被低估了，然后它达到了之前未见过的数据并开始正确报告。换句话说，它过度拟合并报告了虚假的损失。

问题的原因是数据重复，因为它显然记住了其中的一部分，从而报告了更好的损失。

问题是由于[pytorch-lightning](https://github.com/lightning-ai/lightning) 在处理恢复时没有正确处理DataSampler——基本上每次恢复时，你都会从头开始数据流。当然，这要求用户自己修复这种情况。你可以改变种子来一定程度上缓解这种情况，避免完全相同的数据顺序，但这仍然会让你面临重复数据的问题，这不是任何严肃训练所需要的（或消融实验，因为你假设的独立同分布数据分布观察将是无效的）。

脚注：我与PTL开发者讨论了[这个问题](https://github.com/Lightning-AI/lightning/issues/18780)，他们说他们尽力想出一个通用解决方案，但未能实现。所以用户需要自行解决。

确保检查你的训练框架文档，看它是否正确处理了DataSampler恢复。确保你在训练完成后没有发现问题，导致你最终训练了6次同样的50B令牌，而不是计划中的300B每个只出现一次的令牌。

在开始真正的训练之前，做一些早期恢复也可以暴露是否存在问题。尽管如果每次恢复时数据都被重新洗牌，你不太可能看到这个问题。只有当种子相同时才会看到。