# make-tiny-models-tokenizers-datasets - 中文翻译

使用小型模型、分词器和数据集进行更快的调试和开发

如果你在调试问题并使用全尺寸模型和分词器进行开发，那么你的工作方式可能效率不高。不仅解决问题更加困难，程序重启和到达理想状态所需的时间可能会很长——这会大大消耗你的动力和生产力，更不用说解决时间会更长，甚至无法解决。

解决方案很简单：

**除非你在测试模型的质量，否则始终使用一个随机的小型模型和潜在的小型分词器。**

此外，大型模型通常需要大量的资源，这些资源通常是昂贵的，并且也可能使调试过程变得非常复杂。例如，任何调试器都可以处理单个进程，但如果你的模型无法适应并且需要某种形式的并行化，即需要多个进程，则大多数调试器要么会中断，要么会出现问题。理想的开发环境是一个进程，而一个小模型肯定可以适应最便宜的消费级GPU。即使没有GPU，你也可以使用免费的[Google Colab](https://colab.research.google.com/)进行开发。

因此，更新后的ML开发口诀是：

- 模型越大，最终产品的生成效果越好。
- 模型越小，最终产品的训练启动得越快。

脚注：最新的研究表明，大并不总是更好，但这足以传达我的沟通的重要性。

一旦代码运行正常，切换到真实模型以测试生成质量。但在这种情况下，仍然先尝试使用最小的能产生高质量结果的模型。只有当你能看到生成内容基本正确时，才使用最大的模型来验证你的工作是否完美。

## 创建小型模型

重要提示：鉴于其流行程度和设计良好的简单API，我将讨论HF的[`transformers`](https://github.com/huggingface/transformers/)模型。但同样的原则也可以应用于其他任何模型。

简而言之：创建一个小型HF `transformers`模型非常简单：

1. 获取全尺寸模型的配置对象。
2. 缩减隐藏层大小和其他一些构成模型主体的参数。
3. 使用缩减后的配置创建模型。
4. 保存该模型。完成！

脚注：需要注意的是，这将生成一个随机模型，所以不要期望其输出有任何质量。

脚注：这些笔记是基于HF Transformers模型编写的。如果你使用的是不同的建模库，你可能需要调整其中的一些内容。

现在让我们通过实际代码将["google/mt5-small"](https://huggingface.co/google/mt5-small/tree/main)转换为其随机的小型版本。

```python
from transformers import MT5Config, MT5ForConditionalGeneration

mname_from = "google/mt5-small"
mname_very_small = "mt5-tiny-random"

config = MT5Config.from_pretrained(mname_from)

config.update(dict(
    d_model=64,
    d_ff=256,
))
print("新的配置:", config)

very_small_model = MT5ForConditionalGeneration(config)
print(f"参数数量: {very_small_model.num_parameters()}")

very_small_model.save_pretrained(mname_very_small)
```

如你所见，这非常简单。如果你不需要隐藏层大小至少为64，你可以使其更小。例如尝试8——只需确保注意力头的数量不超过隐藏层大小。

请注意，你不需要任何GPU就可以这样做，即使是对像[BLOOM-176B](https://huggingface.co/bigscience/bloom)这样的巨大参数模型（176B）也可以这样做。因为你从未加载实际的原始模型，除了它的配置对象。

在修改配置之前，你可以导出原始参数并选择缩小更多维度。例如，使用更少的层数可以使模型更小，更易于调试。所以你可以这样做：

```python
config.update(dict(
    d_model=64,
    d_ff=256,
    d_kv=8,
    num_layers=8,
    num_decoder_layers=8,
    num_heads=4,
    relative_attention_num_buckets=32,
))
```

原版["google/mt5-small"](https://huggingface.co/google/mt5-small/tree/main)模型文件大小为1.2GB。经过上述更改（以及以下部分中说明的词汇表缩减），我们将其缩小到126MB。

如果你处理的是多级嵌套配置，你需要单独更新每个子级的配置对象。例如，在[IDEFICS](https://huggingface.co/HuggingFaceM4/idefics-9b/blob/main/config.json)中，我们有1个主对象和2个嵌套对象：
```python
config
config.perceiver_config
config.vision_config
```
如果你想缩减这个模型，你需要用较小的值更新`config`和`config.vision_config`：
```python
config.update(dict(
    hidden_size=64,
    intermediate_size=37,
    num_hidden_layers=5,
    num_attention_heads=4,
    max_position_embeddings=64,
    max_sequence_length=64,

))
# 子对象需要直接更新
config.vision_config.update(dict(embed_dim=64))
```
查看[idefics-make-tiny-model.py](tiny-scripts/idefics-make-tiny-model.py)以获取一个完全工作的脚本（这里我没有添加词汇表缩减，因为我是为了演示如何更新嵌套配置对象）。

我们可以通过在保存之前将模型转换为fp16或bf16（根据目标不同）进一步减小小型模型的大小：

```python
very_small_model.half() # 转换为fp16
#very_small_model.bfloat16() # 转换为bf16
very_small_model.save_pretrained(mname_very_small)
```
这将使文件大小减小到64M。

到这里你就可以停止了，程序已经可以快速启动。

还可以采取一个额外步骤使其真正小巧。

到目前为止，我们还没有缩减词汇维度，因此64x250k（隐藏层*词汇量）仍然很大。尽管这个250k词汇量的模型不典型——通常模型的词汇量约为3万到5万，即使3万也很多，如果我们想要模型真正小巧的话。

接下来我们将探讨各种技术来缩减分词器，因为它定义了我们的词汇量大小。