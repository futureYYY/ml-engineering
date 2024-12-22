# dtype - 中文翻译

# 张量精度 / 数据类型

以下是在撰写本文时在机器学习中常见的数据类型（通常称为 `dtype`）：

浮点格式：
- fp32 - 32位
- tf32 - 19位（NVIDIA Ampere+）
- fp16 - 16位
- bf16 - 16位
- fp8 - 8位（E4M3和E5M2格式）

为了进行视觉比较，请参考以下表示：

![fp32-tf32-fp16-bf16](images/fp32-tf32-fp16-bf16.png)

([来源](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/))

![fp16-bf16-fp8](images/fp16-bf16-fp8.png)

([来源](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html))

用于量化中的整数格式：

- int8 - 8位
- int4 - 4位
- int1 - 1位

## 机器学习数据类型进展

最初，机器学习使用的是fp32，但它的速度非常慢。

接下来，发明了混合精度技术，使用fp16和fp32的组合，这大大加快了训练速度。

![fp32/fp16混合精度](images/mixed-precision-fp16.png)

([来源](https://developer.nvidia.com/blog/video-mixed-precision-techniques-tensor-cores-deep-learning/))

但是fp16被证明不够稳定，并且训练大型语言模型非常困难。

幸运的是，bf16出现了，并用相同的混合精度协议取代了fp16。这使得大型语言模型的训练更加稳定。

随后，fp8出现了，并且混合精度切换到了该格式，这使得训练更快。参见论文：[深度学习中的FP8格式](https://arxiv.org/abs/2209.05433)。

为了欣赏不同格式之间的加速效果，请查看以下NVIDIA A100 TFLOPS规格表（无稀疏性）：

| 数据类型              | TFLOPS |
| :---                  |    --: |
| FP32                  |   19.5 |
| Tensor Float 32 (TF32) |    156 |
| BFLOAT16 Tensor Core  |    312 |
| FP16 Tensor Core      |    312 |
| FP8 Tensor Core       |    624 |
| INT8 Tensor Core      |    624 |

每种后续的数据类型比前一种快约2倍（除了fp32，它比其他类型慢得多）。

在混合训练制度的同时，机器学习社区开始提出各种量化方法。可能最好的例子是Tim Dettmers的[bitsandbytes](https://github.com/TimDettmers/bitsandbytes)，它提供了许多4位和8位量化解决方案。Deepspeed团队也有一些有趣的量化解决方案。

## TF32

TF32是一种神奇的数据类型，自Ampere以来在NVIDIA GPU上可用，它允许在比普通fp32 `matmul` 快得多的速度下执行fp32 `matmul`，并且只有很小的精度损失。

以下是A100 TFLOPS示例（无稀疏性）：

| 数据类型              | TFLOPS |
| :---                  |    --: |
| FP32                  |   19.5 |
| Tensor Float 32 (TF32) |    156 |

如您所见，TF32比FP32快8倍！

默认情况下它是禁用的。要在程序开始时启用它，请添加以下代码：

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

有关实际精度损失的更多信息，请参阅[此处](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)。

## 在低精度数据类型中使用fp32累加器

无论何时使用低精度数据类型，都需要小心不要在该数据类型中累积中间结果。

像`LayerNorm`这样的操作不能在半精度中进行，否则可能会丢失大量数据。因此，当这些操作正确实现时，它们会在输入数据类型的dtype中高效地进行内部计算，但使用fp32累加寄存器，然后将其输出向下转换为输入的精度。

通常只是累加部分在fp32中进行，因为否则对多个低精度数字进行累加会非常有损。

以下是一些示例：

1. 减少收集操作

* fp16：如果存在损失缩放，则可以在fp16中进行

* bf16：仅可在fp32中进行

2. 梯度累积

* 对于fp16和bf16，最好在fp32中进行，但对于bf16来说必须这样做

3. 优化器步骤 / 消失的梯度

* 当向一个大数字添加一个小梯度时，这种加法通常会被抵消，因此通常使用fp32主权重和fp32优化状态。

* 当使用[Kahan求和算法](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)或[随机舍入](https://en.wikipedia.org/wiki/Rounding)（在[重新审视BFloat16训练](https://arxiv.org/abs/2010.06192)中引入）时，可以使用f16主权重和优化状态。

对于后者的一个示例，请参阅：[AnyPrecision优化器](https://github.com/pytorch/torchdistx/pull/52)，最新版本可以在[这里](https://github.com/facebookresearch/multimodal/blob/6bf3779a064dc72cde48793521a5be151695fc62/torchmultimodal/modules/optimizers/anyprecision.py#L17)找到。

## 训练后改变精度

有时在模型训练之后改变精度是可以的。

- 使用bf16预训练模型在fp16模式下通常会失败——由于fp16可表示的最大数字是64k。关于深入讨论和可能的解决方法，请参阅这个[PR](https://github.com/huggingface/transformers/pull/10956)。

- 使用fp16预训练模型在bf16模式下通常可以工作——虽然在转换时会失去一些性能，但应该可以工作——最好在使用之前进行微调。