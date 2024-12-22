# underflow_overflow - 中文翻译

## 检测下溢和溢出

在本节中，我们将使用 `[underflow_overflow](./underflow_overflow.py)` 库。

如果你开始遇到 `loss=NaN` 或者模型由于激活值或权重中的 `inf` 或 `nan` 而表现出某些异常行为，那么需要发现第一个下溢或溢出发生的位置及其原因。幸运的是，你可以通过激活一个特殊模块来轻松完成这项工作，该模块会自动进行检测。

我们以 `t5-large` 模型为例进行演示。

```python
from .underflow_overflow import DebugUnderflowOverflow
from transformers import AutoModel

model = AutoModel.from_pretrained("t5-large")
debug_overflow = DebugUnderflowOverflow(model)
```

[`underflow_overflow.DebugUnderflowOverflow`] 在模型中插入钩子，这些钩子会在每次前向调用后立即测试输入和输出变量以及相应的模块的权重。一旦在激活值或权重中的至少一个元素中检测到 `inf` 或 `nan`，程序将断言并打印如下报告（这是在 `fp16` 混合精度下使用 `google/mt5-small` 捕获的）：

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
                  encoder.block.1.layer.1.DenseReluDense.dropout Dropout
0.00e+00 2.57e+02 input[0]
0.00e+00 2.85e+02 output
[...]
                  encoder.block.2.layer.0 T5LayerSelfAttention
6.78e-04 3.15e+03 input[0]
2.65e-04 3.42e+03 output[0]
             None output[1]
2.25e-01 1.00e+04 output[2]
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.dropout Dropout
0.00e+00 8.76e+03 input[0]
0.00e+00 9.74e+03 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

示例输出为了简洁起见进行了裁剪。

第二列显示了绝对最大值，因此如果仔细查看最后几帧，输入和输出的范围为 `1e4`。所以在这种 `fp16` 混合精度下，训练的最后一步发生了溢出（因为在 `fp16` 下，最大的数是 `64e3`）。为了避免 `fp16` 下的溢出，激活值必须远低于 `1e4`，因为 `1e4 * 1e4 = 1e8`，所以任何具有大激活值的矩阵乘法都会导致数值溢出条件。

在跟踪的开始部分，你可以发现问题发生在哪个批次号上（这里 `Detected inf/nan during batch_number=0` 表示问题发生在第一个批次）。

每个报告帧以声明对应模块的完全限定名称开头。如果我们只看这个帧：

```
                  encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```

这里，`encoder.block.2.layer.1.layer_norm` 表示它是一个编码器第二块第一层的层归一化。具体的 `forward` 调用是 `T5LayerNorm`。

让我们看看报告的最后一部分：

```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
[...]
                  encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+01 output
                  encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+01 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+02 output
                  encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
                  encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```

最后一帧报告了 `Dropout.forward` 函数的第一个输入和唯一的输出。可以看出，它是从 `DenseReluDense` 类中的 `dropout` 属性调用的。我们可以看到它发生在第二个块的第一层，在第一个批次。最后，绝对最大的输入元素为 `6.27e+04`，输出也是 `inf`。

在这里，`T5DenseGatedGeluDense.forward` 导致输出激活值的绝对最大值约为 62.7K，这接近 `fp16` 的上限 64K。在下一帧中，`Dropout` 在将一些元素置零后重新规范化权重，使绝对最大值超过 64K，从而导致溢出 (`inf`)。

如你所见，当数值开始变得非常大时，我们需要查看前几帧。

让我们将报告与 `models/t5/modeling_t5.py` 中的代码进行匹配：

```python
class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
```

现在很容易看到 `dropout` 调用以及所有之前的调用。

由于检测是在前向钩子中发生的，这些报告会在每次 `forward` 返回后立即打印出来。

回到完整的报告，要采取行动并解决问题，我们需要回溯几帧，找到数值开始增大的地方，并很可能在此切换到 `fp32` 模式，以便在乘法或加法时不会溢出。当然，也可能有其他解决方案。例如，如果启用了 `amp`，我们可以临时关闭它，方法是将原始的 `forward` 放入一个辅助包装函数中，如下所示：

```python
import torch

def _forward(self, hidden_states):
    hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
    hidden_linear = self.wi_1(hidden_states)
    hidden_states = hidden_gelu * hidden_linear
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.wo(hidden_states)
    return hidden_states

def forward(self, hidden_states):
    if torch.is_autocast_enabled():
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward(hidden_states)
    else:
        return self._forward(hidden_states)
```

由于自动检测器仅报告完整帧的输入和输出，一旦你知道在哪里查找，你可能还想分析特定 `forward` 函数的中间阶段。在这种情况下，你可以使用 `detect_overflow` 辅助函数将检测器注入你想要的位置，例如：

```python
from underflow_overflow import detect_overflow


class T5LayerFF(nn.Module):
    [...]

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        detect_overflow(forwarded_states, "after layer_norm")
        forwarded_states = self.DenseReluDense(forwarded_states)
        detect_overflow(forwarded_states, "after DenseReluDense")
        return hidden_states + self.dropout(forwarded_states)
```

你可以看到我们添加了两个这样的检测点，现在我们跟踪 `forwarded_states` 是否在某处检测到了 `inf` 或 `nan`。

实际上，检测器已经报告了这些内容，因为上面示例中的每个调用都是一个 `nn.Module`，但如果有一些局部直接计算，这就是如何做到的。

此外，如果你在自己的代码中实例化调试器，可以调整打印的最大帧数，例如：

```python
from .underflow_overflow import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```

## 特定批次绝对最小值和最大值跟踪

同样的调试类也可以用于禁用下溢/溢出检测功能的每批次跟踪。

假设你想观察给定批次中每个 `forward` 调用的所有成分的绝对最小值和最大值，并且只想对批次 1 和 3 进行此操作。然后实例化该类如下：

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

现在批次 1 和 3 将使用与下溢/溢出检测器相同的格式进行跟踪。

批次是 0 索引的。

这在你知道程序开始出现问题的批次号之后很有帮助，因此你可以直接跳到该区域。以下是此类配置的样本截断输出：

```
                  *** Starting batch number=1 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.47e+04 input[0]
5.36e-05 7.92e+02 output
[...]
                  decoder.dropout Dropout
1.60e-07 2.27e+01 input[0]
0.00e+00 2.52e+01 output
                  decoder T5Stack
     not a tensor output
                  lm_head Linear
1.01e-06 7.92e+02 weight
0.00e+00 1.11e+00 input[0]
6.06e-02 8.39e+01 output
                   T5ForConditionalGeneration
     not a tensor output

                  *** Starting batch number=3 ***
abs min  abs max  metadata
                  shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.78e+04 input[0]
5.36e-05 7.92e+02 output
[...]
```

你会得到大量帧的转储——正如你的模型中的前向调用数量一样多，所以这可能是你想要的，但有时对于调试目的来说，比普通调试器更容易使用。例如，如果问题开始发生在批次号 150。因此，你可以转储批次 149 和 150 的跟踪并比较数字何时开始偏离。

你还可以指定在哪个批次号后停止训练：

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```