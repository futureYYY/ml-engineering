# README - 中文翻译

## 可重复性

### 在基于随机性的软件中实现确定性

在调试时，为所有使用的随机数生成器（RNG）设置固定的种子，以便每次重新运行时都能得到相同的数据/代码路径。

尽管有如此多不同的系统，覆盖所有情况可能会很棘手。以下是一些尝试覆盖的情况：

```python
import random, torch, numpy as np
def enforce_reproducibility(use_seed=None):
    seed = use_seed if use_seed is not None else random.randint(1, 1000000)
    print(f"使用种子: {seed}")

    random.seed(seed)    # python RNG
    np.random.seed(seed) # numpy RNG

    # pytorch RNGs
    torch.manual_seed(seed)          # cpu + cuda
    torch.cuda.manual_seed_all(seed) # 多GPU - 即使没有GPU也可以调用
    if use_seed: # 较慢的速度！https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    return seed
```

如果你使用这些子系统/框架，可能还有其他一些：

```python
    torch.npu.manual_seed_all(seed)
    torch.xpu.manual_seed_all(seed)
    tf.random.set_seed(seed)
```

当你反复运行相同的代码以解决某些问题时，在代码开始处设置特定的种子：

```python
enforce_reproducibility(42)
```

如上所述，这仅适用于调试，因为它激活了各种有助于确定性的PyTorch标志，但会降低速度，因此在生产环境中不希望这样。

然而，你可以调用以下内容以在生产中使用：

```python
enforce_reproducibility()
```

即不使用显式的种子。然后它会选择一个随机种子并记录下来！因此如果在生产中出现问题，现在可以重现相同的RNG设置。而且这次没有性能损失，因为只有在显式提供种子时才会设置`torch.backends.cudnn`标志。假设它记录了：

```python
使用种子: 1234
```

你只需将代码更改为：

```python
enforce_reproducibility(1234)
```

即可获得相同的RNG设置。

如前几段所述，系统中可能涉及许多其他RNG，例如，如果你想让`DataLoader`中的数据按相同的顺序输入，则需要[设置其种子](https://pytorch.org/docs/stable/notes/randomness.html#dataloader)。

更多资源：
- [PyTorch中的可重复性](https://pytorch.org/docs/stable/notes/randomness.html)

### 复现软件和系统环境

当发现结果中的某些差异（例如质量或吞吐量）时，这种方法很有用。

其想法是记录启动训练（或推理）所使用的环境的关键组件，以便在以后阶段需要完全再现时可以做到。

由于系统和组件种类繁多，无法规定一种始终有效的方法。因此，让我们讨论一种可能的方案，然后你可以根据自己的具体环境进行调整。

这被添加到你的Slurm启动脚本中（或其他任何用于启动训练的方式）——这是一个Bash脚本：

```bash
SAVE_DIR=/tmp # 编辑为实际路径
export REPRO_DIR=$SAVE_DIR/repro/$SLURM_JOB_ID
mkdir -p $REPRO_DIR
# 1. 模块（写入stderr）
module list 2> $REPRO_DIR/modules.txt
# 2. 环境变量
/usr/bin/printenv | sort > $REPRO_DIR/env.txt
# 3. pip（这包括开发安装的SHA）
pip freeze > $REPRO_DIR/requirements.txt
# 4. 在conda中安装的git克隆的未提交更改
perl -nle 'm|"file://(.*?/([^/]+))"| && qx[cd $1; if [ ! -z "\$(git diff)" ]; then git diff > \$REPRO_DIR/$2.diff; fi]' $CONDA_PREFIX/lib/python*/site-packages/*.dist-info/direct_url.json
```

如你所见，这个方案是在Slurm环境中使用的，因此每个新的训练都会将特定于Slurm作业的环境信息保存下来。

1. 我们保存加载的`模块`，例如在云集群/HPC设置中，你可能会使用这种方式加载CUDA和cuDNN库。

   如果你不使用`模块`，则删除该条目。

2. 我们转储环境变量。这可能是至关重要的，因为在某些环境中，单个环境变量如`LD_PRELOAD`或`LD_LIBRARY_PATH`可能对性能产生巨大影响。

3. 然后我们转储conda环境包及其版本——这应该适用于任何虚拟Python环境。

4. 如果你使用`pip install -e .` 进行开发安装，它除了知道从哪个git SHA安装的之外，对git克隆仓库没有任何了解。但问题是，你可能已经本地修改了文件，现在`pip freeze`会忽略这些更改。所以这部分会遍历所有未安装到conda环境中的包（我们通过查找`site-packages/*.dist-info/direct_url.json`来找到它们）。

另一个有用的工具是[conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md)，它可以帮助你找出两个conda环境之间的精确差异。

据我同事和我的经验，在同一个云集群上运行完全相同的代码时，我们得到了非常不同的训练TFLOPs——几乎是从同一共享目录中运行相同的slurm脚本。我们首先使用[conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md)比较我们的conda环境，发现了一些差异——我安装了她拥有的确切包以匹配她的环境，但仍显示巨大的性能差异。然后我们比较了`printenv`的输出，发现我设置了`LD_PRELOAD`而她没有——这在特定的云供应商中由于需要设置多个环境变量到自定义路径才能充分利用其硬件，因此产生了巨大差异。