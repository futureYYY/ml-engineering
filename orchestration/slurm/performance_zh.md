# performance - 中文翻译

# SLURM 性能

在这里，您会找到一些影响性能的SLURM特定设置的讨论。

## `srun` 的 `--cpus-per-task` 可能需要明确指定

您需要确保通过 `srun` 启动的程序接收到预期数量的CPU核心。例如，在典型的机器学习训练程序中，每个GPU至少需要一个用于驱动它的CPU核心以及几个额外的核心用于 `DataLoader`。您需要多个核心以便每个任务可以并行执行。如果您有8个GPU且每个GPU有两个 `DataLoader` 工作者，则您需要每个节点至少 `3*8=24` 个CPU核心。

`--cpus-per-task` 定义了每个任务的CPU核心数，该设置通过 `sbatch` 或 `salloc` 传递，并且原来的 `srun` 会继承此设置。然而，最近这种行为发生了变化：

`sbatch` 手册页中的一段引文：

> 注意：从22.05版本开始，`srun` 不再继承 `salloc` 或 `sbatch` 请求的 `--cpus-per-task` 值。必须在调用 `srun` 时再次请求，或者如果希望为任务设置此值，可以通过 `SRUN_CPUS_PER_TASK` 环境变量来设置。

这意味着在过去，您的SLURM脚本可能是这样的：

```bash
#SBATCH --cpus-per-task=48
[...]
srun myprogram
```

并且由 `srun` 启动的程序会接收到48个CPU核心，因为 `srun` 继承了 `sbatch` 或 `salloc` 设置中的 `--cpus-per-task=48`。根据引文中的文档，自SLURM 22.05版本起，这种行为已不再有效。

注脚：我在SLURM@22.05.09上进行了测试，发现旧的行为仍然有效，但这种情况肯定发生在23.x系列中。因此，变化可能发生在22.05系列的后期版本中。

因此，如果您保持原样，现在程序只会接收到一个CPU核心（除非 `srun` 的默认设置已被修改）。

您可以使用 `os.sched_getaffinity(0)` 轻松测试您的SLURM设置是否受到影响，因为它显示当前进程可以使用的CPU核心。因此，使用 `len(os.sched_getaffinity(0))` 来计数这些核心应该是很容易的。

以下是测试您是否受到影响的方法：
```bash
$ cat test.slurm
#!/bin/bash
#SBATCH --job-name=test-cpu-cores-per-task
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48   # 根据您的环境调整为少于48个CPU核心
#SBATCH --time=0:10:00
#SBATCH --partition=x        # 根据您的环境调整到正确的分区名称
#SBATCH --output=%x-%j.out

srun python -c 'import os; print(f"可见的CPU核心数: {len(os.sched_getaffinity(0))}")'
```

如果您得到
```bash
可见的CPU核心数: 48
```
那么您不需要做任何事情，但如果得到
```bash
可见的CPU核心数: 1
```
或另一个小于48的值，那么您会受到影响。

要解决这个问题，您需要更改您的SLURM脚本，方法如下：

```bash
#SBATCH --cpus-per-task=48
[...]
srun --cpus-per-task=48 myprogram
```
或：
```bash
#SBATCH --cpus-per-task=48
[...]
SRUN_CPUS_PER_TASK=48
srun myprogram
```

或者自动化处理，一次编写并遗忘：
```bash
#SBATCH --cpus-per-task=48
[...]
SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
srun myprogram
```