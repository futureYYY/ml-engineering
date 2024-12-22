# torch-distributed-hanging-solutions - 中文翻译

## 诊断多节点多GPU Python程序中的挂起和死锁

虽然本文中的方法是在处理基于PyTorch的多节点多GPU训练时开发的，但它们当然也可以帮助任何多进程多节点的Python程序。

## 辅助工具

尝试使用以下脚本 [torch-distributed-gpu-test.py](torch-distributed-gpu-test.py) 来诊断情况。

这将主要帮助发现与网络相关的问题，并快速了解多GPU通信的工作原理。

对于代码相关的问题，请阅读文档的其余部分。

## 多GPU挂起/死锁的诊断方法

### py-spy

首先运行 `pip install py-spy`。

现在你可以附加到每个进程中：

``` 
py-spy dump -n -p PID
```
它会告诉你进程在何处挂起（通常是nccl集体函数或 `barrier`）。

- `PID` 是挂起的Python进程的进程ID。
- `-n` 如果你想查看用C、C++等编写的Python扩展中的堆栈跟踪，这是有用的，因为程序可能在这些扩展中挂起。
- 你可能需要在命令前添加 `sudo` - 更多详细信息请参阅 [此注释](https://github.com/benfred/py-spy/blob/master/README.md#when-do-you-need-to-run-as-sudo)。

如果你没有 `sudo` 访问权限，你的系统管理员可能会为你执行：
``` 
sudo echo 0 > /proc/sys/kernel/yama/ptrace_scope
```
这将允许你在不需要 `sudo` 的情况下运行 `py-spy`（以及 `strace`）。请注意可能的 [安全影响](https://wiki.ubuntu.com/SecurityTeam/Roadmap/KernelHardening#ptrace_Protection)，但如果计算节点无法从互联网访问，则通常风险较小。

要使此更改永久化，请编辑 `/etc/sysctl.d/10-ptrace.conf` 并设置：
``` 
kernel.yama.ptrace_scope = 0
```

以下是 `py-spy dump` 的Python堆栈跟踪示例：
``` 
Thread 835995 (active): "MainThread"
    broadcast (torch/distributed/distributed_c10d.py:1191)
    _aggregate_total_loss (deepspeed/runtime/pipe/engine.py:540)
    train_batch (deepspeed/runtime/pipe/engine.py:330)
    train_step (megatron/training.py:436)
    train (megatron/training.py:851)
    pretrain (megatron/training.py:187)
    <module> (pretrain_gpt.py:239)
```
第一行就是程序卡住的地方。

如果挂起发生在CPP扩展中，请添加 `--native` 到 `py-spy` 中，它将显示任何非Python代码。

#### 多进程 py-spy

现在，如何对多个进程进行操作？一次一个地做太慢了。所以让我们一次搞定。

如果启动命令是 `python`，那么你应该做的是：

``` 
pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}
```

如果是 `deepspeed`：

``` 
pgrep -P $(pgrep -o deepspeed) | xargs -I {} py-spy dump --pid {}
```

对于 `accelerate`：

``` 
pgrep -P $(pgrep -o accelerate) | xargs -I {} py-spy dump --pid {}
```

你明白这个逻辑。

这种方法只会分析主进程，而不是这些进程生成的各种子进程/线程。所以如果你有8个GPU和8个进程，上述命令将生成8个堆栈跟踪。

如果你想要所有进程及其子进程，那么只需运行：

``` 
pgrep -f python | xargs -I {} py-spy dump --pid {}
```
（如前所述，如果启动程序不是 `python`，请相应地调整）

该方法将提供所有Python进程的跟踪。

如果你什么也没得到，先从基本调试开始：

``` 
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'date'
```
一旦你知道自己在和所有节点通信，然后你可以逐步解开调用的深度，如下：

``` 
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'date'
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -o python'
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -P $(pgrep -o python) '
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}'
```
在每个阶段检查输出是否合理——例如，第2和第3次调用应该得到进程的PID。

#### 多节点 py-spy 通过 srun

如果你有多节点怎么办？

你可以当然通过 `ssh` 交互式地登录到每个节点并转储堆栈跟踪。

如果你使用的是SLURM环境，你可以使用 `srun` 为所有节点执行此操作。

现在在另一个控制台获取 `SLURM_JOBID`（或从 `salloc` 日志中获取）：
``` 
squeue -u `whoami` -o "%.16i %9P %26j %.8T %.10M %.8l %.6D %.20S %R"
```

现在使用以下 `srun` 命令，调整 `SLURM_JOBID` 以匹配上面这一句的结果：
``` 
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'ps aux | grep python | egrep -v "grep|srun" | grep `whoami` | awk "{print \$2}" | xargs -I {} py-spy dump --native --pid {}' || echo "failed"
```

注意事项：
- 你必须使用 `--gres=gpu:0` 对于监控 `srun`，否则它将一直阻塞直到主 `srun`（运行训练的那个）退出。
- 每个节点将生成其独特的日志文件，名为 `trace-nodename.out`——因此这有助于识别哪些节点存在问题。如果你想让所有内容都转储到标准输出，可以删除 `--output=trace-%N.out`。
- 在某些SLURM版本中，你可能还需要添加 `--overlap`。
- 在某些SLURM版本中，作业ID可能与 `squeue` 报告的不同，因此你必须从你试图“附加”的作业的日志中获取正确的 `SLURM_JOB_ID` ——即分配GPU的 `srun` 作业。
- 有时 `bash` 不起作用，但 `sh` 起作用。我认为这与加载哪些点文件有关。
- 你可能还需要激活自定义的Python环境，可以通过以下方式完成：
``` 
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'conda activate myenvname; ps auxc | ... ' || echo "failed"
```
或者你可以在 `~/.bashrc` 或你决定使用的任何shell的rc文件中完成。

如前所述，如果你只想主进程，你可以使用这个替代命令：
``` 
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}' || echo "failed"
```
根据上述多GPU部分中的说明调整 `python`。

之前的较长命令将为所有Python进程提供跟踪。

如果你什么也没得到，先从基本调试开始：

``` 
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'date'
```
一旦你知道自己在和所有节点通信，然后你可以逐步解开调用的深度，如下：

``` 
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'date'
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -o python'
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -P $(pgrep -o python) '
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}'
```
在每个阶段检查输出是否合理——例如，第2和第3次调用应该得到进程的PID。