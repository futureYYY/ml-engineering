# pytorch - 中文翻译

# 调试 PyTorch 程序

## 让节点互相通信

一旦需要使用多个节点来扩展训练，例如想要使用 DDP 来加快训练速度，就需要让这些节点互相通信，以便通信集合能够彼此发送数据。这通常通过像 [NCCL](https://github.com/nVIDIA/nccl) 这样的通信库来实现。在我们的 DDP 示例中，在训练步骤结束时，所有 GPU 必须执行一个 `all_reduce` 调用来跨所有 rank 同步梯度。

在本节中，我们将讨论一个非常简单的案例，即只有两个节点（每个节点有 8 个 GPU）互相通信，然后可以轻松扩展到任意数量的节点。假设这些节点的 IP 地址是 10.0.0.1 和 10.0.0.2。

有了 IP 地址后，我们还需要选择一个用于通信的端口。

Unix 中有 64k 个端口。前 1k 个端口被保留给常见的服务，因此互联网上的任何计算机都可以连接到任何其他计算机，并且事先知道要连接到哪个端口。例如，端口 22 被保留给 SSH。因此，每当你在终端输入 `ssh example.com` 时，实际上是打开了与 `example.com:22` 的连接。

由于存在成千上万的服务，预留的 1k 个端口是不够的，所以各种服务可以使用几乎任何端口。但不必担心，当你在云或高性能计算环境中获取 Linux 服务器时，不太可能有预装的服务占用高编号端口，因此大多数端口应该是可用的。

因此，我们选择端口 6000。

现在我们有：`10.0.0.1:6000` 和 `10.0.0.2:6000`，我们需要它们能够互相通信。

首先要做的是在两个节点上打开端口 `6000` 以允许进出连接。它可能是已经打开的，或者你可能需要查阅特定设置的文档来了解如何打开某个端口。

以下是你可以使用的多种方法来测试端口 6000 是否已经打开：

```bash
telnet localhost:6000
nmap -p 6000 localhost
nc -zv localhost 6000
curl -v telnet://localhost:6000
```

大多数这些命令可以通过 `apt install` 或你的包管理器安装。

让我们在这个例子中使用 `nmap`。如果我运行：

```bash
$ nmap -p 22 localhost
[...]
PORT   STATE SERVICE
22/tcp open  ssh
```

我们可以看到端口是开放的，并且会告诉我们分配了哪种协议和服务作为额外信息。

现在让我们运行：
```bash
$ nmap -p 6000 localhost
[...]
PORT     STATE  SERVICE
6000/tcp closed X11
```

在这里你可以看到端口 6000 是关闭的。

现在你已经理解了如何测试，可以继续测试 `10.0.0.1:6000` 和 `10.0.0.2:6000`。

首先在终端 A 中 ssh 到第一个节点并测试第二个节点上的端口 6000 是否打开：

```bash
ssh 10.0.0.1
nmap -p 6000 10.0.0.2
```

如果一切正常，那么在终端 B 中 ssh 到第二个节点并反向进行相同的检查：

```bash
ssh 10.0.0.2
nmap -p 6000 10.0.0.1
```

如果两个端口都已打开，你现在就可以使用这个端口。如果任何一个或两个都关闭，则需要打开这些端口。由于大多数云服务提供商使用专有解决方案，只需在网上搜索“打开端口”和你的云提供商名称即可。

接下来要理解的重要一点是，计算节点通常会有多个网络接口卡（NIC）。你可以通过运行以下命令来发现这些接口：

```bash
$ sudo ifconfig
```

一个接口通常用于用户通过 ssh 或各种非计算相关服务连接到节点——例如，发送电子邮件或下载数据。通常这个接口被称为 `eth0`，其中 `eth` 表示以太网，但它可以用其他名称表示。

然后是节点间的接口，可以是 InfiniBand、EFA、OPA、HPE Slingshot 等。可能有一个或几十个这样的接口。

以下是 `ifconfig` 输出的一些示例：

```bash
$ sudo ifconfig
enp5s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.23  netmask 255.255.255.0  broadcast 10.0.0.255
        [...]
```

我移除了大部分输出，只显示了一些信息。这里的关键信息是在 `inet` 之后列出的 IP 地址。在上面的例子中是 `10.0.0.23`。这是 `enp5s0` 接口的 IP 地址。

如果有另一个节点，它可能是 `10.0.0.24` 或 `10.0.0.21` 或类似的东西——最后一个数字会有所不同。

让我们看另一个例子：

```bash
$ sudo ifconfig
ib0     Link encap:UNSPEC  HWaddr 00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00
        inet addr:172.0.0.50  Bcast: 172.0.0.255  Mask:255.255.255.0
        [...]
```

这里的 `ib` 通常表示这是一个 InfiniBand 卡，但实际上它可以是其他供应商的产品。我见过例如 [OmniPath](../network#omni-path) 使用 `ib`。同样，`inet` 告诉我们这个接口的 IP 地址是 `172.0.0.50`。

如果你跟不上思路，我们需要 IP 地址以便测试每个节点上的 `ip:port` 是否打开。

最后回到我们的 `10.0.0.1:6000` 和 `10.0.0.2:6000` 对，让我们使用两个终端做一次 `all_reduce` 测试，我们选择 `10.0.0.1` 作为主主机来协调其他节点。
为了测试，我们将使用这个辅助调试程序 [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py)。

在终端 A 中：

```bash
$ ssh 10.0.0.1
$ python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --master_addr 10.0.0.1 --master_port 6000 torch-distributed-gpu-test.py
```

在终端 B 中：

```bash
$ ssh 10.0.0.2
$ python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --master_addr 10.0.0.1 --master_port 6000 torch-distributed-gpu-test.py
```

请注意，我在两种情况下都使用了相同的 `--master_addr 10.0.0.1 --master_port 6000`，因为我们检查了端口 6000 是开放的，并且我们使用 `10.0.0.1` 作为协调主机。

手动从每个节点运行这种方式虽然痛苦，但有一些工具可以自动在多个节点上启动相同的命令。

**pdsh**

`pdsh` 就是一个这样的解决方案——类似于 `ssh`，但可以自动在多个节点上运行相同的命令：

```bash
PDSH_RCMD_TYPE=ssh pdsh -w 10.0.0.1,10.0.0.2 \
"python -m torch.distributed.run --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --master_addr 10.0.0.1 --master_port 6000 torch-distributed-gpu-test.py"
```

你可以看到我如何将两组命令合并为一组。如果有更多的节点，只需添加更多的节点作为 `-w` 参数即可。


**SLURM**

如果你使用 SLURM，几乎可以肯定的是，设置环境的人已经为你打开了所有端口，所以它应该能直接工作。但如果不行，本节中的信息可以帮助你调试问题。

以下是如何使用 SLURM 的示例。

```bash
#!/bin/bash
#SBATCH --job-name=test-nodes        # 名称
#SBATCH --nodes=2                    # 节点数
#SBATCH --ntasks-per-node=1          # 关键 - 每个节点的任务数!
#SBATCH --cpus-per-task=10           # 每任务的内核数
#SBATCH --gres=gpu:8                 # GPU 数量
#SBATCH --time 0:05:00               # 最大执行时间 (HH:MM:SS)
#SBATCH --output=%x-%j.out           # 输出文件名
#
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
#
srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
torch-distributed-gpu-test.py'
```
如果你有超过两个节点，只需要更改节点数量，上述脚本将自动适用于任意数量的节点。


**MPI**:

另一种流行的方法是使用 [消息传递接口 (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface)。有许多可用的开源实现。

要使用此工具，你首先创建一个包含目标节点及其应在此主机上运行的进程数的 `hostfile`。在这个部分的例子中，对于有两个节点和每个节点有 8 个 GPU 的情况，将是：

```bash
$ cat hostfile
10.0.0.1:8
10.0.0.2:8
```

然后运行时，只需：

```bash
$ mpirun --hostfile  -np 16 -map-by ppr:8:node python my-program.py
```

请注意，我在这里使用了 `my-program.py`，因为 [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py) 是为与 `torch.distributed.run`（也称为 `torchrun`）一起工作而编写的。使用 `mpirun` 时，你需要检查你的特定实现使用哪个环境变量来传递程序的 rank 并替换 `LOCAL_RANK`，其余部分应该大致相同。

注意事项：
- 你可能需要明确告诉它使用哪个接口，通过添加 `--mca btl_tcp_if_include 10.0.0.0/24` 匹配我们的示例。如果你有很多网络接口，它可能会使用一个未打开的接口或仅仅是错误的接口。
- 你也可以反过来排除一些接口。例如，如果你有 `docker0` 和 `lo` 接口，可以通过添加 `--mca btl_tcp_if_exclude docker0,lo` 来排除它们。

`mpirun` 有无数的标志，我建议阅读它的手册页以获取更多信息。我的意图只是向你展示如何使用它。此外，不同的 `mpirun` 实现可能使用不同的命令行选项。