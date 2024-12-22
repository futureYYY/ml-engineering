# README - 中文翻译

## 网络调试

很多时候，你不需要成为网络工程师就能解决网络问题。一些常见的问题可以通过阅读以下笔记来解决。

---

## 术语表

- OOB：带外（通常是较慢的以太网适配器）
- 绑定：使用多个网卡以提高速度或作为备份
- IB：InfiniBand（最初由Mellanox开发，现已被NVIDIA收购）
- NIC：网络接口卡

---

## 如何诊断NCCL多GPU和多节点连接问题

本节内容不是详尽无遗的，旨在涵盖我经常遇到的一些最常见的设置问题。对于更复杂的问题，请参阅[NCCL仓库问题](https://github.com/NVIDIA/nccl/issues)。如果找不到匹配情况的问题，可以提交一个新的问题。NCCL还包含了一个简短的[故障排除部分](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html)，但通常从阅读[问题](https://github.com/NVIDIA/nccl/issues)中学到的东西更多。

对于网络诊断工作，与其使用一个可能需要很长时间启动并且有无关问题的完整应用程序，我建议使用这个特别设计的测试脚本：[torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py)。

首先，在设置后运行基于nccl的程序：

```bash
export NCCL_DEBUG=INFO
```

这将打印大量关于NCCL设置及其网络流量的调试信息。

例如，如果你正在使用上述调试脚本，并且在一个具有8个GPU的单节点上运行，你可以这样做：

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 torch-distributed-gpu-test.py
```

要在多个节点上启动它，你需要使用某种编排软件如SLURM或Kubernetes，或者手动在每个节点上启动（`pdsh`会非常有用）——详情见[torch-distributed-gpu-test.py](../../debug/torch-distributed-gpu-test.py)中的说明。但是为了理解其工作原理，我建议先从一个节点开始，然后逐步进展到两个节点，再扩展到更多的节点。

现在，检查程序的输出并查找以以下行开头的内容：
```bash
NCCL INFO NET/
```
然后检查它使用的是哪种协议以及哪些接口。

例如，以下输出：
```bash
NCCL INFO NET/FastSocket : Using [0]ibs108:10.0.19.12<0> [1]ibs109:10.0.19.13<0> [2]ibs110:10.0.19.14<0> [3]ibs111:10.0.19.15<0> [4]ibs112:10.0.19.16<0> [5]ibs113:10.0.19.17<0> [6]ibs114:10.0.19.18<0> [7]ibs115:10.0.19.19<0>
```

告诉我们[nccl-fastsocket](https://github.com/google/nccl-fastsocket)传输层插件被使用，并且发现了8个`ibs*`网络接口（网卡）。如果你正在使用Google Cloud，这是正确的，你的NCCL很可能配置正确了。但是，如果你正在使用InfiniBand（IB），并且得到了上述输出，那么你可能会得到非常低的节点间速度，因为这意味着你激活了错误的插件。

对于IB来说，你应该看到的是`NET/IB`及其IB接口：
```bash
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/IB [RO]; OOB eno1:101.262.0.9<0>
```

在这里，你可以看到IB被用于8个`mlx5_*`接口的集体通信，还有一个OOB，代表带外，通常使用一个较慢的以太网NIC（有时是几个网卡绑定在一起的——如果你想知道接口名称中的`bond`是什么意思）。

要了解你的节点有哪些TCP/IP接口，可以在其中一个节点上运行`ifconfig`命令（通常所有类似的节点会有相同的接口名称，但并不总是这样）。

如果你的集体通信网络是IB，而不是运行`ifconfig`，你应该运行`ibstat`。上面的`NCCL INFO NET`最后示例将对应于以下输出：

```bash
$ ibstat | grep mlx5
CA 'mlx5_0'
CA 'mlx5_1'
CA 'mlx5_2'
CA 'mlx5_3'
CA 'mlx5_4'
CA 'mlx5_5'
CA 'mlx5_6'
CA 'mlx5_7'
```

由于除了快速节点间连接网卡之外，你还可能有一个缓慢的管理以太网网卡（甚至可能是多个这样的网卡），这些网卡是为了能够配置节点、使用共享文件系统、访问互联网，所以`ifconfig`很可能也会包括额外的网卡。你也可能有一个Docker网络接口、`lo`回环接口和其他一些接口。例如在我的电脑上，我可能会得到以下输出：

```bash
$ ifconfig
docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.99.0.1  netmask 255.255.0.0  broadcast 172.99.255.255
        inet6 f330::42:fe33:f335:7c94  prefixlen 64  scopeid 0x20<link>
        ether 02:42:fe:15:1c:94  txqueuelen 0  (Ethernet)
        RX packets 219909  bytes 650966314 (650.9 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 262998  bytes 20750134 (20.7 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 1147283113  bytes 138463231270 (138.4 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 1147283113  bytes 138463231270 (138.4 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.23  netmask 255.255.255.0  broadcast 10.0.0.255
        inet6 2601:3108:1c71:600:4224:7e4b:13e4:7b54  prefixlen 64  scopeid 0x0<global>
        ether 04:41:1a:16:17:bd  txqueuelen 1000  (Ethernet)
        RX packets 304675330  bytes 388788486256 (388.7 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 74956770  bytes 28501279127 (28.5 GB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
        device memory 0xa3b00000-a3bfffff
```

我提到这些的原因是关键部分是确保NCCL在其`Using`调试行中仅报告正确的接口。如果任何接口如`docker0`、`lo`或`eth0`最终被报告，例如：

```bash
NCCL INFO NET/Socket : Using [0]eth0:10.0.0.23<0>
```

如果不是你希望使用的更快网络接口，则很可能是不正确的。但是，当然，在某些情况下，以太网网卡是你唯一拥有的选择，在这种情况下，上述情况只是很好——只是会非常慢。

有时，如果使用了错误的接口，应用程序可能会挂起。

如果你拥有所有正确的接口加上一些不正确的接口，NCCL可能会工作但速度较慢。

如果是云环境，通常你的云提供商应该给你如何正确设置的说明。如果没有，那么你需要至少询问他们你需要使用哪些网络接口来设置NCCL。

虽然NCCL尽力自动发现它应该使用哪些接口，但如果它无法正确地做到这一点，你可以通过告诉它使用或不使用哪些接口来帮助它：

- `NCCL_SOCKET_IFNAME` 可用于指定在不使用InfiniBand时应包括或排除的`ifconfig`接口。以下是一些示例：

```bash
export NCCL_SOCKET_IFNAME=eth:        使用所有以eth开头的接口，例如eth0, eth1, ...
export NCCL_SOCKET_IFNAME==eth0:      仅使用接口eth0
export NCCL_SOCKET_IFNAME==eth0,eth1: 仅使用接口eth0和eth1
export NCCL_SOCKET_IFNAME=^docker:    不使用任何以docker开头的接口
export NCCL_SOCKET_IFNAME=^=docker0:  不使用接口docker0。
```

完整的文档位于[这里](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname)。

- 当使用IB RDMA（IB Verbs接口）时，而不是使用`NCCL_SOCKET_IFNAME`，应使用`NCCL_IB_HCA`环境变量，该变量选择用于集体通信的接口。示例：

```bash
export NCCL_IB_HCA=mlx5 :               使用所有以mlx5开头的卡的所有端口
export NCCL_IB_HCA==mlx5_0:1,mlx5_1:1 : 使用卡片mlx5_0和mlx5_1的端口1。
export NCCL_IB_HCA=^=mlx5_1,mlx5_4 :    不使用卡片mlx5_1和mlx5_4。
```

完整的文档位于[这里](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-ib-hca)。

例如，通常在IB中，会有一些你不希望包含在NCCL通信中的额外接口，如`mlx5_bond_0`。例如，此报告表明错误的`[8]mlx5_bond_0:1/RoCE`接口被包括了，这几乎肯定会导致带宽低：

```bash
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB [4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/I [8]mlx5_bond_0:1/RoCE [RO]; OOB ibp25s0:10.0.12.82<0>
```

在这种情况下，你可以排除它：

```bash
export NCCL_IB_HCA=^mlx5_bond_0:1
```

或者，你可以明确列出你想要的接口，例如：

```bash
export NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

如前所述，使用IB连接的其中一个节点上的`ibstat`会显示可用的IB接口。

由于NCCL试图自动选择最佳网络接口，只有在NCCL无法正常工作或速度很慢时，你才需要执行上述操作。在正常情况下，NCCL应该可以开箱即用，无需用户进行特殊设置。

此外，根据使用的云环境，很可能会有很多环境变量需要设置。如果你设置了一些变量不正确，NCCL可能会工作得很慢或根本不起作用。

另一个用户经常遇到的典型问题是，当他们尝试在云A上工作的NCCL设置在云B上重用时。通常情况下，事情不会顺利转移，你需要仔细删除以前设置的环境变量，并为新云环境重新正确设置它们。即使你使用的是同一云环境，但在不同类型的实例之间，也可能会出现这个问题，因为某些网络设置对于特定实例非常具体，不会在其他地方工作。

一旦认为NCCL已正确设置，下一步就是对你的连接进行基准测试，确保它与广告的速度相符（大约80%）。继续进行[基准章节](../benchmarks)。

---

## 在Docker容器中使用NCCL

* 通过添加以下参数到Docker `run`命令中，提供足够的资源：`–shm-size=1g –ulimit memlock=-1`（[更多信息](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html#sharing-data)）
* 特权访问：有时你可能需要向Docker `run`命令中添加`--privileged`。
* Docker镜像中包含正确的包，例如，如果使用IB，你需要至少安装`libibverbs1 librdmacm1`

---

## 如何检查是否支持P2P

有时你需要知道你的计算节点上的GPU是否支持P2P访问（Peer2Peer）。禁用P2P通常会导致节点内连接速度变慢。

你可以看到在这个特定的8x NVIDIA H100节点上P2P是支持的：

```bash
$ nvidia-smi topo -p2p r
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
 GPU0   X       OK      OK      OK      OK      OK      OK      OK
 GPU1   OK      X       OK      OK      OK      OK      OK      OK
 GPU2   OK      OK      X       OK      OK      OK      OK      OK
 GPU3   OK      OK      OK      X       OK      OK      OK      OK
 GPU4   OK      OK      OK      OK      X       OK      OK      OK
 GPU5   OK      OK      OK      OK      OK      X       OK      OK
 GPU6   OK      OK      OK      OK      OK      OK      X       OK
 GPU7   OK      OK      OK      OK      OK      OK      OK      X

Legend:

  X    = 自身
  OK   = 状态正常
  CNS  = 芯片组不支持
  GNS  = GPU不支持
  TNS  = 拓扑结构不支持
  NS   = 不支持
  U    = 未知
```

另一方面，在这个特定的2x NVIDIA L4节点上P2P不被支持：

```bash
$ nvidia-smi topo -p2p r
        GPU0    GPU1
 GPU0   X       CNS
 GPU1   CNS     X
```

从图例中可以看出，`CNS`表示“芯片组不支持”。

如果你使用的是高端数据中心GPU，这种情况非常不可能发生。尽管一些低端数据中心GPU可能不支持P2P，就像上面L4的例子一样。

对于消费级GPU，你的GPU不被支持可能有各种原因，通常是IOMMU和/或ACS功能被启用。有时只是驱动版本的问题。如果你花时间搜索，可能会找到有人在黑客方式下修改驱动以在不应支持P2P的GPU上启用P2P，比如这个[4090 P2P支持仓库](https://github.com/tinygrad/open-gpu-kernel-modules)。

要检查PCI访问控制服务（ACS）是否启用并按照[此指南](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/troubleshooting.html#pci-access-control-services-acs)禁用它们。

IOMMU可以在BIOS中禁用。

你也可以使用torch检查特定GPU之间的P2P支持，这里是检查GPU 0和1之间的支持：

```bash
python -c "import torch; print(torch.cuda.can_device_access_peer(torch.device('cuda:0'), torch.device('cuda:1')))"
```

如果没有P2P支持，上述代码会打印`False`。

---

## 如何统计NCCL调用次数

启用NCCL调试日志记录以收集子系统 - 集合：

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL
```

如果你在SLURM环境中使用许多节点，你可能只想在rank 0上执行此操作，如下所示：

```bash
if [[ $SLURM_PROCID == "0" ]]; then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=COLL
fi
```

假设你的日志都被发送到了`main_log.txt`，你可以使用以下命令来统计每种集合调用执行了多少次：

```bash
grep -a "NCCL INFO Broadcast" main_log.txt     | wc -l
2590
grep -a "NCCL INFO AllReduce" main_log.txt     | wc -l
5207
grep -a "NCCL INFO AllGather" main_log.txt     | wc -l
1849749
grep -a "NCCL INFO ReduceScatter" main_log.txt | wc -l
82850
```

最好先隔离特定阶段的训练，因为加载和保存会有与训练迭代非常不同的模式。

因此，我通常会先切分出一次迭代。例如，如果每次迭代日志都以`iteration: ...`开头，那么我会先做：

```bash
csplit main_log.txt '/iteration: /' "{*}"
```

然后分析其中一个对应于迭代的结果文件。默认情况下，它将被命名为类似`xx02`的东西。