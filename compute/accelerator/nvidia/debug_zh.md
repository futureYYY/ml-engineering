# debug - 中文翻译

## 解决NVIDIA GPU问题

## 术语表

- DBE：双比特错误纠正码（Double Bit ECC Error）
- DCGM：（NVIDIA）数据中心GPU管理器
- ECC：错误校正码
- FB：帧缓冲区
- SBE：单比特错误纠正码（Single Bit ECC Error）
- SDC：静默数据损坏

## Xid 错误

没有任何硬件是完美的，有时由于制造问题或因磨损（特别是因为暴露在高温下），GPU可能会遇到各种硬件问题。许多这些问题会自动得到纠正，而不需要真正理解发生了什么。如果应用程序继续运行，则通常无需担心。如果应用程序因硬件问题而崩溃，则重要的是要了解为什么会这样以及如何处理。

普通用户如果只使用少数几个GPU，可能永远不需要理解与GPU相关的硬件问题，但如果你接近大规模机器学习训练，可能需要使用数百到数千个GPU，那么你肯定希望了解不同的硬件问题。

在你的系统日志中，你可能会偶尔看到Xid错误，如：

```
NVRM: Xid (PCI:0000:10:1c): 63, pid=1896, Row Remapper: New row marked for remapping, reset gpu to activate.
```

要获取这些日志，可以使用以下任一方法：
```
sudo grep Xid /var/log/syslog
sudo dmesg -T | grep Xid
```

通常情况下，只要训练没有崩溃，这些错误往往表示由硬件自动纠正的问题。

完整的Xid错误及其解释列表可以在[这里](https://docs.nvidia.com/deploy/xid-errors/index.html)找到。

你可以运行`nvidia-smi -q`并查看是否报告了任何错误计数。例如，在这种Xid 63的情况下，你会看到类似的内容：

```
Timestamp                                 : Wed Jun  7 19:32:16 2023
Driver Version                            : 510.73.08
CUDA Version                              : 11.6

Attached GPUs                             : 8
GPU 00000000:10:1C.0
    Product Name                          : NVIDIA A100-SXM4-80GB
    [...]
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 177
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 177
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 1
        Uncorrectable Error               : 0
        Pending                           : Yes
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 639 bank(s)
            High                          : 1 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
[...]
```

我们可以看到，Xid 63对应于：

```
ECC页面退役或行重映射记录事件
```

这可能有三个原因：硬件错误、驱动程序错误或帧缓冲区（FB）损坏。

此错误意味着其中一行内存出现故障，并且在重新启动和/或GPU重置后，将使用640个备用内存行之一替换该坏行。因此，在上述报告中我们看到只剩下639个银行（总共640个）。

`ECC错误`报告中的`Volatile`部分指的是自上次重新启动/GPU重置以来记录的错误。`Aggregate`部分记录的是自GPU首次使用以来的相同错误。

现在，有两种类型的错误——可纠正和不可纠正。可纠正的一个是单比特错误纠正码（SBE），即使内存有故障，驱动程序仍可以恢复正确的值。不可纠正的是多个比特故障，称为双比特错误纠正码（DBE）。通常，如果同一内存地址发生1个DBE或2个SBE错误，驱动程序会退役整个内存页。有关详细信息，请参阅[此文档](https://docs.nvidia.com/deploy/dynamic-page-retirement/index.html)。

可纠正的错误不会影响应用程序，而非可纠正的错误会导致应用程序崩溃。包含不可纠正ECC错误的内存页将被黑名单并不可访问，直到GPU重置。

如果有计划退役的页面，你将在`nvidia-smi -q`的输出中看到如下内容：

```
    Retired pages
        Single Bit ECC             : 2
        Double Bit ECC             : 0
        Pending Page Blacklist    : Yes
```

每个退役的页面都会减少应用程序可用的总内存。但是，退役的最大页面总数仅为4MB，因此它不会显著减少总的可用GPU内存。

要更深入地进行GPU调试，请参考[此文档](https://docs.nvidia.com/deploy/gpu-debug-guidelines/index.html)——它包括一个有用的诊断图表，帮助确定何时更换GPU。该文档还包括关于类似Xid 63错误的更多信息。

例如，它建议：

> 如果与Xid 94相关联，遇到错误的应用程序需要重新启动。系统上的所有其他应用程序可以保持原样，直到方便的时候重启以激活行重映射。
> 请参阅下面的指南，根据行重映射失败情况确定何时更换GPU。

如果在重新启动后，相同的条件发生在相同的内存地址上，这意味着内存重映射失败，并且将再次发出Xid 64。如果这种情况持续发生，这意味着你有一个无法自动纠正的硬件问题，需要更换GPU。

在其他情况下，你可能会遇到Xid 63或64，并且应用程序崩溃。这通常会生成额外的Xid错误，但大多数时候这意味着错误是不可纠正的（即，这是一个DBE类的错误，然后是Xid 48）。

如前所述，要重置GPU，你可以简单地重新启动机器，或者运行：

``` 
nvidia-smi -r -i gpu_id
```

其中`gpu_id`是你想要重置的GPU的序列号，例如`0`代表第一个GPU。如果没有`-i`，所有GPU都将被重置。

### 遇到不可纠正的ECC错误

如果你遇到错误：
``` 
CUDA错误：遇到不可纠正的ECC错误
```

如前一节所述，检查`nvidia-smi -q`的输出中的`ECC错误`条目将告诉你哪个GPU有问题。但是，如果你需要快速检查以便回收节点，如果至少有一个GPU有这个问题，你可以这样做：

``` 
$ nvidia-smi -q | grep -i correctable | grep -v 0
            SRAM Uncorrectable            : 1
            SRAM Uncorrectable            : 5
```

在一个好的节点上，这应该返回空结果，因为所有的计数都应该是0。但在上面的例子中，我们有一个损坏的GPU——有两个条目，因为完整记录是：

``` 
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 1
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 5
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
```

第一个条目是`Volatile`（自上次GPU驱动程序重载以来的错误）和第二个是`Aggregate`（GPU整个生命周期的总错误计数）。在这个例子中，我们看到`Volatile`的SRAM不可纠正错误计数为1，而整个生命周期的计数为5——这意味着这不是这个GPU第一次遇到这个问题。

这通常对应于Xid 94错误（参见：[Xid错误](#xid-errors)，最有可能没有Xid 48。

要克服这个问题，就像前面提到的一样，重置有问题的GPU：
``` 
nvidia-smi -r -i gpu_id
```

重新启动机器会有同样的效果。

现在，当涉及到`Aggregate` SRAM不可纠正的错误时，如果你有超过4个，那通常是更换这个GPU的理由。