# admin - 中文翻译

## SLURM 管理

## 在多个节点上运行命令

1. 为了避免每次登录到新节点时被提示：
    ```
    Are you sure you want to continue connecting (yes/no/[fingerprint])?
    ```
    可以通过以下方式禁用此检查：
    ``` 
    echo "Host *" >> ~/.ssh/config
    echo "  StrictHostKeyChecking no" >> ~/.ssh/config
    ```
    
    当然，要确保这在你的需求下是足够安全的。我假设你已经登录到 SLURM 集群，并且没有 SSH 到集群之外的地方。你可以选择不设置这个选项，那么你需要手动批准每个新节点。

2. 安装 `pdsh`

现在你可以在多个节点上运行所需的命令了。

例如，让我们运行 `date` 命令：

``` 
$ PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26] date
node-25: Sat Oct 14 02:10:01 UTC 2023
node-21: Sat Oct 14 02:10:02 UTC 2023
node-23: Sat Oct 14 02:10:02 UTC 2023
node-24: Sat Oct 14 02:10:02 UTC 2023
node-26: Sat Oct 14 02:10:02 UTC 2023
```

让我们做一些更有用和复杂的事情。让我们终止所有与 GPU 绑定但未在 SLURM 作业取消时退出的进程：

首先，这条命令会给出所有占用 GPU 的进程 ID：

``` 
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq
```

因此，我们现在可以一次性终止所有这些进程：

``` 
PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26] "nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq | xargs -n1 sudo kill -9"
```

## SLURM 设置

显示 SLURM 设置：

``` 
sudo scontrol show config
```

配置文件位于控制器节点上的 `/etc/slurm/slurm.conf`。

一旦更新了 `slurm.conf` 文件，可以通过以下命令重新加载配置：
``` 
sudo scontrol reconfigure
```

从控制器节点运行。

## 自动重启

如果需要安全地重启节点（例如，如果镜像已更新），则可以调整节点列表并运行：

``` 
scontrol reboot ASAP node-[1-64]
```

对于每个非空闲节点，此命令将等待当前任务结束，然后重启节点并将其状态设置为 `idle`。

请注意，你需要在控制器节点的 `/etc/slurm/slurm.conf` 中设置：
``` 
RebootProgram = "/sbin/reboot"
```
并且如果刚添加了此条目到配置文件，则需要重新配置 SLURM 守护程序。

## 更改节点的状态

更改状态由 `scontrol update` 执行。

示例：

要使已准备好使用的节点处于空闲状态：
``` 
scontrol update nodename=node-5 state=idle
```

要将节点从 SLURM 的资源池中移除：
``` 
scontrol update nodename=node-5 state=drain
```

## 由于慢速进程退出导致节点被排空后的恢复

有时进程在作业取消时退出缓慢。如果 SLURM 配置为不会永远等待，它将自动排空这些节点。但是，这些节点没有理由不可供用户使用。

所以这里是如何自动化它。

关键是获取因 `"Kill task failed"` 而被排空的节点列表，可以通过以下命令获取：

``` 
sinfo -R | grep "Kill task failed"
```

现在提取并扩展节点列表，检查这些节点是否确实没有用户进程（或者先尝试杀死它们），然后解除排空状态。

之前你已经学习了如何在多个节点上运行命令，我们将在此脚本中使用该方法。

这是一个完成所有工作的脚本：[undrain-good-nodes.sh](./undrain-good-nodes.sh)

现在你可以直接运行此脚本，任何基本上准备好服务但目前处于排空状态的节点将切换到 `idle` 状态，并可供用户使用。

## 修改作业的时间限制

要设置新的时间限制，例如 2 天：
``` 
scontrol update JobID=$SLURM_JOB_ID TimeLimit=2-00:00:00
```

要向先前的设置添加额外的时间，例如再增加 3 小时。
``` 
scontrol update JobID=$SLURM_JOB_ID TimeLimit=+10:00:00
```

## 当 SLURM 出现问题时

分析 SLURM 日志文件中的事件日志：
``` 
sudo cat /var/log/slurm/slurmctld.log
```

例如，这可以帮助理解为什么某个节点在预定时间之前就取消了其作业或完全被移除。