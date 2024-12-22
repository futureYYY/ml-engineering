# users - 中文翻译

## SLURM 用户指南

## 快速开始

只需复制此 [example.slurm](./example.slurm)，并根据需要进行调整。

## SLURM 分区

在本文件中，我们将使用以下两个集群名称的示例设置：

- `dev`
- `prod`

要找出节点的主机名及其可用性，请使用：

```
sinfo -p dev
sinfo -p prod
```

Slurm 配置位于 `/opt/slurm/etc/slurm.conf`。

要查看所有分区的配置：

```
scontrol show partition
```

## 资源分配等待时间

```
squeue -u `whoami` --start
```

将显示任何待处理作业预计何时开始运行。

如果其他人在此之前取消了他们的预订，它们可能会更早开始。

## 通过依赖关系请求分配

要在当前正在运行的任务结束后（无论它是否仍在运行或尚未开始）调度新任务，可以使用依赖机制，通过告知 `sbatch` 在当前正在运行的任务成功完成后启动新任务，使用：

```
sbatch --dependency=CURRENTLY_RUNNING_JOB_ID tr1-13B-round1.slurm
```

使用 `--dependency` 可能比使用 `--begin` 会减少等待时间，因为如果 `--begin` 指定的时间允许甚至几分钟的延迟，调度器可能会先开始一些优先级较低的其他任务。这是因为调度器会忽略任何带有 `--begin` 的任务，直到指定时间到达。