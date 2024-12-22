# README - 中文翻译

在SLURM环境中工作

除非你很幸运，拥有完全受你控制的专用集群，否则你可能需要与其他用户共享GPU资源。但即便如此，如果你在高性能计算（HPC）环境中工作，并且被分配了一个专用分区，你仍然需要使用SLURM。

SLURM是**Simple Linux Utility for Resource Management**（Linux资源管理简单工具）的缩写，现在称为Slurm作业管理器。它是用于Linux和类Unix内核的免费开源作业调度程序，被世界上许多超级计算机和计算机集群所采用。

这些章节不会试图详尽地教你SLURM，因为网络上有很多相关手册，但会涵盖一些有助于训练过程的具体细节。

- [SLURM 为用户](./users.md) - 在SLURM环境中进行训练所需了解的所有内容。
- [SLURM 管理](./admin.md) - 如果你需要除了使用SLURM外还管理SLURM集群，此文档中有一系列不断增加的解决方案可以帮助你更快地完成任务。
- [性能](./performance.md) - SLURM的性能细节。
- [启动脚本](./launchers) - 如何在SLURM环境中使用`torchrun`、`accelerate`、PyTorch Lightning等启动程序。