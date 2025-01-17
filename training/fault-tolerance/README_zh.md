# README - 中文翻译

容错

无论你是拥有ML训练硬件还是按小时租用，随着ML领域的快速发展，在规定时间内完成训练非常重要。因此，如果你在熟睡时其中一个GPU故障或者检查点存储空间不足导致训练崩溃，醒来后会发现许多宝贵的训练时间已经损失。

由于ML硬件的成本高得令人望而却步，提供像Web服务那样的冗余故障转移解决方案非常困难。然而，通过一些简单的技巧，使训练具有容错性是可行的。

由于大多数重要的训练任务都在SLURM环境中执行，这里会经常提到，但本章中的见解适用于任何其他训练环境。

## 总是准备比需要更多的节点

GPU设备的现实情况是它们往往会失效。有时它们只是过热并自动关闭，但可以恢复；而在其他情况下，它们可能完全损坏需要更换。

使用同一组节点几周或几个月后，这种情况往往会有所改善，因为有问题的节点逐渐被替换掉。但是，如果你幸运地收到了一批新的GPU，尤其是刚刚推出的新GPU型号，预计会有相当比例的GPU失效。

因此，如果你需要64个节点进行训练，确保你有足够的备用节点，并研究一旦备用节点不够时，你能够多快地更换故障节点。

很难准确预测应该有多少冗余百分比，但5-10%的比例应该是合理的。你越急于按时完成训练，安全余量就应该越高。

一旦你有备用节点可用，验证你的SLURM环境是否会自动从可用节点池中移除任何问题节点，以便它可以自动用好的节点替换坏的节点。

如果你使用的是非SLURM调度器，请验证它是否也能无人工干预地进行不良节点替换。

你还需要至少一个额外的节点来运行各种预防性看门狗（稍后章节会讨论），可能用于卸载检查点和执行清理工作。

## 排队多个训练任务

下一个关键步骤是确保如果训练崩溃，有一个新的任务准备好接替前一个任务。

因此，当开始训练时，不要使用：
```
sbatch train.slurm
```

而是替换为：

``` 
sbatch --array=1-10%1 train.slurm
```

这告诉SLURM预订一个包含10个任务的作业数组，如果其中任何一个任务正常完成或崩溃，它会立即安排下一个任务。

注释：在 `--array=1-10%1` 中的 `%1` 告诉SLURM按顺序启动作业数组——一次一个任务。

如果你已经开始了一个没有这个功能的训练，可以通过使用 `--dependency` 参数轻松修复，而无需中断当前任务：
```
sbatch --array=1-10%1 --dependency=CURRENTLY_RUNNING_JOB_ID train.slurm
```

假设你启动的任务如下所示：

``` 
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
     JOBID PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
       87    prod    my-training-10b  RUNNING 2-15:52:19 1-16:00:00   64    2023-10-07T01:26:28 node-[1-63]
```

你可以看到当前的 `JOBID=87`，现在可以使用它：
``` 
sbatch --array=1-10%1 --dependency=87 train.slurm
```

然后新的状态将会显示为：
``` 
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
     JOBID PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
       87    prod    my-training-10b  RUNNING 2-15:52:19 1-16:00:00   64    2023-10-07T01:26:28 node-[1-63]
 88_[10%1]   prod    my-training-10b  PENDING       0:00 1-16:00:00   64                    N/A (Dependency)
```

可以看到，一个包含10个任务的数组（`88_[10%1]`）将在当前任务（`87`）完成后或失败时立即启动。

当然，如果导致崩溃的情况仍然存在，后续任务也会失败。例如，如果存储设备已满，无论重启多少次都无法继续训练。我们稍后将讨论如何避免这种情况。

但由于训练崩溃的主要原因是GPU故障，确保自动移除故障节点，并且新任务从一组新的节点开始，可以使崩溃后的恢复过程更顺利。

在SLURM术语中，被移除的节点被赋予一个新的状态，称为“drained”。以下是一个假设的SLURM集群示例：

``` 
$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
prod*       up   infinite       4  drain node-[0-3]
prod*       up   infinite      47  alloc node-[4-51]
prod*       up   infinite      23   idle node-[52-73]
```

这里我们有47个节点正在使用（`alloc`），23个节点可用（`idle`），4个节点不可用（`drained`）。

系统管理员应定期检查被隔离的节点，修复或替换它们，然后通过将其状态改为`idle`使其再次可用于使用。

另一种方法是通过`--dependency`链接任务，如[这里](../../orchestration/slurm/users.md#request-allocation-via-dependency)所述。这两种方法也可以结合使用。

你怎么知道什么时候任务数组或链式任务不应该恢复呢？通常，如果训练循环立即退出并知道任务已完成，就不会继续运行。但你还可以添加诸如[开关](#kill-switch)等功能，以防止任务数组继续运行。

## 更倾向于固定加速器分配而非动态分配

通常在获得一组新的加速器节点时，特别是最近推出的新型加速器，很多加速器可能会失效，使得大型语言模型（LLM）的训练变得相当棘手。对于新推出的加速器，早期的失败率可能高达10%，即使在后期阶段，失败率仍然很高。记住，如果你有8个加速器，即使只有一个加速器失效，从训练程序的角度来看，就好像是所有8个都失效了。

如果你使用固定的节点分配，几个月后，坏的加速器会被淘汰，剩下的加速器失败的可能性会大大降低。尽管仍会发生，但这种情况会变得非常罕见。

确保你的提供商在加速器失效时给你新的加速器，而不是让它们冷却后再返回（字面意思）。例如，参见如何跟踪[NVIDIA GPU的UUID](../../compute/accelerator/nvidia/debug.md#如何检测你是否反复收到同一个坏节点)。这些短暂的失效很可能会在重负载下重复出现，所以你需要确保这些加速器得到真正的更换。

如果你使用动态分配，即使在一个新加速器类型发布一年后，你仍然会遇到大量失效的加速器，因为你可能会从其他用户那里得到被拒绝的节点。当然，有些云服务提供商比其他提供商更擅长及时更换坏硬件，问题是有很多加速器不会立即失效，当某个节点出现问题时，技术人员在尝试使用该节点时可能看不到任何问题。如果用户只是释放了该节点而没有报告它是坏的，而云服务提供商在将该节点交给下一个用户之前没有重新检查其是否正常工作，那么获取到坏节点的概率会非常高。

## 频繁保存检查点

每当训练任务失败时，可能会丢失许多小时的训练数据。这个问题可以通过频繁保存检查点来缓解。当训练恢复时，它将从最后保存的检查点处继续。如果失败发生在上次保存检查点后的12小时，那么12小时的训练就会丢失并需要重新进行。这对于使用数百个GPU的训练来说可能非常昂贵。

理论上，每10分钟保存一次检查点，只会丢失10分钟的训练时间，但这也会大幅延迟达到终点的时间，因为大型模型无法快速保存，如果保存时间开始成为训练的瓶颈，这种方法就变得适得其反。

根据你的检查点保存方法以及IO存储分区的速度，保存一个大型模型可能需要几十秒到几分钟不等。因此，最佳的保存频率介于两者之间。

计算很简单——测量保存检查点所需的时间，乘以你想保存的次数，看看保存检查点将对总训练时间贡献多少额外的延迟。

使用场景：在训练BLOOM-176B时，我们使用了非常快的GPFS NVMe文件系统，保存一个2.3TB的检查点仅需40秒，并且我们在大约3小时内保存一次检查点。由于我们训练了约3个月，这意味着我们保存了大约720个检查点（90天 × 24小时 ÷ 3小时）——这额外花费了8小时用于保存检查点（720次 × 40秒 ÷ 3600秒）——即占总训练时间的0.37%（8小时 ÷ （90天 × 24小时）。假设IO速度慢5倍，这在云上并不罕见，除非支付更高的IO费用，这将占到训练时间的2%，这将相当显著。

注释：如果没有大量的本地存储空间，你不得不将检查点卸载到云端，确保最频繁的两个检查点保留在本地，以便快速恢复。保留两个而不是一个的原因是，如果最后一个检查点在保存过程中因崩溃而损坏或未完成保存，这样可以确保你至少有一个可用的检查点。

虽然这种方法为训练引入了一些开销，但拥有训练检查点是非常有用的。因为这些检查点允许你在发生偏离时回滚很多步骤，也便于分析各种事件。如今，许多训练任务从训练期间单一损失衡量评估转向在每个检查点应用到每个基准上的全数据集评估。后者可以在附加节点上进行，而不减缓训练速度来进行训练期间评估。

## 基于多副本的容错

处理加速器故障的另一种方法是不保存检查点。这种方法只在训练期间至少使用两个模型副本的情况下有效。

请先回顾各种[模型并行技术](../model-parallelism)，以便更好地理解。

- 如果使用某种形式的三维模型并行，即你有张量并行（TP）和/或管道并行（PP）和/或数据并行（DP），副本的数量等于DP的程度。
- 如果使用混合ZeRO-DP并行，副本的数量等于混合副本的程度。

例如，假设你有一个使用TP=4、PP=2、DP=2的训练设置，那么你有两个副本，每个使用8个GPU（`node0`和`node1`，TP=4，PP=2 => `4*2=8`），实际上每个副本使用了一整个8-GPU节点。

此外，你还有一个备用备份节点`node2`，有8个空闲的GPU随时待命。

现在，假设在训练过程中`node0.gpu0`失效。由于你还有第二个数据完好的副本，你可以切换到备用的8-GPU节点，通过RDMA复制第二个副本的GPU数据，从而继续从上次中断的地方开始训练。这是一个非常简化的解释，因为根据故障发生时迭代循环所处的不同阶段，恢复的具体算法会涉及多个复杂细节。

当然，在大规模训练中，你可能会有上百个活跃节点和一小部分备份节点。

这种方法优于文件系统检查点保存，因为每次只会丢失一次迭代，而使用文件系统检查点保存则可能丢失数百次迭代。

我不了解有任何开源实现这种高级容错方法，但我们知道一些大公司内部使用这种方法。

## 杀死开关

在许多SLURM环境中，用户没有`sudo`权限，如果一个人启动了训练然后去睡觉，之后发现了问题，其他人不能轻易停止训练并重新启动。

这就是在BLOOM-176B训练期间的情况，我们实现了一个杀死开关来解决这个问题。机制非常简单。训练循环会在开始新迭代之前轮询特定文件是否存在，如果文件存在，则程序会保存检查点并退出，允许其他用户改变设置并重新启动。在`main`函数的最开始还增加了一个轮询，这样如果用户在睡觉时排队了一个长时间的任务数组，他们可以通过快速结束每个任务来“烧穿”这些任务。

这也在这里讨论过[这里](../../orchestration/slurm/users.md#克服缺乏组SLURM作业所有权)。

这一功能有助于减少浪费的训练时间。

## 保存开关

在提到杀死开关时，可能最好快速提一下它的表亲——保存开关。类似于杀死开关，保存开关是一种变体，不同于停止训练，如果训练循环发现保存开关文件出现——它会保存检查点，但会继续训练。它还会自动从文件系统中删除保存开关，以免在每次迭代后意外开始保存检查点。

对于那些关注训练图表的人来说，这个功能非常有用。如果在训练损失或其他训练指标中看到有趣或关键的情况，可以迅速要求训练程序保存感兴趣的检查点，以便日后能够随意再现当前情况。

此功能的主要用途是观察训练损失的尖峰和发散。

（注：更好的归类在不稳定性章节）

## 防止

避免失去训练时间的最简单方法是防止某些类型的问题发生。虽然你不能阻止GPU失效，除了确保提供足够的冷却外，你完全可以确保在接下来几天的训练中有足够的磁盘空间。这通常是通过运行计划好的看门狗来实现的，这些看门狗监控各种资源并在问题发生之前很久就向操作员发出警报。

### 计划好的看门狗

在讨论各种看门狗之前，至关重要的是你必须有一个机制来运行计划好的任务。在Unix世界中，这是通过[`crontab`设施](https://en.wikipedia.org/wiki/Cron)实现的。

这是一个例子，如何每小时运行`~/bin/watch-fs.sh`：
``` 
0 * * * * ~/bin/watch-fs.sh
```
上面的链接解释了如何配置crontab任务以在其他频率运行。

要设置crontab，执行`crontab -e`并检查哪些任务被安排`crontab -l`。

我没有详细说明的原因是因为许多SLURM环境不提供访问`crontab`设施的功能。因此，你需要使用其他方法来调度任务。

关于[Crontab模拟](../../orchestration/slurm/users.md#crontab-emulation)的部分讨论了如何实现类似crontab的SLURM模拟，以及[自持续的SLURM作业](../../orchestration/slurm/users.md#self-perpetuating-slurm-jobs)。

### 通知设施

然后你需要有一个或多个通知设施。

最简单的方法是使用电子邮件发送警报。为了让这个方法工作，你需要确保你有一种从SLURM作业发送电子邮件的方式。如果还没有可用，你可以向系统管理员请求此功能，或者你可能能够使用外部SMTP服务器提供商。

除了电子邮件，你还可以设置其他通知，比如短信提醒和/或将Slack通知发送到你选择的频道。

一旦你了解了如何调度看门狗并且有通知设施正在工作，让我们接下来讨论关键的看门狗。

### 检查作业是否运行的看门狗

最明显的看门狗是检查是否有训练SLURM作业正在运行或更多正在排队运行。

这是一个在BLOOM-176B训练中使用的示例[slurm-status.py](slurm-status.py)。如果检测到作业既没有运行也没有排队，这个看门狗会发送一封电子邮件，并且它还将检查结果输出到主训练的日志文件中。因为我们使用了[Crontab模拟](../../orchestration/slurm/users.md#crontab-emulation)，我们只需将[slurm-status.slurm](slurm-status.slurm)放入`cron/cron.hourly/`文件夹，之前启动的SLURM crontab模拟调度器就会大约每小时运行一次这个检查。

SLURM作业的关键部分是：
``` 
tools/slurm-status.py --job-name $WATCH_SLURM_NAME 2>&1 | tee -a $MAIN_LOG_FILE
```
它告诉脚本要监视哪个作业名称，并且你还可以看到它将日志记录到文件中。

例如，如果你使用以下命令启动脚本：
``` 
tools/slurm-status.py --job-name my-training-10b
```
并且当前状态报告如下：
``` 
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
  JOBID    PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
    87     prod      my-training-10b  RUNNING 2-15:52:19 1-16:00:00  64    2023-10-07T01:26:28 node-[1-63]
```
那么一切正常。但如果`my-training-10b`作业没有显示，就会发送警报。

你现在可以根据需要最小化编辑路径和电子邮件地址来适应这些脚本。如果启动作业的人不是你，那么将`whoami`替换为启动作业的用户名。只有当你自己启动了作业时，`whoami`才有效。


### 检查作业是否卡住的看门狗

如果应用程序正在进行`torch.distributed`之类的操作，并且在某次集体操作中出现挂起，最终超时并抛出异常，这会导致训练重新启动，可以发送警报说作业被重新启动。

然而，如果挂起发生在其他没有超时的系统调用中，例如从磁盘读取数据，应用程序可能会在那里挂起几个小时而没有人察觉。

大多数应用程序都会定期记录日志，例如，大多数训练每隔几分钟就会记录最近N步的状态。那么你就可以检查日志文件是否在预期的时间范围内更新，如果没有更新则发送警报。你可以编写自己的脚本，或者使用[io-watchdog](https://github.com/grondo/io-watchdog)来实现这一点。

### 低磁盘空间警报

下一个最大的问题就是磁盘空间不足。如果你的检查点很大且频繁保存，并且没有卸载到其他地方，很容易很快耗尽磁盘空间。此外，通常多个团队成员共享同一个集群，同事可能会快速消耗大量磁盘空间。理想情况下，你应该有一个专用的存储分区用于训练，但通常这很难实现。不管怎样，你需要知道何时磁盘空间不足，并采取措施释放空间。

那么警报应该在什么阈值触发呢？它们不能太早触发，否则用户会开始忽略这些警报。但如果磁盘空间百分比并不总是适用，因为如果你有一个与他人共享的巨大磁盘空间，5%的磁盘空间可能意味着许多TB的可用空间。但在一个小的分区中，即使是25%也可能只是一些TB。因此，你需要知道你多久写一次检查点，每天需要多少TB的磁盘空间以及目前可用多少磁盘空间。

使用案例：在BLOOM训练期间，我们每3小时写一个2.3TB的检查点，因此我们每天消耗2.6TB的空间！

而且，通常会有多个分区——用于检查点写的高速IO分区，以及用于代码和库的低速分区，以及其他可能使用的分区，所有这些都需要监控，如果它们的可用性对于训练不崩溃是必需的。

还有一个注意事项——当涉及到分布式文件系统时，并不是所有的文件系统都能可靠地提供你所购买的100%的磁盘空间。事实上，有些文件系统最多只能可靠地使用分配存储空间的大约80%。问题在于这些系统使用物理磁盘，它们会在预定时期或触发事件时重新平衡，因此任何一个磁盘可能会达到100%的容量，导致写入失败，这会崩溃训练进程，即使`df`报告分区的磁盘使用率为80%。我们在训练BLOOM-176B时没有遇到这个问题，但在训练IDEFICS-80B时遇到了——80%在那里成了新的100%。你如何知道自己是否有这个问题——通常你会在准备训练时发现它。

还有另一个关于inode可用性的问题，一些存储分区并没有很大的inode配额。Python包以数百到数千个小文件的形式存在，这些文件加起来占用的空间很小，但会累积成数万个文件。因此，即使你有TB级的空闲磁盘空间，也可能会突然因为没有空闲的inode而导致训练崩溃。

最后，许多分布式分区不会实时显示磁盘使用统计信息，更新可能需要长达30分钟。

注释：使用`df -ih`查看inode配额和当前使用情况。

注释：某些文件系统使用内部压缩，因此如果将文件复制到其他地方，报告的磁盘使用量可能会小于实际值，这可能会造成混淆。

这是[BLOOM-176B训练期间使用的fs-watchdog.py](./fs-watchdog.py)。如果任何存储需求阈值未满足，这个看门狗会发送电子邮件，这里是驱动它的相应[fs-watchdog.slurm](./fs-watchdog.slurm)。

如果你研究了看门狗代码，可以看到我们为每个分区都监测了磁盘使用率和inode。我们使用HPC提供的特殊配额工具来即时获取某些分区的统计信息，但这些工具并不适用于所有分区，所以我们不得不退回到使用`df`甚至更慢的`du`。因此，应该很容易适应你的使用情况。