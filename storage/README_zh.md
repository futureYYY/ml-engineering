# README - 中文翻译

# 存储：文件系统和输入输出

## 机器学习的输入输出需求

在机器学习工作负载中有三种不同的输入输出需求：

1. 你需要能够快速喂入DataLoader——（超级快读取，不关心写入速度）——需要可持续数小时甚至数天的负载。
2. 你需要能够快速写入检查点——（超级快写入，中等速度读取，因为你会多次恢复）——需要突发写入能力——你不希望训练长时间阻塞（除非你使用某种CPU卸载来快速解除训练阻塞）。
3. 你需要能够加载和维护你的代码库——（中等速度读写）——这还需要共享，因为你希望所有节点看到相同的代码库——因为它只会在开始或恢复时发生，所以发生的频率很低。

如你所见，这三种需求在速度和可持续负载方面有着非常不同的要求，因此理想情况下，你应该有三个不同的文件系统，每个都针对所需的用例进行优化。

如果你有无限的资金，当然可以得到一个单个的超快读取、超快写入的文件系统，可以在几天内不间断地运行。但对于大多数人来说，这是不可能的，因此选择两种或三种不同类型的分区，最终支付的成本会少得多，这是一个更明智的选择。

## 术语表

- NAS：网络附加存储
- SAN：存储区域网络
- DAS：直接连接存储
- NSD：网络共享磁盘
- OSS：对象存储服务器
- MDS：元数据服务器
- MGS：管理服务器

## 选择哪种文件系统

**分布式并行文件系统是最快的解决方案**

分布式并行文件系统显著提高了性能，在数百到数千个客户端同时访问共享存储的情况下表现尤为突出。它们还大大减少了热点问题（某些数据被访问的频率远高于其他数据）。 

我有经验的两个高性能并行文件系统是：

- [Lustre FS](https://www.lustre.org/)（开源）（[维基](https://wiki.lustre.org/Main_Page)）
- [GPFS](https://en.wikipedia.org/wiki/GPFS)（IBM），最近更名为IBM Storage Scale，之前称为IBM Spectrum Scale。

这两个解决方案都有超过20年的历史。两者都符合POSIX标准。创建这些解决方案并不简单——你必须设置一个专门用于这些文件系统的独立集群，其中包含多个仅用于这些文件系统的CPU虚拟机——只有这样你才能挂载这些文件系统。相比之下，较弱的云提供的“内置”解决方案只需要回答几屏的问题就可以激活。而在创建存储集群时，选择哪些虚拟机用于哪种功能是一门科学。例如，这里有一个关于GCP上的Lustre指南（https://cloud.google.com/architecture/lustre-architecture）。

案例研究：在JeanZay高性能计算（法国），我们在384个进程中并行保存了2.3TB的检查点，仅用了40秒！这快得惊人——而且是在NVME驱动器上的GPFS。

NASA的集群在使用Lustre时有一长串需要注意的问题（https://www.nas.nasa.gov/hecc/support/kb/lustre-best-practices_226.html）。

GPFS的一些非常有用的优点：
- 如果你有很多小文件，你可能会很快耗尽索引节点（使用`df -i`检查）。GPFS 5.x版本永远不会耗尽索引节点，它会根据需要动态创建更多。
- GPFS不会出现Lustre的问题，即如果其中一个子磁盘满了且没有及时重新平衡，则在使用到80%时可能会耗尽磁盘空间。你可以可靠地使用分配存储的全部100%。
- GPFS不使用中央元数据服务器（或元数据服务器集群），在处理小文件时这通常会成为瓶颈。就像数据一样，元数据也由存储集群中的每个节点处理。
- GPFS自带原生的NSD客户端，优于通用的NFS客户端，但任一都可以与之配合使用。

其他我没有直接经验的并行文件系统：

- [BeeGFS](https://www.beegfs.io/)
- [WekaIO](https://www.weka.io/)
- [DAOS](https://docs.daos.io/)（分布式异步对象存储）（英特尔）
- [NetApp](https://www.netapp.com)

大多数云提供商至少提供这些文件系统的一种实现，但并非所有云都提供。如果你的云提供商不提供这些文件系统之一，并且他们没有足够快的替代方案以满足你的需求，你应该重新考虑。

## 还可以接受的解决方案

许多云提供商提供了许多还可以接受的解决方案（<云共享存储解决方案>）。在承诺任何方案之前，请认真对这些方案进行基准测试。这些方案通常对于处理大文件相当不错，但对于小文件则不太合适。

案例研究：截至本文撰写时，使用GCP的Zonal FileStore NFS解决方案时，`python -c "import torch"`执行需要20秒，这非常慢！一旦文件被缓存，它只需约2秒。使用预构建的Python包安装conda环境可能需要20-30分钟！我们开始使用的这种解决方案给我们的工作带来了极大的痛苦和反效果。这对有很多Python包和conda环境的人来说影响很大。但是，当然，GCP也提供了更快的解决方案。

## 远程文件系统客户端

你需要选择哪个客户端来连接文件系统到你的虚拟机。

最常见的选择是：[NFS](https://en.wikipedia.org/wiki/Network_File_System)——它已有40年的历史。它引入了额外的开销，减慢了速度。所以如果有原生客户端支持你的虚拟机，使用它将比NFS获得更高的整体性能。例如，GPFS带有[NDS](https://www.ibm.com/docs/en/linux-on-systems?topic=configurations-network-shared-disk-nsd)客户端，优于NFS。

## 文件块大小

如果你使用的文件系统使用16MB的块大小，而你的文件平均大小为16KB，那么你将使用的磁盘空间是实际使用的1000倍。例如，你将看到100TB的磁盘空间被使用，但实际上只使用了100MB的磁盘空间。

注释：在Linux上，本地文件系统通常使用4KB的块大小。

因此，你可能有两个非常不同的需求，并需要两个不同的分区来优化不同的需求。

1. 数千到数百万个微小文件——4-8KB块大小
2. 几个大文件——2-16MB块大小

案例研究：Python在处理数千个微小文件方面非常糟糕，如果你有很多conda环境，你可能会在某些情况下耗尽索引节点。在JeanZay高性能计算中心，我们不得不请求一个专用的分区来安装所有的conda环境，因为在正常的GPFS分区上我们不断耗尽索引节点。我认为问题在于那些GPFS分区配置了16MB的块大小，这不适合4KB大小的文件。

好消息是现代解决方案开始引入动态块大小。例如，最新的GPFS支持子块。因此，例如，可以将GPFS配置为2MB的块大小，8KB的子块，然后将微小文件打包在一起作为子块，从而不会浪费太多磁盘空间。

## 分布式存储服务器与客户端的距离

使用共享分布式存储的集群应该将存储服务器放置在靠近使用这些服务器的集群的位置。如果运行存储服务器的虚拟机位于多个交换机之外，IO延迟可能会很高，交互式使用存储可能会令人沮丧地缓慢。例如，当你尝试运行`du`和其他访问大量文件元数据的工具时。

因此，如果你有控制权，请向云提供商要求尽可能将仅用于CPU的存储服务器虚拟机分配在网络距离上尽可能接近你的加速器虚拟机。

## 云共享存储解决方案

以下是各种云提供商提供的共享文件系统存储解决方案：

- [GCP](https://cloud.google.com/architecture/filers-on-compute-engine)
- [Azure](https://learn.microsoft.com/en-us/azure/virtual-machines/disks-shared)
- [AWS](https://aws.amazon.com/what-is/nas/#seo-faq-pairs#how-can-aws-help-with-storage-solutions)


## 本地存储优于云存储

虽然云存储更便宜，但在训练过程中动态获取和处理训练数据流的想法在数据量巨大的情况下存在很多潜在问题。

同样的，动态将检查点卸载到云端也是如此。

最好是有足够的本地磁盘空间用于数据加载。

对于检查点，应该有足够的本地磁盘空间以快速可靠地保存检查点，然后通过crontab作业或slurm作业将其卸载到云端。始终保留最后几个检查点以供快速恢复，以防任务崩溃，否则从云端获取检查点进行恢复将非常昂贵。

案例研究：由于我们几乎没有本地存储，并且由于是多模态数据，所以我们不得不在IDEFICS-80B训练期间使用云存储。我们花了数周时间试图使这个解决方案变得健壮，但最终效果不佳。最大的问题是当时很难跟踪DataSampler的随机数生成器状态，因为我们使用的解决方案并没有在这方面做出努力。因此，花费大量时间创建的数据被浪费（未使用），并且很多数据被重复，所以我们没有一个独特数据的完整周期。

在某些情况下，人们找到了一些好的解决方案来处理基于云的数据集，我个人还没有获得顺畅的体验，这就是为什么我倡导本地存储。如果你找到了一种良好的流媒体解决方案，该方案可以在不丢失数据和重复数据的情况下正确恢复，不需要巨大的本地工作者，那么它可能会工作良好。

## 警惕你经常只能获得你支付存储的80%

在计算节点上使用的分布式共享存储有一个隐性问题。由于大多数用于构建大型文件系统的物理磁盘只有0.3-2TB大小，任何一个这样的物理磁盘可能会在联合存储达到满载之前就先满。因此，它们需要持续的再平衡，以确保不会出现某个磁盘99%已满而其他磁盘只有50%已满的情况。由于再平衡是一个成本高昂的操作，类似于大多数编程语言的垃圾回收，它很少发生。因此，如果你运行`df`命令并报告90%已满，很可能任何程序在任何时候都会失败。

从与IO工程师交谈得知，接受的现实（不知为何客户没有被告知）是，分布式大型存储中大约只有80%是可靠的。

这意味着如果你想拥有100TB的可靠云存储，实际上你需要购买125TB的存储，因为80%的存储量将是100TB。所以你需要计划支付比你实际需要的25%更多的费用。我不确定为什么客户应该为技术缺陷买单，但这正是现状。

例如，GCP声明只有[89%](https://cloud.google.com/filestore/docs/known-issues#capacity_errors_before_reaching_full_provisioned_capacity)的存储是可靠的，尽管在我那里存储在达到83%时已经失败过一次。谷歌披露这个问题值得称赞，尽管不是在用户购买存储时告知。也就是说，我们建议您购买12%更多的存储，因为您可以可靠地使用89%的存储。

我还与[Sycomp](https://sycomp.com/)工程师交谈，他们提供托管的IBM Storage Scale（GPFS）解决方案，据他们说，GPFS没有这个问题，100%的存储都是可靠的。

在某些设置中，如果您通过云提供商API进行备份（而不是直接在文件系统上），它们可能会使用相同的分区，并且当然会消耗磁盘空间，但在运行`df`时它不会显示真实的磁盘使用情况——它可能不会包括备份。因此，如果您的备份占用了分区的50%。

无论您选择哪种存储解决方案，都要询问提供商可以可靠使用的存储量，以免以后出现意外。

## 警惕在某些云提供商中备份使用相同的分区

对我来说这毫无意义，但有些提供商在使用其工具备份分区时，备份会占用同一分区的空间。在某些提供商中，直到您用尽磁盘空间时才发现这种情况，而您实际只使用了分配分区的30%。在这些提供商中运行`df`毫无意义，因为它会告诉您可用的磁盘空间，但它不会包括任何备份。因此，您无法了解实际情况。

如果您开始制作备份，突然一切失败，因为所有进程都无法写入，但`df`报告使用率为30%，您现在就会知道为什么会这样。快照也会使用相同的分区。

假设您支付了100TB的分区，您使用了95TB，现在想备份它——好吧，您不能这样做——即使压缩，95TB的数据放到哪里呢，如果它只有5TB的剩余空间？

当我发现特定的解决方案具有这种不直观的行为时，我会添加有关如何查看实际磁盘使用率的提示：
- [GCP FileStore](https://cloud.google.com/filestore/docs/monitoring-instances#free-raw-capacity-percent)（但不适用于基本层）

## 别忘了校验和

在同步数据到和从云存储时，请务必研究您使用的工具是否检查校验和，否则您可能会在传输过程中收到损坏的数据。一些工具会自动检查校验和，而另一些则需要启用此功能（因为这通常会带来额外的计算成本和传输减速）。最好慢一点，但要安全。

这些通常是MD5和SHA256校验和。如果您的环境安全，通常MD5就足够了，但如果需要额外的安全性，可以使用SHA256校验和。

