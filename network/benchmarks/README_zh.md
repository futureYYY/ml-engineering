# README - 中文翻译

# 网络基准测试

**工具**:

- [all_reduce_bench.py](all_reduce_bench.py) - 在大量数据上执行 `all_reduce` 操作时，用于衡量实际网络带宽的工具。这有助于了解实际情况与广告规格之间的差异。

- [all_gather_object_vs_all_reduce.py](all_gather_object_vs_all_reduce.py) - 一个快速基准测试，展示了在从进程组收集完成状态时，从 `all_gather_object` 切换到 `all_reduce` 可以获得 23 倍的速度提升。例如，在实现某种所有进程都完成的标志时。此技术通常用于同步GPU，当它们可能在不同数量的迭代中完成时，这在多DP通道推理或希望同步 `StopIteration` 事件时非常有用。参见 [all_gather_object_vs_all_gather.py](./all_gather_object_vs_all_gather.py)。

- [all_reduce_latency_comp.py](all_reduce_latency_comp.py) - 示例说明 1x 4GB 减少操作比 1000x 4MB 减少操作要快得多。