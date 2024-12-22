# tools - 中文翻译

# 调试工具

## 与git相关的工具


### 有用的别名

显示当前分支中所有与HEAD相比已修改的文件差异：
```bash
alias brdiff="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); git diff origin/\$def_branch..."
```

忽略空白差异，添加`--ignore-space-at-eol`或`-w`：
```bash
alias brdiff-nows="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); git diff -w origin/\$def_branch..."
```

列出当前分支与HEAD相比新增或修改的所有文件：
```bash
alias brfiles="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); git diff --name-only origin/\$def_branch..."
```

获取列表后，我们可以自动打开编辑器来加载仅新增和修改的文件：
```bash
alias bremacs="def_branch=\$(git symbolic-ref refs/remotes/origin/HEAD | sed 's@^refs/remotes/origin/@@'); emacs \$(git diff --name-only origin/\$def_branch...) &"
```


### git-bisect

（注：这是从`the-art-of-debugging/methodology.md`同步过来的内容，该文件是真正的源文件）

接下来讨论的方法适用于任何支持二分查找的版本控制系统。我们将在此讨论中使用`git bisect`。

`git bisect`有助于快速找到导致特定问题的提交。

用例：假设你使用了`transformers==4.33.0`，然后你需要一个较新的功能，所以升级到了最新版本的`transformers@main`，结果代码崩溃了。在这两个版本之间可能有数百个提交，逐个查看这些提交以找到导致崩溃的那个提交是非常困难的。以下是你可以快速找出是哪个提交导致问题的方法。

脚注：HuggingFace Transformers 的确在不经常破坏方面做得很好，但由于其复杂性和巨大规模，它仍然会发生问题，并且一旦报告，这些问题会很快得到修复。由于它是非常流行的机器学习库，因此它是一个很好的调试用例。

解决方案：对已知的好提交和坏提交之间的所有提交进行二分查找，以找到那个应该受到谴责的提交。

我们将使用两个shell终端：A和B。终端A用于`git bisect`，终端B用于测试你的软件。虽然没有技术原因要求你不能用单个终端完成这一切，但使用两个终端更容易。

1. 在终端A中克隆git仓库并安装到开发模式（`pip install -e .`）到你的Python环境。
```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install -e .
```
现在当你运行应用程序时，会自动使用这个克隆中的代码，而不是之前从PyPi、Conda或其他地方安装的版本。

为了简单起见，我们假设所有依赖项都已经安装完毕。

2. 接下来启动二分查找 - 在终端A中，运行：

```bash
git bisect start
```

3. 发现最后一个已知的好提交和第一个已知的坏提交

`git bisect`需要两个数据点才能工作。它需要知道一个较早的已知工作良好的提交（“好”），以及一个较晚的已知导致问题的提交（“坏”）。因此，如果你查看给定分支上的提交序列，它会有两个已知点和许多未知质量的中间提交：

```bash
...... orig_good ..... .... .... .... ..... orig_bad ....
------------->---------------->----------------> 时间
```

例如，如果你知道`transformers==4.33.0`是好的，而`transformers@main` (`HEAD`) 是坏的，可以通过访问[发布页面](https://github.com/huggingface/transformers/releases)并搜索`4.33.0`来找到对应于标签`4.33.0`的提交。我们发现它对应的SHA是`5a4f340d`。

脚注：通常前8个十六进制字符足以作为给定仓库的唯一标识符，但你可以使用完整的40字符字符串。

所以我们现在指定第一个已知的好提交：
```bash
git bisect good 5a4f340d
```

正如我们所说，我们将使用`HEAD`（最新的提交）作为坏的提交，在这种情况下，我们可以直接使用`HEAD`而不是查找对应的SHA字符串：
```bash
git bisect bad HEAD
```

如果你知道问题出现在`4.34.0`版本中，可以按照上述方法找到其最新的提交并用它代替`HEAD`。

我们现在准备找出导致问题的提交。

在你告诉`git bisect`好提交和坏提交之后，它已经切换到了中间的一个提交：

```bash
...... orig_good ..... .... current .... .... ..... orig_bad ........
------------->--------------->---------------->----------------> 时间
```

你可以运行`git log`来查看它切换到了哪个提交。

提醒一下，我们使用`pip install -e .`安装了这个仓库，因此Python环境会即时更新到当前提交的代码版本。

4. 好或坏

下一步是告诉`git bisect`当前提交是“好”还是“坏”：

为此，在终端B中运行你的程序一次。

然后在终端A中运行：
```bash
git bisect bad
```
如果失败，或者：
```bash
git bisect good
```
如果成功。

例如，如果结果显示为坏，`git bisect`会将最后一个提交标记为新坏提交，并再次对剩余的提交进行二分查找，切换到一个新的当前提交：

```bash
...... orig_good ..... current .... new_bad .... ..... orig_bad ....
------------->--------------->---------------->----------------> 时间
```

反之，如果结果显示为好，则会有：

```bash
...... orig_good ..... .... new_good .... current ..... orig_bad ....
------------->--------------->---------------->----------------> 时间
```

5. 重复直到没有更多提交

持续重复步骤4，直到找到有问题的提交。

一旦完成二分查找，`git bisect`会告诉你哪个提交导致了问题。

```bash
...... orig_good ..... .... last_good first_bad .... .. orig_bad ....
------------->--------------->---------------->----------------> 时间
```

如果你遵循了小提交图示，它对应的就是`first_bad`提交。

然后你可以访问`https://github.com/huggingface/transformers/commit/`并在URL后面附加提交的SHA，这将带你到该提交页面（例如`https://github.com/huggingface/transformers/commit/57f44dc4288a3521bd700405ad41e90a4687abc0`），然后链接到它所源自的PR。然后你可以通过跟进该PR请求帮助。

如果你的程序即使在有数千个提交需要搜索的情况下也能快速运行，那么你面临的是`n`次二分查找步骤，即`2**n`。因此，1024个提交可以在10步内被搜索到。

如果程序运行非常慢，尝试将其简化为一个小的复现程序，以快速展示问题。通常，注释掉大量你认为与问题无关的代码块就足够了。

如果你想查看进度，可以要求它显示剩余要检查的提交范围：
```bash
git bisect visualize --oneline
```

6. 清理

所以现在恢复git仓库克隆到你开始时的状态（最有可能是`HEAD`）：
```bash
git bisect reset
```

并在向维护人员报告问题的同时重新安装好版本的库。

有时，问题可能源于有意的向后兼容性破坏的API更改，这时你可能只需要阅读项目的文档以了解发生了什么变化。例如，如果你从`transformers==2.0.0`切换到`transformers==3.0.0`，几乎可以肯定你的代码会崩溃，因为主版本号的不同通常意味着引入了重大的API更改。

7. 可能的问题及其解决方案：

a. 跳过

如果出于某种原因当前提交无法测试 - 可以跳过它：
```bash
git bisect skip
```
然后`git bisect`将继续二分查找剩余的提交。

这在API在提交范围中发生变化并且你的程序因完全不同的原因而失败时非常有用。

你也可以尝试制作一个适应新API的程序变体，并用它替代原来的程序，但这并不总是容易做到。

b. 反转顺序

通常git期望“坏”提交在“好”提交之后。

```bash
...... orig_good ..... .... .... .... ..... orig_bad ....
------------->--------------->---------------->----------------> 时间
```

现在，如果“坏”提交发生在“好”提交的时间线上更早的位置，并且你想找到第一个修复了先前存在的问题的提交 - 你可以反转“好”和“坏”的定义 - 使用新状态可能会混淆逻辑，因此建议使用一组新的状态 - 例如，“固定”和“破坏”。以下是具体操作：

```bash
git bisect start --term-new=fixed --term-old=broken
git bisect fixed
git bisect broken 6c94774
```

然后使用：
```bash
git fixed / git broken
```
代替：
```bash
git good / git bad
```

c. 复杂情况

有时还有其他复杂情况，比如不同版本的依赖项不相同，例如一个版本可能需要`numpy=1.25`，而另一个版本则需要`numpy=1.26`。如果依赖包的版本是向后兼容的，安装新版本应该可以解决问题。但这并不总是有效。因此有时在重新测试程序之前需要重新安装正确的依赖项。

有时，当有多个不同方式破坏的提交范围时，你可以找到一个不包括其他坏提交范围的`good...bad`提交范围，或者你可以尝试按上述方法跳过其他坏提交。