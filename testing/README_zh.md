# README - 中文翻译

## 运行测试

### 运行所有测试

```console
pytest
```

我使用以下别名：
```bash
alias pyt="pytest --disable-warnings --instafail -rA"
```

这告诉pytest：

- 禁用警告
- `--instafail` 实时显示失败情况，而不是在最后显示
- `-rA` 生成简短的测试摘要信息

需要安装：
```bash
pip install pytest-instafail
```

### 获取所有测试列表

显示测试套件中的所有测试：

```bash
pytest --collect-only -q
```

显示给定测试文件中的所有测试：

```bash
pytest tests/test_optimization.py --collect-only -q
```

我使用以下别名：
```bash
alias pytc="pytest --disable-warnings --collect-only -q"
```

### 运行特定测试模块

运行单个测试模块：

```bash
pytest tests/utils/test_logging.py
```

### 运行特定测试

如果使用`unittest`，则需要知道包含这些测试的`unittest`类的名称。例如：

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

这里：

- `tests/test_optimization.py` - 测试文件
- `OptimizationTest` - 测试类的名称
- `test_adam_w` - 具体测试函数的名称

如果文件包含多个类，可以选择仅运行特定类中的测试。例如：

```bash
pytest tests/test_optimization.py::OptimizationTest
```

将运行该类中的所有测试。

如前所述，可以通过运行以下命令查看`OptimizationTest`类中包含的测试：

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

可以按关键字表达式运行测试。

仅运行名称包含`adam`的测试：

```bash
pytest -k adam tests/test_optimization.py
```

逻辑`and`和`or`用于指示是否所有关键字都必须匹配或任一匹配。`not`可以用来否定。

仅运行名称不包含`adam`的测试：

```bash
pytest -k "not adam" tests/test_optimization.py
```

可以将两种模式组合在一起：

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

例如，要运行`test_adafactor`和`test_adam_w`，可以使用：

```bash
pytest -k "test_adafactor or test_adam_w" tests/test_optimization.py
```

注意这里我们使用`or`，因为只要其中一个关键字匹配就包括两者。

如果要确保只有同时包含两个模式的测试被包含，则应使用`and`：

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### 仅运行修改过的测试

可以使用[pytest-picked](https://github.com/anapaulagomes/pytest-picked)来运行与未提交文件或当前分支相关的测试（根据Git）。这是快速测试您的更改是否破坏任何内容的好方法，因为它不会运行您没有触碰的文件相关的测试。

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

所有测试都将从已修改但尚未提交的文件和文件夹中运行。

### 自动重新运行失败的测试

[pytest-xdist](https://github.com/pytest-dev/pytest-xdist) 提供了一个非常有用的功能，即检测所有失败的测试，并在您修改文件并持续重新运行这些失败的测试直到它们通过时等待您修复它们。因此，在修复后您不需要重新启动pytest。这个过程会重复进行，直到所有测试通过后再次执行完整的运行。

```bash
pip install pytest-xdist
```

进入模式：`pytest -f` 或 `pytest --looponfail`

通过查找 `looponfailroots` 根目录及其所有内容（递归地）来检测文件更改。
如果此值的默认设置对您不起作用，您可以在项目中通过在 `setup.cfg` 中设置配置选项来自定义它：

```ini
[tool:pytest]
looponfailroots = transformers tests
```

或者在 `pytest.ini`/``tox.ini`` 文件中：

```ini
[pytest]
looponfailroots = transformers tests
```

这将只在指定的目录中查找文件更改，这些目录相对于 ini 文件所在目录。

[pytest-watch](https://github.com/joeyespo/pytest-watch) 是这一功能的替代实现。

### 跳过测试模块

如果您想运行所有测试模块，但排除某些模块，可以通过提供要运行的测试列表显式地排除它们。例如，要运行除了 `test_modeling_*.py` 测试之外的所有测试：

```bash
pytest $(ls -1 tests/*py | grep -v test_modeling)
```

### 清除状态

CI构建和当隔离很重要时（以牺牲速度为代价），应清除缓存：

```bash
pytest --cache-clear tests
```

### 并行运行测试

如前所述，`make test` 通过 `pytest-xdist` 插件（使用 `-n X` 参数，例如 `-n 2` 来运行2个并行作业）并行运行测试。

`pytest-xdist` 的 `--dist=` 选项允许控制如何分组测试。`--dist=loadfile` 将位于同一文件中的测试放在同一个进程中。

由于执行顺序不同且不可预测，如果使用 `pytest-xdist` 运行测试套件时出现故障（意味着我们有一些未检测到的耦合测试），可以使用 [pytest-replay](https://github.com/ESSS/pytest-replay) 按相同的顺序重播测试，这应该有助于减少该故障序列。

### 测试顺序和重复

多次重复测试是好的，以便检测潜在的相互依赖性和状态相关错误（清理）。直接多次重复测试只是好在随机性中发现一些问题。

#### 重复测试

- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder)：

```bash
pip install pytest-flakefinder
```

然后每次运行每个测试多次（默认为50次）：

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

注释：此插件不能与 `pytest-xdist` 的 `-n` 标志一起使用。

注释：还有一个插件 `pytest-repeat`，但它不适用于 `unittest`。

#### 以随机顺序运行测试

```bash
pip install pytest-random-order
```

重要：`pytest-random-order` 的存在会自动随机化测试，无需配置更改或命令行选项。

如前所述，这允许检测耦合测试 - 其中一个测试的状态会影响另一个的状态。安装 `pytest-random-order` 后，它将打印用于该会话的随机种子，例如：

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

如果给定特定序列失败，可以使用该确切种子重现它，例如：

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

只有当你使用完全相同的测试列表（或根本没有列表）时，它才能重现精确顺序。

一旦开始手动缩小列表范围，你就不能再依赖种子，而必须以它们失败的确切顺序手动列出它们，并告诉 pytest 不要随机化它们，例如：

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

要为所有测试禁用洗牌：

```bash
pytest --random-order-bucket=none
```

默认情况下，`--random-order-bucket=module` 被隐含，它将在模块级别上洗牌。它也可以在 `class`、`package`、`global` 和 `none` 级别上洗牌。有关完整细节，请参阅其[文档](https://github.com/jbasko/pytest-random-order)。

另一种随机化选择是：[`pytest-randomly`](https://github.com/pytest-dev/pytest-randomly)。此模块具有非常相似的功能/接口，但它没有 `pytest-random-order` 中可用的桶模式。它也有同样的问题，一旦安装就会强加于人。

### 查看和感觉的变化

#### pytest-sugar

[pytest-sugar](https://github.com/Frozenball/pytest-sugar) 是一个改进外观的插件，添加了进度条，并即时显示失败的测试和断言。安装后会自动激活。

```bash
pip install pytest-sugar
```

要不使用它运行测试，运行：

```bash
pytest -p no:sugar
```

或卸载它。

#### 报告每个子测试名称及其进度

对于单个或一组测试（通过 `pytest` 安装 `pytest-pspec` 后）：

```bash
pytest --pspec tests/test_optimization.py
```

#### 即时显示失败的测试

[pytest-instafail](https://github.com/pytest-dev/pytest-instafail) 即时显示失败和错误，而不是等到测试会话结束。

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```

### 使用GPU还是不使用GPU

在启用GPU的设置中，要在仅CPU模式下进行测试，添加 `CUDA_VISIBLE_DEVICES=""`：

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

或者如果你有多个GPU，你可以指定哪个GPU将被 `pytest` 使用。例如，如果你有GPU `0` 和 `1`，并且你想只使用第二个GPU，你可以运行：

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

这在你想在不同的GPU上运行不同的任务时非常方便。

某些测试必须在仅CPU上运行，有些可以在CPU或GPU或TPU上运行，还有一些需要多GPU。以下跳过装饰器用于设置测试的CPU/GPU/TPU要求：

- `require_torch` - 此测试将在安装了torch的情况下运行
- `require_torch_gpu` - 作为 `require_torch` 加上至少需要1个GPU
- `require_torch_multi_gpu` - 作为 `require_torch` 加上至少需要2个GPU
- `require_torch_non_multi_gpu` - 作为 `require_torch` 加上需要0或1个GPU
- `require_torch_up_to_2_gpus` - 作为 `require_torch` 加上需要0或1或2个GPU
- `require_torch_tpu` - 作为 `require_torch` 加上需要至少1个TPU

让我们用以下表格展示GPU需求：

| GPU数量 | 装饰器                    |
|---------|--------------------------|
| `>= 0`  | `@require_torch`         |
| `>= 1`  | `@require_torch_gpu`     |
| `>= 2`  | `@require_torch_multi_gpu`|
| `< 2`   | `@require_torch_non_multi_gpu`|
| `< 3`   | `@require_torch_up_to_2_gpus`|

例如，这是一个必须在有两个或更多GPU可用时运行的测试：

```python no-style
from testing_utils import require_torch_multi_gpu

@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

这些装饰器可以堆叠：

```python no-style
from testing_utils import require_torch_gpu

@require_torch_gpu
@some_other_decorator
def test_example_slow_on_gpu():
```

某些装饰器如 `@parametrized` 会重写测试名称，因此 `@require_*` 跳过装饰器必须列在最后以正确工作。这是一个正确的用法示例：

```python no-style
from testing_utils import require_torch_multi_gpu
from parameterized import parameterized

@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

这种顺序问题在 `@pytest.mark.parametrize` 中不存在，你可以将其放在首位或末尾，它仍然有效。但它只适用于非 `unittest`。

在测试内部：

- 可用的GPU数量：

```python
from testing_utils import get_gpu_count

n_gpu = get_gpu_count()
```


### 分布式训练

`pytest` 无法直接处理分布式训练。如果尝试这样做——子进程不会做正确的事情，最终会认为自己是 `pytest` 并开始循环运行测试套件。然而，如果先启动一个正常进程，然后再启动多个工作进程并管理输入输出管道，这种方法是可以工作的。

以下是一些使用它的测试：

- [test_trainer_distributed.py](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/trainer/test_trainer_distributed.py)
- [test_deepspeed.py](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/deepspeed/test_deepspeed.py)

要直接进入执行点，在这些测试中搜索 `execute_subprocess_async` 调用，你将在 [testing_utils.py](testing_utils.py) 中找到它。

你需要至少两个GPU才能看到这些测试的实际效果：

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

(`RUN_SLOW` 是HF Transformers使用的特殊装饰器，通常会跳过耗时测试)


### 输出捕获

在测试执行期间，发送到 `stdout` 和 `stderr` 的任何输出都会被捕获。如果测试或设置方法失败，其相应的捕获输出通常会与失败跟踪一起显示。

要禁用输出捕获并使 `stdout` 和 `stderr` 正常运行，使用 `-s` 或 `--capture=no`：

```bash
pytest -s tests/utils/test_logging.py
```

将测试结果发送到JUnit格式输出：

```bash
py.test tests --junitxml=result.xml
```


### 颜色控制

禁用颜色（例如黄色在白色背景上不可读）：

```bash
pytest --color=no tests/utils/test_logging.py
```


### 将测试报告发送到在线粘贴服务

为每个测试失败创建URL：

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

这将把测试运行信息提交到远程Paste服务，并为每个失败提供一个URL。您可以像往常一样选择测试，或者添加例如 `-x` 如果您只想发送一个特定的失败。

为整个测试会话日志创建URL：

```bash
pytest --pastebin=all tests/utils/test_logging.py
```