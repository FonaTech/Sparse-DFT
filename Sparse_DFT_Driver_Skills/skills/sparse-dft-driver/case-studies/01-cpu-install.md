# 01 · CPU 安装与首个静态计算

类型：原生。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

从干净源码建立可用 CPU 环境，完成用户结构的第一份静态 SCF 结果。适合迁移到新电脑或首次部署。

## 输入与结构策略

- 源码目录和可用 C++20/CMake/FFTW 环境
- Python ≥3.11 与目标安装目录
- 真实周期结构、用户 PAW 归档及线程预算

安装和计算使用不同目录。先确认结构与 PAW，再使用安装器写出的实际环境，不从旧 PATH 猜测内核。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 构建并安装 CPU | tool:scripts/install.py | installation.json 和可调用的 sdft_scf |
| S02 | 检查编译后端和线程 | backend.info | CPU/OpenMP 实际探测信息 |
| S03 | 检查结构与势选择 | structure.parse + paw.resolve | 结构、元素与 record ID 一致 |
| S04 | 执行完整静态 SCF | job.run | scf_result.json、实际输入和正常收敛记录 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

交付执行的 cutoff、网格、电子数、能量及残差，不把安装器测试数量当作材料计算结果。安装失败定位到 installation.json 中首个失败命令。

- 显式 SDFT_SCF_EXE
- OpenMP 与 BLAS 线程避免嵌套超配
- 单独运行目录及既有项目精度依据

## 失败与恢复

- 缺编译依赖时先补实际依赖
- OpenMP 请求失败不能静默改成串行
- 缺 PAW 时不能伪造 numeric.bin

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。向研究流程交接环境、二进制哈希与首个 SCF 阶段；写作只引用真实输出。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

CPU 环境可调用，所请求静态作业收敛，输入、输出、安装日志与参数可追溯。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 01 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
