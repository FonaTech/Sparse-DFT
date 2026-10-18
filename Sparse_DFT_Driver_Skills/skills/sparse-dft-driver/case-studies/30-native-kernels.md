# 30 · 原生核函数、优化器与数值工具调用

类型：原生。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

调用源码实际提供的 kernel smoke、XC、PAW ledger、optimizer 或 benchmark 工具，取得指定功能输出。

## 输入与结构策略

- 对应 CMake target 的真实二进制
- 程序实际要求的输入/PAW cache
- 所选 CPU/GPU 运行环境和用途

按 capability catalogue 找到 target 和 C++ main，读取实际参数后调用 native 模式。与固定历史实验相关的 tool 先在隔离源码中准备真实依赖。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 枚举实际程序 | driver catalog | 实时枚举的原生 target 与源码位置 |
| S02 | 读取参数和数据依赖 | source review | 目标程序实际 argv |
| S03 | 调用指定原生功能 | driver native | 真实 stdout/JSON 与退出码 |
| S04 | 解释输出范围 | project-python:tool result extraction | 指定 kernel/optimizer/ledger 数据表 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

XC 点求值、L-BFGS 方向、FFTW/BLAS/PBE 算子测试和实际 PAW账本分别按各自输入解释。这里交付被请求的工具结果，不要求重新执行整套软件精度验收。

- CMake 构建条件与实际 target 文件
- 工具输入单位和格式
- 原生退出码不套 CLI 含义

## 失败与恢复

- 缺 data cache 先提供真实缓存
- 未构建 target 回到对应构建
- 历史 tool 固定路径在隔离副本配置而非覆盖活跃源码

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接程序、参数、输入哈希及工具特定结果。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

请求的实际原生程序完成，输出和解释绑定真实功能与输入。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 30 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
