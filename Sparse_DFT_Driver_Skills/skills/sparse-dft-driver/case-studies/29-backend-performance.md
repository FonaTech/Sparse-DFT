# 29 · 完整 SCF 的后端和线程性能表

类型：原生＋项目统计。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

对同一真实任务运行指定 CPU 线程数及 GPU 配置，得到完整作业耗时与实际调度表。

## 输入与结构策略

- 固定结构、PAW、控制和候选二进制
- 指定后端/线程/重复数
- 目标硬件、竞争负载记录及计时方法

将每种 mode/repeat 放入唯一目录，按既定项目协议执行。只做请求的性能测量，已有精度验证可引用；计算本身仍要真实收敛。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 冻结控制与 mode 表 | project-python:performance recipe | 实际输入和 binary hashes |
| S02 | 执行各 mode/repeat | job.run with runtime environment | 完整 SCF 与 performance sidecars |
| S03 | 收集墙钟与 dispatch | project-python:performance collection | 后端、线程、耗时和负载表 |
| S04 | 统计速度比 | project-python:performance statistics | 重复统计与图 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

S=Treference/Tmode，CPU 并行效率 E=S/n。采用明确的重复聚合方式；SCF 墙钟与内部嵌套计时分开。微基准不冒充完整作业速度，竞争负载保留在样本标记中。

- 相同物理/数值输入
- 实际 OpenMP 与 BLAS 环境
- GPU 调度和负载来源

## 失败与恢复

- 未收敛样本不给有效速度比
- mode 发生 CPU-only 调度如实标记
- 硬件/输入变化分组比较

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接性能统计与原始计时记录；不把硬件性能结论扩展到未测任务。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

指定配置全部完成或明确失败，统计可从原计时和协议重算。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 29 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
