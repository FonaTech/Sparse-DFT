# 25 · 静态晶相与构型排序

类型：原生＋项目差分。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

比较同组成多晶相或多构型的静态 PBE 能量，输出明确状态与计量的排序。

## 输入与结构策略

- 真实候选结构和出处
- 相同化学组成与归一化单位
- 几何优化协议和候选预算

各晶相单独 cell/ionic 优化，最后按相同协议静态计算。每 formula unit 的原子数量从真实组成取得；不同组成需显式化学势账本，另用缺陷/反应案例。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 登记候选与计量 | project-python:candidate ledger | 结构 ID 与 formula-unit 数 |
| S02 | 优化候选几何 | relax.cell + relax.ionic | 最终晶格/坐标 |
| S03 | 共同静态计算 | job.run | 各候选 E |
| S04 | 归一化并排序 | project-python:relative energy table | ΔE/atom 或 ΔE/f.u. 与排名 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

静态排序只使用该定义下 E，不加入未计算的声子/振动熵。近简并候选记录本任务参数敏感性，不用小数位数虚构误差界。 能量字段保持一致：internal_energy_ev 为 E，total_energy_ev 在有限展宽时为 F=E−TS。

- 统一组成与归一化
- 候选结构保持真实分支
- 同状态 E/F 选择

## 失败与恢复

- 弛豫合并到同相时合并候选 lineage
- 未收敛候选保留失败不强排行
- 不同 f.u. 数修归一化

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接相/构型排序、优化几何与源结果。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

所有指定可执行候选已完成或明确失败，排序的组成、状态和单位可核对。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 25 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
