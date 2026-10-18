# 23 · 非磁性界面结构与粘附能

类型：原生＋项目差分。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

对给定非磁性、非极性界面候选完成 registry 弛豫和一致参照的粘附能分析。

## 输入与结构策略

- 真实两相晶胞及界面匹配方案
- 共同 in-plane cell、面积和界面数
- 分离参照、约束与候选 registry

构造 joined 与相同应变状态 separated 参照，保持面积、原子计量和固定层规则。不同失配方案独立记录，不能把 strained constituent 参照换成 relaxed bulk。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 准备 registry 和参照 | project-python:interface structures | joined/separated 结构表 |
| S02 | 优化活动界面原子 | relax.ionic | 各真实 registry 几何 |
| S03 | 统一静态能量 | job.run | joined 和各 constituent 能量 |
| S04 | 计算粘附能/面积 | project-python:adhesion calculation | Wad 与 registry 排序 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

正粘附功定义 Wad=[Esep,A+Esep,B−Ejoined]/(ninterfaces·A)。面积和界面数来自实际周期模型。只做可由能量与几何完成的粘附分析，不声称已计算未实现的势 lineup 或 Bader 转移。 能量字段保持一致：internal_energy_ev 为 E，total_energy_ev 在有限展宽时为 F=E−TS。

- 共同应变与参照状态
- 面积/界面数/计量
- 约束和层间初始距离

## 失败与恢复

- 分離参照计量不符修模型
- 某 registry 重构则报告最终 motif
- 极性补偿需求超出当前接口时选择本模型可处理输入

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接粘附能、面积和真实几何，作为界面或 mechanics 主线输入。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

界面候选和匹配分离参照均完成，排序和归一化可重算。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 23 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
