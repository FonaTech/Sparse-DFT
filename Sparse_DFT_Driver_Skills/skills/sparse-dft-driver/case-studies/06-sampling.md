# 06 · k 点推荐、显式网格与任意权重

类型：原生。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

把体系精度目标转换成实际执行的 SCF 采样，支持 Gamma、Monkhorst–Pack 和显式加权点列表。

## 输入与结构策略

- 真实晶格与周期方向
- 金属/非金属状态和任务用途
- 采样密度目标与计算预算

用 recommend_kpoints 的 spacing/metallic 选项时调用 Python。先记录推荐，再把最终 mesh 或 points/weights 明确传给 SCF。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 生成推荐 | sparse_dft.kpoints:recommend_kpoints | 网格、spacing 与推荐理由 |
| S02 | 产生目标点与权重 | sparse_dft.kpoints:uniform_kpoint_mesh | 实际 fractional 点与 weights |
| S03 | 用指定采样运行 SCF | job.run | 执行的 scf_kpoints 与 normalized weights |
| S04 | 汇总推荐和执行差别 | project-python:sampling table | 输入、推荐、执行三者一致记录 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

网格与任意点列表互斥。正有限权重由内核归一化，不能将路径点等权当作均匀 Brillouin-zone 采样。查看结果 provenance 而非仅看 prepare 的建议。

- scf_grid 与显式点二选一
- mesh_mode 明确
- 三维点坐标和正权重

## 失败与恢复

- 自动网格预算调整需保留 reason
- 非法点/权重修正输入后重新运行
- 实际仍为单 Gamma 时修正未传入的 mesh

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接执行采样和项目控制矩阵。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

最终选定采样实际执行，点数、权重与模式有输出依据。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 06 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
