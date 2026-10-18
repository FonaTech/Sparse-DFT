# 12 · 能带、真实网格 DOS 与联合绘图

类型：原生。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

从已计算谱生成能带、真实带权网格 DOS 和联合科学图。

## 输入与结构策略

- 实际 bands artifact 与结构
- converged weighted-grid SCF 或实际 DOSCAR
- 费米能、显示区间与绘图风格

能带路径和 DOS 网格分开准备；必要时为 DOS 单独运行密网格 SCF。没有 DOS 数据时按任务补做网格计算，再绘图。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 取得 DOS 网格谱 | job.run 或实际 DOSCAR 输入 | 带权 SCF/总 DOS |
| S02 | 载入真实能带 | sparse_dft.bands:load_bands | 路径与能量列 |
| S03 | 计算/解析 DOS | sparse_dft.bands:density_of_states_from_scf 或 parse_doscar | DOS 数组、归一化和来源 |
| S04 | 联合分析与绘图 | bands-analysis | analysis.json 和 PNG/PDF/SVG |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

高斯 DOS 记录 sigma、积分归一化、网格和能量区间。带隙描述注明取样域；高对称路径带隙与整个 BZ 带隙不是同一个估计量。

- DOS 只用真实网格/权重
- 费米能来源与 energy zero
- 实际 renderer 记录

## 失败与恢复

- 路径数据不冒充 DOS
- matplotlib 不可用使用原生 SVG 路线
- 外部格式列缺失回到真实数据

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接 analysis.json、图和源数值，写作引用可定位图注条件。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

请求的能带与 DOS 有真实来源，图、归一化、显示参数完整。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 12 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
