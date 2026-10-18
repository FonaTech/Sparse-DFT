# 13 · 外部自旋、DOS 与投影谱解析

类型：原生解析功能。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

读取已有 EIGENVAL、vasprun.xml、DOSCAR 和 PROCAR，完成自旋通道与投影谱分析。

## 输入与结构策略

- 用户提供的完整外部输出
- 相同作业的结构、费米能和自旋约定
- 所需元素/轨道/通道与图目标

本案例驱动的是解析与分析功能，无需把外部自旋计算改由非自旋 Sparse-DFT 内核求解。原始输入完整保存，文件关联到同一外部作业。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 载入能带格式 | sparse_dft.bands:load_bands | 能量、占据、spin channels |
| S02 | 解析 DOS 和投影 | sparse_dft.bands:parse_doscar + parse_procar | 原始通道与投影数组 |
| S03 | 生成带/DOS/投影报告 | bands.analyze | 有来源的 analysis.json |
| S04 | 导出图和精确数据 | project-python:analysis table | 元素/轨道/自旋映射表 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

保留上下自旋的能量/占据列，记录各投影通道的来源与归一化。VASP XML 使用最终计算的谱和实际 reciprocal basis；不能把第一轮迭代当最终结果。

- 外部文件属于同一几何和状态
- 费米能一致
- 投影标识与原子索引一致

## 失败与恢复

- 损坏/截断文件回到原输出
- 部分占据不误判绝缘体
- 能量列与占据列混淆时核查解析结果

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接原始外部文件哈希和解析后的通道数据，保留外部引擎 provenance。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

所选格式解析完整，实际自旋和投影列正确传入图与表。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 13 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
