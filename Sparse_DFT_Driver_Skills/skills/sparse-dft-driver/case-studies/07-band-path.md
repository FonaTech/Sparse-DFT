# 07 · 高对称路径生成与标签保持

类型：原生。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

将用户确定的晶体路径顶点转换成完整 NSCF 点列，保留段端点和图中标签。

## 输入与结构策略

- 实际晶胞和 reciprocal convention
- 所选顶点坐标、标签与路径顺序
- 每段点数和是否重复公共顶点

路径顶点来自本项目实际晶体约定。make-band-path 只插值，标签由真实点索引写入 path mapping；不自动发明高对称路径。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 验证顶点与晶格 | structure.parse + project-python:path input | 真实 vertices JSON |
| S02 | 生成显式点列 | make-band-path | path.json 与实际总点数 |
| S03 | 写标签与端点索引 | project-python:path labels | 与点数组同长的 labels |
| S04 | 执行/分析路径 | bands.run + bands.analyze | 实际 kpoints 与一致标注 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

公共顶点是否重复影响索引和图中的段断点。波矢距离用实际倒格子计算，图中折线路径长度不能代替物理采样一致性。

- 2–32768 个有限路径点
- 每段点数与公共顶点规则
- labels 长度和实际点列一致

## 失败与恢复

- 标签越界修路径输入
- 路径和晶格约定不一致先换算
- 同长度不同路径不当作等价

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接路径文件、标签、晶格来源与能带 artifacts。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

路径已实际生成并执行，标签对应真实点索引。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 07 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
