# 14 · Sparse-DFT 与现有参考能带比较

类型：原生。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

对实际已有的参考谱与 Sparse-DFT 谱比较路径、带边和能量差，作为用户所需的分析输出。

## 输入与结构策略

- 实际 reference/candidate 本征值文件
- 相同结构、路径、波矢约定与谱范围
- raw/fermi/vbm 对齐选择

先检查 path geometry 和谱维度。输出 raw 与指定对齐的差异，保留对齐平移；不把不同路径强行插值成一个对照。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 载入两个谱 | sparse_dft.bands:load_bands | reciprocal/path metadata |
| S02 | 检查几何和通道 | bands.compare | 可比性结果或明确不匹配错误 |
| S03 | 计算指定对齐差异 | bands-compare --align none/vbm/fermi | 真实误差与带边表 |
| S04 | 生成对照图 | project-python:matched plot | 相同几何轴的差异图 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

用实际算法返回的误差指标与能量窗口，记录带数与点数。对齐后的误差不能替代 raw 能量偏移信息。已有精度验证可直接引用，本案例只执行请求的谱分析。

- max_energy_ev 和选取窗口
- 相同路径原点/段几何
- spin channels 和带对应

## 失败与恢复

- 同长度不同路径明确拒绝
- 带数不一致先限定共同目标谱
- 缺 reciprocal 信息先取得真实晶胞

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接比较表与参考/candidate 哈希，可并入既有精度证据或论文分析。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

指定可比谱已完成差异分析，无法比较的输入给出准确原因。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 14 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
