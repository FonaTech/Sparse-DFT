# 09 · 波函数与细密度截断能扫描

类型：原生＋项目分析。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

针对当前观测量自动运行 cutoff 与 density_cutoff 控制矩阵，选定本材料的计算参数。

## 输入与结构策略

- 固定结构、PAW、采样与观测量
- 用户指定截断能候选序列
- 误差阈值、参考最密设置与预算

先保持细密度支持足够，扫描波函数；再保持波函数基组，扫描独立 fine-density。每点运行目录和控制标识唯一。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 构造两类参数矩阵 | project-python:scan recipe | 参数表与一一对应的 run IDs |
| S02 | 批量执行静态点 | job.run via run_recipe.py | 各点收敛结果 |
| S03 | 读取能量/力/应力/本征值 | project-python:observable table | 按一致单位归一化的表 |
| S04 | 选择满足目标的设置 | project-python:scan decision | 选择理由和控制图 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

对目标量定义 ΔQ=Q_i−Q_ref。总能按 cell/atom 的归一化明确；力应使用相同原子索引，应力保持相同张量和符号。扫描是在选本任务参数，不是重做软件整体精度验证。

- 除扫描轴外所有控制固定
- density_cutoff≥4×cutoff
- 自动 ENMAX 建议仅作起点

## 失败与恢复

- 某点未收敛不得插值为合格
- 归一化对象变化时重建可比表
- 参考点费用超预算先调整参数设计

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接参数选择表、原始结果与计算协议。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

完成指定参数网格并选出满足用户目标的设置；失败点有独立记录。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 09 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
