# 10 · k 网格与金属展宽控制扫描

类型：原生＋项目分析。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

以总能、自由能、费米能或目标能带边为准选择当前体系的网格与 smearing。

## 输入与结构策略

- 固定周期结构和 PAW
- 网格与 Fermi–Dirac kBT 候选值
- 目标物理状态和选择阈值

先扫描 k mesh，再在合适 mesh 上检查 smearing 敏感性；保留零温与有限 kBT 的能量定义。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 构造 mesh/kBT recipe | project-python:parameter matrix | 有限网格与展宽表 |
| S02 | 执行每个配置 | job.run | 共用费米能的加权 SCF |
| S03 | 分析占据与目标量 | project-python:SCF analysis | occupations、E/F 与采样统计 |
| S04 | 记录最终协议 | project-python:decision export | 固定 mesh 与 kBT 设置 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

smearing_ev 是 kBT，bands-analysis 的 sigma 是 DOS 高斯宽度，两者分别记录。金属占据不能被误读成有带隙。 实际字段：internal_energy_ev=E，entropy_energy_ev=−TS，total_energy_ev=F=E−TS。

- 相同 cutoff/fine-density
- Gamma/MP 网格模式一致
- 零温和有限温能量账本区分

## 失败与恢复

- 占据未覆盖所需 bands 时增加带数
- 采样上限超出时重新设计矩阵
- 不同 smearing 量误混时修正分析

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。向 DFT_Skills 方法控制表交接 mesh 与占据协议。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

完成指定扫描，最终采样和展宽对目标量的影响有数据依据。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 10 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
