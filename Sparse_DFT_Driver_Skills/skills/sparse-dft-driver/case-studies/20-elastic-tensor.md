# 20 · 应变—应力弹性张量

类型：原生＋项目拟合。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

使用 Sparse-DFT 应力计算小应变响应，得到约定清晰的弹性矩阵及所需模量。

## 输入与结构策略

- 已优化参考晶胞和实际对称性
- 独立应变方向、幅度与应力单位
- clamped-ion 或 relaxed-ion 定义

先建立实际 strain basis 和 Voigt order，再对每个方向取对称 ± 点。relaxed-ion 分支对每个应变晶胞做 ionic relax 后静态计算。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 生成独立应变结构 | project-python:symmetric deformation | strain matrix、参考体积与 POSCAR |
| S02 | 取得各点应力 | job.run 或 relax.ionic→job.run | 同约定 stress tensor |
| S03 | 拟合张量分量 | project-python:stress-strain fit | Cij、残差与对称性信息 |
| S04 | 输出模量和稳定性分析 | project-python:tensor analysis | 单位、约定、特征值与模量表 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

σ_i=C_ij ε_j；工程剪切采用 γ_ij=2ε_ij 时必须一致记录。将 eV/Å³ 转为目标单位，并说明参考状态残余应力处理。只输出已由独立应变覆盖的分量。

- Voigt order 和剪切定义
- fixed-ion/relaxed-ion 不混用
- 应变幅度和各点状态记录

## 失败与恢复

- 应变点发生几何跳变时缩小范围或分支
- 秩不足的拟合不输出完整张量
- 应力符号不一致先纠正约定

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接带 basis/sign/units 的 tensor 表，可进入 DFT_Skills FEM 属性交接。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

所需独立分量可识别，原应变/应力与拟合输出可追溯。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 20 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
