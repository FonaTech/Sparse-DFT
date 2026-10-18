# 18 · 有限差分力与应变响应计算

类型：原生＋项目结构操作。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

按用户需要用中心差分计算局部力或应变响应，作为可直接交付的导数量。

## 输入与结构策略

- 真实周期结构与 PAW
- 实际扰动步长和导数方向
- 匹配 SCF 参数与差分计算预算

独立 ± 扰动配置不共享输出目录。原子位移可直接使用 finite_difference_forces；晶胞应变使用已有 strain validator 或生成结构后静态计算。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 构造/检查扰动 | project-python:displacement/strain generation | ±结构及算法来源 |
| S02 | 计算中心差分力 | sparse_dft.relax:finite_difference_forces | forces 与各 SCF samples |
| S03 | 计算选定应变响应 | tool:validate_paw_strain.py | 实际导数及 basis policy |
| S04 | 整理响应结果 | project-python:derivative extraction | 导数、步长、单位和源能量表 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

F_i=−[E(+δ)−E(−δ)]/(2δ)。应力从能量应变导数除以体积；同时变化对称 off-diagonal 元素时按实现中的双项约定处理。此案例提供请求的数值导数，不要求重复已有全软件精度验证。

- step_angstrom 与 strain 分别记录
- 相同 PAW、SCF 和采样
- 固定参考/当前 cutoff 基组策略明确

## 失败与恢复

- 任一侧 SCF 失败则不输出该导数
- cache 与扰动结构不匹配不得复用
- 剪切因子或单位不一致修分析后重算对应量

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接实际导数、原能量与坐标变换，必要时作为 elastic case 的输入。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

指定方向导数已按真实收敛能量算出，各项来源完整。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 18 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
