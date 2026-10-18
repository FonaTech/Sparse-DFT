# 16 · 固定晶胞离子弛豫

类型：原生。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

在固定晶胞下优化原子位置，包括真实 Selective Dynamics 可移动子空间。

## 输入与结构策略

- 初始周期结构、PAW 与约束
- 同目录的 scf/relax_step 二进制
- force tolerance、step bounds 与 SCF 设置

非正交晶胞中 T/F 表示 Direct 晶格方向；保留限制，不把它们当 Cartesian 轴固定。每次接受态记录真实能量与哈希。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 核对初始结构和约束 | structure.parse | 可移动原子/方向记录 |
| S02 | 执行解析优化 | relax.ionic | relax_result 与 checkpoint |
| S03 | 必要时兼容重启 | relax.ionic with driver --resume | 从最后 accepted state 继续 |
| S04 | 导出最终结构和轨迹 | project-python:relax analysis | 几何、力、能量与接受步表 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

收敛检查采用可移动子空间的每原子力范数。line-search trial 和 accepted steps 分开，不把一次试探写成已接受结构。

- force_tolerance 单位 eV/Å
- max_displacement 和步数预算
- smooth/onsite 和 cutoff 固定

## 失败与恢复

- SCF 失败保留最近接受态
- 输入/PAW/二进制变化用新目录
- 约束导致剩余受限力不误判活动空间未收敛

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接最终结构作为下游静态/能带输入，并记录 initial→accepted lineage。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

活动空间达到用户力阈值，最终结构、历史和兼容重启信息完整。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 16 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
