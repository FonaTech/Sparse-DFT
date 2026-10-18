# 19 · 体积扫描与状态方程

类型：原生＋项目拟合。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

对固定组成和结构分支进行体积扫描，求平衡体积、能量曲线与体积模量。

## 输入与结构策略

- 实际参考结构与组成
- 用户体积或晶格缩放序列
- 固定离子/内坐标弛豫选择与拟合范围

各晶格按保留形状的缩放或指定形变生成，记录体积和每点内坐标处理；不要把不同晶相的点拟成一个分支。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 生成体积结构集合 | eos_workflow.py prepare | 各点 POSCAR 与 V |
| S02 | 执行各点 ionic/static | job.run/relax.ionic via run_recipe.py | 收敛 E(V) 点 |
| S03 | 拟合选定 EOS 或局部多项式 | eos_workflow.py analyze (quadratic) 或 project-python 指定 EOS | 拟合代码、参数与残差 |
| S04 | 求极值和模量 | project-python:derivative of fit | V0、E0、B0 与图 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

在稳定极小值附近 B0=V0·d²E/dV²。拟合方程和单位换算留在代码，不在未采样区外推结论；比较拟合范围与阶数对所得参数的影响。 已附 eos_workflow.py，可生成实际缩放结构和 batch recipe，并以保存的真实输出完成二次拟合。使用 --energy-field 显式选择 total_energy_ev（F）或 internal_energy_ev（E）；零展宽下两者相等。离子弛豫时读取最后 accepted SCF，而非不存在的顶层能量字段。

- 统一每 cell/每 formula unit 归一化
- mesh/cutoff 协议一致
- 固定或 relaxed-ion 分支分开

## 失败与恢复

- 最低点在边界时扩大实际采样
- 相变或优化跳支拆开分支
- 未收敛点不进入拟合

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接原始 E(V) 表、拟合公式与导出参数，derived evidence 绑定每个源点。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

完成指定体积网格，平衡点位于采样域且拟合方法和参数可重算。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 19 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
