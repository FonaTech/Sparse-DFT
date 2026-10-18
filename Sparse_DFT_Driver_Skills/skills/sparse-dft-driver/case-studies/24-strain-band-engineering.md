# 24 · 应变能带工程与有效质量

类型：原生＋项目分析。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

计算非自旋材料的应变能带变化，并在孤立可追踪能带极值附近拟合有效质量。

## 输入与结构策略

- 真实参考晶胞和 strain states
- 实际 band-edge 位置/带索引
- 局部波矢 stencil、单位和拟合范围

对每个应变态先确定真实最终结构，再 SCF/NSCF。质量计算使用极值附近局部点列，不从稀疏高对称整条路径猜二阶曲率。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 生成应变结构与内坐标 | project-python:strain structures + relax.ionic | 各状态几何 |
| S02 | 取得状态密度和能带 | job.run + bands.run | 各状态真实 eigenvalues |
| S03 | 计算局部曲率点列 | bands.run with local kpoints | 同一冻结状态的局部谱 |
| S04 | 拟合曲率/质量与形变势 | project-python:band curvature fit | Hessian、mass tensor 和变化表 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

孤立带的逆质量张量为 ħ⁻²∂²E/∂ki∂kj；分数坐标先用实际 reciprocal basis 转到 Å⁻¹。能量与坐标单位换算在脚本中明确。简并/交叉不能用未经追踪的单带二次拟合。

- 真实 band index 与能量零点
- 局部 stencil 完整覆盖独立分量
- 应变改变后 reciprocal basis 更新

## 失败与恢复

- 带交叉时缩小适用域或只输出谱
- 极值未在采样域时扩展真实点列
- 不把 PBE gap 改写为实验光学 gap

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接应变—谱—曲率的完整派生链，可进入上游 band engineering 分支。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

请求应变谱与可识别的孤立带曲率完成，拟合范围和单位完整。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 24 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
