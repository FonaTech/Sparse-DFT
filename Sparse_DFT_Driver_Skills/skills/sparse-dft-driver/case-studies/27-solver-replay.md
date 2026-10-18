# 27 · 真实 H/S 快照与求解器重放

类型：原生。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

捕获并重放一次实际矩阵自由 H/S 求解，用于复现运行问题或分析迭代行为。

## 输入与结构策略

- 实际 native 输入、二进制与求解任务
- 唯一 snapshot 目录和目标调用序号
- CPU/GPU 模式及重放迭代预算

先捕获实际算符，再按快照 magic 解释 PWO1/PWO2。求解调用序号不等于 SCF 外迭代号，多 k 点尤其需读取真实记录。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 捕获实际求解状态 | native:sdft_scf with snapshot env | pwop 与 pwop.json |
| S02 | 核查快照身份 | tool:audit_solver_snapshot.py | 哈希与算符尺寸 |
| S03 | 按原容差重放 | native:sdft_solver_replay | replay JSON 与真实 residual |
| S04 | 整理迭代与耗时 | project-python:solver replay analysis | 每根结果、归一化和重放协议 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

可增加最大迭代数而保留原快照残差门槛。PWO1 的历史折叠矩阵与 PWO2 的实际 B/D/Q 不互换；重放结果作为本次算符分析交付。

- 快照目录不覆盖
- native 退出码按 replay 定义
- 同快照 identity 与原阈值

## 失败与恢复

- 文件哈希/尺寸错误先修真实文件
- fresh residual 失败保留报告
- 不能将重放输出当作完整 SCF checkpoint

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接真实算符状态和重放结果到计算诊断分支。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

目标求解可由已保存的真实算符重放，成功或失败均有可定位结果。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 27 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
