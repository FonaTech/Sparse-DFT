# 26 · 用户结构库批量计算与筛选

类型：原生＋项目调度。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

对用户真实结构库完成统一计算协议、结果收集和按目标观测量筛选。

## 输入与结构策略

- 真实结构列表、唯一 ID 和 provenance
- 统一 PAW/数值协议与筛选规则
- 并发、内存、线程和计算预算

先做数据清点和去重，再按预算生成显式 recipe。并发控制应考虑每作业内存和 OpenMP；默认串行执行器避免复用输出或超配。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 建立候选 ledger | project-python:input audit | 结构 ID、哈希与元素表 |
| S02 | 生成实际作业 recipe | project-python:recipe generation | 每候选唯一输出 |
| S03 | 逐作业计算并记录状态 | run_recipe.py with job.run/relax | 真实成功、失败和退出记录 |
| S04 | 汇总与筛选 | project-python:result collection | 目标量、条件和 shortlist |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

排名只比较同定义/同状态的观测量；失败、缺失和未计算分别记录。筛选阈值来自用户目标，不因某候选未收敛而自动改变。

- 输入和 run ID 一一对应
- 内存/线程预算
- 共同物理和统计协议

## 失败与恢复

- 一条失败停止其依赖链但不制造结果
- 重复 ID 修 ledger
- 断点续算核查 recipe 与输出哈希

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接筛选表与完整结果 ledger，研究主线保留淘汰原因。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

指定候选的状态与目标量均有来源，shortlist 可从规则和原始表重算。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 26 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
