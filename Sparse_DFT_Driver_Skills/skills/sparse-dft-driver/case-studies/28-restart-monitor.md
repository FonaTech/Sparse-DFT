# 28 · 作业进度、失败处理与兼容重启

类型：原生＋驱动记录。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

保持长作业可观测，处理异常并从可兼容状态继续，而不重复启动已有任务。

## 输入与结构策略

- 实际进程/session 与结果目录
- driver run.json 和 iteration JSONL
- 用户已有任务范围与可续算 checkpoint

先确定实际进程身份和输出关联，再查看完整 JSONL 迭代；只对已结束/确需续算的任务发出下一次调用。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 查看进度和最新完整迭代 | driver monitor | 残差、记录数与结果状态 |
| S02 | 核实进程/session | host process/session tool | 真实 running/ended 信息 |
| S03 | 处理失败与兼容 checkpoint | relax.ionic/relax.cell --resume | 最后接受态继续 |
| S04 | 同步 recipe 和研究状态 | run_recipe.py --resume-state | 输入/输出哈希一致的状态 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

进程退出、SCF 收敛、geometry 停止状态分别报告。只有实际 completed outputs 的哈希匹配才跳过 recipe 步骤；PID 数字本身不证明存活。

- 最后半行不作完成迭代
- 重启签名与实际源文件一致
- 一次任务对应一个结果目录

## 失败与恢复

- 进程身份未知先查 session
- 结果已存在不静默覆盖
- 改变二进制/输入时新建状态而非强续算

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接 current gate、branch status 和唯一 next action。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

本任务进度及最后结果状态明确，可续算任务恢复并保留历史。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 28 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
