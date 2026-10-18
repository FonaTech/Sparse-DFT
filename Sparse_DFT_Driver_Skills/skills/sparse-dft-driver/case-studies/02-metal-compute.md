# 02 · Apple Metal 计算与实际 GPU 调度

类型：原生。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

在 Apple 设备上选择 Metal 构建，完成既定 SCF/NSCF 任务并读取实际 GPU 使用证据。

## 输入与结构策略

- 支持 Metal 的 macOS 设备
- 同版本源码和 Metal 构建目录
- 结构、PAW、计算参数及主机线程预算

安装器 backend 使用 metal，运行环境使用 mps。沿用项目已确认的数值方案，不因 GPU 调度变化修改物理输入。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 建立 Metal 安装 | tool:scripts/install.py --backend metal | 可运行 Metal 内核 |
| S02 | 检查编译功能 | backend.info | metal_compiled 与实际设备条件 |
| S03 | 执行 SCF 和冻结路径 | job.run + bands.run | 收敛 SCF、真实 NSCF 本征值 |
| S04 | 汇总实际调度 | project-python:read performance JSON | device、GPU 调用计数与各阶段耗时表 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

读每次 OUTPUT.performance.json，而不是凭 macOS 平台或 backend.info 的保守 active 字段判断。嵌套计时不直接求和，CPU 最终处理属于实际混合执行。

- SDFT_COMPUTE_BACKEND=mps
- 显式 GPU_FFT/GPU_PBE 选择与线程环境
- 相同输入和独立输出

## 失败与恢复

- 设备不可用时定位实际后端错误
- 小任务 GPU 调度为零时如实记 CPU 执行
- 冻结 SCF 来源不一致时重新选匹配结果

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接计算结果和设备/调度表；性能解释读取案例 29。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

请求计算完成，GPU 实际 dispatch 与结果来源可核实，未调用 GPU 的部分也明确记录。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 02 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
