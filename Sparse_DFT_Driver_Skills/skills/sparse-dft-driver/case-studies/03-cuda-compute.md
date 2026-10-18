# 03 · NVIDIA CUDA 构建与计算

类型：原生，需对应硬件。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

在有 NVIDIA 驱动与 CUDA Toolkit 的主机完成 CUDA 构建并运行现有计算入口。

## 输入与结构策略

- 真实 NVIDIA 设备、驱动和 toolkit
- 目标主机 C++/FFTW/Python 环境
- 用户结构、PAW 与计算资源预算

此案例在目标 NVIDIA 主机执行，交付真实编译与计算记录；不从另一平台的源码阅读推断已完成构建。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | CUDA 构建安装 | tool:scripts/install.py --backend cuda | installation.json、CUDA 二进制 |
| S02 | 检查编译和设备 | backend.info + native backend programs | 目标主机实际状态 |
| S03 | 运行请求的 SCF | job.run | 真实 SCF 输出 |
| S04 | 读取设备与性能记录 | project-python:read performance JSON | 实际 CUDA 调度和耗时 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

构建、设备与一次物理任务是三个操作结果。记录完整内核路径、工具链和实际 GPU 使用，项目精度依据沿用已有验证。

- SDFT_COMPUTE_BACKEND=cuda
- CPU/BLAS/设备线程配置
- 唯一输出及目标主机二进制哈希

## 失败与恢复

- 缺 toolkit 先修构建环境
- 无设备或驱动不匹配记录运行错误
- 不把 CPU 回退记录为 CUDA 执行

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接 CUDA 主机的计算结果、环境和调度记录。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

在实际 NVIDIA 主机成功构建并完成请求作业，记录可重放。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 03 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
