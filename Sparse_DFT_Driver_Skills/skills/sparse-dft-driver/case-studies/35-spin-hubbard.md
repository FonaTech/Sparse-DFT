# 35 · 共线自旋与 Dudarev U

类型：真实接口＋可复用脚本。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

共线自旋与 Dudarev U，以真实输入执行并交付可追溯结果。

## 输入与结构策略

- 逐原子初始磁矩、spin_channels=2 和显式 Ueff 模型

读取对应 references/sparsity.md、workflows.md 和实际接口签名，使用独立输出目录。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 核对输入与契约 | driver catalog | 输入身份和精确参数 |
| S02 | 执行真实功能 | job.run / sparse_dft.scf:run_scf | 两自旋谱、磁矩、Hubbard 能量和收敛记录 |
| S03 | 检查输出 | project-python:artifact review | 收敛、证书和适用范围 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

不自动套用推荐 U；保留初始态和竞争磁态。SOC、非共线与独立 U/J 不支持。

- 实际 PAW/结构与二进制哈希
- 保持请求的阈值与模型
- 记录运行与验证范围

## 失败与恢复

- 缺数据或硬件时报告具体缺口
- 未实现或忽略的控制不得宣称成功

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交付原始字段、条件、单位、SHA-256 与限制。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

两自旋谱、磁矩、Hubbard 能量和收敛记录

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 35 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
