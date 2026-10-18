# 08 · 多 k 点自洽静态计算

类型：原生。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

对真实周期结构完成完整 PAW-PBE 自洽计算，输出能量、占据、密度、一中心态与导数。

## 输入与结构策略

- 结构与本地 PAW
- 既定 cutoff、fine-density 与采样
- 电子迭代、混合和线程设置

使用独立 output 保存每种计算配置。完整 PAW 路线明确开启 closure；所需占据与 smearing 设置随本项目状态记录。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 准备结构和 PAW | job.prepare | resolved_input、selection 与本地 cache |
| S02 | 运行已定采样的 SCF | job.run | scf.json 与 scf_result.json |
| S03 | 读取密度/占据/能量字段 | project-python:SCF extraction | 单位明确的静态结果表 |
| S04 | 汇总作业记录 | research_bridge.py handoff | methods、evidence 与研究阶段 sidecar |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

报告实际总能/自由能字段、电子数、费米能、scf_kpoints 和残差；有限 smearing 的能量比较说明选择 E 还是 F。本次收敛检查仍需执行，不重复项目软件精度验收。 能量字段保持一致：internal_energy_ev 为 E，total_energy_ev 在有限展宽时为 F=E−TS。

- SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC=1
- smooth/onsite 控制显式
- cutoff 与 density_cutoff 独立

## 失败与恢复

- 内层求解错误先分析 eigen residual
- smooth/onsite 未收敛保留失败
- 缺输出不读取旧结果冒充本次成功

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。返回可追溯 SCF 数值和实际条件。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

本次 SCF 收敛，真实字段和参数可定位，结果完成导出。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 08 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
