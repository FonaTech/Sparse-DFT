# 15 · Materials Project 结构与谱快照

类型：原生＋现有工具。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

通过可用 MP 凭据获取真实结构和原始能带，并生成离线快照与批量比较输入。

## 输入与结构策略

- 可用 MP_API_KEY 或 MAPI_KEY
- 所需真实 material IDs 与 count/criteria
- 网络预算、目标快照和候选谱目录

凭据经环境提供。结构 summary 和 raw bandstructure 分别获取，原始响应与记录 ID 关联保存。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 获取结构快照 | mp.fetch_structures | structures、manifest 与来源 URL |
| S02 | 获取指定 raw bands | mp.fetch_bands | mp_bandstructures.json 与原始本征值 |
| S03 | 准备对应 Sparse-DFT 候选 | job.run + bands.run | 按实际 material ID 命名的候选 |
| S04 | 离线批量比较 | tool:compare_mp_band_batch.py | 逐材料比较及缺失/失败列表 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

检查请求数与实际记录数、谱维度和真实路径。API 超时按交互需求显式设置；原始谱缺失时不使用带隙摘要替代。

- 凭据不写公开日志
- 快照 ID 与候选 ID 一致
- 端点/路径及实际物理条件记录

## 失败与恢复

- summary-only 响应保留失败
- 缺 ID/候选不产生零误差
- 网络中断从已冻结快照继续分析

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接 dataset manifest、原始快照和比较结果，写作引用数据库出处而非编造论文。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

请求的实际数据已获取并冻结，可离线使用；未返回记录明确列出。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 15 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
