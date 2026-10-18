# 05 · PAW 清单、选型、还原与数值缓存

类型：原生。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

检查本地 PBE.54 JSON 归档，按真实元素和选择策略准备完整 PAW 缓存。

## 输入与结构策略

- 合法本地 manifest/index/potentials 归档
- 结构所需元素列表
- standard/semicore/gw/gw_semicore 选择依据

先列 inventory 和候选 record，再按策略 resolve。需要非 standard 家族计算时使用 Python run_scf 的 family，而不把 API 中被忽略的 family 当已执行。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 归档清点与完整性信息 | paw.inventory | 文件哈希与可用记录 |
| S02 | 按策略解析候选 | paw.resolve | 实际选择报告 |
| S03 | 还原原始记录 | paw.restore | 指定本地目的路径 |
| S04 | 解析完整数值数据 | paw.parse | numeric.bin 与 numeric_summary |
| S05 | 带指定家族准备计算 | sparse_dft.workflow:prepare_job 或 sparse_dft.scf:run_scf | 实际 PAW report 与缓存路径 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

保留 ENMAX、价电子、record ID 与缓存哈希；还原文件不进入公开 skill 分发包。PAW 元数据是 cutoff 起始依据，实际参数取执行输入。

- 所需元素必须全部匹配
- 选择家族与实际调用接口一致
- 恢复目的目录独立

## 失败与恢复

- 归档哈希不一致时停止使用损坏数据
- 缺某元素不从别的元素借用缓存
- family 选项在 API 未支持时切换 Python

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接选型报告、哈希和本地路径，不交接受限制的势内容。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

目标元素的真实势记录已解析，所需缓存可供计算入口读取。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 05 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
