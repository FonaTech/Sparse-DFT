# 11 · 冻结 SCF 复用与多路径 NSCF

类型：原生。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

对已完成的 SCF 复用真实密度与一中心状态，计算多条能带路径或不同带数。

## 输入与结构策略

- 实际收敛 scf.json 与可读 PAW caches
- 相同结构、cutoff、单位与插值约定
- 真实 NSCF 点列、labels 与带数

冻结源固定，仅改变允许的 NSCF 路径/谱设置。明确传源 cutoff 和 atomic_units，避免默认 ENMAX 与源输入不一致。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 导出并校验 frozen 状态 | sparse_dft.checkpoint:export_frozen_scf | frozen 格式与 SHA-256 |
| S02 | 逐条执行 NSCF | bands.run with scf_checkpoint | bands.json 与冻结来源 |
| S03 | 检查路径/带数/标签 | project-python:band artifact inventory | 实际采样和谱维度表 |
| S04 | 分析多路径结果 | bands.analyze | 分析 JSON 与出版图文件 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

冻结会核查缓存、结构、单位、投影插值和独立密度支持。该 candidate 的冻结接口要求保留源 JSON 中已有的诊断状态字段，因此不要为了外部精度验证擅改这些字段。

- 源 cutoff 显式
- 源 caches 存在且哈希一致
- 路径 output 独立

## 失败与恢复

- cache 路径断开恢复真实文件
- 来源参数不符回到匹配 SCF
- 缺 fine-density provenance 用真实原输出而非猜值

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接冻结身份、各路径本征值与图。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

指定路径全部完成 NSCF，复用来源与谱数据可核对。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 11 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
