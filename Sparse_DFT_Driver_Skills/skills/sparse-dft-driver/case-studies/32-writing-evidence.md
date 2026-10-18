# 32 · Academic_Writing_Skills 数值、方法与图表交接

类型：驱动＋证据适配。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

从实际 Sparse-DFT 输出生成学术写作可用的原始/派生证据、方法与图表来源，不改写原计算数据。

## 输入与结构策略

- 真实 SCF/能带/弛豫及分析 artifacts
- 实际实体、参数和计量范围
- 项目已有精度验证、所需稿件证据字段

原始标量按 JSON locator 导出；派生量从保存的分析公式和输入证据键绑定。写作层复用已有精度依据，数据提取与稿件语义仍按真实来源核对。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 导出原始证据 | research_bridge.py handoff | evidence.json、methods.md |
| S02 | 生成派生证据与图数据 | project-python:derived evidence | formula、inputs、hashes 与数据图 |
| S03 | 绑定稿件 claim keys | Academic_Writing_Skills content contract | claims/cell_claims 与条件 |
| S04 | 交付可用方法/图表包 | project-file evidence review | 原始数据、图和方法来源 |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

原计算 result_method=simulated；测试人工数据始终 synthetic。原结果不附虚构 paper/DOI/page。派生记录声明 formula、源 evidence keys 和实际代码。EN/ZH 数值和范围由同一源块绑定。

- 单位、conditions、accounting_scope
- 原始和派生来源区分
- 图直接使用真实数据

## 失败与恢复

- 缺字段保持未知而不补造
- 复制项目时修路径并核哈希
- 稿件语义和证据范围不一致修正文稿

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。交接 sci-writing-orchestrator、sci-figure-design 和 sci-manuscript-build；无 companion 安装时仍生成完整原始交接包。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

请求的证据、方法和图来源已导出，可通过写作仓库字段检查并追溯到原文件。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 32 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
