# 31 · DFT_Skills 研究阶段与结果交接

类型：驱动＋契约适配。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

将 Sparse-DFT 真实计算纳入现有研究主线，或独立建立相同格式的 DFT 项目并交接结果。

## 输入与结构策略

- 实际研究问题、claim/stage IDs
- 已有 plan/spine/manifest 或新项目目录
- Sparse-DFT 已产生的结果与项目验证依据

已有项目以其控制平面为权威，新项目才 init。执行阶段仍用 Sparse-DFT driver，不使用 VASP runner；sidecar 合并保留原 dependencies 和 claim_ids。

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
| S01 | 建立或读取项目权威 | research_bridge.py init 或现有 workflow | schema 1.1 计划与任务 |
| S02 | 完成 Sparse-DFT 阶段 | job.run/bands.run/relax methods | 真实输出 |
| S03 | 导出结果 sidecar | research_bridge.py handoff | handoff 与 stage-proposal |
| S04 | 同步研究状态 | project-file merge + upstream validator | 一致的 plan/spine/registers |

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

外部已验证精度与本次计算的 convergence、claim interpretation 分别记录。阶段依赖、单位、适用状态和下一行动沿上游契约保存。

- kind=dft engine=Sparse-DFT
- 原 claim/stage 关系
- 不覆盖另一会话 workflow

## 失败与恢复

- 已有 workflow 不重新 init
- sidecar 不整段替换既有 stage
- 状态冲突先同步真实执行证据

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。本案例本身就是 dft-workflow-orchestrator 的 Sparse-DFT 适配路线；独立运行不依赖它的安装。

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

Sparse-DFT 输出已进入一致的研究控制平面，所有交接路径与状态可定位。

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \
  --case 31 --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
