#!/usr/bin/env python3
"""Render and materialize the executable Sparse-DFT case library from one source."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from research_bridge import scaffold, write_new

SKILL = Path(__file__).resolve().parents[1]


def library():
    return json.loads((SKILL / 'assets/case-library.json').read_text())


def render_case(case):
    rows = '\n'.join(f'| S{i:02d} | {stage[0]} | {stage[1]} | {stage[2]} |' for i, stage in enumerate(case['stages'], 1))
    bullets = lambda xs: '\n'.join('- ' + x for x in xs)
    return f"""# {case['id']} · {case['title']}

类型：{case['mode']}。这是可复用的驱动案例，未在此文档中填入任何虚构计算结果。复用有来源的项目精度记录；本次运行仍须核查输入、收敛与交付物。

## 目标与选择条件

{case['intent']}

## 输入与结构策略

{bullets(case['inputs'])}

{case['strategy']}

## 可执行任务阶梯

| 阶段 | 要完成的操作 | 实际入口 | 输出与阶段完成条件 |
|---|---|---|---|
{rows}

API 通过 `sparse_dft_driver.py api METHOD --params FILE` 调用；CLI 通过 `cli -- COMMAND ...` 调用；Python 通过 `python MODULE:FUNCTION --params FILE` 或项目内 Python 脚本调用；native/tool 使用对应 helper 模式。所有路径与数值控制来自真实项目。`project-python` 表示用已有输出做确定性计算或创建结构的项目脚本，并非新增 Sparse-DFT 原生功能。

## 控制变量与分析方法

{case['analysis']}

{bullets(case['controls'])}

## 失败与恢复

{bullets(case['failures'])}

保留失败输出与退出码。输入或二进制改变后使用新运行目录；兼容几何重启才复用目录。修复本次调用后从受影响阶段继续，不重复项目已完成的软件精度验证。

## 独立运行和 skills 联动

独立运行时由本 skill 完成上述全部 Sparse-DFT 操作与项目脚本分析。联动时保留 `dft-workflow-orchestrator` 的 claim/stage ID，将每个 Sparse-DFT 阶段记录为 `kind=dft, engine=Sparse-DFT`，分析阶段记录为 `kind=analysis`。{case['handoff']}

记录源文件 SHA-256、实际参数、单位、公式及 JSON/表格定位。项目已有精度验证作为外部验证依据保留；原始输出中的历史状态字段原样保存。需要论文时交接给 Academic_Writing_Skills，原始结果不编造 DOI 或页码。

## 完成定义

{case['done']}

## 实施入口

```bash
python3 /absolute/sparse-dft-driver/scripts/case_studies.py scaffold \\
  --case {case['id']} --project /actual/project --system "实际体系与状态"
```

生成详细案例说明、项目控制平面与 `workflow/experiment_matrix.csv`。补齐真实路径、作业参数与资源限制后按任务阶梯调用驱动。批量执行使用 [run_recipe.py](../scripts/run_recipe.py)，请求格式见 [recipe-runner.md](../references/recipe-runner.md)。
"""


def render():
    cases = library()['cases']
    folder = SKILL / 'case-studies'
    folder.mkdir(exist_ok=True)
    index = ['# Sparse-DFT 可执行案例库', '', '按科学问题或操作目标选择一个主案例，再组合必要的输入准备或分析案例。每个案例都能通过现有 Sparse-DFT 接口与项目脚本完成。软件精度采用项目已有验证，不把重复精度验收作为案例主线。', '', '案例来源是实际源码接口与可由其输出完成的计算，不包含尚无实现的 NEB、AIMD、SOC、杂化、Berry 相位或响应函数案例。', '', '## 使用方式', '', '1. 先阅读所选案例，确定真实输入和交付物。', '2. 用 `case_studies.py scaffold` 将案例写入项目计划与任务矩阵。已有研究项目则仅合并相关阶段，不重建控制平面。', '3. 按实际入口执行；批量任务用 `run_recipe.py` 保存状态、输入哈希和输出记录。', '4. 保留项目已有精度验证，本次失败只回到受影响的计算阶段。', '5. 从实际输出提取数值和分析，交接研究或写作 skills。', '', '## 案例路由', '', '| ID | 案例 | 实现形式 | 交付目标 |', '|---|---|---|---|']
    for case in cases:
        name = f"{case['id']}-{case['slug']}.md"
        (folder / name).write_text(render_case(case))
        index.append(f"| {case['id']} | [{case['title']}]({name}) | {case['mode']} | {case['deliverable']} |")
    (folder / 'INDEX.md').write_text('\n'.join(index) + '\n')
    return {'status': 'RENDERED', 'cases': len(cases)}


def materialize(case, root, system):
    root = Path(root).resolve()
    result = scaffold(root, system, case['intent'], case['deliverable'], case['done'])
    workflow = root / 'workflow'
    guide = render_case(case)
    guide = guide.replace('[run_recipe.py](../scripts/run_recipe.py)', f'[run_recipe.py](<{SKILL / "scripts/run_recipe.py"}>)')
    guide = guide.replace('[recipe-runner.md](../references/recipe-runner.md)', f'[recipe-runner.md](<{SKILL / "references/recipe-runner.md"}>)')
    guide = guide.replace('python3 /absolute/sparse-dft-driver/scripts/case_studies.py', f'python3 "{SKILL / "scripts/case_studies.py"}"')
    write_new(workflow / 'case_method.md', guide)
    write_new(workflow / 'sdft_case.json', {'case_id': case['id'], 'case_title': case['title'], 'system': system, 'status': 'design', 'validation_basis': 'Use this project\'s existing Sparse-DFT accuracy validation; no new validation result is asserted by scaffolding.', 'required_inputs': case['inputs'], 'controls': case['controls'], 'execution_stages': case['stages']})
    with (workflow / 'experiment_matrix.csv').open('x', newline='') as file:
        headers = ['stage_id', 'purpose', 'entrypoint', 'output_and_gate', 'status']
        writer = csv.DictWriter(file, fieldnames=headers); writer.writeheader()
        for i, row in enumerate(case['stages'], 1):
            writer.writerow(dict(zip(headers, [f'S{i:02d}', *row, 'planned'])))
    # Incorporate the real case roadmap, while leaving scientific controls open.
    manifest_path = workflow / 'experiment_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    stages = []
    for i, row in enumerate(case['stages'], 1):
        analysis = row[1].startswith(('project-python', 'eos_workflow.py')) or row[1] in ('bands.analyze', 'bands.compare', 'bands-analysis', 'research_bridge.py handoff')
        stage = {'id': f'S{i:02d}', 'kind': 'analysis' if analysis else 'dft', 'name': row[0], 'engine': 'project-python' if row[1].startswith('project-python') else 'Sparse-DFT', 'claim_ids': ['C01'], 'depends_on': [f'S{i-1:02d}'] if i > 1 else [], 'inputs': case['inputs'], 'outputs': [row[2]], 'controls': case['controls'], 'validation': ['本次作业收敛与输出检查；采用项目已有软件精度验证'], 'pass_condition': row[2], 'status': 'planned', 'execution_metadata': {'entrypoint': row[1]}}
        if row[1].startswith('eos_workflow.py'): stage['engine'] = 'project-python'
        stages.append(stage)
    manifest['stages'] = stages
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    plan_path = workflow / 'integrated_research_plan.md'
    plan = plan_path.read_text()
    roadmap = '\n'.join(f"- {x['id']}: {x['name']}; entry={x['execution_metadata']['entrypoint']}; gate={x['pass_condition']}" for x in stages)
    def replace_section(text, heading, body):
        pattern = rf'(?ms)^## {re.escape(heading)}\n.*?(?=^## |\Z)'
        return re.sub(pattern, lambda match: f'## {heading}\n\n{body}\n\n', text)
    plan = replace_section(plan, 'Stage Roadmap', roadmap + '\n\n' + case['strategy'])
    gates = '\n'.join(f"- {x['id']}: {x['pass_condition']}" for x in stages)
    plan = replace_section(plan, 'Stage Gates And Deliverables', gates + '\n\n采用项目已有 Sparse-DFT 精度验证；各阶段只核查本次真实输入、收敛和交付物。')
    plan = replace_section(plan, 'Evidence, Models, And Assumptions', '\n'.join('- ' + x for x in case['inputs']) + '\n\n' + case['analysis'] + '\n\n项目尚需提供上述实际路径和控制。已有精度验证可直接采用；本次 scaffold 未运行材料计算。')
    plan = replace_section(plan, 'Definition Of Done', case['done'])
    plan_path.write_text(plan)
    result['case_id'] = case['id']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    sub.add_parser('list'); sub.add_parser('render')
    q = sub.add_parser('scaffold'); q.add_argument('--case', required=True); q.add_argument('--project', type=Path, required=True); q.add_argument('--system', required=True)
    a = p.parse_args()
    try:
        cases = library()['cases']
        if a.mode == 'list': result = [{k: c[k] for k in ('id', 'title', 'mode', 'deliverable')} for c in cases]
        elif a.mode == 'render': result = render()
        else:
            case = next((c for c in cases if c['id'] == a.case.zfill(2)), None)
            if not case: raise ValueError('Unknown case ID')
            result = materialize(case, a.project, a.system)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({'status': 'FAILED', 'message': str(exc)})); return 2


if __name__ == '__main__':
    raise SystemExit(main())
