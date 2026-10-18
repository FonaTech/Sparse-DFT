# Sparse-DFT Driver Skills

技能名称为 `sparse-dft-driver`，驱动脚本为 `sparse_dft_driver.py`，调用真实的 `sparse_dft` / `sparse-dft` / `sdft_*` / `SDFT_*` 接口。

[技能入口](skills/sparse-dft-driver/SKILL.md) · [稀疏化操作](skills/sparse-dft-driver/references/sparsity.md) · [38 个科学工作流案例](skills/sparse-dft-driver/case-studies/INDEX.md) · [实际验证报告](validation/REPORT.md)

接口目录覆盖 22 个顶层 CLI 命令、21 个 API 分支、377 个公开 Python 定义、53 个原生目标和 11 个源码脚本。

## 安装

Python ≥3.11。完整安装相对资源，默认不覆盖已有目标：

```bash
python3 skills/sparse-dft-driver/scripts/install_skill.py --skills-dir ~/.codex/skills
```

迁移旧名称时显式保存备份（backup-dir 必须位于 skills-dir 外）：

```bash
python3 skills/sparse-dft-driver/scripts/install_skill.py \
  --skills-dir ~/.codex/skills --migrate-from dftx-driver \
  --backup-dir /actual/new-backup-directory
```

可使用 `--skills-dir` 指定其他运行环境；Clouds_Coder 可加 `--clouds`。本包不修改其他科学技能。新安装下一轮对话可用。

## 调用

```text
$sparse-dft-driver 使用本地 Sparse-DFT 和 PAW，预览 Si 的密度谱，执行 off/auto/S1–S6 对照，检查收敛和实际误差，再复用检查点生成能带和 DOS。
```

先设置真实 SDFT_SOURCE、SDFT_SCF_EXE、SDFT_PAW_ARCHIVE；完整示例和 JSON 契约见技能引用文档。`examples/` 的路径和数值是待替换的输入形状，不代表已收敛设置。软件使用 `scripts/install.py` 构建；技能不捆绑二进制、PAW 或凭据。

案例 scaffold 生成设计与任务矩阵，不等于运行计算。批量执行使用 `run_recipe.py`；稀疏化对照使用 `sparsity_workflow.py`；体积结构/拟合使用 `eos_workflow.py`。其他派生问题按案例指定的实际原始量与公式编写项目分析脚本，再以 project-python 模式记录执行。

测试命令、环境、通过项、未运行项和真实结果摘要保存在 validation/；完整原始计算保留在本次任务工作目录，打包时排除本地 PAW 数据。
