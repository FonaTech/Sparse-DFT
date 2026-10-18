# Sparse-DFT Driver Skills 2.0.0

基于用户提供的 **Sparse-DFT 1.0.0** 源码更新。技能名称为 `sparse-dft-driver`，驱动脚本为 `sparse_dft_driver.py`，调用真实的 `sparse_dft` / `sparse-dft` / `sdft_*` / `SDFT_*` 接口。

[技能入口](skills/sparse-dft-driver/SKILL.md) · [稀疏化操作](skills/sparse-dft-driver/references/sparsity.md) · [38 个科学工作流案例](skills/sparse-dft-driver/case-studies/INDEX.md) · [实际验证报告](validation/REPORT.md)

接口目录覆盖 22 个顶层 CLI 命令、21 个 API 分支、377 个公开 Python 定义、53 个原生目标和 11 个源码脚本。接口发现、真实执行、数值收敛与科学精度分别记录；目录数量不是已通过计算的功能数量。

本次重点更新：

- S1–S6 独立控制、原生密度谱预览、auto/off/custom、预算/fallback/规则/计数账本及可执行对照脚本。
- 冻结 NSCF 自动配置提取：去掉 auto 控制及被软件禁止的 density_cutoff_factor，从检查点继承密度支持。
- 共线自旋、Dudarev U、均匀网格 DOS/PDOS、PAW-UPF、自动对称性路径、绘图、工作进程及 MPI/Metal 路由。
- 修复间接 API 自旋参数发现和子包/委托命令扫描；拒绝 API 实际忽略的 solver/family 等参数。
- 运行记录、输入哈希、失败退出码、批量恢复、EOS 与研究/写作证据交接。

S7 FP32 尚无软件实现，明确拒绝。软件中的占位 job.status 和未实现 results.summarize/export 使用驱动的 monitor 与 research_bridge.py 处理。UPF、投影精度、硬件及数据缺口详见验证报告；不把旧包中的声明移作新版全功能验证。

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
