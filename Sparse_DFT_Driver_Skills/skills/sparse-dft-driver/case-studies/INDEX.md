# Sparse-DFT 可执行案例库

按科学问题或操作目标选择一个主案例，再组合必要的输入准备或分析案例。每个案例都能通过现有 Sparse-DFT 接口与项目脚本完成。软件精度采用项目已有验证，不把重复精度验收作为案例主线。

案例来源是实际源码接口与可由其输出完成的计算，不包含尚无实现的 NEB、AIMD、SOC、杂化、Berry 相位或响应函数案例。

## 使用方式

1. 先阅读所选案例，确定真实输入和交付物。
2. 用 `case_studies.py scaffold` 将案例写入项目计划与任务矩阵。已有研究项目则仅合并相关阶段，不重建控制平面。
3. 按实际入口执行；批量任务用 `run_recipe.py` 保存状态、输入哈希和输出记录。
4. 保留项目已有精度验证，本次失败只回到受影响的计算阶段。
5. 从实际输出提取数值和分析，交接研究或写作 skills。

## 案例路由

| ID | 案例 | 实现形式 | 交付目标 |
|---|---|---|---|
| 01 | [CPU 安装与首个静态计算](01-cpu-install.md) | 原生 | 可用 CPU 安装与 SCF 结果包 |
| 02 | [Apple Metal 计算与实际 GPU 调度](02-metal-compute.md) | 原生 | Metal 作业与调度报告 |
| 03 | [NVIDIA CUDA 构建与计算](03-cuda-compute.md) | 原生，需对应硬件 | CUDA 构建与结果包 |
| 04 | [多格式结构导入与坐标规范化](04-structure-intake.md) | 原生 | 结构库与来源记录 |
| 05 | [PAW 清单、选型、还原与数值缓存](05-paw-archive.md) | 原生 | PAW 选择与缓存报告 |
| 06 | [k 点推荐、显式网格与任意权重](06-sampling.md) | 原生 | 可执行采样与 SCF 采样表 |
| 07 | [高对称路径生成与标签保持](07-band-path.md) | 原生 | 完整路径与标注能带 |
| 08 | [多 k 点自洽静态计算](08-weighted-scf.md) | 原生 | 完整静态结果包 |
| 09 | [波函数与细密度截断能扫描](09-cutoff-scan.md) | 原生＋项目分析 | 截断能扫描表与所选参数 |
| 10 | [k 网格与金属展宽控制扫描](10-mesh-smearing-scan.md) | 原生＋项目分析 | 网格展宽选择报告 |
| 11 | [冻结 SCF 复用与多路径 NSCF](11-frozen-nscf.md) | 原生 | 多路径 NSCF 结果与图 |
| 12 | [能带、真实网格 DOS 与联合绘图](12-bands-dos.md) | 原生 | 能带 DOS 图与分析包 |
| 13 | [外部自旋、DOS 与投影谱解析](13-import-projections.md) | 原生解析功能 | 外部谱解析与投影报告 |
| 14 | [Sparse-DFT 与现有参考能带比较](14-matched-band-analysis.md) | 原生 | 能带对照表与图 |
| 15 | [Materials Project 结构与谱快照](15-materials-project.md) | 原生＋现有工具 | MP 数据快照与批量对照 |
| 16 | [固定晶胞离子弛豫](16-ionic-relax.md) | 原生 | 离子弛豫结构与轨迹 |
| 17 | [晶胞优化与交替离子—晶胞收敛](17-cell-ionic-relax.md) | 原生＋项目调度 | 优化晶胞与联合平衡记录 |
| 18 | [有限差分力与应变响应计算](18-finite-difference.md) | 原生＋项目结构操作 | 有限差分力/应变响应表 |
| 19 | [体积扫描与状态方程](19-equation-of-state.md) | 原生＋项目拟合 | E–V 曲线与体积模量 |
| 20 | [应变—应力弹性张量](20-elastic-tensor.md) | 原生＋项目拟合 | 弹性矩阵与模量表 |
| 21 | [闭壳层吸附与解离端点能](21-adsorption-endpoints.md) | 原生＋项目差分 | 吸附与解离端点排序 |
| 22 | [中性缺陷、掺杂与复合体结合能](22-neutral-defects.md) | 原生＋项目差分 | 中性缺陷与结合能表 |
| 23 | [非磁性界面结构与粘附能](23-interface-adhesion.md) | 原生＋项目差分 | 界面 registry 与粘附能 |
| 24 | [应变能带工程与有效质量](24-strain-band-engineering.md) | 原生＋项目分析 | 应变能带与有效质量表 |
| 25 | [静态晶相与构型排序](25-polymorph-ranking.md) | 原生＋项目差分 | 静态晶相/构型相对能表 |
| 26 | [用户结构库批量计算与筛选](26-high-throughput.md) | 原生＋项目调度 | 批量结果库与 shortlist |
| 27 | [真实 H/S 快照与求解器重放](27-solver-replay.md) | 原生 | H/S 快照与重放报告 |
| 28 | [作业进度、失败处理与兼容重启](28-restart-monitor.md) | 原生＋驱动记录 | 可观测作业与重启记录 |
| 29 | [完整 SCF 的后端和线程性能表](29-backend-performance.md) | 原生＋项目统计 | 完整 SCF 性能与线程图 |
| 30 | [原生核函数、优化器与数值工具调用](30-native-kernels.md) | 原生 | 原生工具结果包 |
| 31 | [DFT_Skills 研究阶段与结果交接](31-research-handoff.md) | 驱动＋契约适配 | 联动研究计划与阶段记录 |
| 32 | [Academic_Writing_Skills 数值、方法与图表交接](32-writing-evidence.md) | 驱动＋证据适配 | 写作证据、方法与图表包 |
| 33 | [稀疏密度谱预览](33-sparsity-preview.md) | 真实接口＋可复用脚本 | density_probe.json、sparsity_preview.json；未执行 SCF |
| 34 | [S1–S6 稀疏化组件与 off/auto 对照](34-sparsity-components.md) | 真实接口＋可复用脚本 | 逐项真实 SCF、原始证书、sparsity_ledger 与偏差报告 |
| 35 | [共线自旋与 Dudarev U](35-spin-hubbard.md) | 真实接口＋可复用脚本 | 两自旋谱、磁矩、Hubbard 能量和收敛记录 |
| 36 | [均匀 NSCF DOS 与 lm-PDOS](36-uniform-dos-pdos.md) | 真实接口＋可复用脚本 | dos.json、spin DOS、lm-PDOS 与完整性信息 |
| 37 | [UPF v2 PAW 数据路线](37-upf-paw.md) | 真实接口＋可复用脚本 | 解析数据、实际 SCF 或精确的不支持错误 |
| 38 | [并行、工作进程与结果图](38-parallel-plot-worker.md) | 真实接口＋可复用脚本 | 布局、进程状态、实际计数及图形来源 |
