# Sparse-DFT Driver Skills 2.0.0 验证报告

日期：2026-10-18（Asia/Shanghai）。软件基线：用户提供的 Sparse-DFT 1.0.0 源码。归档 SHA-256 与接口文件哈希见 source-provenance.json。

技能已安装到 `/Users/Fona/.codex/skills/sparse-dft-driver`。61 个技能文件与交付源文件逐字节一致。旧版 dftx-driver 已备份到本任务 installed-backups-20261018，退出技能发现目录；未修改其他技能。下一轮对话可调用 `$sparse-dft-driver`。

## 已实际通过

| 验证 | 真实结果 |
|---|---|
| 技能格式、相对链接、文档调用名称、源码目录契约 | 通过；catalog 与实际源码一致 |
| 驱动回归 | 33/33；安装副本再次 33/33 |
| 上游 Python 测试 | 317 项中 312 项执行通过、5 项跳过；无失败 |
| CPU 原生构建、测试、安装 | 84/84 |
| Metal 构建 | 同构建 CPU 测试 84/84；4 个 GPU 原生测试实际通过 |
| MPI 构建 | 84 个普通测试通过；3 个 MPI 测试在宿主机重测通过 |
| Python 软件安装 | 真实构建 wheel 并安装 sparse-dft 1.0.0，安装包成功解析 Si |
| 稀疏密度谱预览 | 原生 probe 与三个等级选择真实执行，未标为 SCF |
| S1–S6 / off / auto / 开关变体 / deterministic / fallback | 13 组 Si SCF 均收敛；匹配能量、力、应力及原始本征残差门限均通过 |
| 冻结能带和均匀 DOS/PDOS | 3 个路径点、2×2×2 DOS 网格真实执行 |
| 自旋与 Dudarev U | 双自旋和显式 Si-p U 模型通过；仅验证驱动，不推荐此模型作材料结论 |
| 离子/晶胞弛豫 | 离子弛豫收敛；2×2×2 网格晶胞弛豫 5 步收敛；安装副本 API relax.cell 也收敛 |
| 有限差分力 | 12 个真实位移 SCF，中心差分，保留原始净力 |
| Solver snapshot/replay | 捕获实际算子，native replay 按快照原容差通过；不是全局 SCF 精度证书 |
| 对称性路径、worker、绘图、结果交接 | 实际执行；12 张 PNG；SCF/DOS 证据导出保留源哈希 |
| 安装后实际计算 | 新安装脚本再次执行 auto SCF 并收敛 |
| 从压缩包安装 | 解包后通过包内安装器独立安装，再执行真实 auto SCF 并收敛 |

GPU 实际 SCF 性能文件标识 `mps` 和 Apple M5 Max，并含非零 FFT、PBE、PAW 链和 GEMM 调用。MPI 的 k 分组与 band 分组各执行了双进程 SCF。详见 host-scf-tests.json、backend-host-retest.json 和 real-results-summary.json。

## 本次修复的关键问题

1. 软件/技能/命名空间/环境/可执行文件切换为 Sparse-DFT、sparse-dft-driver、sparse_dft、SDFT_*、sdft_*。
2. 源码扫描改为新目录，识别 plot/worker 委托命令、子包和 API 间接消费的 spin/U 参数。
3. API 不消费的 solver/family、relax sparsity 等参数明确拒绝；需要时使用真实 CLI/Python。
4. 冻结 NSCF 不接受 auto 或 density_cutoff_factor；提取脚本删除这两类控制，从检查点继承支持。
5. 新增真实稀疏化矩阵执行与对照工具，区分截断/网格误差和搜索证书，保留 fallback 与 warnings。
6. 增加自旋/U、DOS/PDOS、UPF、MPI/Metal、对称性/绘图/worker 的操作说明；删除旧版无自旋/无 U 限制和已移除工具路线。
7. DOS 交接验证其 NSCF 源文件哈希；损坏来源不提升为完成阶段。新安装器校验副本并安全迁移旧名称。

## 失败记录与恢复

初始 Metal 设备访问和 MPI 本机套接字受沙箱限制；原始失败日志保留，随后在宿主机完成对应测试及真实 SCF，未把沙箱失败改写为通过。早期 Python 测试因执行目录/可选依赖失败；按源码指定版本安装到独立测试目录后，最终完整运行无失败。

单 Gamma 点的晶胞弛豫在几何变化后出现电子不收敛。原目录与失败 API 响应保留；使用新目录、明确 2×2×2 采样和 auto 冻结方案后，晶胞弛豫收敛。没有通过放松原始残差门限掩盖失败。

## 尚未验证或软件未实现

- CUDA 没有本机 NVIDIA 运行环境；保留真实构建/调用说明，不标为运行通过。
- 没有实际 PAW-UPF 数据集，未声称完成 UPF 全 SCF；POTCAR 路线的精度不能外推到 UPF。
- Materials Project 的两个认证接口未执行在线真实取数；缺少指定材料及凭据。错误和凭据隐藏路径有测试。
- S7 FP32 无实现，必须明确拒绝。job.status 是占位；results.summarize/export 是上游未实现分支。驱动另提供真实监控和证据导出，不伪装这些分支已实现。
- 历史 benchmark 脚本所需原始 workflow 数据不随源码分发；目录可发现不等于 benchmark 已跑通。
- 上游 5 个跳过测试涉及缺少的历史图形/扫描夹具或特定 build/ 目录；未计入通过。当前本地实际结果已独立覆盖多种图形和扫描路线。
- 上游 GaAs 与 Al slab 的 lm 投影精度问题仍存在；不承诺任意材料 PDOS 已验证。
- 旧 evidence-v2 外部验证器未安装，当前只验证本地导出契约和真实源哈希，未宣称完成该上游 schema 认证。

这些测试是软件驱动与数值流程的真实验证。300 eV、小网格 Si pilot 不是 cutoff/k-mesh 收敛研究或所有材料精度证明。原生 `total_energy_valid=false`、导数和 production 状态均保留。13 组对照默认门限为 1 meV/atom、1e-3 eV/Å、0.01 GPa，证书为 1e-12 Ha；详见 sparsity-comparison.json 的真实原值、条件与 SHA-256。

完整接口覆盖与运行边界见 coverage.json。完整原始计算位于本任务 sparse-dft-upgrade-20261018；交付包不包含 POTCAR、PAW numeric cache、凭据或主机二进制。

本机可用环境文件：`/Users/Fona/Documents/ChatGPT/Sparse Density Function/Sparse-DFT-runtime.env.sh`。source 后可直接运行 sparse-dft；未修改 shell profile。
