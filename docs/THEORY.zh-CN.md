[English](THEORY.md) · **简体中文** · [README](../README.zh-CN.md) · [稀疏化](SPARSIFICATION.zh-CN.md) · [精度](ACCURACY.zh-CN.md) · [性能](PERFORMANCE.zh-CN.md) · [用户指南](USER_GUIDE.zh-CN.md)

# 理论与实现

本文从 Sparse Density Function 方法讲起，这是 Sparse-DFT 特有的部分：密度保留哪些傅里叶支撑，计算前如何估计被略去的部分，计算后如何认证结果（§1）。其余各节按文献 [1–3, 16, 17] 及教科书 [19] 的形式，给出该方法所作用的、采用 Perdew–Burke–Ernzerhof（PBE）泛函的标准平面波投影缀加波（projector augmented-wave，PAW）方程，并指出每个方程在 Sparse-DFT 中的源文件。方法的完整描述、标定和实测节省见 [SPARSIFICATION.zh-CN.md](SPARSIFICATION.zh-CN.md)。

除非另注单位，方程均采用原子单位。路径均指本版本的源代码树。

## 1. Sparse Density Function 方法

平面波 PAW 计算把密度保存在傅里叶球 $\tfrac12|\mathbf G|^2\le E_\rho$ 上。光滑的能带密度需要 $E_\rho = 4E_\mathrm{cut}$。补偿电荷 $\hat n$ 和部分芯 $\tilde n_c$ 局域在缀加球内，因此它们的谱很长。所以 Sparse-DFT 的精确参考保留 $E_\rho = 16E_\mathrm{cut}$。本方法用对给定数据集而言被略去谱可以忽略的最小支撑因子 $f = E_\rho/E_\mathrm{cut}$ 取代这一固定选择。

**误差估计。** 在自洽场（self-consistent field，SCF）循环之前，谱探针对 §4 的补偿形状因子 $g_{s,L}(q)$ 在各候选球之外积分，并以自由原子多极矩 $w_{s,L}$ 加权：

$$T(f) = \sum_s\sum_L w_{s,L}^2\int_{q(fE_\mathrm{cut})}^{q(24E_\mathrm{cut})}|g_{s,L}(q)|^2\,q^2\,dq,\qquad q(E) = \sqrt{2E}.$$

求和遍历不同的数据集，因此 $T$ 不随晶胞增大而增长。

**选择。** auto 模式从 $\{4,5,6,8,10,12,16\}$ 中取满足 $T(f)\le\tau$ 的最小 $f$，balanced 级别下 $\tau = 2.5\times10^{-7}$，金属取 $f\ge6$。密度模式数随之按 $(f/16)^{3/2}$ 下降，$f = 4$ 时降到 1/8。按实测误差拟合的规则还设定交换关联（exchange-correlation，XC）积分填充和本征求解器的搜索控制。

**证书。** 只改变本征求解器搜索方式的组件，只有在每个本征对都在不变的算符上通过 §9 的 FP64 残差检验 $\|\tilde Hx-\varepsilon\tilde Sx\|\le10^{-12}$ Ha，且 SCF 满足其不变的密度阈值（§8）时才被接受。每个结果都在稀疏化账本中记录所作选择、尾部和算符计数。

**证据。** 在六个拟合体系和四个留出体系上，auto 通过与 $f = 16$ 相同的门限，哈密顿算符调用次数减少到 1/2.1–1/3.9（实测；[SPARSIFICATION.zh-CN.md §5](SPARSIFICATION.zh-CN.md#5-auto-节省了什么)）。

## 2. 单位与常数

内部所有项都以 Hartree 原子单位计算。两套常数用于与 Å 和 eV 之间的换算（`src/atomic_units.hpp`）：

| 约定 | Å/bohr | eV/Hartree | 用途 |
|---|---|---|---|
| `codata`（默认） | 1/1.8897261254578281 | 27.211386245988 | CODATA 2018 数值 [18] |
| `vasp_legacy` | 0.529177249 | 2 × 13.605826 | VASP 6.4.1 使用的数值；用于匹配对比 |

同一套约定同时作用于径向网格、原子参考、冻结芯、平面波、FFT 网格和 Ewald 求和；混用会被拒绝。两套常数的 eV/Hartree 相对相差 $-9.77\times10^{-6}$。PAW 数据集以 eV 给出原子参考能量，而其余各项都随 Hartree 常数缩放。因此总能的偏移随这些项的大小增长；实测偏移见 [ACCURACY.zh-CN.md §3](ACCURACY.zh-CN.md#3-codata-与-vasp_legacy)。用 `--atomic-units` 选择常数集。

## 3. 平面波基组

波矢 $\mathbf k$ 处的波函数用平面波展开

$$\tilde\psi_{n\mathbf k}(\mathbf r) = \frac{1}{\sqrt\Omega}\sum_{\mathbf G} c_{n\mathbf k}(\mathbf G)\,e^{i(\mathbf k+\mathbf G)\cdot\mathbf r},\qquad \tfrac12|\mathbf k+\mathbf G|^2 \le E_\mathrm{cut},$$

其中 $\Omega$ 为晶胞体积（`enumerate_g_vectors`，`src/pw_core.cpp`）。不给 `--cutoff` 时，$E_\mathrm{cut} = \max(300\ \mathrm{eV},\ m\cdot\max_s E_{\max,s})$ 并向上取整；精度级别 `low`、`standard`、`high`、`production` 对应 $m$ = 1.15、1.30、1.40、1.40，$E_{\max,s}$ 为各数据集的推荐截断能（`automatic_cutoff`，`sparse_dft/settings.py`）。

密度使用单独的球 $\tfrac12|\mathbf G|^2\le E_\rho$。光滑密度 $\tilde n$ 需要 $E_\rho\ge4E_\mathrm{cut}$；补偿电荷和部分芯电荷需要更多。默认 $E_\rho = 16E_\mathrm{cut}$（`run_scf`，`src/scf_core.cpp`），可用 `--density-cutoff` 设定，或用 `--density-cutoff-factor` 以因子设定。低于 $4E_\mathrm{cut}$ 的值会被拒绝。默认基组在当前晶格上选取（`basis_policy: current_lattice_cutoff`）。

## 4. 投影缀加波

Blöchl 的 PAW 方法 [1]，按 Kresse 和 Joubert 的形式 [2]，用线性变换把光滑赝波函数 $\tilde\psi$ 映射为全电子波函数 $\psi$：

$$|\psi\rangle = |\tilde\psi\rangle + \sum_{a,i}\big(|\phi^a_i\rangle - |\tilde\phi^a_i\rangle\big)\langle\tilde p^a_i|\tilde\psi\rangle .$$

这里 $a$ 遍历原子，$i = (\ell, m, \text{径向指标})$ 遍历分波。$\phi^a_i$ 和 $\tilde\phi^a_i$ 是全电子分波和赝分波，$\tilde p^a_i$ 是投影子，均取自 PAW 数据集。在每个缀加球内，对数据集所针对的束缚态，该展开是完备的。

**广义本征问题。** Kohn–Sham 方程 [16, 17] 变为

$$\tilde H|\tilde\psi_{n\mathbf k}\rangle = \varepsilon_{n\mathbf k}\,\tilde S|\tilde\psi_{n\mathbf k}\rangle,\qquad \tilde S = 1 + \sum_{a,ij}|\tilde p^a_i\rangle\,Q^a_{ij}\,\langle\tilde p^a_j|,$$

$$\tilde H = -\tfrac12\nabla^2 + \tilde v_\mathrm{eff} + \sum_{a,ij}|\tilde p^a_i\rangle\,D^a_{ij}\,\langle\tilde p^a_j|,$$

其中缀加电荷 $Q^a_{ij} = \langle\phi^a_i|\phi^a_j\rangle - \langle\tilde\phi^a_i|\tilde\phi^a_j\rangle$ 在球内积分。非局域项以低秩乘积 $B^\dagger D B$ 和 $B^\dagger Q B$ 作用，$B$ 为平面波基下的投影子矩阵；不构造 $N_\mathrm{PW}\times N_\mathrm{PW}$ 矩阵（`add_paw_low_rank_operator`，`src/pw_core.cpp`；`src/plane_wave_operator.cpp`）。

**占据矩阵。** 原子 $a$ 的单中心密度矩阵为

$$\rho^a_{ij} = \sum_{n\mathbf k} w_{\mathbf k}\,f_{n\mathbf k}\,\langle\tilde\psi_{n\mathbf k}|\tilde p^a_i\rangle\langle\tilde p^a_j|\tilde\psi_{n\mathbf k}\rangle .$$

**密度。** 全电子密度分为傅里叶网格上的光滑部分和径向网格上的单中心修正：

$$n = \tilde n + \hat n + \tilde n_c + \sum_a\big(n^1_a - \tilde n^1_a - \hat n_a\big),\qquad \tilde n(\mathbf r)=\sum_{n\mathbf k}w_{\mathbf k}f_{n\mathbf k}|\tilde\psi_{n\mathbf k}(\mathbf r)|^2 .$$

$\tilde n_c$ 是部分（赝）芯电荷。补偿电荷恢复每个球内全电子密度与赝密度之差的多极矩：

$$\hat n_a(\mathbf r) = \sum_{LM}\rho^a_{LM}\,\bar g_{a,L}(|\mathbf r-\mathbf R_a|)\,Y_{LM}(\widehat{\mathbf r-\mathbf R_a}),\qquad \rho^a_{LM} = \sum_{ij}\rho^a_{ij}\,\hat Q^{a,LM}_{ij},$$

其中 $\bar g_{a,L}$ 是数据集的补偿形状，归一化为单位 $L$ 阶矩，$\hat Q^{a,LM}_{ij}$ 是 $\phi^a_i\phi^a_j-\tilde\phi^a_i\tilde\phi^a_j$ 的多极矩。在倒空间中 $\hat n_a(\mathbf G) = e^{-i\mathbf G\cdot\mathbf R_a}\sum_{LM}(-i)^L\rho^a_{LM}\,g_{a,L}(|\mathbf G|)\,Y_{LM}(\hat{\mathbf G})/\Omega$（相差 $Y_{LM}$ 的归一化），径向形式为 $g_{a,L}(q) = \int\bar g_{a,L}(r)\,r^2j_L(qr)\,dr$（`src/paw_compensation.cpp`、`src/paw_augmentation.cpp`）。稀疏化探针积分的正是这个径向形式。

**总能。** 记 $n_v = \tilde n + \hat n$ 为价电子赝密度，

$$E = \sum_{n\mathbf k}w_{\mathbf k}f_{n\mathbf k}\langle\tilde\psi_{n\mathbf k}|-\tfrac12\nabla^2|\tilde\psi_{n\mathbf k}\rangle + \int\tilde v_\mathrm{loc}\,n_v + E_\mathrm H[n_v] + E_\mathrm{xc}[\tilde n+\hat n+\tilde n_c] + \sum_a\big(E^1_a - \tilde E^1_a\big) + E_\mathrm{Ewald} + \sum_a\big(E^\mathrm{ref}_a + \Delta E^\mathrm{cut}_a\big) - TS .$$

Hartree 能在倒空间计算，$E_\mathrm H = 2\pi\Omega\sum_{\mathbf G\neq0}|n_v(\mathbf G)|^2/|\mathbf G|^2$（`src/density_functionals.cpp`）。单中心能量 $E^1_a$ 和 $\tilde E^1_a$ 由 $\rho^a_{ij}$ 在径向网格上用同一泛函计算（`src/paw_onecenter.cpp`、`src/paw_augmentation.cpp`）。$E^\mathrm{ref}_a$ 是数据集中为自由原子存储的参考值，因此 $E$ 以参考原子为基准。$\Delta E^\mathrm{cut}_a$ 是数据集在 $E_\mathrm{cut}$ 处列出的动能误差，$-TS$ 是 §7 的 Fermi–Dirac 熵项。代码以等价的能带能（双重计数）形式组装同一能量（`run_scf`，`src/scf_core.cpp`），并把该形式的每一项写入 `scf.json`。

**非局域强度。** $E$ 对 $\rho^a_{ij}$ 求变分得到

$$D^a_{ij} = \sum_{LM}\int\tilde v_\mathrm{eff}(\mathbf r)\,\hat Q^{a,LM}_{ij}\,\bar g_{a,L}Y_{LM}\,d\mathbf r + D^{1,a}_{ij} - \tilde D^{1,a}_{ij},$$

其中第一项把光滑势投影到补偿形状上，$D^1$、$\tilde D^1$ 是单中心导数（`src/paw_augmentation.cpp`）。单中心项总是包含在内；`SDFT_ENABLE_PAW_CLOSURE_DIAGNOSTIC=0` 仅用于诊断时去掉它们，自旋极化计算中会被拒绝。

## 5. 交换关联

泛函为 PBE [3]，分非自旋形式（`src/pbe_xc.cpp`）和共线自旋形式（`src/spin_pbe_xc.cpp`）。在傅里叶网格上，它作用于 $\tilde n+\hat n+\tilde n_c$，梯度在倒空间求得。实空间积分网格沿轴 $a$ 有 $N_a = p\max(3, 2m_a+1)$ 个点，$m_a$ 为密度支撑的最大 Miller 指数，填充默认 $p = 2$（稀疏化组件 S2）。在径向网格上，单中心 XC 能用角向积分求得（`angular_pbe_xc`，`src/paw_augmentation.cpp`）。

## 6. 离子–离子能量

Ewald 求和 [4] 把离子点电荷 $Z_a$ 的库仑能分为实空间和倒空间两部分，

$$E_\mathrm{Ewald} = \frac12\sum_{a,b,\mathbf T}{}^{'}\frac{Z_aZ_b\,\mathrm{erfc}(\eta|\mathbf R_a-\mathbf R_b+\mathbf T|)}{|\mathbf R_a-\mathbf R_b+\mathbf T|} + \frac{2\pi}{\Omega}\sum_{\mathbf G\neq0}\frac{e^{-G^2/4\eta^2}}{G^2}\Big|\sum_aZ_ae^{i\mathbf G\cdot\mathbf R_a}\Big|^2 - \frac{\eta}{\sqrt\pi}\sum_aZ_a^2 - \frac{\pi}{2\Omega\eta^2}\Big(\sum_aZ_a\Big)^2,$$

默认 $\eta = \sqrt\pi/\Omega^{1/3}$。两个求和在尾部低于所需容差处截断，截断用倒格子范数确定，使斜晶胞也保留所有必需的平移（`evaluate_ewald`，`src/ewald.cpp`；力与应力见 `src/ionic_ewald.cpp`）。

## 7. 占据

所有 k 点共享一个化学势 $\mu$（`fill_occupations`，`src/occupations.cpp`）。给定 `--smearing-ev` $\sigma = k_BT > 0$ 时，占据服从 Fermi–Dirac 统计，即 Mermin 的有限温度泛函 [5]：

$$f_{n\mathbf k} = \frac{2}{1+e^{(\varepsilon_{n\mathbf k}-\mu)/\sigma}},\qquad \sum_{n\mathbf k}w_{\mathbf k}f_{n\mathbf k} = N_e,$$

$$-TS = \sigma\sum_{n\mathbf k}2w_{\mathbf k}\big[\tilde f\ln\tilde f + (1-\tilde f)\ln(1-\tilde f)\big],\qquad \tilde f = f_{n\mathbf k}/2 .$$

$\mu$ 用二分法求得。报告的总能是自由能 $E - TS$，`internal_energy_ev` 为 $E$。$\sigma = 0$ 时按零温从低到高填充能级，简并能级等占据。共线自旋使用同一求解器，每个通道取一半权重。若某 k 点最高的计算能带占据仍高于 $10^{-6}$，自动能带数增加 $\max(4, 25\,\%)$ 并重新开始 SCF；显式给出 `--bands` 时则直接报错。

## 8. 自洽

SCF 在联合状态 $X = (\tilde n(\mathbf G), \rho^a_{ij})$ 上迭代。每一步由 $X_\mathrm{in}$ 构造 $\tilde H$ 和 $\tilde S$，求解本征问题，得到 $X_\mathrm{out}$。多割线（Anderson/Broyden 类）更新 [6, 7] 把 $(X_\mathrm{in}, X_\mathrm{out}-X_\mathrm{in})$ 对的历史组合成下一步的输入（`src/joint_multisecant.hpp`、`src/scf_core.cpp`）。光滑密度用 Kerker 形式 [8] 预条件：

$$\chi(G) = \max\Big(a_\mathrm{min},\ a\,\frac{G^2}{G^2+q_0^2}\Big),$$

其中 $a$ = `--mixing`（默认 0.2），$q_0$ = 1 Å$^{-1}$，下限 $a_\mathrm{min} = \min(0.1, a, a_\mathrm{mag})$（`kerker_response`，`src/collinear_mixing.hpp`）。共线自旋时，磁化在同一历史中使用自己的混合系数 $a_\mathrm{mag} = \min(2, 4a)$。若提议的单中心矩阵不是半正定的，则沿整个步长回退，从不逐原子或逐通道回退。

两个残差都低于阈值时 SCF 停止：

- 密度残差，即输入与输出之间任一密度系数的最大变化，低于 `--density-tolerance`（默认 $10^{-6}$；弛豫中为 $10^{-8}$）；
- 单中心残差，即 $\rho^a_{ij}$ 变化的逐原子归一化 Frobenius 范数，低于 `--onecenter-density-tolerance`（默认 $10^{-6}$）。

`--scf-acceptance-margin` $\in(0,1]$ 同时缩放两个阈值。

## 9. 本征求解器与证书

每个 k 点用块 Davidson 方法 [9] 在无矩阵算符 $\tilde H$ 和 $\tilde S$ 上求解（`solve_generalized_hermitian_davidson`，`src/pw_core.cpp`）。局域势用 FFT 作用；动能是对角的；非局域部分是 §4 的低秩乘积。修正向量用 Teter–Payne–Allan 形式 [10] 预条件：

$$K(x) = \frac{2}{s}\,\frac{27+18x+12x^2+8x^3}{27+18x+12x^2+8x^3+16x^4},\qquad x = \tfrac12|\mathbf k+\mathbf G|^2/s,$$

其中 $s = 1.5\max\big(\sum_{\mathbf G}|x_{\mathbf G}|^2\,\tfrac12|\mathbf k+\mathbf G|^2,\ 2\ \mathrm{eV}\big)$ 由当前向量 $x$ 求得。搜索子空间默认最多容纳能带数的 8 倍（`--davidson-subspace-factor`，4–16）。

只有在原始算符上以 FP64（IEEE 双精度）通过证书时，Ritz 对 $(\varepsilon, x)$ 才被接受：

$$\|\,\tilde Hx - \varepsilon\tilde Sx\,\|_2 \le \epsilon_\mathrm{eig},\qquad x^\dagger\tilde Sx = 1,$$

重新归一化后作为一个块计算（`davidson_fp64_certification`）。SCF 最后几次迭代中 $\epsilon_\mathrm{eig}\le10^{-12}$ Ha。稀疏化组件和求解器控制可以改变寻找 Ritz 对的方式，但从不改变这一检验。收敛的计算以 `eigensolver_max_residual_hartree` 报告最大残差。

## 10. 力与应力

**力。** 自洽时，原子 $a$ 上的力是 $E$ 在固定占据和固定平面波集合下的导数。它包含局域势、非局域投影子、补偿电荷、部分芯和 Ewald 贡献；投影子项为

$$\mathbf F^{\mathrm{nl}}_a = -\sum_{n\mathbf k}w_{\mathbf k}f_{n\mathbf k}\sum_{ij}\big(D^a_{ij}-\varepsilon_{n\mathbf k}Q^a_{ij}\big)\,2\,\mathrm{Re}\,\langle\tilde\psi_{n\mathbf k}|\partial_{\mathbf R_a}\tilde p^a_i\rangle\langle\tilde p^a_j|\tilde\psi_{n\mathbf k}\rangle .$$

由于平面波不随原子移动，不存在基组（Pulay）力。投影子导数取实际所用投影子插值的导数（`src/projector_derivative.cpp`；组装见 `src/scf_core.cpp`）。

**应力。** 应力是对均匀应变 $\epsilon_{\alpha\beta}$ 的导数，$\sigma_{\alpha\beta} = \Omega^{-1}\,\partial E/\partial\epsilon_{\alpha\beta}$，按 Nielsen 和 Martin [11] 的方法求得。求导时保持平面波的整数指标不变，因此它是有限基组能量的精确导数（`electronic_paw_stress`，`src/scf_core.cpp`）。固定截断能下改变晶胞会增减平面波，这一不连续性不在解析应力中。两者之差即基组（Pulay）应力，随 $E_\mathrm{cut}$ 增大而减小；定量使用应力前请先收敛截断能。

## 11. 弛豫

`sparse-dft relax` 在原子位置上（固定晶胞）最小化能量，加 `--cell` 时则在固定分数坐标下对晶胞最小化（`sparse_dft/analytic_relax.py`）。每一步在原生程序 `sdft_relax_step` 中使用有限内存 BFGS（L-BFGS）双循环递推 [12]（`lbfgs_direction`，`src/relax_optimizer.cpp`）：

- 满足 $\mathbf s^\top\mathbf y \le 10^{-10}\|\mathbf s\|\|\mathbf y\|$ 的曲率对 $(\mathbf s,\mathbf y)$ 被舍弃；
- 初始逆 Hessian 为 $\gamma I$，$\gamma = \mathbf s^\top\mathbf y/\mathbf y^\top\mathbf y$ 取自最新一对，并裁剪到 $[10^{-6},10^{6}]$；
- 非下降方向退回最速下降，步长上限为 `--max-displacement`（默认 0.15 Å）或 `--max-strain`（默认 0.02）。

在试探点完成一次完整 SCF 并满足 Armijo 条件 $E(\mathbf x+\alpha\mathbf p)\le E(\mathbf x)+10^{-4}\alpha\,\nabla E^\top\mathbf p$ 后接受该步，依次尝试 $\alpha = 1, \tfrac12, \tfrac14,\dots$（最多 16 次）。历史保留 8 对。收敛要求最大力的模低于 `--force-tolerance`（默认 0.02 eV/Å），或最大应力分量低于 `--stress-tolerance`（默认 0.05 GPa）。每个被接受的状态都会写检查点。

## 12. 能带与 DOS

**非自洽（NSCF）计算。** `bands` 和 `dos` 先收敛一次 SCF（或用 `--scf-checkpoint` 复用），然后冻结其密度、$\rho^a_{ij}$ 和势，在新的 k 点上求解本征问题（`run_nscf`，`src/scf_core.cpp`）。同样的证书适用。默认能带数为占据能带数加 4。

**能带路径。** `make-band-path --auto` 由 spglib [14] 和 SeeK-path 给出 Hinuma 等 [13] 的标准路径；采样密度由 `--spacing` 设定，单位 Å$^{-1}$（含 $2\pi$）。每一段都包含两个端点。

**DOS。** `sparse-dft dos` 在不做对称约化的完整均匀网格上求解，并用归一化高斯函数展宽每个能级，

$$D_\sigma(E) = \sum_{n\mathbf k}w_{\mathbf k}\,g\,\frac{1}{\sigma\sqrt{2\pi}}\,e^{-(E-\varepsilon_{n\mathbf k})^2/2\sigma^2},$$

其中 $g$ 为自旋简并度，$\sigma$ = `--dos-sigma`（`gaussian_broaden`，`sparse_dft/bands.py`）。能量步长不得超过 $\sigma/3$。DOS 只在最高求解能带起始能量加 $3\sigma$ 以下是完整的；`dos.json` 将其记录为 `complete_below_ev`。

## 13. 轨道投影

态 $|\tilde\psi_{n\mathbf k}\rangle$ 在原子 $a$ 上的 lm 分辨权重是其在缀加球内按 $(\ell, m)$ 分解的全电子电荷：

$$w^a_{n\mathbf k,\ell m} = \sum_{ij\in\ell}\overline{c^a_{i,\ell m}}\;q^a_{ij}\;c^a_{j,\ell m},\qquad c^a_{i,\ell m} = \langle\tilde p^a_{i}|\tilde\psi_{n\mathbf k}\rangle,\qquad q^a_{ij} = \int_0^{r_a}\phi^a_i(r)\,\phi^a_j(r)\,dr .$$

求和遍历同一 $\ell$ 的径向通道 $i, j$，$\phi^a_i$ 是全电子径向分波（已含 $r^2$）。积分在截断于数据集球半径 $r_a$ 的径向网格上用 Simpson 权重计算（`paw_sphere_ae_overlap`，`src/paw_hubbard.cpp`；`run_nscf`）。这是 Sparse-DFT 的球内电荷定义，在每个输出中记录为 `projection_weight_definition`。[ACCURACY.zh-CN.md](ACCURACY.zh-CN.md#5-能带路径dos-与投影) 把它与 VASP 6.4.1 在 `LORBIT = 11` 下写入 PROCAR 的球投影权重比较。实球谐函数按 $m = -\ell\dots\ell$ 排列，$m<0$ 取 $\sin(|m|\varphi)$，$m>0$ 取 $\cos(m\varphi)$，顺序为 s；p$_y$、p$_z$、p$_x$；d$_{xy}$、d$_{yz}$、d$_{z^2}$、d$_{xz}$、d$_{x^2-y^2}$。

权重未归一化：球内不含整个态，高能态的权重可能大于一。因此总量只在能量窗口内比较。

## 14. 共线自旋与 DFT+U

`--spin-channels 2` 用一个化学势和自旋极化 PBE 泛函求解两个自旋通道。`--initial-moments` 设定每个原子的初始磁矩（单位 $\mu_B$）。`--hubbard EL:L:UEFF` 加入 Dudarev 等 [15] 的转动不变修正，

$$E_U = \frac{U_\mathrm{eff}}{2}\sum_\sigma\Big[\mathrm{Tr}\,n^{\sigma} - \mathrm{Tr}\big(n^{\sigma}n^{\sigma}\big)\Big],$$

其中 $n^\sigma$ 是壳层 $\ell$ 的在位占据矩阵，用同样的全电子交叠 $q^a_{ij}$ 构造（`dudarev_paw`，`src/paw_hubbard.cpp`）。

## 参考文献

1. P. E. Blöchl, Projector augmented-wave method, *Phys. Rev. B* **50**, 17953–17979 (1994). doi:[10.1103/PhysRevB.50.17953](https://doi.org/10.1103/PhysRevB.50.17953)
2. G. Kresse, D. Joubert, From ultrasoft pseudopotentials to the projector augmented-wave method, *Phys. Rev. B* **59**, 1758–1775 (1999). doi:[10.1103/PhysRevB.59.1758](https://doi.org/10.1103/PhysRevB.59.1758)
3. J. P. Perdew, K. Burke, M. Ernzerhof, Generalized gradient approximation made simple, *Phys. Rev. Lett.* **77**, 3865–3868 (1996). doi:[10.1103/PhysRevLett.77.3865](https://doi.org/10.1103/PhysRevLett.77.3865)
4. P. P. Ewald, Die Berechnung optischer und elektrostatischer Gitterpotentiale, *Ann. Phys.* **369**, 253–287 (1921). doi:[10.1002/andp.19213690304](https://doi.org/10.1002/andp.19213690304)
5. N. D. Mermin, Thermal properties of the inhomogeneous electron gas, *Phys. Rev.* **137**, A1441–A1443 (1965). doi:[10.1103/PhysRev.137.A1441](https://doi.org/10.1103/PhysRev.137.A1441)
6. P. Pulay, Convergence acceleration of iterative sequences. The case of SCF iteration, *Chem. Phys. Lett.* **73**, 393–398 (1980). doi:[10.1016/0009-2614(80)80396-4](https://doi.org/10.1016/0009-2614(80)80396-4)
7. L. D. Marks, D. R. Luke, Robust mixing for ab initio quantum mechanical calculations, *Phys. Rev. B* **78**, 075114 (2008). doi:[10.1103/PhysRevB.78.075114](https://doi.org/10.1103/PhysRevB.78.075114)
8. G. P. Kerker, Efficient iteration scheme for self-consistent pseudopotential calculations, *Phys. Rev. B* **23**, 3082–3084 (1981). doi:[10.1103/PhysRevB.23.3082](https://doi.org/10.1103/PhysRevB.23.3082)
9. E. R. Davidson, The iterative calculation of a few of the lowest eigenvalues and corresponding eigenvectors of large real-symmetric matrices, *J. Comput. Phys.* **17**, 87–94 (1975). doi:[10.1016/0021-9991(75)90065-0](https://doi.org/10.1016/0021-9991(75)90065-0)
10. M. P. Teter, M. C. Payne, D. C. Allan, Solution of Schrödinger's equation for large systems, *Phys. Rev. B* **40**, 12255–12263 (1989). doi:[10.1103/PhysRevB.40.12255](https://doi.org/10.1103/PhysRevB.40.12255)
11. O. H. Nielsen, R. M. Martin, Quantum-mechanical theory of stress and force, *Phys. Rev. B* **32**, 3780–3791 (1985). doi:[10.1103/PhysRevB.32.3780](https://doi.org/10.1103/PhysRevB.32.3780)
12. D. C. Liu, J. Nocedal, On the limited memory BFGS method for large scale optimization, *Math. Program.* **45**, 503–528 (1989). doi:[10.1007/BF01589116](https://doi.org/10.1007/BF01589116)
13. Y. Hinuma, G. Pizzi, Y. Kumagai, F. Oba, I. Tanaka, Band structure diagram paths based on crystallography, *Comput. Mater. Sci.* **128**, 140–184 (2017). doi:[10.1016/j.commatsci.2016.10.015](https://doi.org/10.1016/j.commatsci.2016.10.015)
14. A. Togo, K. Shinohara, I. Tanaka, Spglib: a software library for crystal symmetry search, *Sci. Technol. Adv. Mater.: Methods* **4**, 2384822 (2024). doi:[10.1080/27660400.2024.2384822](https://doi.org/10.1080/27660400.2024.2384822)
15. S. L. Dudarev, G. A. Botton, S. Y. Savrasov, C. J. Humphreys, A. P. Sutton, Electron-energy-loss spectra and the structural stability of nickel oxide: an LSDA+U study, *Phys. Rev. B* **57**, 1505–1509 (1998). doi:[10.1103/PhysRevB.57.1505](https://doi.org/10.1103/PhysRevB.57.1505)
16. P. Hohenberg, W. Kohn, Inhomogeneous electron gas, *Phys. Rev.* **136**, B864–B871 (1964). doi:[10.1103/PhysRev.136.B864](https://doi.org/10.1103/PhysRev.136.B864)
17. W. Kohn, L. J. Sham, Self-consistent equations including exchange and correlation effects, *Phys. Rev.* **140**, A1133–A1138 (1965). doi:[10.1103/PhysRev.140.A1133](https://doi.org/10.1103/PhysRev.140.A1133)
18. E. Tiesinga, P. J. Mohr, D. B. Newell, B. N. Taylor, CODATA recommended values of the fundamental physical constants: 2018, *Rev. Mod. Phys.* **93**, 025010 (2021). doi:[10.1103/RevModPhys.93.025010](https://doi.org/10.1103/RevModPhys.93.025010)
19. R. M. Martin, *Electronic Structure: Basic Theory and Practical Methods*, 2nd ed. (Cambridge University Press, 2020). doi:[10.1017/9781108555586](https://doi.org/10.1017/9781108555586)
