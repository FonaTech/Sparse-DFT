"""Axis-label and legend translations (en, zh_CN, zh_TW, ja).

Physics symbols stay as math text in every language; units stay in
parentheses.  Missing translations fall back to English, never to a key.
"""
from __future__ import annotations

from .style import current_style

EF = r'$E - E_\mathrm{F}$'

# key -> {lang: text}; '{...}' placeholders are filled by tr(**kwargs).
LABELS: dict[str, dict[str, str]] = {
    'energy_rel_ef': {'en': EF + ' (eV)', 'zh_CN': '能量 ' + EF + ' (eV)',
                      'zh_TW': '能量 ' + EF + ' (eV)', 'ja': 'エネルギー ' + EF + ' (eV)'},
    'energy_rel_vbm': {'en': r'$E - E_\mathrm{VBM}$ (eV)', 'zh_CN': r'能量 $E - E_\mathrm{VBM}$ (eV)',
                       'zh_TW': r'能量 $E - E_\mathrm{VBM}$ (eV)', 'ja': r'エネルギー $E - E_\mathrm{VBM}$ (eV)'},
    'energy_abs': {'en': 'Energy (eV)', 'zh_CN': '能量 (eV)', 'zh_TW': '能量 (eV)', 'ja': 'エネルギー (eV)'},
    'energy_ev': {'en': 'Energy (eV)', 'zh_CN': '能量 (eV)', 'zh_TW': '能量 (eV)', 'ja': 'エネルギー (eV)'},
    'contribution_ev': {'en': 'Contribution (eV)', 'zh_CN': '能量贡献 (eV)', 'zh_TW': '能量貢獻 (eV)', 'ja': '寄与 (eV)'},
    'kpath': {'en': 'Wave vector', 'zh_CN': '波矢', 'zh_TW': '波向量', 'ja': '波数ベクトル'},
    'dos': {'en': 'DOS (states eV$^{-1}$)', 'zh_CN': '态密度 (states eV$^{-1}$)',
            'zh_TW': '態密度 (states eV$^{-1}$)', 'ja': '状態密度 (states eV$^{-1}$)'},
    'dos_short': {'en': 'DOS', 'zh_CN': '态密度', 'zh_TW': '態密度', 'ja': '状態密度'},
    'integrated_states': {'en': 'Integrated states', 'zh_CN': '积分态数', 'zh_TW': '積分態數', 'ja': '積分状態数'},
    'pdos': {'en': 'PDOS (states eV$^{-1}$)', 'zh_CN': '投影态密度 (states eV$^{-1}$)',
             'zh_TW': '投影態密度 (states eV$^{-1}$)', 'ja': '射影状態密度 (states eV$^{-1}$)'},
    'iteration': {'en': 'SCF iteration', 'zh_CN': '自洽迭代步', 'zh_TW': '自洽迭代步', 'ja': 'SCF 反復回数'},
    'residual': {'en': 'Residual', 'zh_CN': '残差', 'zh_TW': '殘差', 'ja': '残差'},
    'eig_residual': {'en': 'Eigen residual (Ha)', 'zh_CN': '本征残差 (Ha)', 'zh_TW': '本徵殘差 (Ha)', 'ja': '固有残差 (Ha)'},
    'density_residual': {'en': r'Density $\max|\Delta\rho|$', 'zh_CN': r'密度 $\max|\Delta\rho|$',
                         'zh_TW': r'密度 $\max|\Delta\rho|$', 'ja': r'密度 $\max|\Delta\rho|$'},
    'onecenter_residual': {'en': 'One-centre (relative)', 'zh_CN': '单中心（相对）', 'zh_TW': '單中心（相對）', 'ja': '一中心（相対）'},
    'eigensolver': {'en': 'Eigensolver', 'zh_CN': '本征求解器', 'zh_TW': '本徵求解器', 'ja': '固有値ソルバー'},
    'tolerance': {'en': 'tolerance', 'zh_CN': '收敛阈值', 'zh_TW': '收斂閾值', 'ja': '許容値'},
    'moment': {'en': r'Moment ($\mu_\mathrm{B}$)', 'zh_CN': r'磁矩 ($\mu_\mathrm{B}$)',
               'zh_TW': r'磁矩 ($\mu_\mathrm{B}$)', 'ja': r'磁気モーメント ($\mu_\mathrm{B}$)'},
    'moment_output': {'en': 'output', 'zh_CN': '输出', 'zh_TW': '輸出', 'ja': '出力'},
    'moment_mixed': {'en': 'mixed', 'zh_CN': '混合后', 'zh_TW': '混合後', 'ja': '混合後'},
    'moment_change': {'en': r'$|\Delta m|$ ($\mu_\mathrm{B}$)', 'zh_CN': r'$|\Delta m|$ ($\mu_\mathrm{B}$)',
                      'zh_TW': r'$|\Delta m|$ ($\mu_\mathrm{B}$)', 'ja': r'$|\Delta m|$ ($\mu_\mathrm{B}$)'},
    'atom': {'en': 'Atom', 'zh_CN': '原子', 'zh_TW': '原子', 'ja': '原子'},
    'force_abs': {'en': r'$|F|$ (eV $\mathrm{\AA}^{-1}$)', 'zh_CN': r'力 $|F|$ (eV $\mathrm{\AA}^{-1}$)',
                  'zh_TW': r'力 $|F|$ (eV $\mathrm{\AA}^{-1}$)', 'ja': r'力 $|F|$ (eV $\mathrm{\AA}^{-1}$)'},
    'force_component': {'en': r'$F$ (eV $\mathrm{\AA}^{-1}$)', 'zh_CN': r'力 $F$ (eV $\mathrm{\AA}^{-1}$)',
                        'zh_TW': r'力 $F$ (eV $\mathrm{\AA}^{-1}$)', 'ja': r'力 $F$ (eV $\mathrm{\AA}^{-1}$)'},
    'component': {'en': 'Component', 'zh_CN': '分量', 'zh_TW': '分量', 'ja': '成分'},
    'stress_gpa': {'en': r'$\sigma$ (GPa)', 'zh_CN': r'应力 $\sigma$ (GPa)', 'zh_TW': r'應力 $\sigma$ (GPa)',
                   'ja': r'応力 $\sigma$ (GPa)'},
    'time_s': {'en': 'Time (s)', 'zh_CN': '时间 (s)', 'zh_TW': '時間 (s)', 'ja': '時間 (s)'},
    'wall_time_s': {'en': 'Wall time (s)', 'zh_CN': '墙钟时间 (s)', 'zh_TW': '牆鐘時間 (s)', 'ja': '経過時間 (s)'},
    'calls': {'en': 'calls', 'zh_CN': '调用', 'zh_TW': '呼叫', 'ja': '呼出'},
    'nested_note': {'en': 'nested timers, not additive', 'zh_CN': '计时器嵌套，不可相加',
                    'zh_TW': '計時器巢狀，不可相加', 'ja': 'タイマーは入れ子、加算不可'},
    'diagnostic': {'en': 'diagnostic', 'zh_CN': '诊断值', 'zh_TW': '診斷值', 'ja': '診断値'},
    'diagnostic_note': {'en': 'diagnostic: total_energy_valid = false',
                        'zh_CN': '诊断值：total_energy_valid = false',
                        'zh_TW': '診斷值：total_energy_valid = false',
                        'ja': '診断値：total_energy_valid = false'},
    'unvalidated_note': {'en': 'not validated', 'zh_CN': '未经验证', 'zh_TW': '未經驗證', 'ja': '未検証'},
    'total': {'en': 'Total', 'zh_CN': '总计', 'zh_TW': '總計', 'ja': '合計'},
    'total_dos': {'en': 'Total', 'zh_CN': '总态密度', 'zh_TW': '總態密度', 'ja': '全状態密度'},
    'spin_up': {'en': 'spin up', 'zh_CN': '自旋向上', 'zh_TW': '自旋向上', 'ja': 'スピン上向き'},
    'spin_down': {'en': 'spin down', 'zh_CN': '自旋向下', 'zh_TW': '自旋向下', 'ja': 'スピン下向き'},
    'gap_direct': {'en': 'direct gap', 'zh_CN': '直接带隙', 'zh_TW': '直接能隙', 'ja': '直接ギャップ'},
    'gap_indirect': {'en': 'indirect gap', 'zh_CN': '间接带隙', 'zh_TW': '間接能隙', 'ja': '間接ギャップ'},
    'metal': {'en': 'metallic', 'zh_CN': '金属性', 'zh_TW': '金屬性', 'ja': '金属的'},
    'reference': {'en': 'reference', 'zh_CN': '参考', 'zh_TW': '參考', 'ja': '参照'},
    'candidate': {'en': 'candidate', 'zh_CN': '对比', 'zh_TW': '對比', 'ja': '比較対象'},
    'band_error_mev': {'en': r'$\Delta E$ (meV)', 'zh_CN': r'误差 $\Delta E$ (meV)',
                       'zh_TW': r'誤差 $\Delta E$ (meV)', 'ja': r'誤差 $\Delta E$ (meV)'},
    'ref_energy_rel': {'en': r'Reference $E - E_0$ (eV)', 'zh_CN': r'参考能量 $E - E_0$ (eV)',
                       'zh_TW': r'參考能量 $E - E_0$ (eV)', 'ja': r'参照エネルギー $E - E_0$ (eV)'},
    'relax_step': {'en': 'Relaxation step', 'zh_CN': '弛豫步', 'zh_TW': '弛豫步', 'ja': '緩和ステップ'},
    'energy_change_mev': {'en': r'$E - E_\mathrm{final}$ (meV)', 'zh_CN': r'能量 $E - E_\mathrm{final}$ (meV)',
                          'zh_TW': r'能量 $E - E_\mathrm{final}$ (meV)', 'ja': r'エネルギー $E - E_\mathrm{final}$ (meV)'},
    'max_force': {'en': r'$\max|F|$ (eV $\mathrm{\AA}^{-1}$)', 'zh_CN': r'最大力 $\max|F|$ (eV $\mathrm{\AA}^{-1}$)',
                  'zh_TW': r'最大力 $\max|F|$ (eV $\mathrm{\AA}^{-1}$)', 'ja': r'最大力 $\max|F|$ (eV $\mathrm{\AA}^{-1}$)'},
    'max_stress': {'en': r'$\max|\sigma|$ (GPa)', 'zh_CN': r'最大应力 $\max|\sigma|$ (GPa)',
                   'zh_TW': r'最大應力 $\max|\sigma|$ (GPa)', 'ja': r'最大応力 $\max|\sigma|$ (GPa)'},
    'step_length': {'en': r'Step (${\mathrm{\AA}}$)', 'zh_CN': r'步长 (${\mathrm{\AA}}$)',
                    'zh_TW': r'步長 (${\mathrm{\AA}}$)', 'ja': r'ステップ長 (${\mathrm{\AA}}$)'},
    'step_length_strain': {'en': 'Step (strain)', 'zh_CN': '步长（应变）', 'zh_TW': '步長（應變）', 'ja': 'ステップ長（ひずみ）'},
    'line_search': {'en': 'line-search trials', 'zh_CN': '线搜索次数', 'zh_TW': '線搜尋次數', 'ja': '直線探索回数'},
    'gate_use': {'en': 'Gate use (max over gates)', 'zh_CN': '误差门限占用（各门限最大值）',
                 'zh_TW': '誤差門檻佔用（各門檻最大值）', 'ja': 'ゲート使用率（全ゲートの最大）'},
    'work_reduction': {'en': 'Work reduction ({metric}, ×)', 'zh_CN': '计算量缩减（{metric}，×）',
                       'zh_TW': '計算量縮減（{metric}，×）', 'ja': '計算量削減（{metric}、×）'},
    'gate_limit': {'en': 'gate limit', 'zh_CN': '门限', 'zh_TW': '門檻', 'ja': 'ゲート上限'},
    'pass': {'en': 'pass', 'zh_CN': '通过', 'zh_TW': '通過', 'ja': '合格'},
    'fail': {'en': 'fail', 'zh_CN': '未通过', 'zh_TW': '未通過', 'ja': '不合格'},
    'threads': {'en': 'Threads', 'zh_CN': '线程数', 'zh_TW': '執行緒數', 'ja': 'スレッド数'},
    'ranks': {'en': 'MPI ranks', 'zh_CN': 'MPI 进程数', 'zh_TW': 'MPI 行程數', 'ja': 'MPI ランク数'},
    'workers': {'en': 'Threads × ranks', 'zh_CN': '线程 × 进程', 'zh_TW': '執行緒 × 行程', 'ja': 'スレッド × ランク'},
    'speedup': {'en': 'Speed-up (×)', 'zh_CN': '加速比 (×)', 'zh_TW': '加速比 (×)', 'ja': '高速化率 (×)'},
    'efficiency': {'en': 'Parallel efficiency', 'zh_CN': '并行效率', 'zh_TW': '平行效率', 'ja': '並列効率'},
    'occupancy': {'en': 'CPU occupancy', 'zh_CN': 'CPU 占用率', 'zh_TW': 'CPU 佔用率', 'ja': 'CPU 占有率'},
    'ideal': {'en': 'ideal', 'zh_CN': '理想', 'zh_TW': '理想', 'ja': '理想'},
    'weight': {'en': 'Projection weight', 'zh_CN': '投影权重', 'zh_TW': '投影權重', 'ja': '射影重み'},
    'orbital': {'en': 'Orbital', 'zh_CN': '轨道', 'zh_TW': '軌域', 'ja': '軌道'},
    'raises_total': {'en': 'raises total', 'zh_CN': '使总能升高', 'zh_TW': '使總能升高', 'ja': '全エネルギーを上げる'},
    'lowers_total': {'en': 'lowers total', 'zh_CN': '使总能降低', 'zh_TW': '使總能降低', 'ja': '全エネルギーを下げる'},
    'contribution_ev_atom': {'en': r'Contribution (eV atom$^{-1}$)', 'zh_CN': r'能量贡献 (eV atom$^{-1}$)',
                             'zh_TW': r'能量貢獻 (eV atom$^{-1}$)', 'ja': r'寄与 (eV atom$^{-1}$)'},
    'non_additive': {'en': 'terms do not sum to the total: bar view', 'zh_CN': '各项之和不等于总能：仅作条形图',
                     'zh_TW': '各項之和不等於總能：僅作條形圖', 'ja': '各項の和が全エネルギーと一致しない：棒グラフ表示'},
    'not_converged': {'en': 'not converged', 'zh_CN': '未收敛', 'zh_TW': '未收斂', 'ja': '未収束'},
    'tension_positive': {'en': r'$\sigma > 0$: tension', 'zh_CN': r'$\sigma > 0$：拉伸',
                         'zh_TW': r'$\sigma > 0$：拉伸', 'ja': r'$\sigma > 0$：引張'},
    'pressure': {'en': '$P$', 'zh_CN': '压强 $P$', 'zh_TW': '壓力 $P$', 'ja': '圧力 $P$'},
    'openmp_threads': {'en': 'OpenMP threads', 'zh_CN': 'OpenMP 线程', 'zh_TW': 'OpenMP 執行緒', 'ja': 'OpenMP スレッド'},
    'converged_state': {'en': 'converged', 'zh_CN': '已收敛', 'zh_TW': '已收斂', 'ja': '収束'},
    'mae': {'en': 'MAE', 'zh_CN': '平均绝对误差', 'zh_TW': '平均絕對誤差', 'ja': '平均絶対誤差'},
    'max_error': {'en': 'max', 'zh_CN': '最大', 'zh_TW': '最大', 'ja': '最大'},
    'inferred': {'en': 'from $E_\\mathrm{F}$, no occupations', 'zh_CN': '由 $E_\\mathrm{F}$ 推断，无占据数',
                 'zh_TW': '由 $E_\\mathrm{F}$ 推斷，無佔據數', 'ja': '$E_\\mathrm{F}$ から推定、占有数なし'},
    'width_note': {'en': 'line width ∝ weight (common scale)', 'zh_CN': '线宽 ∝ 权重（统一标度）',
                   'zh_TW': '線寬 ∝ 權重（統一標度）', 'ja': '線幅 ∝ 重み（共通スケール）'},
    'gaussian_note': {'en': 'Gaussian σ = {sigma:g} eV', 'zh_CN': '高斯展宽 σ = {sigma:g} eV',
                      'zh_TW': '高斯展寬 σ = {sigma:g} eV', 'ja': 'ガウス幅 σ = {sigma:g} eV'},
    # Inequality signs are math text: CJK gothic faces (e.g. Hiragino Sans) lack U+2264.
    'omitted_rows': {'en': '{n} rows at $\\leq$1.02× omitted', 'zh_CN': '已略去 {n} 个缩减 $\\leq$1.02× 的算例',
                     'zh_TW': '已略去 {n} 個縮減 $\\leq$1.02× 的算例', 'ja': '削減 $\\leq$1.02× の {n} 件を省略'},
    'baseline_missing': {'en': 'no 1-worker run: speed-up undefined', 'zh_CN': '缺少单线程基准：加速比无定义',
                         'zh_TW': '缺少單執行緒基準：加速比無定義', 'ja': '1 ワーカー基準なし：高速化率は未定義'},
    'not_converged_rows': {'en': '{n} unconverged runs omitted', 'zh_CN': '已略去 {n} 个未收敛算例',
                           'zh_TW': '已略去 {n} 個未收斂算例', 'ja': '未収束の {n} 件を省略'},
}


from .atlas_labels import ATLAS_LABELS  # noqa: E402  (atlas panel text)
LABELS.update(ATLAS_LABELS)

def tr(key: str, lang: str | None = None, **kwargs) -> str:
    """Translate ``key`` for ``lang`` (default: the active figure style)."""
    lang = lang or current_style().lang
    entry = LABELS.get(key)
    if entry is None:
        text = key
    else:
        text = entry.get(lang) or entry['en']
    return text.format(**kwargs) if kwargs else text


# Energy-ledger term names (sparse_dft.analysis.scf.ENERGY_LEDGER) -> math labels.
ENERGY_TERM_LABELS = {
    'band': r'$E_\mathrm{band}$', 'hartree': r'$-E_\mathrm{H}$', 'xc': r'$E_\mathrm{xc}$',
    'vxc_density': r'$-\int v_\mathrm{xc}\rho$', 'paw_onecenter': r'$E^{1}_\mathrm{PAW}$',
    'paw_expectation': r'$-\langle D^{1}\rangle$', 'paw_core_xc': r'$-E_\mathrm{xc}^\mathrm{core}$',
    'ion_ion': r'$E_\mathrm{Ewald}$', 'local_g0': r'$E_\mathrm{loc}(G{=}0)$',
    'atomic_reference': r'$E_\mathrm{atom}$', 'cutoff_correction': r'$\Delta E_\mathrm{cut}$',
    'entropy': r'$-TS$',
}
