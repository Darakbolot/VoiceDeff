from __future__ import annotations
from .common import *

METHODS = (
    "Undefended",
    "Random Noise",
    "Static PGD",
    "Dynamic PGD (No U-Shape)",
    "High-Frequency PGD (3.75-8 kHz)",
    "Proposed (Dynamic + U-Shape)",
)

def calibrate_asv(
    selection: pd.DataFrame,
    cache: Dict[str, Dict[str, torch.Tensor]],
    cfg: ExperimentConfig,
) -> Tuple[float, pd.DataFrame, float]:
    calib = selection[selection.split == "calibration"]
    genuine_rows, impostor_candidates = [], []
    for spk, group in calib.groupby("speaker_id"):
        paths = list(group.path)
        for a, b in combinations(paths, 2):
            genuine_rows.append({"label": "genuine", "speaker_a": spk, "speaker_b": spk, "score": float(cosine(cache[a]["wavlm"], cache[b]["wavlm"]).cpu())})
    records = calib.to_dict("records")
    for a, b in combinations(records, 2):
        if a["speaker_id"] != b["speaker_id"]:
            impostor_candidates.append({
                "label": "impostor", "speaker_a": a["speaker_id"], "speaker_b": b["speaker_id"],
                "score": float(cosine(cache[a["path"]]["wavlm"], cache[b["path"]]["wavlm"]).cpu()),
            })
    rng = random.Random(cfg.selection_seed)
    rng.shuffle(impostor_candidates)
    impostor_rows = impostor_candidates[: len(genuine_rows)]
    calibration = pd.DataFrame(genuine_rows + impostor_rows)
    threshold, eer = eer_threshold(
        calibration.loc[calibration.label == "genuine", "score"],
        calibration.loc[calibration.label == "impostor", "score"],
    )
    return threshold, calibration, eer

class PGDProtector:
    def __init__(self, cfg: ExperimentConfig, models: RepresentationModels, device: torch.device, routing_threshold: float):
        self.cfg, self.models, self.device = cfg, models, device
        self.routing_threshold = routing_threshold
        self.window = torch.hann_window(cfg.win_length, device=device)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.win_length,
            hop_length=cfg.hop_length, f_min=cfg.mel_fmin, f_max=cfg.mel_fmax,
            n_mels=cfg.n_mels, center=cfg.stft_center, normalized=cfg.stft_normalized,
            mel_scale=cfg.mel_scale, norm=cfg.mel_norm, power=cfg.mel_power,
        ).to(device)
        mel_centers = librosa.mel_frequencies(
            n_mels=cfg.n_mels, fmin=cfg.mel_fmin, fmax=cfg.mel_fmax,
            htk=cfg.mel_scale == "htk",
        )
        self.mel_centers_hz = mel_centers
        self.formant_bins = cfg.formant_mel_bins
        self.freqs = torch.fft.rfftfreq(cfg.n_fft, d=1.0 / cfg.sample_rate).to(device)

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            x, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length, window=self.window,
            center=self.cfg.stft_center, normalized=self.cfg.stft_normalized,
            onesided=True, return_complex=True,
        )

    def istft(self, x: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            x, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length,
            win_length=self.cfg.win_length, window=self.window,
            center=self.cfg.stft_center, normalized=self.cfg.stft_normalized,
            onesided=True, length=length,
        )

    def filter_gradient(self, gradient: torch.Tensor, phase: int) -> torch.Tensor:
        spectrum = self.stft(gradient)
        scales = torch.ones_like(self.freqs)
        if phase == 1:
            # Paper Section III-D: bins >= 80 (2.5 kHz) are scaled by 0.05.
            scales[self.freqs >= self.cfg.phase1_high_cut_hz] = self.cfg.phase1_high_scale
        else:
            # Paper Section III-D: <45 unchanged, 45..120 -> 0.01, >120 -> 0.15.
            middle = (self.freqs >= self.cfg.ushape_low_end_hz) & (self.freqs <= self.cfg.ushape_mid_end_hz)
            high = self.freqs > self.cfg.ushape_mid_end_hz
            scales[middle] = self.cfg.ushape_mid_scale
            scales[high] = self.cfg.ushape_high_scale
        spectrum = spectrum * scales.view(1, -1, 1)
        return self.istft(spectrum, gradient.shape[-1])

    def filter_high_frequency_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Keep only 3.75-8 kHz at 16 kHz; this is high-frequency, not ultrasonic."""
        spectrum = self.stft(gradient)
        keep = (self.freqs >= self.cfg.high_frequency_start_hz).to(spectrum.dtype)
        spectrum = spectrum * keep.view(1, -1, 1)
        return self.istft(spectrum, gradient.shape[-1])

    def protect(
        self,
        mode: str,
        clean: torch.Tensor,
        reference: Dict[str, torch.Tensor],
        opposite: Dict[str, torch.Tensor],
        seed: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if mode not in {"static", "dynamic_no_ushape", "high_frequency", "proposed"}:
            raise ValueError(f"Unknown PGD mode: {mode}")
        seed_everything(seed)
        clean_norm = normalize_waveform(clean)
        mel_ref = torch.log(self.mel(clean_norm) + 1e-6).detach()
        f_ref = self.models.extract(clean, with_grad=False)["semantic"]
        envelope = clean.abs() / (clean.abs().amax(dim=-1, keepdim=True) + 1e-7)
        dynamic_bound = self.cfg.epsilon_max * envelope + self.cfg.baseline_noise
        delta = ((torch.rand_like(clean) * 2.0 - 1.0) * (self.cfg.baseline_noise * 0.5)).requires_grad_(True)
        phase2_steps = 0
        phase_switches = 0
        previous_phase = None
        initial_score = float(cosine(self.models.extract(clean)["wavlm"], reference["wavlm"]).cpu())

        for _ in range(self.cfg.num_iterations):
            adversarial = torch.clamp(clean + delta, -1.0, 1.0)
            reps = self.models.extract(adversarial, with_grad=True)
            s_asv = cosine(reps["wavlm"], reference["wavlm"])

            if mode in {"static", "high_frequency"}:
                phase, weights = 0, self.cfg.static_weights
            elif float(s_asv.detach().cpu()) > self.routing_threshold:
                phase, weights = 1, self.cfg.phase1_weights
            else:
                phase, weights = 2, self.cfg.phase2_weights
                phase2_steps += 1
            if previous_phase is not None and phase != previous_phase:
                phase_switches += 1
            previous_phase = phase

            loss_identity = cosine(reps["wavlm"], reference["wavlm"]) - cosine(reps["wavlm"], opposite["wavlm"])
            loss_hubert = cosine(reps["hubert"], reference["hubert"]) - cosine(reps["hubert"], opposite["hubert"])
            mel_adv = torch.log(self.mel(normalize_waveform(adversarial)) + 1e-6)
            loss_formant = cosine(
                mel_adv[:, : self.formant_bins].flatten(1),
                mel_ref[:, : self.formant_bins].flatten(1),
            )
            # F is explicitly HuBERT hidden layer `semantic_layer`; minimizing
            # 1-cosine preserves its content representation.
            loss_semantic = 1.0 - cosine(reps["semantic"], f_ref)
            loss = (
                weights["identity"] * loss_identity
                + weights["hubert_identity"] * loss_hubert
                + weights["formant"] * loss_formant
                + weights["semantic"] * loss_semantic
            )
            loss.backward()

            with torch.no_grad():
                # Equations (12)-(14): filter the raw loss gradient in STFT,
                # return with ISTFT, then clip its L-infinity norm. This keeps
                # both the declared band scales and the gradient information.
                raw_gradient = delta.grad
                if mode == "proposed":
                    direction = self.filter_gradient(raw_gradient, phase)
                    direction = direction / direction.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
                elif mode == "high_frequency":
                    direction = self.filter_high_frequency_gradient(raw_gradient)
                    direction = direction / direction.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
                else:
                    direction = raw_gradient.sign()
                delta -= self.cfg.step_size * direction
                delta.copy_(torch.maximum(torch.minimum(delta, dynamic_bound), -dynamic_bound))
                delta.grad.zero_()

        protected = torch.clamp(clean + delta.detach(), -1.0, 1.0)
        final_score = float(cosine(self.models.extract(protected)["wavlm"], reference["wavlm"]).cpu())
        return protected, {
            "initial_routing_s_asv": initial_score,
            "final_s_asv": final_score,
            "phase2_fraction": phase2_steps / self.cfg.num_iterations,
            "phase_switches": phase_switches,
        }

def evaluate_candidate(
    clean: torch.Tensor,
    candidate: torch.Tensor,
    enrollment: Dict[str, torch.Tensor],
    models: RepresentationModels,
    quality: SquimQuality,
    asr: Optional[ASREvaluator],
    transcript: Optional[str],
    cfg: ExperimentConfig,
    asv_threshold: float,
    calibrated_asv_threshold: float,
    quality_threshold: float,
    utility_threshold: float,
) -> Dict[str, float]:
    clean_np = clean.squeeze().detach().cpu().numpy()
    test_np = candidate.squeeze().detach().cpu().numpy()
    s_asv = float(cosine(models.extract(candidate)["wavlm"], enrollment["wavlm"]).cpu())
    squim_mos = quality(candidate, clean)
    dtw, mcd = calculate_dtw_and_mcd(clean_np, test_np, cfg.sample_rate)
    hypothesis = asr.transcribe(candidate, cfg.sample_rate) if asr is not None and transcript else None
    stoi_score = float(stoi(clean_np, test_np, cfg.sample_rate, extended=False))
    result = {
        "s_asv": s_asv,
        "asv_accept": int(s_asv >= asv_threshold),
        "asv_obfuscated": int(s_asv < asv_threshold),
        "asv_obfuscated_calibration_audit": int(s_asv < calibrated_asv_threshold),
        "squim_mos": squim_mos,
        "quality_degraded": int(squim_mos < quality_threshold),
        "snr_db": calculate_snr(clean_np, test_np),
        "pesq": safe_pesq(clean_np, test_np, cfg.sample_rate),
        "stoi": stoi_score,
        "quality_preserved": int(stoi_score >= utility_threshold),
        "ssim": calculate_ssim(clean_np, test_np, cfg.sample_rate),
        "mcd_db": mcd,
        "dtw_cost": dtw,
        "wer": word_error_rate(transcript, hypothesis),
        **perturbation_band_energy(clean_np, test_np, cfg),
    }
    # Exact sample-level SQUIM-proxy definition; this is not called NISQA.
    result["dsr_success"] = int(result["asv_obfuscated"] and result["quality_degraded"])
    # Primary privacy-utility criterion: the source must remain intelligible.
    result["privacy_utility_success"] = int(result["asv_obfuscated"] and result["quality_preserved"])
    return result

def build_manifest(
    cfg: ExperimentConfig,
    selection: pd.DataFrame,
    device: torch.device,
    asv_threshold: float,
    calibrated_asv_threshold: float,
    asv_eer: float,
    quality_threshold: float,
    quality_eer: float,
    utility_threshold: float,
    utility_eer: float,
    protector: PGDProtector,
) -> Dict:
    freq_resolution = cfg.sample_rate / cfg.n_fft
    cutoffs = {
        name: {"hz": hz, "nearest_bin": int(round(hz / freq_resolution))}
        for name, hz in {
            "phase1_high_cut": cfg.phase1_high_cut_hz,
            "ushape_low_end": cfg.ushape_low_end_hz,
            "ushape_mid_end": cfg.ushape_mid_end_hz,
            "high_frequency_start": cfg.high_frequency_start_hz,
        }.items()
    }
    return {
        "config": asdict(cfg),
        "dataset": {
            "name": cfg.dataset_name,
            "selected_speakers": int(selection.speaker_id.nunique()),
            "selected_utterances": int(len(selection)),
            "counts_by_split_role": selection.groupby(["split", "role"]).size().astype(int).to_dict(),
            "selection_rule": "speaker-disjoint; complete utterances in [minimum_seconds, segment_seconds]; original duration retained; no truncation or padding",
            "input_level_protocol": f"DC removal and RMS normalization to {cfg.input_target_rms_dbfs:g} dBFS with peak ceiling {cfg.input_peak_limit:g}; identical for both datasets; pitch and timing unchanged",
            "speaker_id_rule": "parent directory pXXX" if cfg.dataset_name.lower() == "vctk" else "grandparent directory in LibriTTS layout",
            "duplicate_recording_rule": "VCTK _mic1/_mic2 recordings sharing a source utterance ID are deduplicated before sampling; mic1 is retained deterministically",
            "statistical_scope": "pilot/debug experiment only when probe count < 30; do not use 4-probe results for paper-level significance claims",
        },
        "operational_definitions": {
            "R": f"L2-normalized embeddings from {cfg.wavlm_checkpoint}",
            "H": f"mean-pooled last hidden state of {cfg.hubert_checkpoint}",
            "F": f"mean-pooled hidden_states[{cfg.semantic_layer}] of {cfg.hubert_checkpoint}; this is not called a U-Net",
            "R_ref_H_ref": "speaker-reference centroids computed only from the test speaker's enrollment utterance(s); the protected probe itself is excluded",
            "C_opp_H_opp": "same hardest calibration-speaker centroid selected by maximum WavLM cosine similarity; calibration speakers are disjoint from test speakers",
            "evaluation_reference": "the same held-out enrollment centroid used by the ASV verifier and routing score, preventing train/evaluation reference mismatch",
            "s_asv_scale": "raw cosine similarity in [-1, 1]",
            "asv_accept_rate": "100 * mean(s_asv >= calibrated ASV threshold)",
            "SQUIM_proxy_DSR": "100 * mean((s_asv < fixed paper ASV threshold 0.25) AND (SQUIM_MOS < calibration-derived SQUIM threshold))",
            "privacy_utility_success": "100 * mean((s_asv < calibrated ASV threshold) AND (STOI >= calibration-derived utility threshold))",
            "primary_endpoint": "privacy_utility_success; SQUIM proxy DSR is secondary because degrading the protected source conflicts with transparency",
            "quality_note": "SQUIM-MOS is reported by name; it is not NISQA and is never synthetically imputed. Therefore this notebook does not claim to reproduce the paper's NISQA<3.0 DSR exactly.",
            "L_identity": "cos(R_adv,R_ref)-cos(R_adv,C_opp), minimized",
            "L_hubert_identity": "cos(H_adv,H_ref)-cos(H_adv,H_opp), minimized",
            "L_formant": "cosine similarity of the first 45 log-Mel bins, minimized to cause formant distortion",
            "L_semantic": "implemented as 1-cos(F_adv,F_ref) with positive weights 5/20; gradient-equivalent to the paper's -5/-20 times cosine similarity",
        },
        "method_ablation": {
            "Undefended": "clean probe without perturbation",
            "Random Noise": "uniform random perturbation projected to the same sample-wise dynamic bound",
            "Static PGD": "fixed Phase-1 loss weights for every iteration; no frequency-domain gradient filter",
            "Dynamic PGD (No U-Shape)": "the paper's threshold-routed Phase-1/Phase-2 loss weights without a frequency-domain gradient filter",
            "High-Frequency PGD (3.75-8 kHz)": "fixed-weight PGD whose STFT gradient is zeroed below 3.75 kHz; not ultrasonic at 16 kHz",
            "Proposed (Dynamic + U-Shape)": "the paper's threshold-routed Phase-1/Phase-2 weights and exact STFT gradient scales: Phase 1 >=2.5 kHz x0.05; Phase 2 <1.4 kHz x1, 1.4-3.75 kHz x0.01, >3.75 kHz x0.15",
            "fairness": "all PGD variants use identical iteration count, step size, perturbation bound, sample order, and paired initialization seed",
        },
        "stft": {
            "sample_rate_hz": cfg.sample_rate, "n_fft": cfg.n_fft, "win_length": cfg.win_length,
            "hop_length": cfg.hop_length, "window": "periodic Hann", "center": cfg.stft_center,
            "normalized": cfg.stft_normalized, "onesided": True,
            "pgd_filter_order": "raw loss gradient -> STFT band scaling -> ISTFT -> L-infinity gradient clipping -> alpha-scaled update, matching Equations (12)-(14) and the narrated gradient-clipping loop",
            "frequency_resolution_hz": freq_resolution, "nyquist_hz": cfg.sample_rate / 2,
            "cutoffs": cutoffs,
            "terminology": "All bands are audible at 16 kHz; the upper band is called high-frequency, not ultrasonic.",
            "formant_mel_bins": protector.formant_bins,
            "formant_mel_range_hz": [float(protector.mel_centers_hz[0]), float(protector.mel_centers_hz[protector.formant_bins - 1])],
        },
        "calibration": {
            "paper_asv_evaluation_threshold": asv_threshold,
            "calibration_eer_threshold_for_audit_only": calibrated_asv_threshold,
            "asv_eer": asv_eer,
            "routing_threshold": protector.routing_threshold,
            "threshold_policy": "The fixed 0.24 routing and 0.25 evaluation operating points from the paper are preserved. The speaker-disjoint calibration EER threshold is reported for sensitivity/audit and is not substituted into the method.",
            "quality_metric": cfg.quality_metric_name, "quality_threshold": quality_threshold,
            "quality_eer": quality_eer, "degraded_anchor_snr_db": cfg.quality_calibration_snr_db,
            "utility_metric": "STOI", "utility_threshold": utility_threshold,
            "utility_eer": utility_eer,
        },
        "runtime": {
            "device": str(device), "platform": platform.platform(), "python": platform.python_version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": package_versions(),
        },
    }

def save_json(data: Dict, path: Path) -> None:
    def convert(value):
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value
    path.write_text(json.dumps(convert(data), indent=2, ensure_ascii=False), encoding="utf-8")

def aggregate_statistics(results: pd.DataFrame, cfg: ExperimentConfig, output: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "s_asv", "snr_db", "pesq", "stoi", "ssim", "mcd_db", "dtw_cost", "squim_mos", "wer",
        "band_energy_low_fraction", "band_energy_mid_fraction", "band_energy_high_fraction",
        "phase2_fraction", "phase_switches", "routing_s_asv_initial", "routing_s_asv_final",
    ]
    per_run = results.groupby(["method", "run_seed"])[metrics].mean().reset_index()
    rates = results.groupby(["method", "run_seed"])[["asv_accept", "asv_obfuscated", "quality_preserved", "privacy_utility_success", "dsr_success"]].mean().mul(100).reset_index()
    per_run = per_run.merge(rates, on=["method", "run_seed"])
    per_run.to_csv(output / "run_level_summary.csv", index=False)

    rows = []
    for method, group in per_run.groupby("method"):
        for metric in metrics + ["asv_accept", "asv_obfuscated", "quality_preserved", "privacy_utility_success", "dsr_success"]:
            values = group[metric].dropna().to_numpy(float)
            mean = float(np.mean(values)) if len(values) else float("nan")
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            half = float(stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values))) if len(values) > 1 else float("nan")
            rows.append({"method": method, "metric": metric, "n_runs": len(values), "mean": mean, "std": sd, "ci95_low": mean - half, "ci95_high": mean + half})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "summary_statistics.csv", index=False)
    return per_run, summary

def paired_significance(results: pd.DataFrame, output: Path) -> pd.DataFrame:
    proposed = "Proposed (Dynamic + U-Shape)"
    rows = []
    for baseline in ("Static PGD", "Dynamic PGD (No U-Shape)", "High-Frequency PGD (3.75-8 kHz)"):
        for metric in ("s_asv", "pesq", "stoi", "ssim", "squim_mos"):
            # Seeds are repeated measurements, not independent utterances.
            averaged = results.groupby(["speaker_id", "sample_id", "method"], as_index=False)[metric].mean()
            wide = averaged.pivot_table(index=["speaker_id", "sample_id"], columns="method", values=metric).dropna()
            if baseline not in wide or proposed not in wide or len(wide) < 2:
                continue
            try:
                test = stats.wilcoxon(wide[proposed], wide[baseline], zero_method="zsplit")
                statistic, pvalue = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, pvalue = float("nan"), 1.0
            rows.append({"proposed": proposed, "baseline": baseline, "metric": metric, "n_pairs": len(wide), "test": "two-sided paired Wilcoxon on per-utterance seed means", "statistic": statistic, "p_value": pvalue})
        binary_average = results.groupby(["speaker_id", "sample_id", "method"], as_index=False)["privacy_utility_success"].mean()
        binary_average["majority_success"] = (binary_average["privacy_utility_success"] >= 0.5).astype(int)
        binary = binary_average.pivot_table(index=["speaker_id", "sample_id"], columns="method", values="majority_success").dropna()
        if baseline in binary and proposed in binary:
            proposed_only = int(((binary[proposed] == 1) & (binary[baseline] == 0)).sum())
            baseline_only = int(((binary[proposed] == 0) & (binary[baseline] == 1)).sum())
            discordant = proposed_only + baseline_only
            pvalue = float(stats.binomtest(proposed_only, discordant, 0.5).pvalue) if discordant else 1.0
            rows.append({"proposed": proposed, "baseline": baseline, "metric": "privacy_utility_success", "n_pairs": len(binary), "test": "exact paired McNemar on per-utterance majority across seeds", "statistic": proposed_only - baseline_only, "p_value": pvalue})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["holm_significant_0.05"] = False
        ordered = table.p_value.sort_values().index
        m = len(ordered)
        still_rejecting = True
        for rank, idx in enumerate(ordered):
            reject = still_rejecting and table.loc[idx, "p_value"] <= 0.05 / (m - rank)
            table.loc[idx, "holm_significant_0.05"] = reject
            still_rejecting = reject
    table.to_csv(output / "paired_significance_tests.csv", index=False)
    return table

def pareto_analysis(results: pd.DataFrame, output: Path) -> pd.DataFrame:
    table = results.groupby("method").agg(
        asv_rejection_rate=("asv_obfuscated", "mean"), pesq=("pesq", "mean"), stoi=("stoi", "mean")
    ).reset_index()
    values = table[["asv_rejection_rate", "pesq", "stoi"]].to_numpy(float)
    optimal = []
    for i, point in enumerate(values):
        dominated = any(np.all(other >= point) and np.any(other > point) for j, other in enumerate(values) if i != j)
        optimal.append(not dominated)
    table["pareto_optimal"] = optimal
    table.to_csv(output / "pareto_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    for row in table.itertuples():
        ax.scatter(row.asv_rejection_rate * 100, row.pesq, s=80 + 180 * max(row.stoi, 0), marker="*" if row.pareto_optimal else "o")
        ax.annotate(row.method, (row.asv_rejection_rate * 100, row.pesq), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("ASV rejection / identity-obfuscation rate (%) - higher is better")
    ax.set_ylabel("PESQ of protected source - higher is better")
    ax.set_title("Protection-quality Pareto analysis (marker size = STOI)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "pareto_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return table

def plot_results_table(summary: pd.DataFrame, output: Path) -> None:
    chosen = {
        "s_asv": "s_ASV cosine (down)", "asv_obfuscated": "ASV reject % (up)",
        "pesq": "PESQ (up)", "stoi": "STOI (up)", "wer": "WER (down)",
        "ssim": "SSIM (up)", "squim_mos": "SQUIM-MOS (up)",
        "privacy_utility_success": "Privacy-utility success % (up)",
        "dsr_success": "SQUIM proxy DSR % (up)",
    }
    rows = []
    for method in METHODS:
        row = {"Method": method}
        sub = summary[summary.method == method].set_index("metric")
        for metric, label in chosen.items():
            if metric not in sub.index or pd.isna(sub.loc[metric, "mean"]):
                row[label] = "NA"
            else:
                scale = 1.0
                row[label] = f"{sub.loc[metric, 'mean'] * scale:.3f} [{sub.loc[metric, 'ci95_low'] * scale:.3f}, {sub.loc[metric, 'ci95_high'] * scale:.3f}]"
        rows.append(row)
    table_data = pd.DataFrame(rows)
    table_data.to_csv(output / "paper_table_values.csv", index=False)
    fig, ax = plt.subplots(figsize=(21, 3.8))
    ax.axis("off")
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 2.0)
    for j in range(len(table_data.columns)):
        table[0, j].set_facecolor("#d9eaf7")
        table[0, j].set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(output / "tabel_hasil_dengan_ci95.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_metric_distributions(results: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, metric, title in zip(axes.flat, ("s_asv", "pesq", "stoi", "squim_mos"), ("ASV cosine", "PESQ", "STOI", "SQUIM-MOS")):
        data = [results.loc[results.method == method, metric].dropna() for method in METHODS]
        ax.boxplot(data, tick_labels=[m.replace(" (", "\n(") for m in METHODS], showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "distribusi_metrik.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_example_audio(example: Dict[str, np.ndarray], cfg: ExperimentConfig, output: Path) -> None:
    fig, axes = plt.subplots(len(example), 1, figsize=(11, 2.3 * len(example)), sharex=True)
    for ax, (method, waveform) in zip(axes, example.items()):
        spectrum = librosa.feature.melspectrogram(y=waveform, sr=cfg.sample_rate, n_mels=128, fmax=cfg.sample_rate / 2)
        librosa.display.specshow(librosa.power_to_db(spectrum, ref=np.max), sr=cfg.sample_rate, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        ax.set_title(method)
    fig.tight_layout()
    fig.savefig(output / "contoh_mel_spectrogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    clean = example["Undefended"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for method, waveform in example.items():
        if method == "Undefended":
            continue
        perturbation = waveform - clean
        freqs = np.fft.rfftfreq(len(perturbation), 1 / cfg.sample_rate)
        power = 20 * np.log10(np.abs(np.fft.rfft(perturbation)) + 1e-10)
        ax.plot(freqs, power, label=method, alpha=0.8)
    ax.axvline(cfg.ushape_low_end_hz, color="black", linestyle="--", linewidth=1)
    ax.axvline(cfg.ushape_mid_end_hz, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Frequency (Hz)", ylabel="Perturbation magnitude (dB)", title="Perturbation spectrum and declared U-Shape cutoffs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "spektrum_perturbasi_ushape.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def run_experiment(cfg: ExperimentConfig) -> Dict[str, pd.DataFrame]:
    cfg.validate()
    seed_everything(cfg.selection_seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "audio_examples").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selection = build_speaker_disjoint_split(cfg)
    selection.to_csv(output / "sample_selection_manifest.csv", index=False)
    level_audit = audit_audio_levels(selection, cfg, device)
    level_audit.to_csv(output / "audio_level_audit.csv", index=False)
    level_audit.groupby(["split", "role"]).agg(
        utterances=("utterance_id", "size"), raw_rms_dbfs_mean=("raw_rms_dbfs", "mean"),
        raw_peak_mean=("raw_peak", "mean"), processed_rms_dbfs_mean=("processed_rms_dbfs", "mean"),
        processed_peak_mean=("processed_peak", "mean"),
    ).reset_index().to_csv(output / "audio_level_summary.csv", index=False)
    models = RepresentationModels(cfg, device)
    cache = precompute_representations(selection, cfg, models, device)
    calibration_centroids = speaker_centroids(selection, cache, "calibration")
    enrollment_centroids = speaker_centroids(selection, cache, "test", role="enrollment")

    calibrated_asv_threshold, asv_calibration, asv_eer = calibrate_asv(selection, cache, cfg)
    asv_calibration.to_csv(output / "asv_calibration_scores.csv", index=False)
    asv_threshold = cfg.asv_evaluation_threshold
    routing_threshold = cfg.routing_threshold
    threshold_rows = []
    for threshold_name, threshold_value in (
        ("paper_routing", routing_threshold),
        ("paper_evaluation", asv_threshold),
        ("calibration_eer_audit_only", calibrated_asv_threshold),
    ):
        genuine = asv_calibration.loc[asv_calibration.label == "genuine", "score"].to_numpy(float)
        impostor = asv_calibration.loc[asv_calibration.label == "impostor", "score"].to_numpy(float)
        threshold_rows.append({
            "threshold_name": threshold_name, "threshold": threshold_value,
            "false_reject_rate": float(np.mean(genuine < threshold_value)),
            "false_accept_rate": float(np.mean(impostor >= threshold_value)),
            "n_genuine": len(genuine), "n_impostor": len(impostor),
        })
    pd.DataFrame(threshold_rows).to_csv(output / "asv_threshold_audit.csv", index=False)

    quality = SquimQuality()
    quality_threshold, quality_calibration, quality_eer, utility_threshold, utility_eer = calibrate_quality(selection, cfg, quality, device)
    quality_calibration.to_csv(output / "quality_calibration_scores.csv", index=False)
    protector = PGDProtector(cfg, models, device, routing_threshold)
    manifest = build_manifest(cfg, selection, device, asv_threshold, calibrated_asv_threshold, asv_eer, quality_threshold, quality_eer, utility_threshold, utility_eer, protector)
    save_json(manifest, output / "experiment_manifest.json")

    asr = ASREvaluator(cfg.asr_checkpoint, device) if cfg.run_asr_wer else None
    probes = selection[(selection.split == "test") & (selection.role == "probe")].reset_index(drop=True)
    result_rows: List[Dict] = []
    example_waveforms: Dict[str, np.ndarray] = {}

    for run_seed in cfg.run_seeds:
        for sample_index, row in enumerate(probes.to_dict("records")):
            clean = load_audio(row["path"], cfg, device)
            enrollment = enrollment_centroids[row["speaker_id"]]
            # R_ref/H_ref operationalize the claimed ASV task: a centroid
            # built from separate enrollment speech, never from this probe.
            optimization_reference = enrollment
            opposite_speaker, opposite = choose_hard_negative(optimization_reference["wavlm"], calibration_centroids)
            sample_seed = run_seed * 100_000 + sample_index
            dynamic_bound = cfg.epsilon_max * clean.abs() / (clean.abs().amax() + 1e-7) + cfg.baseline_noise
            seed_everything(sample_seed)
            random_noise = (torch.rand_like(clean) * 2 - 1) * dynamic_bound
            candidates = {
                "Undefended": clean,
                "Random Noise": torch.clamp(clean + random_noise, -1, 1),
            }
            phase2 = {"Undefended": float("nan"), "Random Noise": float("nan")}
            phase_switches = {"Undefended": float("nan"), "Random Noise": float("nan")}
            routing_initial = {"Undefended": float("nan"), "Random Noise": float("nan")}
            routing_final = {"Undefended": float("nan"), "Random Noise": float("nan")}
            for method, mode in (
                ("Static PGD", "static"),
                ("Dynamic PGD (No U-Shape)", "dynamic_no_ushape"),
                ("High-Frequency PGD (3.75-8 kHz)", "high_frequency"),
                ("Proposed (Dynamic + U-Shape)", "proposed"),
            ):
                protected, diagnostics = protector.protect(mode, clean, optimization_reference, opposite, sample_seed)
                candidates[method] = protected
                phase2[method] = diagnostics["phase2_fraction"]
                phase_switches[method] = diagnostics["phase_switches"]
                routing_initial[method] = diagnostics["initial_routing_s_asv"]
                routing_final[method] = diagnostics["final_s_asv"]

            transcript = read_transcript(row["path"], cfg)
            for method, candidate in candidates.items():
                metrics = evaluate_candidate(
                    clean, candidate, enrollment, models, quality, asr, transcript, cfg,
                    asv_threshold, calibrated_asv_threshold, quality_threshold, utility_threshold,
                )
                result_rows.append({
                    "sample_id": row["utterance_id"], "speaker_id": row["speaker_id"],
                    "path": row["path"], "run_seed": run_seed, "method": method,
                    "opposite_speaker_id": opposite_speaker, "phase2_fraction": phase2[method],
                    "phase_switches": phase_switches[method],
                    "routing_s_asv_initial": routing_initial[method], "routing_s_asv_final": routing_final[method],
                    "asv_threshold": asv_threshold, "quality_threshold": quality_threshold, "utility_threshold": utility_threshold,
                    **metrics,
                })
                if run_seed == cfg.run_seeds[0] and sample_index == 0:
                    example_waveforms[method] = candidate.squeeze().detach().cpu().numpy()
                    if cfg.save_audio_examples:
                        sf.write(output / "audio_examples" / f"{method.replace(' ', '_').replace('/', '_')}.wav", candidate.squeeze().detach().cpu().numpy(), cfg.sample_rate)

            if (sample_index + 1) % cfg.checkpoint_every == 0:
                pd.DataFrame(result_rows).to_csv(output / "per_sample_metrics_checkpoint.csv", index=False)

    results = pd.DataFrame(result_rows)
    results.to_csv(output / "per_sample_metrics.csv", index=False)
    threshold_sensitivity = results.groupby("method").agg(
        fixed_paper_threshold_rejection_rate=("asv_obfuscated", "mean"),
        calibration_eer_threshold_rejection_rate=("asv_obfuscated_calibration_audit", "mean"),
    ).mul(100).reset_index()
    threshold_sensitivity.to_csv(output / "asv_threshold_sensitivity.csv", index=False)
    per_run, summary = aggregate_statistics(results, cfg, output)
    significance = paired_significance(results, output)
    pareto = pareto_analysis(results, output)
    comparison_rows = []
    comparison_metrics = {
        "s_asv": "lower", "asv_obfuscated": "higher", "pesq": "higher",
        "stoi": "higher", "ssim": "higher", "dsr_success": "higher",
    }
    proposed_name = "Proposed (Dynamic + U-Shape)"
    for baseline in ("Static PGD", "Dynamic PGD (No U-Shape)", "High-Frequency PGD (3.75-8 kHz)"):
        for metric, direction in comparison_metrics.items():
            proposed_mean = float(results.loc[results.method == proposed_name, metric].mean())
            baseline_mean = float(results.loc[results.method == baseline, metric].mean())
            difference = proposed_mean - baseline_mean
            comparison_rows.append({
                "baseline": baseline, "metric": metric, "better_direction": direction,
                "proposed_mean": proposed_mean, "baseline_mean": baseline_mean,
                "proposed_minus_baseline": difference,
                "proposed_wins": bool(difference < 0 if direction == "lower" else difference > 0),
            })
    proposed_comparison = pd.DataFrame(comparison_rows)
    proposed_comparison.to_csv(output / "proposed_vs_baselines.csv", index=False)
    plot_results_table(summary, output)
    plot_metric_distributions(results, output)
    if example_waveforms:
        plot_example_audio(example_waveforms, cfg, output)
    return {"per_sample": results, "per_run": per_run, "summary": summary, "significance": significance, "pareto": pareto, "comparison": proposed_comparison}
