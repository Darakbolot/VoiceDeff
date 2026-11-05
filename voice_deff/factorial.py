from __future__ import annotations
from .common import *

def calibrate_asv(
    selection: pd.DataFrame,
    cache: Dict[str, Dict[str, torch.Tensor]],
    cfg: ExperimentConfig,
) -> Tuple[float, pd.DataFrame, float]:
    # Kalibrasi harus meniru scoring test: centroid dari dua enrollment
    # dibandingkan dengan satu probe yang tidak ikut membentuk centroid.
    calib = selection[selection.split == "calibration"]
    enrollment_size = cfg.enrollment_utterances_per_speaker
    genuine_rows, impostor_candidates, centroid_rows = [], [], []
    probe_rows = calib[["speaker_id", "path", "utterance_id"]].to_dict("records")
    for spk, group in calib.groupby("speaker_id"):
        paths = sorted(group.path.tolist())
        if len(paths) <= enrollment_size:
            raise ValueError(f"Calibration speaker {spk} needs more than {enrollment_size} utterances.")
        for enrollment_paths in combinations(paths, enrollment_size):
            enrollment_set = set(enrollment_paths)
            centroid = F.normalize(
                torch.stack([cache[path]["wavlm"] for path in enrollment_paths]).mean(dim=0),
                dim=-1,
            )
            centroid_rows.append({
                "speaker_id": spk, "enrollment_paths": enrollment_paths, "centroid": centroid,
            })
            for probe_path in paths:
                if probe_path in enrollment_set:
                    continue
                genuine_rows.append({
                    "label": "genuine", "speaker_a": spk, "speaker_b": spk,
                    "enrollment_paths": "|".join(enrollment_paths), "probe_path": probe_path,
                    "score": float(cosine(centroid, cache[probe_path]["wavlm"]).cpu()),
                    "protocol": f"{enrollment_size}-utterance centroid-to-held-out-probe",
                })
    for enrollment in centroid_rows:
        for probe in probe_rows:
            if probe["speaker_id"] == enrollment["speaker_id"]:
                continue
            impostor_candidates.append({
                "label": "impostor", "speaker_a": enrollment["speaker_id"],
                "speaker_b": probe["speaker_id"],
                "enrollment_paths": "|".join(enrollment["enrollment_paths"]),
                "probe_path": probe["path"],
                "score": float(cosine(enrollment["centroid"], cache[probe["path"]]["wavlm"]).cpu()),
                "protocol": f"{enrollment_size}-utterance centroid-to-held-out-probe",
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

def precompute_calibration_wavlm(
    selection: pd.DataFrame, cfg: ExperimentConfig, device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Extract only the 100 calibration embeddings; HuBERT and PGD are skipped."""
    model = WavLMForXVector.from_pretrained(cfg.wavlm_checkpoint).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache = {}
    subset = selection[selection.split == "calibration"]
    for row in tqdm(subset.to_dict("records"), desc="Calibration WavLM embeddings"):
        waveform = normalize_waveform(load_audio(row["path"], cfg, device))
        with torch.inference_mode():
            embedding = F.normalize(model(input_values=waveform).embeddings, dim=-1)
        cache[row["path"]] = {"wavlm": embedding}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache

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
        if mode not in {"static", "static_ushape", "dynamic_no_ushape", "high_frequency", "proposed"}:
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

            if mode == "static_ushape":
                phase, weights = 1, self.cfg.static_weights
            elif mode in {"static", "high_frequency"}:
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
                if mode in {"proposed", "static_ushape"}:
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
        # Endpoint utama memakai threshold EER dari calibration speakers.
        "asv_accept": int(s_asv >= calibrated_asv_threshold),
        "asv_obfuscated": int(s_asv < calibrated_asv_threshold),
        # Threshold 0.25 dipertahankan hanya sebagai audit legacy, bukan ASV EER.
        "asv_accept_fixed_paper": int(s_asv >= asv_threshold),
        "asv_obfuscated_fixed_paper": int(s_asv < asv_threshold),
        "asv_obfuscated_calibrated_eer": int(s_asv < calibrated_asv_threshold),
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

def run(dataset, dataset_glob, old_results_path, output_path, run_asr_wer=True):
    # =========================
    # KONFIGURASI PENGGUNA
    # =========================
    TARGET_DATASET = dataset

    # None = cari hasil lama secara otomatis di /kaggle/input.
    # Dapat diisi folder hasil lama atau file ZIP bila pencarian otomatis tidak sesuai.
    OLD_RESULT_SOURCE = str(old_results_path)

    # True mempertahankan WER untuk konsistensi dengan hasil lama, tetapi menambah waktu.
    RUN_ASR_WER = run_asr_wer

    DATASET_GLOBS = {
        "LibriTTS": "./datasets/libri-tts-train-clean-360-part-2/**/*.wav",
        "VCTK": "./datasets/english-multispeaker-corpus-for-voice-cloning/VCTK-Corpus/VCTK-Corpus/wav48/**/*.wav",
    }

    METHOD_NEW = "Static PGD + U-Shape"
    METHOD_STATIC = "Static PGD"
    METHOD_DYNAMIC_NO_FILTER = "Dynamic PGD (No U-Shape)"
    METHOD_DYNAMIC_FILTER = "Proposed (Dynamic + U-Shape)"

    import glob
    import shutil
    import zipfile


    def _read_manifest(directory: Path) -> Optional[Dict]:
        path = directory / "experiment_manifest.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


    def _valid_old_result(directory: Path, dataset_name: str) -> bool:
        manifest = _read_manifest(directory)
        metrics_path = directory / "per_sample_metrics.csv"
        selection_path = directory / "sample_selection_manifest.csv"
        if manifest is None or not metrics_path.is_file() or not selection_path.is_file():
            return False
        if str(manifest.get("config", {}).get("dataset_name", "")).casefold() != dataset_name.casefold():
            return False
        try:
            selection = pd.read_csv(selection_path, usecols=["speaker_id", "utterance_id", "split", "role"])
            metrics = pd.read_csv(metrics_path, usecols=["sample_id", "run_seed", "method"])
        except Exception:
            return False
        expected_methods = {
            "Undefended", "Random Noise", "Static PGD",
            "Dynamic PGD (No U-Shape)",
            "High-Frequency PGD (3.75-8 kHz)",
            "Proposed (Dynamic + U-Shape)",
        }
        return (
            len(selection) == 500
            and selection.speaker_id.nunique() == 60
            and len(metrics) == 5760
            and set(metrics.method.unique()) == expected_methods
        )


    def resolve_old_result(dataset_name: str, explicit_source: Optional[str]) -> Path:
        if explicit_source:
            source = Path(explicit_source)
            if source.is_file() and source.suffix.lower() == ".zip":
                extracted = Path("/kaggle/working") / f"_old_{dataset_name.lower()}_extracted"
                extracted.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(source) as archive:
                    archive.extractall(extracted)
                search_root = extracted
            elif source.is_dir():
                search_root = source
            else:
                raise FileNotFoundError(f"OLD_RESULT_SOURCE tidak ditemukan: {source}")
            candidates = [search_root] + [p.parent for p in search_root.rglob("experiment_manifest.json")]
        else:
            input_root = Path("/kaggle/input")
            candidates = [p.parent for p in input_root.rglob("experiment_manifest.json")]

        valid = sorted(
            {candidate.resolve() for candidate in candidates if _valid_old_result(candidate, dataset_name)},
            key=lambda path: (int((path / "asv_centroid_recalibration_status.csv").is_file()), len(path.parts), str(path)),
            reverse=True,
        )
        if not valid:
            raise FileNotFoundError(
                f"Hasil final {dataset_name} tidak ditemukan. Pasang ZIP/folder hasil lama "
                "sebagai Kaggle Input atau isi OLD_RESULT_SOURCE secara eksplisit."
            )
        return valid[0]


    def resolve_audio_paths(saved_selection: pd.DataFrame, dataset_name: str, dataset_glob: str) -> pd.DataFrame:
        files = sorted(glob.glob(dataset_glob, recursive=True))
        if not files:
            raise FileNotFoundError(f"Dataset WAV tidak ditemukan dengan pola: {dataset_glob}")

        index: Dict[Tuple[str, str], List[str]] = {}
        for path in files:
            path_obj = Path(path)
            spk = path_obj.parent.name if dataset_name == "VCTK" else path_obj.parent.parent.name
            source_id = re.sub(r"_mic[12]$", "", path_obj.stem) if dataset_name == "VCTK" else path_obj.stem
            index.setdefault((str(spk), source_id), []).append(path)

        resolved = saved_selection.copy()
        resolved_paths = []
        for row in resolved.to_dict("records"):
            source_id = str(row.get("source_utterance_id", row["utterance_id"]))
            if dataset_name == "VCTK":
                source_id = re.sub(r"_mic[12]$", "", source_id)
            candidates = index.get((str(row["speaker_id"]), source_id), [])
            if dataset_name == "VCTK" and len(candidates) > 1:
                mic1 = [path for path in candidates if Path(path).stem.endswith("_mic1")]
                candidates = mic1 or candidates
            if not candidates:
                raise FileNotFoundError(
                    f"Audio manifest tidak ditemukan dalam dataset: speaker={row['speaker_id']}, "
                    f"utterance={source_id}"
                )
            resolved_paths.append(sorted(candidates)[0])

        resolved["path"] = resolved_paths
        if len(resolved) != 500 or resolved.speaker_id.nunique() != 60:
            raise RuntimeError("Manifest hasil lama tidak lagi berisi tepat 500 utterances dari 60 speakers.")
        if resolved.path.duplicated().any():
            raise RuntimeError("Resolusi dataset menghasilkan path audio duplikat.")
        return resolved


    if TARGET_DATASET not in DATASET_GLOBS:
        raise ValueError("TARGET_DATASET harus 'LibriTTS' atau 'VCTK'.")

    old_result_dir = resolve_old_result(TARGET_DATASET, OLD_RESULT_SOURCE)
    old_manifest = _read_manifest(old_result_dir)
    if "primary_asv_evaluation_threshold" not in old_manifest["calibration"]:
        raise ValueError("Mode factorial membutuhkan hasil calibrated_asv, bukan fixed_threshold.")
    old_results = pd.read_csv(old_result_dir / "per_sample_metrics.csv")
    saved_selection = pd.read_csv(old_result_dir / "sample_selection_manifest.csv")

    config_values = {
        key: value for key, value in old_manifest["config"].items()
        if key in ExperimentConfig.__dataclass_fields__
    }
    config_values.update({
        "dataset_name": TARGET_DATASET,
        "dataset_glob": dataset_glob,
        "output_dir": str(output_path),
        "run_asr_wer": RUN_ASR_WER,
        "checkpoint_every": 10,
    })
    if "run_seeds" in config_values:
        config_values["run_seeds"] = tuple(config_values["run_seeds"])
    cfg = ExperimentConfig(**config_values)
    cfg.validate()

    selection = resolve_audio_paths(saved_selection, TARGET_DATASET, cfg.dataset_glob)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_dir / "resolved_sample_selection_manifest.csv", index=False)

    calibration_info = old_manifest["calibration"]
    legacy_asv_threshold = float(
        calibration_info.get("legacy_fixed_score_threshold_audit_only",
                             calibration_info.get("paper_asv_evaluation_threshold", 0.25))
    )
    calibrated_asv_threshold = float(
        calibration_info.get("calibrated_asv_eer_threshold",
                             calibration_info.get("primary_asv_evaluation_threshold",
                                                  calibration_info.get("calibration_eer_threshold_for_audit_only")))
    )
    quality_threshold = float(calibration_info["quality_threshold"])
    utility_threshold = float(calibration_info["utility_threshold"])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU Kaggle harus diaktifkan untuk eksperimen 1000 iterasi.")

    models = RepresentationModels(cfg, device)
    cache = precompute_representations(selection, cfg, models, device)
    calibration_centroids = speaker_centroids(selection, cache, "calibration")
    enrollment_centroids = speaker_centroids(selection, cache, "test", role="enrollment")
    protector = PGDProtector(cfg, models, device, cfg.routing_threshold)
    quality = SquimQuality()
    asr = ASREvaluator(cfg.asr_checkpoint, device) if cfg.run_asr_wer else None

    probes = selection[(selection.split == "test") & (selection.role == "probe")].reset_index(drop=True)
    if len(probes) != 320 or probes.speaker_id.nunique() != 40:
        raise RuntimeError("Eksperimen tambahan wajib menggunakan 320 probe dari 40 test speakers.")

    checkpoint_path = output_dir / "static_ushape_checkpoint.csv"
    result_rows = pd.read_csv(checkpoint_path).to_dict("records") if checkpoint_path.is_file() else []
    completed = {(str(row["sample_id"]), int(row["run_seed"])) for row in result_rows}

    for run_seed in cfg.run_seeds:
        for sample_index, row in enumerate(probes.to_dict("records")):
            key = (str(row["utterance_id"]), int(run_seed))
            if key in completed:
                continue

            clean = load_audio(row["path"], cfg, device)
            enrollment = enrollment_centroids[row["speaker_id"]]
            opposite_speaker, opposite = choose_hard_negative(enrollment["wavlm"], calibration_centroids)
            sample_seed = int(run_seed) * 100_000 + sample_index

            protected, diagnostics = protector.protect(
                "static_ushape", clean, enrollment, opposite, sample_seed
            )
            transcript = read_transcript(row["path"], cfg)
            metrics = evaluate_candidate(
                clean, protected, enrollment, models, quality, asr, transcript, cfg,
                legacy_asv_threshold, calibrated_asv_threshold,
                quality_threshold, utility_threshold,
            )
            result_rows.append({
                "sample_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "path": row["path"],
                "run_seed": int(run_seed),
                "method": METHOD_NEW,
                "opposite_speaker_id": opposite_speaker,
                "phase2_fraction": 0.0,
                "phase_switches": 0,
                "routing_s_asv_initial": diagnostics["initial_routing_s_asv"],
                "routing_s_asv_final": diagnostics["final_s_asv"],
                "asv_threshold": calibrated_asv_threshold,
                "quality_threshold": quality_threshold,
                "utility_threshold": utility_threshold,
                "asv_threshold_calibrated_eer": calibrated_asv_threshold,
                "asv_threshold_fixed_paper": legacy_asv_threshold,
                **metrics,
            })
            completed.add(key)

            if run_seed == cfg.run_seeds[0] and sample_index == 0:
                sf.write(
                    output_dir / "Static_PGD_plus_U_Shape_example.wav",
                    protected.squeeze().detach().cpu().numpy(), cfg.sample_rate,
                )

            if len(result_rows) % cfg.checkpoint_every == 0:
                pd.DataFrame(result_rows).to_csv(checkpoint_path, index=False)

    new_results = pd.DataFrame(result_rows)
    expected_rows = len(probes) * len(cfg.run_seeds)
    if len(new_results) != expected_rows:
        raise RuntimeError(f"Metode tambahan menghasilkan {len(new_results)}, seharusnya {expected_rows} records.")
    if new_results[["sample_id", "run_seed"]].duplicated().any():
        raise RuntimeError("Checkpoint metode tambahan mengandung pasangan sample-seed duplikat.")

    new_results.to_csv(output_dir / "static_ushape_per_sample_metrics.csv", index=False)
    combined_results = pd.concat([old_results, new_results], ignore_index=True, sort=False)
    combined_results.to_csv(output_dir / "combined_per_sample_metrics.csv", index=False)


    CONTINUOUS_METRICS = {
        "s_asv": "lower",
        "pesq": "higher",
        "stoi": "higher",
        "ssim": "higher",
        "squim_mos": "higher",
        "snr_db": "higher",
        "mcd_db": "lower",
        "dtw_cost": "lower",
    }
    BINARY_METRICS = {
        "privacy_utility_success": "higher",
        "dsr_success": "higher",
    }
    FACTORIAL_METHODS = [
        METHOD_STATIC,
        METHOD_NEW,
        METHOD_DYNAMIC_NO_FILTER,
        METHOD_DYNAMIC_FILTER,
    ]
    CONTRASTS = {
        "filter_effect_fixed": (METHOD_NEW, METHOD_STATIC),
        "filter_effect_dynamic": (METHOD_DYNAMIC_FILTER, METHOD_DYNAMIC_NO_FILTER),
        "routing_effect_unfiltered": (METHOD_DYNAMIC_NO_FILTER, METHOD_STATIC),
        "routing_effect_filtered": (METHOD_DYNAMIC_FILTER, METHOD_NEW),
    }


    def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
        values = np.asarray(p_values, dtype=float)
        order = np.argsort(values)
        adjusted = np.empty(len(values), dtype=float)
        running = 0.0
        total = len(values)
        for rank, index in enumerate(order):
            running = max(running, (total - rank) * values[index])
            adjusted[index] = min(running, 1.0)
        return adjusted


    def speaker_level_values(results: pd.DataFrame) -> pd.DataFrame:
        continuous = results.groupby(["speaker_id", "method"], as_index=False)[list(CONTINUOUS_METRICS)].mean()
        rates = results.groupby(["speaker_id", "method"], as_index=False)[list(BINARY_METRICS)].mean()
        return continuous.merge(rates, on=["speaker_id", "method"], how="inner")


    def factorial_summary(results: pd.DataFrame) -> pd.DataFrame:
        per_speaker = speaker_level_values(results[results.method.isin(FACTORIAL_METHODS)])
        rows = []
        for method, group in per_speaker.groupby("method"):
            for metric in list(CONTINUOUS_METRICS) + list(BINARY_METRICS):
                scale = 100.0 if metric in BINARY_METRICS else 1.0
                values = group[metric].dropna().to_numpy(float) * scale
                half = (
                    float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
                    if len(values) > 1 else float("nan")
                )
                rows.append({
                    "dataset": TARGET_DATASET,
                    "method": method,
                    "metric": metric,
                    "n_speakers": len(values),
                    "mean": float(np.mean(values)),
                    "ci95_half_width": half,
                })
        return pd.DataFrame(rows)


    def factorial_pairwise_tests(results: pd.DataFrame) -> pd.DataFrame:
        per_speaker = speaker_level_values(results[results.method.isin(FACTORIAL_METHODS)])
        rows = []
        for contrast, (method_a, method_b) in CONTRASTS.items():
            for metric, direction in CONTINUOUS_METRICS.items():
                wide = per_speaker.pivot(index="speaker_id", columns="method", values=metric).dropna()
                if method_a not in wide or method_b not in wide:
                    continue
                a, b = wide[method_a], wide[method_b]
                try:
                    test = stats.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided")
                    statistic, p_value = float(test.statistic), float(test.pvalue)
                except ValueError:
                    statistic, p_value = float("nan"), 1.0
                rows.append({
                    "dataset": TARGET_DATASET,
                    "family": "primary" if metric in {"s_asv", "pesq", "stoi", "ssim", "squim_mos"} else "distortion",
                    "contrast": contrast,
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "preferred_direction": direction,
                    "n_speakers": len(wide),
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "a_minus_b": float((a - b).mean()),
                    "test": "two-sided paired Wilcoxon on speaker means",
                    "statistic": statistic,
                    "p_value_raw": p_value,
                })

            for metric, direction in BINARY_METRICS.items():
                binary = per_speaker.copy()
                binary["majority"] = (binary[metric] >= 0.5).astype(int)
                wide = binary.pivot(index="speaker_id", columns="method", values="majority").dropna()
                if method_a not in wide or method_b not in wide:
                    continue
                a_only = int(((wide[method_a] == 1) & (wide[method_b] == 0)).sum())
                b_only = int(((wide[method_a] == 0) & (wide[method_b] == 1)).sum())
                discordant = a_only + b_only
                p_value = float(stats.binomtest(a_only, discordant, 0.5).pvalue) if discordant else 1.0
                rows.append({
                    "dataset": TARGET_DATASET,
                    "family": "primary",
                    "contrast": contrast,
                    "method_a": method_a,
                    "method_b": method_b,
                    "metric": metric,
                    "preferred_direction": direction,
                    "n_speakers": len(wide),
                    "mean_a": float(per_speaker.loc[per_speaker.method == method_a, metric].mean() * 100),
                    "mean_b": float(per_speaker.loc[per_speaker.method == method_b, metric].mean() * 100),
                    "a_minus_b": float(
                        (per_speaker.loc[per_speaker.method == method_a, metric].to_numpy()
                         - per_speaker.loc[per_speaker.method == method_b, metric].to_numpy()).mean() * 100
                    ),
                    "test": "exact paired McNemar on speaker-majority decisions",
                    "statistic": a_only - b_only,
                    "p_value_raw": p_value,
                })

        table = pd.DataFrame(rows)
        table["p_value_holm"] = np.nan
        for family, indices in table.groupby("family").groups.items():
            table.loc[indices, "p_value_holm"] = holm_adjust(table.loc[indices, "p_value_raw"])
        table["holm_significant_0.05"] = table.p_value_holm < 0.05
        return table


    def factorial_interactions(results: pd.DataFrame) -> pd.DataFrame:
        per_speaker = speaker_level_values(results[results.method.isin(FACTORIAL_METHODS)])
        rows = []
        for metric, direction in CONTINUOUS_METRICS.items():
            wide = per_speaker.pivot(index="speaker_id", columns="method", values=metric).dropna()
            required = set(FACTORIAL_METHODS)
            if not required.issubset(wide.columns):
                continue
            filter_dynamic = wide[METHOD_DYNAMIC_FILTER] - wide[METHOD_DYNAMIC_NO_FILTER]
            filter_fixed = wide[METHOD_NEW] - wide[METHOD_STATIC]
            interaction = filter_dynamic - filter_fixed
            try:
                test = stats.wilcoxon(interaction, zero_method="zsplit", alternative="two-sided")
                statistic, p_value = float(test.statistic), float(test.pvalue)
            except ValueError:
                statistic, p_value = float("nan"), 1.0
            rows.append({
                "dataset": TARGET_DATASET,
                "metric": metric,
                "preferred_direction": direction,
                "n_speakers": len(interaction),
                "filter_effect_dynamic": float(filter_dynamic.mean()),
                "filter_effect_fixed": float(filter_fixed.mean()),
                "difference_in_differences": float(interaction.mean()),
                "test": "two-sided paired Wilcoxon of speaker-level difference-in-differences",
                "statistic": statistic,
                "p_value_raw": p_value,
            })
        table = pd.DataFrame(rows)
        table["p_value_holm"] = holm_adjust(table.p_value_raw) if len(table) else []
        table["holm_significant_0.05"] = table.p_value_holm < 0.05
        return table


    def plot_factorial(summary: pd.DataFrame, output_path: Path) -> None:
        metrics = ["s_asv", "pesq", "stoi", "ssim", "snr_db", "mcd_db"]
        labels = {
            METHOD_STATIC: "Fixed\nNo filter",
            METHOD_NEW: "Fixed\nU-Shape",
            METHOD_DYNAMIC_NO_FILTER: "Dynamic\nNo filter",
            METHOD_DYNAMIC_FILTER: "Dynamic\nU-Shape",
        }
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, metric in zip(axes.flat, metrics):
            subset = summary[summary.metric == metric].set_index("method").reindex(FACTORIAL_METHODS)
            x = np.arange(len(FACTORIAL_METHODS))
            ax.bar(x, subset["mean"], yerr=subset["ci95_half_width"], capsize=3, color=["#777777", "#4C78A8", "#B279A2", "#E45756"])
            ax.set_xticks(x, [labels[method] for method in FACTORIAL_METHODS], fontsize=8)
            ax.set_title(metric)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(f"2x2 routing-filtering ablation: {TARGET_DATASET}")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)


    factorial_methods_present = set(combined_results.method.unique())
    missing = set(FACTORIAL_METHODS) - factorial_methods_present
    if missing:
        raise RuntimeError(f"Metode faktorial belum lengkap: {sorted(missing)}")

    summary = factorial_summary(combined_results)
    pairwise = factorial_pairwise_tests(combined_results)
    interactions = factorial_interactions(combined_results)

    summary.to_csv(output_dir / "factorial_speaker_summary.csv", index=False)
    pairwise.to_csv(output_dir / "factorial_pairwise_tests.csv", index=False)
    interactions.to_csv(output_dir / "factorial_interaction_tests.csv", index=False)

    table = summary.pivot(index="method", columns="metric", values="mean").reindex(FACTORIAL_METHODS)
    table.to_csv(output_dir / "factorial_results_table.csv")
    plot_factorial(summary, output_dir / "factorial_ablation.png")

    run_manifest = {
        "dataset": TARGET_DATASET,
        "new_method": METHOD_NEW,
        "old_result_source": str(old_result_dir),
        "records_new": int(len(new_results)),
        "records_combined": int(len(combined_results)),
        "new_method_speakers": int(new_results.speaker_id.nunique()),
        "new_method_probes": int(new_results.sample_id.nunique()),
        "run_seeds": [int(seed) for seed in cfg.run_seeds],
        "iterations": int(cfg.num_iterations),
        "weights": dict(cfg.static_weights),
        "routing": "disabled; State-A weights fixed for all iterations",
        "gradient_filter": {
            "0_to_2.5_kHz": 1.0,
            "2.5_to_8_kHz": float(cfg.phase1_high_scale),
        },
        "paired_initialization": "sample_seed = run_seed * 100000 + original probe index",
        "thresholds_reused_from_old_result": {
            "asv_calibrated_eer": calibrated_asv_threshold,
            "asv_legacy_audit_only": legacy_asv_threshold,
            "squim": quality_threshold,
            "stoi": utility_threshold,
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
    }
    (output_dir / "static_ushape_experiment_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )

    validation = {
        "status": "PASS",
        "same_320_probes_as_old_results": bool(
            set(new_results.sample_id.astype(str))
            == set(old_results.loc[old_results.method == METHOD_STATIC, "sample_id"].astype(str))
        ),
        "same_three_seeds": sorted(new_results.run_seed.astype(int).unique().tolist())
            == sorted(int(seed) for seed in cfg.run_seeds),
        "exactly_960_new_records": len(new_results) == 960,
        "exactly_40_test_speakers": new_results.speaker_id.nunique() == 40,
        "no_duplicate_sample_seed": not new_results[["sample_id", "run_seed"]].duplicated().any(),
    }
    validation["status"] = "PASS" if all(value for key, value in validation.items() if key != "status") else "FAIL"
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if validation["status"] != "PASS":
        raise RuntimeError(f"Validasi eksperimen tambahan gagal: {validation}")

    archive_base = str(output_dir.parent / (output_dir.name + "_archive"))
    shutil.make_archive(archive_base, "zip", output_dir)
