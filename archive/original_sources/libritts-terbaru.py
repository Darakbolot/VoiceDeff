#!/usr/bin/env python
# coding: utf-8

# In[ ]:


%%capture
%pip install -q transformers librosa soundfile matplotlib pandas numpy tqdm pesq pystoi jiwer scipy scikit-learn scikit-image

# In[ ]:


"""Reproducible evaluation for Dynamic PGD + U-Shape voice protection.

The module intentionally reports SQUIM-MOS as SQUIM-MOS.  It never labels a
proxy or a hand-written formula as NISQA. Fixed ASV operating points remain
those narrated in the paper; calibration-derived values are reported for audit.
"""

from __future__ import annotations

import json
import math
import os
import warnings

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
import platform
import random
import re
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from matplotlib import pyplot as plt
from pesq import pesq
from pystoi import stoi
from scipy import stats
from scipy.fft import dct
from sklearn.metrics import roc_curve
from skimage.metrics import structural_similarity
from tqdm.auto import tqdm
from transformers import (
    HubertModel,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WavLMForXVector,
)
from transformers.utils import logging as transformers_logging
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

@dataclass
class ExperimentConfig:
    # Dataset protocol: speakers in calibration and test are disjoint.
    dataset_name: str = "LibriTTS"
    dataset_glob: str = "./datasets/libri-tts-train-clean-360-part-2/**/*.wav"
    output_dir: str = "./hasil_revisi_libritts"
    sample_rate: int = 16_000
    segment_seconds: float = 4.0
    minimum_seconds: float = 1.0
    calibration_speakers: int = 20
    calibration_utterances_per_speaker: int = 5
    test_speakers: int = 40
    enrollment_utterances_per_speaker: int = 2
    test_utterances_per_speaker: int = 8
    expected_unique_utterances: int = 500
    selection_seed: int = 2026
    normalize_input_level: bool = True
    input_target_rms_dbfs: float = -26.0
    input_peak_limit: float = 0.95

    # Repeated randomized PGD protocol.
    run_seeds: Tuple[int, ...] = (42, 43, 44)
    num_iterations: int = 1000
    step_size: float = 1e-3
    epsilon_max: float = 0.12
    baseline_noise: float = 0.02
    checkpoint_every: int = 10

    # Target models and the operational definition of F.
    wavlm_checkpoint: str = "microsoft/wavlm-base-plus-sv"
    hubert_checkpoint: str = "facebook/hubert-base-ls960"
    semantic_layer: int = 2
    asr_checkpoint: str = "facebook/wav2vec2-base-960h"
    run_asr_wer: bool = True

    # STFT is fully specified; frequency regions are expressed in Hz.
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 160
    stft_center: bool = True
    stft_normalized: bool = False
    # Exact Hz equivalents of STFT bins 80, 45, and 120 at 16 kHz/512.
    phase1_high_cut_hz: float = 2500.0
    ushape_low_end_hz: float = 1406.25
    ushape_mid_end_hz: float = 3750.0
    phase1_high_scale: float = 0.05
    ushape_mid_scale: float = 0.01
    ushape_high_scale: float = 0.15
    high_frequency_start_hz: float = 3750.0

    # Equation (9) uses Mel bins [0:45] exactly.
    n_mels: int = 128
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0
    mel_scale: str = "htk"
    mel_norm: Optional[str] = None
    mel_power: float = 2.0
    formant_mel_bins: int = 45

    # Static PGD holds Phase-1 weights fixed. Dynamic variants switch weights.
    static_weights: Dict[str, float] = field(default_factory=lambda: {
        "identity": 150.0, "formant": 30.0, "hubert_identity": 20.0, "semantic": 5.0
    })
    phase1_weights: Dict[str, float] = field(default_factory=lambda: {
        "identity": 150.0, "formant": 30.0, "hubert_identity": 20.0, "semantic": 5.0
    })
    phase2_weights: Dict[str, float] = field(default_factory=lambda: {
        "identity": 15.0, "formant": 300.0, "hubert_identity": 20.0, "semantic": 20.0
    })

    # Fixed operating points narrated in Sections III-C and IV-A.
    routing_threshold: float = 0.24
    asv_evaluation_threshold: float = 0.25
    quality_calibration_snr_db: float = 10.0
    quality_metric_name: str = "SQUIM_MOS"
    save_audio_examples: bool = True
    strict_paper_method: bool = True

    def validate(self) -> None:
        assert self.sample_rate == 16_000, "PESQ wide-band evaluation requires 16 kHz here."
        assert self.mel_fmax <= self.sample_rate / 2
        assert 0 < self.ushape_low_end_hz < self.ushape_mid_end_hz <= self.sample_rate / 2
        assert 0 < self.formant_mel_bins <= self.n_mels
        assert self.win_length <= self.n_fft
        assert self.input_target_rms_dbfs < 0 and 0 < self.input_peak_limit <= 1.0
        assert len(self.run_seeds) >= 2, "Use at least two repeated runs; three or more are recommended."
        expected = (
            self.calibration_speakers * self.calibration_utterances_per_speaker
            + self.test_speakers * (self.enrollment_utterances_per_speaker + self.test_utterances_per_speaker)
        )
        assert expected == self.expected_unique_utterances, f"Configured selection is {expected}, not {self.expected_unique_utterances} utterances."
        if self.strict_paper_method:
            assert self.num_iterations == 1000 and self.step_size == 1e-3
            assert self.epsilon_max == 0.12 and self.baseline_noise == 0.02
            assert self.routing_threshold == 0.24 and self.asv_evaluation_threshold == 0.25
            assert self.static_weights == {"identity": 150.0, "formant": 30.0, "hubert_identity": 20.0, "semantic": 5.0}
            assert self.phase1_weights == self.static_weights
            assert self.phase2_weights == {"identity": 15.0, "formant": 300.0, "hubert_identity": 20.0, "semantic": 20.0}
            assert self.phase1_high_scale == 0.05 and self.ushape_mid_scale == 0.01 and self.ushape_high_scale == 0.15
            resolution = self.sample_rate / self.n_fft
            assert self.phase1_high_cut_hz / resolution == 80
            assert self.ushape_low_end_hz / resolution == 45
            assert self.ushape_mid_end_hz / resolution == 120
            assert self.formant_mel_bins == 45


METHODS = (
    "Undefended",
    "Random Noise",
    "Static PGD",
    "Dynamic PGD (No U-Shape)",
    "High-Frequency PGD (3.75-8 kHz)",
    "Proposed (Dynamic + U-Shape)",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def speaker_id(path: str, cfg: ExperimentConfig) -> str:
    """Dataset-specific speaker ID without changing the experiment method."""
    audio = Path(path)
    if cfg.dataset_name.lower() == "vctk":
        # VCTK: <root>/wav48[_silence_trimmed]/pXXX/<utterance>.wav
        return audio.parent.name
    # LibriTTS: <root>/<speaker>/<chapter>/<utterance>.wav
    return audio.parent.parent.name


def utterance_id(path: str) -> str:
    return Path(path).stem


def source_utterance_id(path: str, cfg: ExperimentConfig) -> str:
    stem = utterance_id(path)
    return re.sub(r"_mic[12]$", "", stem) if cfg.dataset_name.lower() == "vctk" else stem


def audio_metadata(path: str) -> Dict[str, float]:
    info = sf.info(path)
    return {
        "duration_s": float(info.frames) / float(info.samplerate),
        "original_sample_rate_hz": int(info.samplerate),
        "original_channels": int(info.channels),
    }


def audio_duration(path: str) -> float:
    return audio_metadata(path)["duration_s"]




# In[ ]:


def build_speaker_disjoint_split(cfg: ExperimentConfig) -> pd.DataFrame:
    """Select complete utterances in the configured duration interval."""
    import glob
    files = sorted(glob.glob(cfg.dataset_glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"No WAV files matched: {cfg.dataset_glob}")

    rows = []
    for path in files:
        metadata = audio_metadata(path)
        duration = metadata["duration_s"]
        if cfg.minimum_seconds <= duration <= cfg.segment_seconds:
            rows.append({
                "path": path, "speaker_id": speaker_id(path, cfg),
                "utterance_id": utterance_id(path), "source_utterance_id": source_utterance_id(path, cfg),
                **metadata,
            })
    eligible = pd.DataFrame(rows)
    if eligible.empty:
        raise RuntimeError("No utterance satisfies the configured duration interval.")
    # Avoid treating VCTK mic1/mic2 copies of the same spoken sentence as
    # independent enrollment/probe samples. Sorting deterministically keeps mic1.
    eligible = eligible.sort_values("path").drop_duplicates(
        subset=["speaker_id", "source_utterance_id"], keep="first"
    ).reset_index(drop=True)

    required_test = cfg.enrollment_utterances_per_speaker + cfg.test_utterances_per_speaker
    counts = eligible.groupby("speaker_id").size()
    calib_candidates = set(counts[counts >= cfg.calibration_utterances_per_speaker].index)
    test_candidates = set(counts[counts >= required_test].index)
    candidates = sorted(calib_candidates | test_candidates)
    rng = random.Random(cfg.selection_seed)
    rng.shuffle(candidates)

    calibration_speakers = [s for s in candidates if s in calib_candidates][: cfg.calibration_speakers]
    test_speakers = [s for s in candidates if s not in calibration_speakers and s in test_candidates][: cfg.test_speakers]
    if len(calibration_speakers) < cfg.calibration_speakers or len(test_speakers) < cfg.test_speakers:
        raise RuntimeError(
            f"Insufficient speakers after filtering: calibration={len(calibration_speakers)}, "
            f"test={len(test_speakers)}. Reduce requested counts or duration restrictions."
        )

    selected = []
    for spk in calibration_speakers:
        group = eligible[eligible.speaker_id == spk].sample(
            n=cfg.calibration_utterances_per_speaker,
            random_state=cfg.selection_seed + sum(map(ord, spk)),
        ).sort_values("utterance_id")
        for row in group.to_dict("records"):
            selected.append({**row, "split": "calibration", "role": "calibration"})
    for spk in test_speakers:
        group = eligible[eligible.speaker_id == spk].sample(
            n=required_test,
            random_state=cfg.selection_seed + sum(map(ord, spk)),
        ).sort_values("utterance_id")
        for j, row in enumerate(group.to_dict("records")):
            role = "enrollment" if j < cfg.enrollment_utterances_per_speaker else "probe"
            selected.append({**row, "split": "test", "role": role})
    result = pd.DataFrame(selected)
    if len(result) != cfg.expected_unique_utterances:
        raise RuntimeError(f"Selection produced {len(result)} instead of exactly {cfg.expected_unique_utterances} utterances.")
    return result


def canonicalize_audio_level(wav: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    """Dataset-independent level normalization; it does not alter pitch or timing."""
    wav = wav - wav.mean(dim=-1, keepdim=True)
    if not cfg.normalize_input_level:
        return wav.clamp(-1.0, 1.0)
    target_rms = 10.0 ** (cfg.input_target_rms_dbfs / 20.0)
    current_rms = torch.sqrt(torch.mean(wav.square(), dim=-1, keepdim=True) + 1e-12)
    wav = wav * (target_rms / current_rms)
    peak = wav.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    peak_scale = torch.minimum(torch.ones_like(peak), torch.full_like(peak, cfg.input_peak_limit) / peak)
    return (wav * peak_scale).clamp(-1.0, 1.0)


def load_audio(path: str, cfg: ExperimentConfig, device: torch.device) -> torch.Tensor:
    waveform_np, sr = sf.read(path, dtype='float32', always_2d=True)
    waveform_np = waveform_np.mean(axis=1)
    wav = torch.from_numpy(waveform_np).unsqueeze(0)
    if sr != cfg.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, cfg.sample_rate)
    maximum_length = round(cfg.segment_seconds * cfg.sample_rate)
    if wav.shape[-1] > maximum_length:
        raise ValueError(f"Selection protocol forbids truncation, but {path} is too long.")
    # Keep the original duration. Padding created artificial noisy tails and
    # biased PESQ/STOI when the perturbation was optimized on padded silence.
    return canonicalize_audio_level(wav, cfg).to(device)


def normalize_waveform(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + 1e-7)


class RepresentationModels:
    """R=WavLM-SV embedding, H=HuBERT last layer, F=HuBERT hidden layer k."""

    def __init__(self, cfg: ExperimentConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.wavlm = WavLMForXVector.from_pretrained(cfg.wavlm_checkpoint).to(device).eval()
        self.hubert = HubertModel.from_pretrained(cfg.hubert_checkpoint).to(device).eval()
        for model in (self.wavlm, self.hubert):
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def extract(self, waveform: torch.Tensor, with_grad: bool = False) -> Dict[str, torch.Tensor]:
        context = torch.enable_grad() if with_grad else torch.no_grad()
        with context:
            x = normalize_waveform(waveform)
            r = F.normalize(self.wavlm(input_values=x).embeddings, dim=-1)
            hubert_out = self.hubert(input_values=x, output_hidden_states=True)
            h = F.normalize(hubert_out.last_hidden_state.mean(dim=1), dim=-1)
            semantic = F.normalize(hubert_out.hidden_states[self.cfg.semantic_layer].mean(dim=1), dim=-1)
        return {"wavlm": r, "hubert": h, "semantic": semantic}


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a, b, dim=-1).mean()


def precompute_representations(
    selection: pd.DataFrame,
    cfg: ExperimentConfig,
    models: RepresentationModels,
    device: torch.device,
) -> Dict[str, Dict[str, torch.Tensor]]:
    cache = {}
    for row in selection.to_dict("records"):
        cache[row["path"]] = models.extract(load_audio(row["path"], cfg, device))
    return cache


def audit_audio_levels(selection: pd.DataFrame, cfg: ExperimentConfig, device: torch.device) -> pd.DataFrame:
    rows = []
    for row in selection.to_dict("records"):
        raw_np, raw_sr = sf.read(row["path"], dtype="float32", always_2d=True)
        raw_mono = raw_np.mean(axis=1)
        processed = load_audio(row["path"], cfg, device).squeeze().detach().cpu().numpy()
        raw_rms = float(np.sqrt(np.mean(raw_mono ** 2) + 1e-12))
        processed_rms = float(np.sqrt(np.mean(processed ** 2) + 1e-12))
        rows.append({
            "path": row["path"], "speaker_id": row["speaker_id"],
            "utterance_id": row["utterance_id"], "split": row["split"], "role": row["role"],
            "original_sample_rate_hz": raw_sr, "original_channels": raw_np.shape[1],
            "raw_rms": raw_rms, "raw_rms_dbfs": float(20 * np.log10(raw_rms + 1e-12)),
            "raw_peak": float(np.max(np.abs(raw_mono))),
            "processed_rms": processed_rms, "processed_rms_dbfs": float(20 * np.log10(processed_rms + 1e-12)),
            "processed_peak": float(np.max(np.abs(processed))),
        })
    return pd.DataFrame(rows)


def speaker_centroids(
    selection: pd.DataFrame,
    cache: Dict[str, Dict[str, torch.Tensor]],
    split: str,
    role: Optional[str] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    subset = selection[selection.split == split]
    if role is not None:
        subset = subset[subset.role == role]
    result = {}
    for spk, group in subset.groupby("speaker_id"):
        result[spk] = {
            key: F.normalize(torch.stack([cache[p][key] for p in group.path]).mean(dim=0), dim=-1)
            for key in ("wavlm", "hubert", "semantic")
        }
    return result


def choose_hard_negative(
    reference: torch.Tensor,
    candidate_centroids: Dict[str, Dict[str, torch.Tensor]],
) -> Tuple[str, Dict[str, torch.Tensor]]:
    scores = {spk: float(cosine(reference, values["wavlm"]).detach().cpu()) for spk, values in candidate_centroids.items()}
    chosen = max(scores, key=scores.get)
    return chosen, candidate_centroids[chosen]


def eer_threshold(genuine: Sequence[float], impostor: Sequence[float]) -> Tuple[float, float]:
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError('Calibration requires both genuine and impostor scores.')
    # When calibration classes are separated, every point inside the gap has
    # zero empirical error. Use the midpoint, not the minimum genuine score.
    if np.max(impostor) < np.min(genuine):
        return float((np.max(impostor) + np.min(genuine)) / 2.0), 0.0
    labels = np.r_[np.ones(len(genuine)), np.zeros(len(impostor))]
    scores = np.r_[genuine, impostor]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float(thresholds[idx]), float((fpr[idx] + fnr[idx]) / 2.0)




# In[ ]:


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


class SquimQuality:
    def __init__(self):
        from torchaudio.pipelines import SQUIM_SUBJECTIVE
        self.model = SQUIM_SUBJECTIVE.get_model().cpu().eval()

    def __call__(self, degraded: torch.Tensor, reference: torch.Tensor) -> float:
        with torch.inference_mode():
            score = self.model(degraded.detach().cpu(), reference.detach().cpu())
        return float(score.squeeze().item())


def add_noise_at_snr(x: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(seed)
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    signal_rms = torch.sqrt(torch.mean(x.square()) + 1e-12)
    noise_rms = torch.sqrt(torch.mean(noise.square()) + 1e-12)
    scaled = noise * (signal_rms / (10 ** (snr_db / 20.0) * noise_rms))
    return torch.clamp(x + scaled, -1.0, 1.0)


def calibrate_quality(
    selection: pd.DataFrame,
    cfg: ExperimentConfig,
    quality: SquimQuality,
    device: torch.device,
) -> Tuple[float, pd.DataFrame, float, float, float]:
    rows = []
    calib = selection[selection.split == "calibration"]
    for idx, row in enumerate(calib.to_dict("records")):
        clean = load_audio(row["path"], cfg, device)
        degraded = add_noise_at_snr(clean, cfg.quality_calibration_snr_db, cfg.selection_seed + idx)
        clean_np = clean.squeeze().detach().cpu().numpy()
        degraded_np = degraded.squeeze().detach().cpu().numpy()
        rows.append({"label": "clean", "utterance_id": row["utterance_id"], "score": quality(clean, clean), "stoi": 1.0})
        rows.append({"label": f"noise_{cfg.quality_calibration_snr_db:g}dB", "utterance_id": row["utterance_id"], "score": quality(degraded, clean), "stoi": float(stoi(clean_np, degraded_np, cfg.sample_rate, extended=False))})
    table = pd.DataFrame(rows)
    threshold, eer = eer_threshold(
        table.loc[table.label == "clean", "score"],
        table.loc[table.label != "clean", "score"],
    )
    utility_threshold, utility_eer = eer_threshold(
        table.loc[table.label == "clean", "stoi"],
        table.loc[table.label != "clean", "stoi"],
    )
    return threshold, table, eer, utility_threshold, utility_eer




# In[ ]:


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


def calculate_snr(clean: np.ndarray, test: np.ndarray) -> float:
    signal = np.mean(clean ** 2)
    noise = np.mean((test - clean) ** 2)
    return float(10.0 * np.log10((signal + 1e-12) / (noise + 1e-12)))


def perturbation_band_energy(clean: np.ndarray, test: np.ndarray, cfg: ExperimentConfig) -> Dict[str, float]:
    perturbation = test - clean
    power = np.abs(np.fft.rfft(perturbation)) ** 2
    frequencies = np.fft.rfftfreq(len(perturbation), d=1.0 / cfg.sample_rate)
    total = float(power.sum()) + 1e-20
    masks = {
        "band_energy_low_fraction": frequencies < cfg.ushape_low_end_hz,
        "band_energy_mid_fraction": (frequencies >= cfg.ushape_low_end_hz) & (frequencies <= cfg.ushape_mid_end_hz),
        "band_energy_high_fraction": frequencies > cfg.ushape_mid_end_hz,
    }
    return {name: float(power[mask].sum() / total) for name, mask in masks.items()}


def safe_pesq(clean: np.ndarray, test: np.ndarray, sr: int) -> float:
    try:
        return float(pesq(sr, clean, test, "wb"))
    except Exception:
        return float("nan")


def calculate_ssim(clean: np.ndarray, test: np.ndarray, sr: int) -> float:
    kwargs = dict(sr=sr, n_fft=512, hop_length=160, n_mels=80, fmax=sr / 2)
    a = librosa.power_to_db(librosa.feature.melspectrogram(y=clean, **kwargs), ref=np.max)
    b = librosa.power_to_db(librosa.feature.melspectrogram(y=test, **kwargs), ref=np.max)
    data_range = max(float(a.max() - a.min()), 1e-8)
    return float(structural_similarity(a, b, data_range=data_range))




# In[ ]:


def calculate_dtw_and_mcd(clean: np.ndarray, test: np.ndarray, sr: int) -> Tuple[float, float]:
    # Natural-log Mel cepstra with orthonormal DCT; coefficients 1..23.
    mel_kwargs = dict(sr=sr, n_fft=512, hop_length=160, n_mels=40, fmax=sr / 2, power=1.0)
    logmel_clean = np.log(librosa.feature.melspectrogram(y=clean, **mel_kwargs) + 1e-10)
    logmel_test = np.log(librosa.feature.melspectrogram(y=test, **mel_kwargs) + 1e-10)
    a = dct(logmel_clean, type=2, axis=0, norm='ortho')[1:24]
    b = dct(logmel_test, type=2, axis=0, norm='ortho')[1:24]
    cost, path = librosa.sequence.dtw(X=a, Y=b, metric="euclidean")
    path = path[::-1]
    differences = a[:, path[:, 0]] - b[:, path[:, 1]]
    frame_distances = np.sqrt(np.sum(differences ** 2, axis=0))
    mcd = (10.0 * np.sqrt(2.0) / np.log(10.0)) * np.mean(frame_distances)
    dtw = float(cost[-1, -1] / max(len(path), 1))
    return dtw, float(mcd)


def read_transcript(audio_path: str, cfg: ExperimentConfig) -> Optional[str]:
    audio = Path(audio_path)
    if cfg.dataset_name.lower() == "vctk":
        base = re.sub(r"_mic[12]$", "", audio.stem)
        candidates = [audio.with_name(base + ".txt")]
        parts = list(audio.parts)
        for index, part in enumerate(parts):
            if part.lower().startswith("wav48") or part.lower() == "wav16":
                candidates.append(Path(*parts[:index], "txt", audio.parent.name, base + ".txt"))
                break
    else:
        candidates = [audio.with_suffix(".normalized.txt")]
    for txt in candidates:
        if txt.exists():
            return txt.read_text(encoding="utf-8").strip()
    return None


def normalize_text(text: str) -> str:
    return re.sub(r"[^A-Z' ]+", " ", text.upper()).strip()


class ASREvaluator:
    def __init__(self, checkpoint: str, device: torch.device):
        self.processor = Wav2Vec2Processor.from_pretrained(checkpoint)
        self.model = Wav2Vec2ForCTC.from_pretrained(checkpoint).to(device).eval()
        self.device = device

    def transcribe(self, waveform: torch.Tensor, sr: int) -> str:
        values = self.processor(waveform.squeeze().detach().cpu().numpy(), sampling_rate=sr, return_tensors="pt").input_values.to(self.device)
        with torch.inference_mode():
            ids = self.model(values).logits.argmax(dim=-1)
        return normalize_text(self.processor.batch_decode(ids)[0])


def word_error_rate(reference: Optional[str], hypothesis: Optional[str]) -> float:
    if not reference or hypothesis is None:
        return float("nan")
    from jiwer import wer
    return float(wer(normalize_text(reference), normalize_text(hypothesis)))


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


def package_versions() -> Dict[str, str]:
    import importlib.metadata
    versions = {}
    for name in ("torch", "torchaudio", "transformers", "librosa", "pesq", "pystoi", "scipy", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions




# In[ ]:


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




# In[ ]:


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




# In[ ]:


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




# In[ ]:


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


# In[ ]:


from collections import Counter

wav_counts = Counter()
for wav_path in Path('/kaggle/input').rglob('*.wav'):
    relative = wav_path.relative_to('/kaggle/input')
    dataset_slug = relative.parts[0] if relative.parts else '(unknown)'
    wav_counts[dataset_slug] += 1

dataset_inventory = pd.DataFrame([
    {'dataset_path': f'/kaggle/input/{slug}/**/*.wav', 'wav_files': count}
    for slug, count in wav_counts.most_common()
])
dataset_inventory.to_csv('/kaggle/working/dataset_wav_inventory.csv', index=False)


# In[ ]:


cfg = ExperimentConfig(
    dataset_name='LibriTTS',
    # Sesuaikan slug bila nama dataset Kaggle berbeda.
    dataset_glob='/kaggle/input/**/libri-tts-train-clean-360-part-2/**/*.wav',
    output_dir='/kaggle/working/hasil_revisi_libritts',
    calibration_speakers=20,
    calibration_utterances_per_speaker=5,
    test_speakers=40,
    enrollment_utterances_per_speaker=2,
    test_utterances_per_speaker=8,
    expected_unique_utterances=500,
    normalize_input_level=True,
    input_target_rms_dbfs=-26.0,
    input_peak_limit=0.95,
    run_seeds=(42, 43, 44),
    num_iterations=1000,
    checkpoint_every=1,
    run_asr_wer=True,
)

# Total unik = (20 x 5) + (40 x 2) + (40 x 8) = 500 utterance.

# Preview data sebelum model dan PGD dijalankan.
preview_output = Path(cfg.output_dir)
preview_output.mkdir(parents=True, exist_ok=True)
selection_preview = build_speaker_disjoint_split(cfg)
data_count_preview = (
    selection_preview.groupby(['split', 'role'])
    .agg(speakers=('speaker_id', 'nunique'), utterances=('utterance_id', 'size'))
    .reset_index()
)
selection_preview.to_csv(preview_output / 'sample_selection_preview.csv', index=False)
data_count_preview.to_csv(preview_output / 'data_count_preview.csv', index=False)
probe_count = int(((selection_preview['split'] == 'test') & (selection_preview['role'] == 'probe')).sum())
pd.DataFrame([{
    'unique_utterances': int(len(selection_preview)),
    'unique_speakers': int(selection_preview['speaker_id'].nunique()),
    'probe_utterances': probe_count,
    'random_seeds': len(cfg.run_seeds),
    'probe_seed_trials': probe_count * len(cfg.run_seeds),
    'pgd_methods': 4,
    'iterations_per_pgd_method': cfg.num_iterations,
    'total_pgd_iterations': probe_count * len(cfg.run_seeds) * 4 * cfg.num_iterations,
    'metric_rows_expected': probe_count * len(cfg.run_seeds) * len(METHODS),
}]).to_csv(preview_output / 'execution_workload.csv', index=False)


# In[ ]:


frequency_resolution = cfg.sample_rate / cfg.n_fft
frequency_rows = []
for label, hz in {
    'Phase-1 high cutoff': cfg.phase1_high_cut_hz,
    'U-Shape low end': cfg.ushape_low_end_hz,
    'U-Shape mid end': cfg.ushape_mid_end_hz,
    'High-frequency start': cfg.high_frequency_start_hz,
}.items():
    frequency_rows.append({
        'label': label,
        'frequency_hz': hz,
        'nearest_stft_bin': round(hz / frequency_resolution),
        'stft_resolution_hz': frequency_resolution,
        'nyquist_hz': cfg.sample_rate / 2,
    })
pd.DataFrame(frequency_rows).to_csv(
    Path(cfg.output_dir) / 'frequency_band_mapping.csv', index=False
)
pd.DataFrame({'method': METHODS}).to_csv(
    Path(cfg.output_dir) / 'evaluated_methods.csv', index=False
)


# In[ ]:


%%capture
results = run_experiment(cfg)

# In[ ]:


import shutil

output_dir = Path(cfg.output_dir)
archive = shutil.make_archive('/kaggle/working/hasil_revisi_libritts', 'zip', output_dir)
_ = archive

