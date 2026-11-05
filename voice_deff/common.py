#!/usr/bin/env python
# coding: utf-8

# In[ ]:



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

def package_versions() -> Dict[str, str]:
    import importlib.metadata
    versions = {}
    for name in ("torch", "torchaudio", "transformers", "librosa", "pesq", "pystoi", "scipy", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions
