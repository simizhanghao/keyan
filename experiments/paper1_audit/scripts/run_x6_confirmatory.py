#!/usr/bin/env python3
"""Frozen-checkpoint X6 evaluator with an explicit pre-blind development mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rfhstu.b1_late_fusion import MultiViewLateFusionCNN
from rfhstu.models import DeviceClassifier, RFPatchEmbedder
from run_x2_formal import macro, views
from run_x4c_shen_port import ShenNet, cis

BLIND = {"b200_2", "b200_mini_2", "b210_2", "n210_2", "n210_3", "pluto_2"}
OOB_MODELS = {"B1-OOB", "C'-OOB"}
CONDITIONS = {
    "clean", "scale_0.5", "scale_0.70710678", "scale_1.41421356", "scale_2.0",
    "shuffle", "neutral", "left_scale_0.5", "right_scale_0.5",
}


class EvalSet(Dataset):
    def __init__(self, path: Path, seed: int, max_packets: int | None):
        self.path = path
        with h5py.File(path, "r") as f:
            self.labels = np.asarray(f["label"][0], dtype=np.int64) - 31
        self.indices = np.arange(len(self.labels))
        if max_packets is not None:
            self.indices = self.indices[:max_packets]
        rng = np.random.default_rng(seed + 9173)
        by_label = {int(y): np.flatnonzero(self.labels == y) for y in np.unique(self.labels)}
        self.donors = []
        for i in self.indices:
            candidates = np.concatenate([v for y, v in by_label.items() if y != int(self.labels[i])])
            self.donors.append(int(candidates[rng.integers(len(candidates))]))
        self.handle = None

    def __len__(self):
        return len(self.indices)

    def _iq(self, index: int):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        raw = np.asarray(self.handle["data"][index], dtype=np.float32)
        half = raw.shape[0] // 2
        z = raw[:half] + 1j * raw[half:]
        z = z / (np.sqrt(np.mean(np.abs(z) ** 2)) + 1e-8)
        return torch.from_numpy(np.stack([z.real, z.imag])).float()

    def __getitem__(self, k):
        i = int(self.indices[k])
        return self._iq(i), self._iq(self.donors[k]), int(self.labels[i]), i


def model_for(name: str, device: torch.device):
    if name == "B1-OOB":
        return MultiViewLateFusionCNN(10).to(device)
    if name in {"C'-OOB", "C'-TrueIB"}:
        oob = name == "C'-OOB"
        embedder = RFPatchEmbedder(
            window_size=8192, patch_size=256, dim=64, cnn_stem_dim=32,
            patch_embed_type="cnn_stem", use_oob=oob,
            oob_fusion_type="cross_attn_oob" if oob else "no_oob",
            use_oob_cross_attention=oob, fft_norm="log_zscore",
            oob_norm="ratio" if oob else "none",
        )
        return DeviceClassifier(embedder, 10, dim=64, depth=2, use_chirp_embedding=False).to(device)
    if name in {"Shen-CIS", "Shen-RA"}:
        return ShenNet(14, name == "Shen-RA").to(device)
    raise ValueError(name)


def modified_oob(iq, donor, condition, neutral):
    z = torch.complex(iq[:, 0], iq[:, 1])
    spectrum = torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1)
    freq = torch.fft.fftshift(torch.fft.fftfreq(iq.shape[-1], d=1e-6)).to(iq.device)
    oob = freq.abs() > 62500
    changed = spectrum.clone()
    if condition.startswith("scale_"):
        changed[:, oob] *= float(condition.removeprefix("scale_"))
    elif condition == "shuffle":
        zd = torch.complex(donor[:, 0], donor[:, 1])
        sd = torch.fft.fftshift(torch.fft.fft(zd, dim=-1), dim=-1)
        changed[:, oob] = sd[:, oob]
    elif condition == "neutral":
        changed[:, oob] = neutral[None, :]
    elif condition == "left_scale_0.5":
        changed[:, freq < -62500] *= 0.5
    elif condition == "right_scale_0.5":
        changed[:, freq > 62500] *= 0.5
    else:
        raise ValueError(condition)
    iq_changed = torch.fft.ifft(torch.fft.ifftshift(changed, dim=-1), dim=-1)
    return changed, oob, torch.stack([iq_changed.real, iq_changed.imag], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["dev-dry-run", "blind-confirmatory"], required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--data-file", type=Path, required=True)
    p.add_argument("--model", choices=["Shen-CIS", "Shen-RA", "B1-OOB", "C'-OOB", "C'-TrueIB"], required=True)
    p.add_argument("--seed", type=int, choices=range(5), required=True)
    p.add_argument("--condition", choices=sorted(CONDITIONS), default="clean")
    p.add_argument("--neutral", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-packets", type=int)
    a = p.parse_args()

    receiver = a.data_file.stem.replace("_train", "").replace("_test", "")
    is_blind = receiver in BLIND
    if (a.phase == "dev-dry-run" and is_blind) or (a.phase == "blind-confirmatory" and not is_blind):
        raise SystemExit("phase/receiver guard rejected this data file")
    if a.phase == "blind-confirmatory" and a.max_packets is not None:
        raise SystemExit("blind confirmation must evaluate every packet")
    if a.model not in OOB_MODELS and a.condition != "clean":
        raise SystemExit("interventions are frozen to B1-OOB and C'-OOB")
    if a.condition == "neutral" and not a.neutral:
        raise SystemExit("neutral condition requires --neutral")

    snapshot = json.loads(a.snapshot.read_text())
    row = next(r for r in snapshot["checkpoints"] if r["model"] == a.model and r["seed"] == a.seed)
    device = torch.device(a.device)
    model = model_for(a.model, device)
    state = torch.load(row["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    neutral = None
    if a.neutral:
        neutral = torch.from_numpy(np.load(a.neutral)).to(device=device, dtype=torch.complex64)

    ds = EvalSet(a.data_file, a.seed, a.max_packets)
    loader = DataLoader(ds, a.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    predictions, labels, packet_indices = [], [], []
    with torch.inference_mode():
        for iq, donor, y, packet_index in loader:
            iq, donor = iq.to(device), donor.to(device)
            if a.model.startswith("Shen-"):
                logits = model(cis(iq))[0]
            elif a.model == "C'-TrueIB":
                z = torch.complex(iq[:, 0], iq[:, 1])
                s = torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1)
                f = torch.fft.fftshift(torch.fft.fftfreq(iq.shape[-1], d=1e-6)).to(device)
                s[:, f.abs() > 62500] = 0
                ib = torch.fft.ifft(torch.fft.ifftshift(s, dim=-1), dim=-1)
                logits = model(torch.stack([ib.real, ib.imag], 1))["logits"]
            elif a.condition == "clean":
                logits = model(views(iq))["logits"] if a.model == "B1-OOB" else model(iq)["logits"]
            else:
                changed, oob, changed_iq = modified_oob(iq, donor, a.condition, neutral)
                if a.model == "B1-OOB":
                    z = torch.complex(iq[:, 0], iq[:, 1])
                    input_views = {
                        "iq": iq, "fft": torch.stack([changed.real, changed.imag], 1),
                        "amp_phase": torch.stack([z.abs(), z.angle()], 1),
                        "oob": changed.abs()[:, oob].unsqueeze(1),
                    }
                    logits = model(input_views)["logits"]
                else:
                    logits = model(iq, oob_iq=changed_iq)["logits"]
            predictions.extend(logits.argmax(1).cpu().tolist())
            labels.extend(y.tolist())
            packet_indices.extend(packet_index.tolist())

    pred = np.asarray(predictions)
    truth = np.asarray(labels)
    cm = np.bincount(truth * 10 + pred, minlength=100).reshape(10, 10)
    payload = {
        "protocol": {
            "phase": a.phase, "training": False, "receiver": receiver,
            "model": a.model, "seed": a.seed, "condition": a.condition,
            "checkpoint_sha256": row["checkpoint_sha256"],
            "intervention_scope": "OOB branch only; clean main IQ retained",
            "blind_opened": a.phase == "blind-confirmatory",
            "aggregation_unit": "receiver, then seed",
        },
        "n_eval": len(truth), "accuracy": float(np.mean(pred == truth)),
        "macro_f1": macro(pred, truth), "confusion_matrix": cm.tolist(),
        "packet_indices": packet_indices, "labels": labels, "predictions": predictions,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("n_eval", "accuracy", "macro_f1")}))


if __name__ == "__main__":
    main()
