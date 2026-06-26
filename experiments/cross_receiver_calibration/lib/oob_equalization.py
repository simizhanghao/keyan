"""OOB representation equalization (embedding-level v1)."""
from __future__ import annotations

import numpy as np


def _safe_std(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.maximum(x.std(axis=0), eps)


def fit_stats(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = z.mean(axis=0)
    std = _safe_std(z)
    cov = np.cov(z, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = cov + np.eye(cov.shape[0]) * 1e-4
    return mu, std, cov


def mean_shift(z: np.ndarray, mu_src: np.ndarray, mu_tgt: np.ndarray) -> np.ndarray:
    return z - mu_tgt + mu_src


def std_alignment(
    z: np.ndarray,
    mu_src: np.ndarray,
    std_src: np.ndarray,
    mu_tgt: np.ndarray,
    std_tgt: np.ndarray,
) -> np.ndarray:
    z_norm = (z - mu_tgt) / std_tgt
    return z_norm * std_src + mu_src


def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, 1e-6)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def coral(
    z: np.ndarray,
    mu_src: np.ndarray,
    cov_src: np.ndarray,
    mu_tgt: np.ndarray,
    cov_tgt: np.ndarray,
) -> np.ndarray:
    cs_rt = _sqrtm_psd(cov_src)
    ct_rt = _sqrtm_psd(cov_tgt)
    ct_inv = np.linalg.inv(ct_rt)
    transform = ct_inv @ cs_rt
    return (z - mu_tgt) @ transform.T + mu_src


def apply_equalization(
    z: np.ndarray,
    method: str,
    mu_src: np.ndarray,
    std_src: np.ndarray,
    cov_src: np.ndarray,
    mu_tgt: np.ndarray,
    std_tgt: np.ndarray,
    cov_tgt: np.ndarray,
) -> np.ndarray:
    if method == "none":
        return z.copy()
    if method == "mean_shift":
        return mean_shift(z, mu_src, mu_tgt)
    if method == "std_alignment":
        return std_alignment(z, mu_src, std_src, mu_tgt, std_tgt)
    if method == "coral":
        return coral(z, mu_src, cov_src, mu_tgt, cov_tgt)
    raise ValueError(f"Unknown equalization method: {method}")


def build_representation(
    main: np.ndarray,
    oob: np.ndarray,
    fused: np.ndarray,
    repr_name: str,
    oob_eq: np.ndarray | None = None,
) -> np.ndarray:
    if repr_name == "main_only":
        return main
    if repr_name == "oob_only":
        return oob_eq if oob_eq is not None else oob
    if repr_name == "fused":
        return fused
    if repr_name == "late_fusion":
        oob_use = oob_eq if oob_eq is not None else oob
        return 0.5 * (main + oob_use)
    raise ValueError(f"Unknown representation: {repr_name}")
