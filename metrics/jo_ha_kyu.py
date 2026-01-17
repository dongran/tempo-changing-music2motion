#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jo–Ha–Kyu metric: tempo-change synchronization score.

This module implements a minimal, dataset-agnostic version of the
Jo–Ha–Kyu metric described in the paper and appendix. It measures the
Pearson correlation between:

- instantaneous tempo (BPM) of the music, and
- overall motion speed per beat segment.

The implementation is intentionally simple and close to the original
analysis scripts: motion speed is computed from frame-to-frame
displacements over all joints, without requiring a specific skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr

from . import tempo_utils
from dataset.python import bvh as bvh_utils  # type: ignore


@dataclass
class JoHaKyuResult:
    """Container for a single-sequence Jo–Ha–Kyu score."""

    r: float
    p_value: float
    tempo: np.ndarray
    motion_speed: np.ndarray


def _motion_speed_per_beat(
    motion_channels: np.ndarray,
    beat_times: np.ndarray,
    frame_time: float,
    smooth_sigma: float = 5.0,
) -> np.ndarray:
    """
    Compute overall motion speed per beat segment.

    Parameters
    ----------
    motion_channels : np.ndarray, shape (T, D)
        Motion channels (typically all non-translation channels). We
        interpret every group of 3 channels as a 3D trajectory.
    beat_times : np.ndarray, shape (n_beats,)
        Beat times in seconds.
    frame_time : float
        Frame time (seconds per frame) of the motion.
    smooth_sigma : float, optional
        Standard deviation for Gaussian smoothing (in frames), by default 5.

    Returns
    -------
    motion_speeds : np.ndarray, shape (n_segments,)
        Mean motion speed per beat segment.
    """
    if motion_channels.ndim != 2:
        raise ValueError("motion_channels must be 2D (T, D).")

    T, D = motion_channels.shape
    if D % 3 != 0:
        raise ValueError("Number of motion channels must be a multiple of 3.")

    n_joints = D // 3
    keypoints = motion_channels.reshape(T, n_joints, 3)

    # Per-frame kinetic velocity over all joints
    vel = np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2))
    kinetic_vel = np.mean(vel, axis=1)  # (T-1,)

    if smooth_sigma > 0:
        kinetic_vel = gaussian_filter1d(kinetic_vel, smooth_sigma)

    # Map beat times to frame indices
    frame_times = np.arange(T) * frame_time
    beat_frames = np.searchsorted(frame_times, beat_times).astype(int)

    speeds = []
    for i in range(len(beat_frames) - 1):
        start = beat_frames[i]
        end = beat_frames[i + 1]
        # kinetic_vel has length T-1, so clip indices
        start = max(0, min(start, kinetic_vel.shape[0] - 1))
        end = max(start + 1, min(end, kinetic_vel.shape[0]))

        segment = kinetic_vel[start:end]
        if segment.size == 0 or np.isnan(segment).any():
            continue
        speeds.append(np.mean(segment))

    return np.asarray(speeds, dtype=np.float32)


def compute_jo_ha_kyu_from_bvh_and_audio(
    bvh_path: str,
    audio_path: str,
    smooth_sigma: float = 5.0,
    fps_madmom: int = 100,
) -> JoHaKyuResult:
    """
    Compute the Jo–Ha–Kyu score for a single motion/audio pair using BVH.

    Parameters
    ----------
    bvh_path : str
        Path to a BVH file (e.g., SMPL BVH).
    audio_path : str
        Path to the corresponding audio file.
    smooth_sigma : float, optional
        Gaussian smoothing sigma (in frames) applied to motion speed,
        by default 5.
    fps_madmom : int, optional
        Frame rate used by Madmom's DBN beat tracker, by default 100.

    Returns
    -------
    JoHaKyuResult
        Contains the Pearson correlation r, p-value, and the aligned
        tempo and motion-speed sequences used for the computation.
    """
    data, fs_str, _ = bvh_utils.bvhreader(bvh_path)
    try:
        frame_time = float(fs_str)
    except ValueError:
        # Fallback: strip potential labels such as "0.0333333"
        frame_time = float(fs_str.strip())

    # Use all channels after the first 3 (root translation) as motion
    motion_channels = data[:, 3:]

    beat_times = tempo_utils.get_beat_times(audio_path, fps=fps_madmom)
    tempo, _ = tempo_utils.tempo_and_diff_from_beats(beat_times)

    motion_speeds = _motion_speed_per_beat(
        motion_channels=motion_channels,
        beat_times=beat_times,
        frame_time=frame_time,
        smooth_sigma=smooth_sigma,
    )

    # Align lengths by truncation to the shorter one
    n = min(len(tempo), len(motion_speeds))
    if n < 2:
        raise RuntimeError(
            "Not enough beat segments to compute Jo–Ha–Kyu (need at least 2)."
        )

    tempo_aligned = tempo[:n]
    speed_aligned = motion_speeds[:n]

    r, p = pearsonr(tempo_aligned, speed_aligned)
    return JoHaKyuResult(
        r=float(r),
        p_value=float(p),
        tempo=tempo_aligned,
        motion_speed=speed_aligned,
    )

