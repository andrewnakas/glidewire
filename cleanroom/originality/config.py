"""Originality thresholds, all in one place. A manifest may override any
value with {"thresholds": {"music": {"ngram_max": {...}}, ...}}.

Calibration (2026-10-19, synthetic): 91 pairs of independent compose()
outputs scored at most ngram 4/5/6 = 0.14/0.08/0.04 and contour DTW 0.36;
a transposed or tempo-doubled copy scores 1.0 everywhere; a copy with every
4th note changed drops to 4-gram 0.16 but keeps DTW 0.81, so it still fails.
Images/geometry thresholds are first guesses checked only on synthetic
cases (tests/test_originality_*.py); re-tune on the first real game.
"""
import copy

CONFIG = {
    "music": {
        "ns": [4, 5, 6],                 # interval n-gram lengths (intervals, not notes)
        "ngram_max": {"4": 0.35, "5": 0.25, "6": 0.15},   # fail if clean containment exceeds
        "min_notes": 8,                  # channels with fewer notes are ignored
        "dtw_notes": 128,                # first N notes of each channel's top line
        "dtw_band": 0.15,                # Sakoe-Chiba band, fraction of length
        "dtw_scale": 1.0,                # semitones: sim = exp(-mean cost / scale)
        "dtw_sim_max": 0.60,
        "min_pitch_std": 1.5,            # pedal / drum lines carry no contour
        "melodic_diversity": 0.5,        # clean lines below this are accompaniment, not judged
        "min_diversity": 0.25,           # retail lines below this skip the DTW
        "chroma_excess_max": 0.25,       # PCM: DTW path cosine minus matrix mean
    },
    "text": {
        "run_words": 5,                  # shared run of this many words fails
        "control_re": r"\{[^}]*\}|<[^>]*>|\[[^\]]*\]|\\[nrt]|\\x[0-9a-fA-F]{2}|[\x00-\x1f]",
        "allowed_caps": [],              # capitalised words that are not names
    },
    "geometry": {
        "voxel": None,                   # world units; None = bbox extent / voxel_div
        "voxel_div": 64,
        "tolerance": 1,                  # voxels: a cell within this distance matches
        "voxel_iou_max": 0.40,           # tolerant voxel IoU fails above this
        "silhouette_max": 0.90,          # mean of 3 silhouette IoUs ...
        "silhouette_voxel_min": 0.20,    # ... fails only together with this voxel IoU
        "anchor_radius": 2,              # voxels around each anchor excluded
    },
    "images": {
        "size": 32,                      # SSIM / hash working size
        "ssim_max": 0.80,
        "phash_min_dist": 8,             # fail if pHash AND dHash distances are both <=
        "dhash_min_dist": 10,
        "flat_std": 4.0,                 # 0..255: flat textures are structural only
    },
    "tilemap": {
        "band_fill": 0.9,                # rows this full are base floor / ceiling (structure)
        "max_shift": 3,                  # columns of horizontal alignment tried
        "min_holes": 3,                  # pit columns needed before pits are compared
        "iou_max": 0.35,                 # feature IoU or pit IoU above this fails
    },
    "blockmap": {
        "equal_max": 0.35,               # top-down block maps: share of equal non-anchor, non-background cells
        "min_cells": 6,
    },
    "code": {
        "window": 256,
        "min_nonzero": 32,               # windows with fewer nonzero bytes are padding
    },
}


def merged(overrides=None):
    cfg = copy.deepcopy(CONFIG)
    for kind, vals in (overrides or {}).items():
        cfg.setdefault(kind, {}).update(vals)
    return cfg
