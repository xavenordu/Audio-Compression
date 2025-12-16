#!/usr/bin/env python3
"""
fractal_wav_compressor_gpu_cli.py
Upgraded fractal compressor/decompressor for WAV files with GPU support, memory mapping, convergence-based iterative decoding,
batch processing, and logging/metrics.
Provides CLI and Python API.

Features added in this version:
- Embedded memmap-backed domain storage during compression (written to temporary file, streamed into .fwav). This
  allows building domains for extremely large audio without holding the full domain matrix in RAM.
- Block-wise statistics and block-wise matching to avoid loading all domains into memory or GPU at once.
- GPU self-test at startup and a startup banner. If GPU fails, automatic fallback to CPU.
"""

import argparse
import hashlib
import wave
import struct
import numpy as np
import os
import time
import json
import logging
import tempfile
from multiprocessing import Pool, cpu_count

import multiprocessing as mp
from functools import partial


# GPU setup
try:
    import cupy as cp # type: ignore
    GPU_AVAILABLE = True
except Exception:
    cp = np
    GPU_AVAILABLE = False
  
# quick GPU self-test
GPU_WORKING = False
if GPU_AVAILABLE:
    try:
        # simple operation to validate CuPy runtime
        _ = cp.arange(2).sum()
        GPU_WORKING = True
    except Exception:
        GPU_WORKING = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('fwavc')

# startup banner
if GPU_WORKING:
    logger.info('[FWAVC] GPU available: CuPy detected and working')
elif GPU_AVAILABLE and not GPU_WORKING:
    logger.warning('[FWAVC] CuPy import succeeded but GPU self-test failed — falling back to CPU')
else:
    logger.info('[FWAVC] GPU not available — running in CPU mode')

FWAV_VERSION = 1

# ----------------------- I/O helpers -----------------------

def read_wav_mono(path, mmap=False):
    with wave.open(path, 'rb') as w:
        nchan = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        comptype = w.getcomptype()
        if comptype != 'NONE':
            raise ValueError(f'Unsupported WAV compression type: {comptype}')
        raw = w.readframes(nframes)

    # 8-bit unsigned, 16-bit signed, 24-bit signed, 32-bit float
    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 3:
        # 24-bit PCM: read 3 bytes per sample
        data = np.frombuffer(raw, dtype=np.uint8)
        data = data.reshape(-1, 3)
        # convert to int32
        data = (data[:,0].astype(np.int32) | (data[:,1].astype(np.int32) << 8) | (data[:,2].astype(np.int32) << 16))
        # sign extend
        mask = data & 0x800000
        data = data - (mask << 1)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f'Unsupported sample width: {sampwidth}')

    if nchan > 1:
        data = data.reshape(-1, nchan).mean(axis=1)
    return data.astype(np.float32), framerate, sampwidth


def write_wav(path, data, framerate, sampwidth):
    if sampwidth == 1:
        out = (data + 128).clip(0, 255).astype(np.uint8)
    elif sampwidth == 2:
        out = data.clip(-32768, 32767).astype(np.int16)
    elif sampwidth == 3:
        # convert to int32 then to bytes
        data32 = data.clip(-2**23, 2**23-1).astype(np.int32)
        b0 = (data32 & 0xFF).astype(np.uint8)
        b1 = ((data32 >> 8) & 0xFF).astype(np.uint8)
        b2 = ((data32 >> 16) & 0xFF).astype(np.uint8)
        out = np.column_stack([b0,b1,b2]).flatten()
    elif sampwidth == 4:
        out = data.astype(np.float32)
    else:
        raise ValueError(f'Unsupported sample width: {sampwidth}')

    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(framerate)
        w.writeframes(out.tobytes())


# ----------------------- Fractal helpers -----------------------

def frame_ranges(signal, range_size, hop=None):
    hop = hop or range_size
    total = len(signal)
    ranges = [signal[i:i+range_size] for i in range(0, total-range_size+1, hop)]
    return np.vstack(ranges) if ranges else np.empty((0, range_size), dtype=signal.dtype)

# ----------------------- Domain memmap building -----------------------

def build_domains_memmap(signal, tile_size, range_size, domain_step=1, block_size=1000, tmpdir=None, use_gpu=False):
    """
    Build domain downsampled tiles and store them in a temporary memmap file.
    Returns (domains_path, n_domains)
    Vectorized version: avoids Python loops for per-tile downsampling.
    """
    xp = cp if (use_gpu and GPU_WORKING) else np

    n = len(signal)
    starts = list(range(0, n - tile_size + 1, domain_step))
    n_domains = len(starts)
    if n_domains == 0:
        return None, 0

    # create temporary memmap file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.domains', dir=tmpdir)
    tmp_path = tmp.name
    tmp.close()
    domains_mm = np.memmap(tmp_path, dtype='float32', mode='w+', shape=(n_domains, range_size))

    # determine block parameters
    block_lengths = tile_size // range_size

    idx = 0
    for i in range(0, n_domains, block_size):
        batch_starts = starts[i:i+block_size]
        batch_len = len(batch_starts)

        # extract tiles for the batch
        tiles = np.array([signal[s:s+tile_size] for s in batch_starts], dtype=np.float32)  # shape (batch_len, tile_size)

        # trim excess samples if tile_size not divisible by range_size
        tiles = tiles[:, :block_lengths*range_size]

        # reshape to (batch_len, range_size, block_length) and compute mean along last axis
        tiles_reshaped = tiles.reshape(batch_len, range_size, block_lengths)
        downsampled = tiles_reshaped.mean(axis=2).astype(np.float32)  # shape (batch_len, range_size)

        # write block to memmap
        domains_mm[idx:idx+batch_len, :] = downsampled
        idx += batch_len

    domains_mm.flush()
    return tmp_path, n_domains



# ----------------------- Matching helpers (block-wise) -----------------------

def compute_stats_memmap(domains_path, n_domains, range_size, batch_size=1024):
    """Compute means and variances for domains stored in a memmap file in blocks."""
    d_means = np.empty(n_domains, dtype=np.float32)
    d_vars = np.empty(n_domains, dtype=np.float32)
    domains_mm = np.memmap(domains_path, dtype='float32', mode='r', shape=(n_domains, range_size))
    idx = 0
    for i in range(0, n_domains, batch_size):
        b = domains_mm[i:i+batch_size]
        means = b.mean(axis=1)
        mean_sq = (b * b).mean(axis=1)
        vars = mean_sq - means * means
        vars = np.where(vars < 0, 0.0, vars)
        d_means[idx:idx+len(means)] = means
        d_vars[idx:idx+len(vars)] = vars
        idx += len(means)
    return d_means, d_vars

def build_domain_clusters(d_means, d_vars, n_clusters=32, max_iter=25):
    """
    K-means clustering over (mean, variance) space.
    Returns:
        cluster_ids: (n_domains,)
        cluster_centers: (n_clusters, 2)
    """
    feats = np.stack([d_means, d_vars], axis=1).astype(np.float32)
    n_domains = feats.shape[0]

    # Init centers by sampling
    rng = np.random.default_rng(0)
    centers = feats[rng.choice(n_domains, size=n_clusters, replace=False)]

    for _ in range(max_iter):
        # assign
        dists = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(dists, axis=1)

        # update
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            members = feats[labels == k]
            if len(members):
                new_centers[k] = members.mean(axis=0)
            else:
                new_centers[k] = centers[k]

        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    return labels.astype(np.int32), centers



def find_best_domain_for_range_blockwise(range_block, domains_path, n_domains, range_size, d_means, d_vars, batch_size=1024, use_gpu=False):
    """
    Vectorized scan of domains stored in memmap file to find best matching domain for the given range_block.
    Fully stays on GPU if requested; returns (best_idx, best_s, best_err)
    """
    xp = cp if (use_gpu and GPU_WORKING) else np
    domains_mm = np.memmap(domains_path, dtype='float32', mode='r', shape=(n_domains, range_size))
    
    rb = xp.asarray(range_block, dtype=xp.float32)
    rb_mean = float(rb.mean())
    m = len(rb)
    sr2 = float((range_block**2).mean())

    if use_gpu and GPU_WORKING:
        best_err = cp.inf
        best_idx = -1
        best_s = 0.0
    else:
        best_err = float('inf')
        best_idx = -1
        best_s = 0.0

    for i in range(0, n_domains, batch_size):
        batch = domains_mm[i:i+batch_size]
        batch_len = len(batch)

        if use_gpu and GPU_WORKING:
            batch_x = cp.asarray(batch, dtype=cp.float32)
            d_means_batch = cp.asarray(d_means[i:i+batch_len], dtype=cp.float32)
            d_vars_batch = cp.asarray(d_vars[i:i+batch_len], dtype=cp.float32)
        else:
            batch_x = batch.astype(np.float32)
            d_means_batch = d_means[i:i+batch_len]
            d_vars_batch = d_vars[i:i+batch_len]

        # Vectorized scaling factor s
        dots = batch_x.dot(rb)
        denom = m * d_vars_batch
        valid = denom > 1e-12
        s = xp.zeros_like(denom)
        s[valid] = (dots[valid] - m * d_means_batch[valid] * rb_mean) / denom[valid]

        # Vectorized chi² calculation
        chi2 = xp.full(batch_len, xp.inf, dtype=xp.float32)
        if xp.any(valid):
            chi2[valid] = sr2 + s[valid] * (s[valid]*d_vars_batch[valid] + 2*d_means_batch[valid]*rb_mean - 2*dots[valid]/m)

        # Determine local best on GPU/CPU
        local_best_idx = int(xp.argmin(chi2))
        local_err = float(xp.sqrt(max(float(chi2[local_best_idx]), 0.0)))
        local_s = float(s[local_best_idx])

        if local_err < best_err:
            best_err = local_err
            best_idx = i + local_best_idx
            best_s = local_s

    return best_idx, best_s, best_err

def find_best_domain_pruned(range_block,
                            domains_path,
                            n_domains,
                            range_size,
                            d_means,
                            d_vars,
                            cluster_ids,
                            cluster_centers,
                            top_k_clusters=3,
                            coarse_downsample=4,
                            top_k_domains=16,
                            batch_size=1024,
                            use_gpu=False):
    """
    Pruned hierarchical domain search using clusters with symmetry-aware affine correction.

    Stages:
      A) Cluster preselection (cheap: O(num_clusters))
      B) Coarse distance (downsampled: O(K*range_size/coarse_downsample))
      C) Exact affine solve with mirror check (O(K))

    Returns:
        best_idx, best_s, best_o, sym_flag, best_err
    """

    xp = cp if (use_gpu and GPU_WORKING) else np
    r = xp.asarray(range_block, dtype=xp.float32)
    r_mean = float(r.mean())
    r_var = float(xp.var(r))
    m = range_size
    eps = 1e-12

    # -------------------------
    # Stage A: Cluster preselection
    # -------------------------
    feat = xp.array([r_mean, r_var], dtype=xp.float32)
    centers = xp.asarray(cluster_centers, dtype=xp.float32)
    dists = xp.sum((centers - feat)**2, axis=1)
    best_clusters = xp.argsort(dists)[:top_k_clusters]

    best_clusters_np = cp.asnumpy(best_clusters) if (use_gpu and GPU_WORKING) else best_clusters # type: ignore
    candidate_idxs = np.where(np.isin(cluster_ids, best_clusters_np))[0]
    if len(candidate_idxs) == 0:
        return -1, 0.0, 0.0, 0, float('inf')

    # -------------------------
    # Stage B: Coarse downsampled distance
    # -------------------------
    ds = max(1, m // coarse_downsample)
    r_ds = r[::ds]

    domains_mm = np.memmap(domains_path, dtype='float32', mode='r', shape=(n_domains, range_size))
    coarse_errs = []
    for idx in candidate_idxs:
        d = domains_mm[int(idx)][::ds]
        coarse_errs.append(xp.linalg.norm(r_ds - xp.asarray(d)))
    coarse_errs = xp.asarray(coarse_errs)
    k = min(top_k_domains, len(coarse_errs))
    shortlist = candidate_idxs[xp.argsort(coarse_errs)[:k]]

    # -------------------------
    # Stage C: Exact affine solve + mirrored
    # -------------------------
    best_err = float('inf')
    best_idx = -1
    best_s = 0.0
    best_o = 0.0
    sym_flag = 0

    for idx in shortlist:
        tile = xp.asarray(domains_mm[int(idx)], dtype=xp.float32)
        tile_mean = float(tile.mean())
        tile_c = tile - tile_mean
        r_c = r - r_mean
        denom = float(xp.sum(tile_c * tile_c))
        if denom < eps:
            continue
        # Original
        s0 = float(xp.sum(r_c * tile_c) / denom)
        o0 = r_mean - s0 * tile_mean
        err0 = float(xp.linalg.norm(r - (s0*tile + o0)))

        # Mirrored
        tile_m = tile[::-1]
        tile_m_mean = float(tile_m.mean())
        tile_m_c = tile_m - tile_m_mean
        denom_m = float(xp.sum(tile_m_c * tile_m_c))
        if denom_m < eps:
            continue
        s1 = float(xp.sum(r_c * tile_m_c) / denom_m)
        o1 = r_mean - s1 * tile_m_mean
        err1 = float(xp.linalg.norm(r - (s1*tile_m + o1)))

        if err1 < err0:
            err = err1
            s = s1
            o = o1
            sym = 1
        else:
            err = err0
            s = s0
            o = o0
            sym = 0

        if err < best_err:
            best_err = err
            best_idx = int(idx)
            best_s = s
            best_o = o
            sym_flag = sym

    return best_idx, best_s, best_o, sym_flag, best_err

# ----------------------- Symmetry -----------------------

def apply_symmetry(tile):
    return [tile, tile[::-1]]  # identity + mirrored

# ----------------------- Compression & Decompression (memmap-enabled) -----------------------

def compress_audio(signal, framerate, sampwidth, tile_size=1024, energy_thresh=1e-4,
                   use_gpu=False, domains_tmpdir=None, batch_size=512):
    """
    Compress audio using fractal domain matching with safe multiprocessing and
    per-range symmetry-aware affine correction integrated into hierarchical pruning.
    Returns:
        matches, domains_array, n_ranges, range_size,
        tile_size, domain_step, energy_threshold, original_len
    """

    xp = cp if (use_gpu and GPU_WORKING) else np

    range_size = max(4, tile_size // 16)
    domain_step = max(1, range_size // 2)

    # -------------------------
    # Voiced detection
    # -------------------------
    def voiced_detection(signal, frame_size=range_size*2,
                         energy_threshold=energy_thresh,
                         smooth_window=5,
                         low_threshold=None):

        signal = np.asarray(signal, dtype=np.float32)
        n = len(signal)
        n_frames = (n + frame_size - 1) // frame_size
        pad_len = n_frames * frame_size - n
        padded = np.pad(signal, (0, pad_len), mode='reflect')
        frames = padded.reshape(n_frames, frame_size)
        energies = np.mean(frames * frames, axis=1)
        if smooth_window > 1:
            kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
            energies = np.convolve(energies, kernel, mode='same')
        if low_threshold is None:
            low_threshold = energy_threshold * 0.5
        voiced_mask = np.zeros_like(energies, dtype=np.uint8)
        voiced = False
        for i, e in enumerate(energies):
            if e > energy_threshold:
                voiced = True
            elif e < low_threshold:
                voiced = False
            voiced_mask[i] = 1 if voiced else 0
        return np.repeat(voiced_mask, frame_size)[:n]

    # -------------------------
    # Apply voiced mask
    # -------------------------
    voiced_mask = voiced_detection(signal)
    weighted_signal = signal * voiced_mask

    original_len = len(weighted_signal)
    pad_len = (range_size - (original_len % range_size)) % range_size
    if pad_len:
        weighted_signal = np.pad(weighted_signal, (0, pad_len), mode='reflect')

    n_ranges = len(weighted_signal) // range_size
    if n_ranges == 0:
        return [], np.zeros((0, range_size), dtype=np.float32), 0, range_size, \
               tile_size, domain_step, energy_thresh, original_len

    ranges = weighted_signal.reshape(n_ranges, range_size)

    # -------------------------
    # Build domains (memmap)
    # -------------------------
    domains_path, n_domains = build_domains_memmap(
        signal, tile_size, range_size, domain_step,
        block_size=500, tmpdir=domains_tmpdir, use_gpu=use_gpu
    )
    if n_domains == 0:
        return [], np.zeros((0, range_size), dtype=np.float32), 0, range_size, \
               tile_size, domain_step, energy_thresh, original_len

    # Compute domain statistics & clusters
    d_means, d_vars = compute_stats_memmap(domains_path, n_domains, range_size, batch_size=1024)
    cluster_ids, cluster_centers = build_domain_clusters(d_means, d_vars, n_clusters=32)

    domains_array = np.memmap(domains_path, dtype='float32', mode='r', shape=(n_domains, range_size)) # type: ignore
    domains_xp = xp.asarray(domains_array) if (use_gpu and GPU_WORKING) else domains_array

    # -------------------------
    # Worker: per-range hierarchical pruning + symmetry-aware affine
    # -------------------------
    def _process_range(idx):
        r = ranges[idx]

        # Pruned hierarchical matcher handles clusters, coarse distance, and exact solve
        idx_d, s, o, sym_flag, err = find_best_domain_pruned( # type: ignore
            r, domains_path, n_domains, range_size,
            d_means, d_vars, cluster_ids, cluster_centers,
            top_k_clusters=3, batch_size=1024, use_gpu=use_gpu
        )

        return (idx_d, float(s), float(o), int(sym_flag), float(err))

    # -------------------------
    # Parallel processing
    # -------------------------
    matches = []
    with mp.Pool(mp.cpu_count()) as pool:
        for res in pool.imap(_process_range, range(n_ranges), chunksize=batch_size):
            matches.append(res)

    # -------------------------
    # Cleanup
    # -------------------------
    domains_array_cpu = np.asarray(domains_array)
    try:
        os.remove(domains_path) # type: ignore
    except Exception:
        pass

    return (
        matches,
        domains_array_cpu,
        n_ranges,
        range_size,
        tile_size,
        domain_step,
        energy_thresh,
        original_len
    )


def save_compressed(filepath, matches, domains_array, range_size, framerate, sampwidth,
                    tile_size, domain_step, energy_threshold, original_len):
    """
    Save .fwav with SHA-256 checksum in a memory-efficient single pass.
    Includes original_len in header to allow exact reconstruction.
    """
    n_ranges = len(matches)
    n_domains = len(domains_array)

    sha = hashlib.sha256()

    with open(filepath, 'wb') as f:
        # Header
        f.write(b'FWAV')
        f.write(struct.pack('<B', FWAV_VERSION))
        f.write(struct.pack('<I', range_size))
        f.write(struct.pack('<I', framerate))
        f.write(struct.pack('<B', sampwidth))
        f.write(struct.pack('<H', tile_size))
        f.write(struct.pack('<H', domain_step))
        f.write(struct.pack('<f', energy_threshold))
        f.write(struct.pack('<I', n_ranges))
        f.write(struct.pack('<I', n_domains))
        f.write(struct.pack('<I', original_len))  # NEW

        # Placeholder for checksum
        checksum_pos = f.tell()
        f.write(b'\0' * 32)

        # Write domains
        for d in domains_array:
            b = d.astype(np.float32).tobytes()
            f.write(b)
            sha.update(b)

        # Write matches
        for m in matches:
            b = struct.pack('<IffBf', int(m[0]), float(m[1]), float(m[2]), int(m[3]), float(m[4]))
            f.write(b)
            sha.update(b)

        # Finalize checksum
        checksum = sha.digest()
        f.seek(checksum_pos)
        f.write(checksum)


def load_compressed(filepath, verify_checksum=True):
    """
    Load a FWAV compressed file with optional SHA-256 verification.
    Returns original_len as well.
    """
    with open(filepath, 'rb') as f:
        if f.read(4) != b'FWAV':
            raise ValueError('Not a FWAV file')

        version = struct.unpack('<B', f.read(1))[0]
        if version != FWAV_VERSION:
            raise ValueError(f'Unsupported FWAV version: {version}')

        range_size = struct.unpack('<I', f.read(4))[0]
        framerate = struct.unpack('<I', f.read(4))[0]
        sampwidth = struct.unpack('<B', f.read(1))[0]
        tile_size = struct.unpack('<H', f.read(2))[0]
        domain_step = struct.unpack('<H', f.read(2))[0]
        energy_threshold = struct.unpack('<f', f.read(4))[0]
        n_ranges = struct.unpack('<I', f.read(4))[0]
        n_domains = struct.unpack('<I', f.read(4))[0]
        original_len = struct.unpack('<I', f.read(4))[0]  # NEW

        stored_checksum = f.read(32)
        sha = hashlib.sha256() if verify_checksum else None

        domains = []
        for _ in range(n_domains):
            b = f.read(4 * range_size)
            arr = np.frombuffer(b, dtype=np.float32)
            domains.append(arr)
            if verify_checksum:
                sha.update(b) # type: ignore

        matches = []
        for _ in range(n_ranges):
            b = f.read(17)
            m = struct.unpack('<IffBf', b)
            matches.append(m)
            if verify_checksum:
                sha.update(b) # type: ignore

        if verify_checksum:
            calc_checksum = sha.digest() # type: ignore
            if calc_checksum != stored_checksum:
                raise ValueError('Checksum mismatch — file may be corrupted')

    domains_array = np.vstack(domains)
    matches_list = [(int(m[0]), float(m[1]), float(m[2]), int(m[3]), float(m[4])) for m in matches]

    return matches_list, domains_array, n_ranges, range_size, framerate, sampwidth, tile_size, domain_step, energy_threshold, original_len


def decompress_audio(matches, domains_array, n_ranges, range_size, iterations=8,
                     convergence_eps=1e-3, use_gpu=False, original_len=None,
                     s_clip=16.0, s_damping=0.0):
    """
    Vectorized fractal audio reconstruction with optional iterative per-iteration
    refinement of the scale. Returns reconstructed 1-D array (NumPy or CuPy array)
    Output trimmed to original_len if provided.
    """
    xp = cp if (use_gpu and GPU_WORKING) else np

    recon_len = n_ranges * range_size
    recon = xp.zeros(recon_len, dtype=xp.float32)

    domain_indices = xp.array([m[0] for m in matches], dtype=xp.int32)
    s_stored = xp.array([m[1] for m in matches], dtype=xp.float32)
    o_stored = xp.array([m[2] for m in matches], dtype=xp.float32)
    sym_flags = xp.array([m[3] for m in matches], dtype=xp.bool_)

    domains_xp = xp.asarray(domains_array, dtype=xp.float32) if (use_gpu and GPU_WORKING) else domains_array

    starts = xp.arange(n_ranges, dtype=xp.int32) * range_size
    scatter_idx = xp.repeat(starts, range_size) + xp.tile(xp.arange(range_size, dtype=xp.int32), n_ranges)
    counts_flat = xp.ones(n_ranges * range_size, dtype=xp.float32)
    eps = 1e-12

    for it in range(iterations):
        recon_ranges = recon.reshape(n_ranges, range_size)

        tiles = domains_xp[domain_indices]
        if sym_flags.any():
            tiles = xp.where(sym_flags[:, None], tiles[:, ::-1], tiles)

        mean_d = tiles.mean(axis=1)
        tiles_centered = tiles - mean_d[:, None]

        mean_r_cur = recon_ranges.mean(axis=1)
        ranges_centered = recon_ranges - mean_r_cur[:, None]

        numerator = xp.sum(ranges_centered * tiles_centered, axis=1)
        denom = xp.sum(tiles_centered * tiles_centered, axis=1)

        valid = denom > eps
        s_opt = xp.zeros_like(denom)
        if xp.any(valid):
            s_opt[valid] = numerator[valid] / denom[valid]

        s_used = ((1.0 - s_damping) * s_stored + s_damping * s_opt) if s_damping > 0 else xp.where(valid, s_opt, s_stored)
        s_used = xp.clip(s_used, -abs(s_clip), abs(s_clip))

        o = o_stored
        transformed = s_used[:, None] * tiles + o[:, None]

        transformed_flat = transformed.ravel()
        out = xp.bincount(scatter_idx, weights=transformed_flat, minlength=recon_len)
        counts = xp.bincount(scatter_idx, weights=counts_flat, minlength=recon_len)

        recon_next = xp.zeros_like(out, dtype=xp.float32)
        nz = counts > 0
        if nz.any():
            recon_next[nz] = out[nz] / counts[nz]

        denom_norm = xp.linalg.norm(recon) if xp.linalg.norm(recon) > 0 else 1.0
        delta = float(xp.linalg.norm(recon_next - recon) / denom_norm)
        recon = recon_next

        logger.debug(f'Iteration {it+1}: delta={delta:.6e}')
        if delta < convergence_eps:
            logger.info(f'Converged after {it+1} iterations (delta={delta:.3e})')
            break

    # Trim to original_len if provided
    if original_len is not None:
        recon = recon[:original_len]

    return recon


# ----------------------- Metrics & Utilities -----------------------

def compute_snr(original, reconstructed):
    n = min(len(original), len(reconstructed))
    orig = np.asarray(original[:n], dtype=np.float64)
    recon = np.asarray(reconstructed[:n], dtype=np.float64)
    noise = orig - recon
    signal_power = np.sum(orig*orig)
    noise_power = np.sum(noise*noise)
    if noise_power <= 0:
        return float('inf')
    return 10.0 * np.log10(signal_power / noise_power)

# ----------------------- Batch Processing -----------------------

def process_file_compress(path, outdir=None, tile=1024, energy_thresh=1e-4, use_gpu=False):
    try:
        start = time.time()
        signal, framerate, sampwidth = read_wav_mono(path)

        if sampwidth == 4: 
            signal = signal.astype(np.float32)
            signal = np.clip(signal, -1.0, 1.0)  # optional safety clipping

        matches, domains, n_ranges, range_size, tile_size, domain_step, energy_threshold, original_len = \
            compress_audio(signal, framerate, sampwidth, tile_size=tile, energy_thresh=energy_thresh, use_gpu=use_gpu)

        # Ensure output directory exists
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir)

        outpath = (os.path.splitext(path)[0] + '.fwav') if outdir is None else os.path.join(outdir, os.path.basename(path) + '.fwav')
        save_compressed(outpath, matches, domains, range_size, framerate, sampwidth,
                        tile_size, domain_step, energy_threshold, original_len)

        elapsed = time.time() - start
        in_size = os.path.getsize(path)
        out_size = os.path.getsize(outpath)
        ratio = in_size / out_size if out_size > 0 else 0
        logger.info(f'Compressed {path} -> {outpath}  time={elapsed:.2f}s  ratio={ratio:.2f}')
        return {'input': path, 'output': outpath, 'time_s': elapsed, 'ratio': ratio}
    except Exception as e:
        logger.exception('Compression failed for %s', path)
        return {'input': path, 'error': str(e)}


def process_file_decompress(path, outdir=None, iterations=8, eps=1e-3, use_gpu=False):
    try:
        start = time.time()
        matches, domains, n_ranges, range_size, framerate, sampwidth, tile_size, domain_step, energy_threshold, original_len = load_compressed(path)
        recon = decompress_audio(matches, domains, n_ranges, range_size,
                                 iterations=iterations, convergence_eps=eps,
                                 use_gpu=use_gpu, original_len=original_len)

        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir)

        if sampwidth == 4:  # float WAV
            recon = np.clip(recon, -1.0, 1.0)

        outpath = (os.path.splitext(path)[0] + '_recon.wav') if outdir is None else os.path.join(outdir, os.path.basename(path) + '_recon.wav')
        write_wav(outpath, np.asarray(recon), framerate, sampwidth)

        elapsed = time.time() - start
        logger.info(f'Decompressed {path} -> {outpath}  time={elapsed:.2f}s')
        return {'input': path, 'output': outpath, 'time_s': elapsed}
    except Exception as e:
        logger.exception('Decompression failed for %s', path)
        return {'input': path, 'error': str(e)}

# ----------------------- CLI -----------------------

def main():
    parser = argparse.ArgumentParser(description='Fractal WAV compressor with GPU, batch processing, and metrics')
    sub = parser.add_subparsers(dest='cmd')

    # ---------------- Compress ----------------
    pc = sub.add_parser('compress')
    pc.add_argument('input', help='input file or directory')
    pc.add_argument('--tile', type=int, default=1024)
    pc.add_argument('--out', default=None, help='output file or directory')
    pc.add_argument('--energy-thresh', type=float, default=1e-4)
    pc.add_argument('--gpu', action='store_true')
    pc.add_argument('--batch', action='store_true', help='treat input as directory and compress all WAV inside')
    pc.add_argument('--workers', type=int, default=4, help='parallel file-level workers for batch')

    # ---------------- Decompress ----------------
    pd = sub.add_parser('decompress')
    pd.add_argument('input', help='input file or directory')
    pd.add_argument('--out', default=None, help='output file or directory')
    pd.add_argument('--iter', type=int, default=8)
    pd.add_argument('--eps', type=float, default=1e-3)
    pd.add_argument('--gpu', action='store_true')
    pd.add_argument('--batch', action='store_true', help='treat input as directory and decompress all FWAV inside')
    pd.add_argument('--workers', type=int, default=4, help='parallel file-level workers for batch')

    args = parser.parse_args()

    if args.cmd == 'compress':
        if args.batch and os.path.isdir(args.input):
            files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith('.wav')]
            # Skip files that already have .fwav output
            files_to_process = []
            for f in files:
                outpath = (os.path.splitext(f)[0] + '.fwav') if args.out is None else os.path.join(args.out, os.path.basename(f) + '.fwav')
                if not os.path.exists(outpath):
                    files_to_process.append(f)
            logger.info(f'Batch compressing {len(files_to_process)}/{len(files)} files using {args.workers} workers')

            if files_to_process:
                pool = Pool(processes=min(args.workers, len(files_to_process)))
                try:
                    jobs = [pool.apply_async(process_file_compress, (f, args.out, args.tile, args.energy_thresh, args.gpu)) for f in files_to_process]
                    results = [j.get() for j in jobs]
                finally:
                    pool.close()
                    pool.join()

                metrics_file = os.path.join(args.out or args.input, 'compression_metrics.json')
                os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
                with open(metrics_file, 'w') as mf:
                    json.dump(results, mf, indent=2)
                logger.info(f'Wrote metrics to {metrics_file}')
            else:
                logger.info('No files to compress — all already exist.')
        else:
            process_file_compress(args.input, args.out, args.tile, args.energy_thresh, args.gpu)

    elif args.cmd == 'decompress':
        if args.batch and os.path.isdir(args.input):
            files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith('.fwav')]
            # Skip files that already have reconstructed output
            files_to_process = []
            for f in files:
                outpath = (os.path.splitext(f)[0] + '_recon.wav') if args.out is None else os.path.join(args.out, os.path.basename(f) + '_recon.wav')
                if not os.path.exists(outpath):
                    files_to_process.append(f)
            logger.info(f'Batch decompressing {len(files_to_process)}/{len(files)} files using {args.workers} workers')

            if files_to_process:
                pool = Pool(processes=min(args.workers, len(files_to_process)))
                try:
                    jobs = [pool.apply_async(process_file_decompress, (f, args.out, args.iter, args.eps, args.gpu)) for f in files_to_process]
                    results = [j.get() for j in jobs]
                finally:
                    pool.close()
                    pool.join()

                metrics_file = os.path.join(args.out or args.input, 'decompression_metrics.json')
                os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
                with open(metrics_file, 'w') as mf:
                    json.dump(results, mf, indent=2)
                logger.info(f'Wrote metrics to {metrics_file}')
            else:
                logger.info('No files to decompress — all already exist.')
        else:
            process_file_decompress(args.input, args.out, args.iter, args.eps, args.gpu)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
