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

# ANN libraries (optional)
try:
    import hnswlib  # type: ignore
    HNSW_AVAILABLE = True
except Exception:
    hnswlib = None
    HNSW_AVAILABLE = False

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False


top_k = 32  # number of candidates to consider per range

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

from scipy.fftpack import dct
# ----------------------- Embedding helpers ----------------------- 
EMBED_K = 32  # number of DCT coefficients for tile embedding

from scipy.fftpack import dct

def tonal_embedding(tile, k=16):
    """Low-D DCT embedding (exclude DC)"""
    v = dct(tile, norm='ortho')
    v = v[1:k+1]  # drop DC, keep k coefficients
    nrm = np.linalg.norm(v)
    if nrm > 1e-8:
        v = v / nrm
    return v.astype(np.float32)

def transient_embedding(tile, k=16):
    """Captures temporal changes / transients"""
    diff = np.diff(tile, prepend=tile[0])
    # optional high-pass weighting
    diff = diff * np.linspace(1.0, 2.0, len(diff))
    v = dct(diff, norm='ortho')
    v = v[:k]
    nrm = np.linalg.norm(v)
    if nrm > 1e-8:
        v = v / nrm
    return v.astype(np.float32)

def multi_head_embedding(tile, tonal_k=8, transient_k=8):
    """Combine tonal + transient embeddings for a tile."""
    tonal = tile_embedding(tile, k=tonal_k)              # tonal content
    transient = transient_embedding(tile, k=transient_k) # temporal changes
    e = np.concatenate([tonal, transient])
    
    # pad if needed
    if len(e) < tonal_k + transient_k:
        e = np.pad(e, (0, tonal_k + transient_k - len(e)), mode='constant')
    return e.astype(np.float32)


def tile_embedding(x, k=EMBED_K):
    """
    Compute shape embedding for a 1D tile.
    - DCT-II, orthonormal
    - DC excluded
    - L2 normalized
    """
    x = np.asarray(x, dtype=np.float32)
    v = dct(x, norm='ortho')

    w = np.linspace(1.0, 2.0, len(v))
    v = v * w

    # Exclude DC term (v[0]) and take up to k coefficients; pad with zeros if needed
    available = max(0, len(v) - 1)
    take = min(k, available)
    if take > 0:
        e = v[1:1+take].astype(np.float32)
    else:
        e = np.zeros((0,), dtype=np.float32)

    if take < k:
        # pad to fixed size k
        pad = np.zeros((k - take,), dtype=np.float32)
        e = np.concatenate([e, pad])

    # L2 normalize (if non-zero)
    nrm = np.linalg.norm(e)
    if nrm > 1e-8:
        e = e / nrm
    return e

def quick_energy(x):
    return float(np.sum(x * x))

def correlation_upper_bound(r_c, d_c, r_norm, d_norm):
    # |corr| ≤ 1
    if d_norm < 1e-12 or r_norm < 1e-12:
        return 0.0
    return abs(float(np.dot(r_c, d_c))) / (r_norm * d_norm)


# ----------------------- Fractal helpers -----------------------

def frame_ranges(signal, range_size, hop=None):
    hop = hop or range_size
    signal = np.asarray(signal)
    total = signal.shape[0]
    
    if total < range_size:
        return np.empty((0, range_size), dtype=signal.dtype)
    
    num_frames = 1 + (total - range_size) // hop
    shape = (num_frames, range_size)
    strides = (signal.strides[0]*hop, signal.strides[0])
    
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

# ----------------------- Domain memmap building -----------------------

def build_domain_embeddings(domains_path,
                            n_domains,
                            range_size,
                            emb_dim=16,
                            block_size=4096,
                            tmpdir=None):
    """
    Build low-D DCT embeddings for each domain (memmap-backed).
    Embedding input is the *downsampled domain* of length range_size.
    """

    emb_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix='.emb',
        dir=tmpdir
    )
    emb_path = emb_file.name
    emb_file.close()

    domains_mm = np.memmap(
        domains_path,
        dtype='float32',
        mode='r',
        shape=(n_domains, range_size)
    )

    emb_mm = np.memmap(
        emb_path,
        dtype='float32',
        mode='w+',
        shape=(n_domains, emb_dim)
    )

    for i in range(0, n_domains, block_size):
        b = domains_mm[i:i+block_size]
        out = np.empty((len(b), emb_dim), dtype=np.float32)
        for j, tile in enumerate(b):
            out[j] = multi_head_embedding(tile, tonal_k=emb_dim//2, transient_k=emb_dim//2)

        emb_mm[i:i+len(out)] = out

    emb_mm.flush()
    return emb_path


from numpy.lib.stride_tricks import sliding_window_view

def build_domains_memmap(signal, tile_size, range_size, domain_step=1,
                         block_size=1000, tmpdir=None, use_gpu=False):
    """
    Build domain downsampled tiles and store them in a temporary memmap file.
    Vectorized using sliding_window_view to avoid Python loops.
    Returns (domains_path, n_domains)
    """
    xp = cp if (use_gpu and GPU_WORKING) else np

    signal = np.asarray(signal, dtype=np.float32)
    n = len(signal)

    if n < tile_size:
        return None, 0

    # ---- Create sliding windows: shape (n_tiles, tile_size)
    windows = sliding_window_view(signal, tile_size)[::domain_step]
    n_domains = windows.shape[0]
    if n_domains == 0:
        return None, 0

    # ---- Create temporary memmap
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.domains', dir=tmpdir)
    tmp_path = tmp.name
    tmp.close()
    domains_mm = np.memmap(tmp_path, dtype='float32', mode='w+',
                           shape=(n_domains, range_size))

    # ---- Downsampling parameters
    block_lengths = tile_size // range_size
    usable_len = block_lengths * range_size

    idx = 0
    for i in range(0, n_domains, block_size):
        batch = windows[i:i+block_size]        # (B, tile_size)
        batch_len = batch.shape[0]

        # Trim if not divisible
        batch = batch[:, :usable_len]

        # Reshape and downsample: (B, range_size, block_lengths) -> mean over last axis
        batch_rs = batch.reshape(batch_len, range_size, block_lengths)
        downsampled = batch_rs.mean(axis=2, dtype=np.float32)

        # Write to memmap
        domains_mm[idx:idx+batch_len, :] = downsampled
        idx += batch_len

    domains_mm.flush()
    return tmp_path, n_domains


def range_candidates_from_embedding(range_block,
                                    domain_embs,
                                    emb_dim=16,
                                    top_k=top_k):
    """
    Fast candidate selection using embedding similarity.
    Returns top_k domain indices sorted by descending similarity.
    """
    r_emb = multi_head_embedding(range_block, tonal_k=emb_dim//2, transient_k=emb_dim//2)

    # cosine similarity (dot product)
    scores = domain_embs @ r_emb  # (n_domains,)

    idxs = np.argpartition(-scores, top_k)[:top_k]
    return idxs[np.argsort(-scores[idxs])]

def build_ann_index(emb_path, n_domains, emb_dim=EMBED_K, index_path=None, method='hnsw', ef=200, M=16):
    """Build an ANN index for embeddings stored in `emb_path` memmap.
    Returns path to saved index file or None if index couldn't be built.
    """
    if method == 'hnsw' and not HNSW_AVAILABLE:
        return None
    if index_path is None:
        index_path = emb_path + '.ann'

    # load embeddings
    try:
        emb_mm = np.memmap(emb_path, dtype='float32', mode='r', shape=(n_domains, emb_dim))
        data = np.asarray(emb_mm)
    except Exception:
        try:
            data = np.fromfile(emb_path, dtype=np.float32).reshape(n_domains, emb_dim)
        except Exception:
            return None

    if method == 'hnsw' and HNSW_AVAILABLE:
        p = hnswlib.Index(space='ip', dim=emb_dim)
        p.init_index(max_elements=n_domains, ef_construction=ef, M=M)
        p.add_items(data, np.arange(n_domains))
        p.set_ef(50)
        p.save_index(index_path)
        return index_path

    # FAISS path can be added here (not implemented by default)
    return None


def ann_query(range_block, index_path, top_k=top_k, emb_dim=EMBED_K):
    """Query saved ANN index and return candidate ids (np.int32).
    Returns empty array on failure.
    """
    if index_path is None:
        return np.empty((0,), dtype=np.int32)
    if HNSW_AVAILABLE:
        try:
            idx = hnswlib.Index(space='ip', dim=emb_dim)
            idx.load_index(index_path)
            q = tile_embedding(range_block, k=emb_dim)
            labels, dists = idx.knn_query(q.astype(np.float32), k=top_k)
            return np.asarray(labels[0], dtype=np.int32)
        except Exception:
            return np.empty((0,), dtype=np.int32)

    # FAISS or other backends could be added here
    return np.empty((0,), dtype=np.int32)


def find_best_domain_affine(range_block, domains_path, candidate_idxs, range_size, use_gpu=False):
    xp = cp if (use_gpu and GPU_WORKING) else np
    rb = xp.asarray(range_block, dtype=xp.float32)
    rb_mean = float(rb.mean())
    sr2 = float((range_block**2).mean())

    best_err = float('inf')
    best_idx = -1
    best_s = 0.0
    best_o = 0.0
    sym_flag = 0

    # If no candidates, return safe sentinel
    if candidate_idxs is None or len(candidate_idxs) == 0:
        return -1, 0.0, 0.0, 0, float('inf')

    # Open a memmap for the full domains file (determine n_domains from file size)
    try:
        file_size = os.path.getsize(domains_path)
        n_domains_file = file_size // (4 * range_size)
    except Exception:
        n_domains_file = None

    if n_domains_file is not None:
        domains_mm = np.memmap(domains_path, dtype='float32', mode='r', shape=(n_domains, range_size))

    else:
        # Fallback: try to open without shape (may raise); keep previous behavior
        domains_mm = np.memmap(domains_path, dtype='float32', mode='r')

    # Filter out any invalid (negative) candidate indices
    try:
        candidate_list = [int(x) for x in candidate_idxs if int(x) >= 0]
    except Exception:
        candidate_list = [int(x) for x in candidate_idxs]

    if len(candidate_list) == 0:
        return -1, 0.0, 0.0, 0, float('inf')

    for idx in candidate_list:
        tile = xp.asarray(domains_mm[int(idx)], dtype=xp.float32)
        tile_mean = float(tile.mean())
        tile_c = tile - tile_mean
        r_c = rb - rb_mean
        denom = float(xp.sum(tile_c * tile_c))
        if denom < 1e-12:
            continue

        # Original
        s0 = float(xp.sum(r_c * tile_c) / denom)
        o0 = rb_mean - s0 * tile_mean
        err0 = float(xp.linalg.norm(rb - (s0*tile + o0)))

        # Mirrored
        tile_m = tile[::-1]
        tile_m_mean = float(tile_m.mean())
        tile_m_c = tile_m - tile_m_mean
        denom_m = float(xp.sum(tile_m_c * tile_m_c))
        if denom_m < 1e-12:
            continue
        s1 = float(xp.sum(r_c * tile_m_c) / denom_m)
        o1 = rb_mean - s1 * tile_m_mean
        err1 = float(xp.linalg.norm(rb - (s1*tile_m + o1)))

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

import librosa

def perceptual_error_batch(candidate_tiles, target_tile, mel_fb=None, transient_mask=None,
                           transient_weight=1.0, use_gpu=False):
    """
    Vectorized perceptual error for multiple candidate tiles.
    candidate_tiles: (n_candidates, range_size)
    target_tile: (range_size,)
    Returns: (n_candidates,) error scores
    """
    xp = cp if (use_gpu and GPU_WORKING) else np
    tiles = xp.asarray(candidate_tiles)
    r = xp.asarray(target_tile)

    # Default Mel weighting
    range_size = r.shape[0]
    if mel_fb is None:
        mel_weights = xp.linspace(1.0, 0.5, range_size).astype(xp.float32)
    else:
        mel_weights = mel_fb

    # Transient envelope
    env = xp.abs(r[1:] - r[:-1])
    env = xp.pad(env, (0, 1))
    if transient_mask is not None:
        env *= transient_mask

    diff = tiles - r[None, :]
    weighted_diff = diff * mel_weights[None, :]
    weighted_diff *= (1.0 + transient_weight * env[None, :])

    return xp.linalg.norm(weighted_diff, axis=1)


def get_mel_filterbank(sr=44100, n_fft=1024, n_mels=40, fmin=20, fmax=None):
    fmax = fmax or sr // 2
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    return mel_fb.astype(np.float32)

def compute_transient_mask(signal, frame_size=256):
    # Simple local energy difference for transient weighting
    signal = np.abs(signal)
    mask = np.zeros_like(signal)
    mask[frame_size:] = np.maximum(0, signal[frame_size:] - signal[:-frame_size])
    mask /= (mask.max() + 1e-8)
    return mask

def range_candidates_from_embedding_emb(q, domain_embs, top_k=64):
    # domain_embs: (n_domains, emb_dim)
    scores = domain_embs @ q
    if top_k >= len(scores):
        return np.argsort(scores)[::-1].astype(np.int32)
    idxs = np.argpartition(scores, -top_k)[-top_k:]
    return idxs[np.argsort(scores[idxs])[::-1]].astype(np.int32)


def pad_candidates(idxs, top_k):
    if idxs is None or len(idxs) == 0:
        return np.full(top_k, -1, dtype=np.int32)
    idxs = np.asarray(idxs, dtype=np.int32)
    if len(idxs) >= top_k:
        return idxs[:top_k]
    out = np.full(top_k, -1, dtype=np.int32)
    out[:len(idxs)] = idxs
    return out


# ---------------- CPU worker ----------------
def cpu_worker(
    *,
    idx_slice,
    ranges,
    range_embs,
    domain_embs_path,
    n_domains,
    emb_dim,
    candidate_queue,
    ann_index_path=None,
    energy_thresh=1e-4,
    fast_mode=True,
    batch_size=32,
):
    import os
    assert isinstance(domain_embs_path, (str, os.PathLike)), (
        f"domain_embs_path must be a path, got {type(domain_embs_path)}: {domain_embs_path}"
    )

    # Load domain embeddings once per worker
    domain_embs = np.memmap(
        domain_embs_path,
        dtype="float32",
        mode="r",
        shape=(n_domains, emb_dim),
    )

    batch = []

    # Load ANN index once per worker
    ann_index = None
    if ann_index_path is not None and HNSW_AVAILABLE:
        try:
            ann_index = hnswlib.Index(space="ip", dim=emb_dim)
            ann_index.load_index(ann_index_path)
            try:
                ann_index.set_ef(50)
            except Exception:
                pass
        except Exception:
            ann_index = None

    for idx in idx_slice:
        r = ranges[idx]

        # ---- Energy prune
        if fast_mode and np.mean(r ** 2) < energy_thresh * 0.75:
            cand = np.empty(0, dtype=np.int32)
        else:
            q = range_embs[idx]
            cand = None

            # ---- ANN first
            if ann_index is not None:
                try:
                    labels, _ = ann_index.knn_query(q.reshape(1, -1), k=top_k)
                    cand = np.asarray(labels[0], dtype=np.int32)
                except Exception:
                    cand = None

            # ---- Fallback: linear
            if cand is None or len(cand) == 0:
                cand = range_candidates_from_embedding_emb(
                    q, domain_embs, top_k=top_k
                )

        cand_pad = pad_candidates(cand, top_k)
        batch.append((idx, cand_pad))

        if len(batch) >= batch_size:
            candidate_queue.put(batch)
            batch = []

    if batch:
        candidate_queue.put(batch)

    candidate_queue.put(None)



# ---------------- GPU worker ----------------
def gpu_worker(
    candidate_queue,
    domains_path,
    ranges,             
    range_size,
    results_queue,
    use_gpu=True,
    batch_size=16,
    num_cpu_workers=None,
    transient_weight=1.0,
    n_mels=40,
    mel_fb=None,
    transient_masks=None,
):
    """
    Robust GPU worker process for fractal audio compression.
    Safely integrates _process_gpu_batch and handles CPU sentinels.
    """
    import logging, os, time
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("gpu_worker")

    xp = cp if (use_gpu and "cp" in globals() and GPU_WORKING) else np
    batch = []

    finished_workers = 0
    expected_workers = num_cpu_workers or 0

    # ---------------- Load domains safely ----------------
    domains_xp = None
    if domains_path:
        try:
            file_size = os.path.getsize(domains_path)
            n_domains_file = file_size // (4 * range_size)
            domains_array = np.memmap(
                domains_path,
                dtype="float32",
                mode="r",
                shape=(n_domains_file, range_size),
            )
            domains_xp = domains_array if (use_gpu and GPU_WORKING) else domains_array
            logger.info(f"[GPU Worker] Loaded domains: {domains_xp.shape}")
        except Exception as e:
            logger.warning(f"[GPU Worker] Failed to load domains memmap: {e}")
            domains_xp = None

    # ---------------- Main loop ----------------
    while True:
        try:
            item = candidate_queue.get(timeout=0.1)
        except Exception:
            continue

        if item is None:  # sentinel from CPU worker
            finished_workers += 1
            logger.debug(f"[GPU Worker] Received sentinel ({finished_workers}/{expected_workers})")
            if finished_workers >= expected_workers:
                # Process any remaining batch before shutdown
                if batch:
                    try:
                        _flush_gpu_batch(
                                batch=batch,
                                ranges=ranges,
                                domains_xp=domains_xp,
                                results_queue=results_queue,
                                use_gpu=use_gpu,
                            )
                        batch.clear()

                    except Exception as e:
                        logger.error(f"[GPU Worker] Error processing final batch: {e}")
                    batch.clear()
                logger.info("[GPU Worker] All CPU workers finished. Exiting.")
                break
            continue

        # Append item and process if batch is full
        batch.extend(item)
        if len(batch) >= batch_size:
            try:
                _flush_gpu_batch(
                        batch=batch,
                        ranges=ranges,
                        domains_xp=domains_xp,
                        results_queue=results_queue,
                        use_gpu=use_gpu,
                    )
                batch.clear()

            except Exception as e:
                logger.error(f"[GPU Worker] Error processing batch: {e}")
            batch.clear()

    # ---------------- Drain remaining items from queue ----------------
    while not candidate_queue.empty():
        try:
            item = candidate_queue.get_nowait()
            if item is not None:
                batch.extend(item)
        except Exception:
            break

    if batch:
        try:
            _flush_gpu_batch(
                batch=batch,
                ranges=ranges,
                domains_xp=domains_xp,
                results_queue=results_queue,
                use_gpu=use_gpu,
            )
            batch.clear()

        except Exception as e:
            logger.error(f"[GPU Worker] Error processing remaining items: {e}")
        batch.clear()

    logger.info("[GPU Worker] Shutdown complete.")


def _process_gpu_batch(
    range_idxs,
    ranges_xp,          # (B, N)
    domain_idxs,        # (B, K) int32, padded with -1
    domains_xp,         # (n_domains, N)
    results_queue,
    use_gpu=True,
    s_clip=16.0,
):
    xp = cp if (use_gpu and "cp" in globals()) else np

    B, N = ranges_xp.shape
    K = domain_idxs.shape[1]

    # ---------------- Safe domain indices ----------------
    domain_safe = domain_idxs.copy()
    domain_safe[domain_safe < 0] = 0

    # ---------------- Gather domains: (B, K, N) ----------------
    domains = domains_xp[domain_safe]

    # ---------------- Symmetry: (B, 2K, N) ----------------
    domains_rev = domains[:, :, ::-1]
    domains_sym = xp.concatenate([domains, domains_rev], axis=1)

    sym_flags = xp.concatenate([
        xp.zeros((B, K), dtype=xp.int8),
        xp.ones((B, K), dtype=xp.int8)
    ], axis=1)

    domain_idxs_sym = xp.concatenate([domain_safe, domain_safe], axis=1)

    # ---------------- Center ranges ----------------
    r_mean = xp.mean(ranges_xp, axis=1, keepdims=True)        # (B, 1)
    r_c = ranges_xp - r_mean                                  # (B, N)
    r_energy = xp.sum(r_c * r_c, axis=1, keepdims=True)       # (B, 1)
    r_norm = xp.sqrt(r_energy) + 1e-12                        # (B, 1)

    # ---------------- Center domains ----------------
    d_mean = xp.mean(domains_sym, axis=2, keepdims=True)      # (B, 2K, 1)
    d_c = domains_sym - d_mean                                 # (B, 2K, N)
    d_energy = xp.sum(d_c * d_c, axis=2, keepdims=True)       # (B, 2K, 1)
    d_norm = xp.sqrt(d_energy) + 1e-12                        # (B, 2K, 1)

    # ---------------- Correlation + bounds ----------------
    num = xp.sum(d_c * r_c[:, None, :], axis=2)               # (B, 2K)
    denom = xp.sum(d_c * d_c, axis=2) + 1e-12                 # (B, 2K)
    s = num / denom                                           # (B, 2K)
    o = r_mean - s * d_mean[:, :, 0]                          # (B, 2K) — broadcasting safe

    corr = xp.abs(num) / (r_norm * d_norm[:, :, 0])           # (B, 2K)
    min_err = r_norm * xp.sqrt(1.0 - xp.clip(corr, 0.0, 1.0) ** 2)  # (B, 2K)

    # ---------------- Reconstruction error ----------------
    recon = s[:, :, None] * domains_sym + o[:, :, None]      # (B, 2K, N)
    diff = recon - ranges_xp[:, None, :]                     # (B, 2K, N)
    err = xp.linalg.norm(diff, axis=2)                       # (B, 2K)

    # Mask padded candidates
    mask_invalid = xp.concatenate([domain_idxs < 0, domain_idxs < 0], axis=1)
    err = xp.where(mask_invalid, xp.inf, err)

    # ---------------- Select best ----------------
    idx = xp.argmin(err, axis=1)
    best_err = err[xp.arange(B), idx]
    best_domain = domain_idxs_sym[xp.arange(B), idx]
    best_s = xp.clip(s[xp.arange(B), idx], -abs(s_clip), abs(s_clip))
    best_o = o[xp.arange(B), idx]
    best_sym = sym_flags[xp.arange(B), idx]

    # ---------------- Emit ----------------
    to_numpy = lambda x: cp.asnumpy(x) if use_gpu and "cp" in globals() else x

    best_domain = to_numpy(best_domain)
    best_s = to_numpy(best_s)
    best_o = to_numpy(best_o)
    best_sym = to_numpy(best_sym)
    best_err = to_numpy(best_err)

    # Build all results at once
    batch_results = list(zip(
        range_idxs.tolist(),
        zip(
            best_domain.tolist(),
            best_s.tolist(),
            best_o.tolist(),
            best_sym.tolist(),
            best_err.tolist()
        )
    ))

    # Push to queue
    for item in batch_results:
        results_queue.put(item)

def _flush_gpu_batch(batch, ranges, domains_xp, results_queue, use_gpu, s_clip=16.0):
    xp = cp if (use_gpu and "cp" in globals() and GPU_WORKING) else np

    # batch: [(idx, cand_pad), ...]
    range_idxs = np.array([b[0] for b in batch], dtype=np.int32)
    domain_idxs = np.stack([b[1] for b in batch], axis=0)

    ranges_xp = xp.asarray(ranges[range_idxs])
    assert isinstance(batch[0], tuple) and len(batch[0]) == 2

    _process_gpu_batch(
        range_idxs=range_idxs,
        ranges_xp=ranges_xp,
        domain_idxs=domain_idxs,
        domains_xp=domains_xp,
        results_queue=results_queue,
        use_gpu=use_gpu,
        s_clip=s_clip,
    )

# ----------------------- Symmetry -----------------------

def apply_symmetry(tile):
    return [tile, tile[::-1]]  # identity + mirrored

# ----------------------- Compression & Decompression (memmap-enabled) -----------------------

# ---------------- Top-level voiced detection ----------------
def voiced_detection(signal, frame_size=64, energy_threshold=1e-4, smooth_window=5, low_threshold=None):
    """
    Detect voiced regions in a signal.
    Returns a binary mask (1=voiced, 0=unvoiced) of the same length as signal.
    """
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


# ---------------- Top-level worker for multiprocessing ----------------
def exact_affine_gpu(
    range_block,
    domain_idxs,
    domains_xp,
    sr=44100,
    transient_weight=1.0,
    mel_weights=None,
    transient_mask=None,
):
    # ---------------- Backend ----------------
    xp = cp.get_array_module(range_block) if "cp" in globals() else np # type: ignore

    r = xp.asarray(range_block, dtype=xp.float32)
    tiles_base = xp.asarray(domains_xp[domain_idxs], dtype=xp.float32)

    range_size = r.shape[0]

    # ---------------- Perceptual weights ----------------
    if mel_weights is None:
        mel_weights = xp.linspace(1.0, 0.5, range_size, dtype=xp.float32)
    else:
        mel_weights = xp.asarray(mel_weights, dtype=xp.float32)

    if transient_mask is None:
        env = xp.abs(r[1:] - r[:-1])
        env = xp.pad(env, (0, 1))
    else:
        env = xp.asarray(transient_mask, dtype=xp.float32)

    weight = mel_weights * (1.0 + transient_weight * env)

    # ---------------- Precompute range stats ----------------
    r_mean = r.mean()
    r_c = r - r_mean
    r_energy = xp.sum(r_c * r_c)
    r_norm = xp.sqrt(r_energy)

    best_err = xp.inf
    best_idx = -1
    best_s = 0.0
    best_o = 0.0
    best_sym = 0

    # ---------------- Symmetry loop ----------------
    for sym in (0, 1):
        tiles = tiles_base if sym == 0 else tiles_base[:, ::-1]

        mean_d = tiles.mean(axis=1)
        d_c = tiles - mean_d[:, None]

        # ---------- ENERGY PRUNING ----------
        d_energy = xp.sum(d_c * d_c, axis=1)
        energy_ratio = d_energy / (r_energy + 1e-12)
        valid_energy = (energy_ratio > 0.25) & (energy_ratio < 4.0)

        if not xp.any(valid_energy):
            continue

        d_c = d_c[valid_energy]
        tiles_v = tiles[valid_energy]
        idxs_v = xp.asarray(domain_idxs)[valid_energy]
        d_energy = d_energy[valid_energy]
        d_norm = xp.sqrt(d_energy)

        # ---------- CORRELATION UPPER BOUND PRUNING ----------
        corr = xp.abs(xp.sum(d_c * r_c, axis=1)) / (r_norm * d_norm + 1e-12)
        min_possible_err = r_norm * xp.sqrt(1.0 - xp.clip(corr, 0.0, 1.0) ** 2)

        keep = min_possible_err < best_err
        if not xp.any(keep):
            continue

        d_c = d_c[keep]
        tiles_v = tiles_v[keep]
        idxs_v = idxs_v[keep]
        d_norm = d_norm[keep]

        # ---------- AFFINE SOLVE ----------
        denom = xp.sum(d_c * d_c, axis=1)
        valid = denom > 1e-8

        s = xp.zeros_like(denom)
        s[valid] = xp.sum(d_c[valid] * r_c, axis=1) / denom[valid]
        o = r_mean - s * tiles_v.mean(axis=1)

        recon = s[:, None] * tiles_v + o[:, None]
        diff = (recon - r) * weight[None, :]
        err = xp.linalg.norm(diff, axis=1)

        i = int(xp.argmin(err))
        if err[i] < best_err:
            best_err = err[i]
            best_idx = int(idxs_v[i])
            best_s = float(s[i])
            best_o = float(o[i])
            best_sym = sym

    return best_idx, best_s, best_o, best_sym, float(best_err)


def _process_range(idx, ranges, domains_path, domain_embs_path, n_domains, range_size, emb_dim=16, use_gpu=False, domains_gpu=None):
    """
    Worker function for one range. Uses embedding shortlist and optionally GPU affine solve.
    """
    r = ranges[idx]

    if domain_embs_path is None:
        return -1, 0.0, 0.0, 0, float('inf')

    domain_embs = np.memmap(domain_embs_path, dtype='float32', mode='r', shape=(n_domains, emb_dim))
    candidate_idxs = range_candidates_from_embedding(r, domain_embs, top_k=64)
    if candidate_idxs is None or len(candidate_idxs) == 0:
        return -1, 0.0, 0.0, 0, float('inf')

    if use_gpu and domains_gpu is not None:
        return exact_affine_gpu(r, candidate_idxs, domains_gpu)
    else:
        return find_best_domain_affine(r, domains_path, candidate_idxs, range_size, use_gpu=False)

def _worker_batch(idx_slice, ranges, domains_path, domain_embs_path, n_domains, range_size,
                  use_gpu=False, domains_gpu=None, energy_thresh=1e-4, fast_mode=True):
    results = []
    for idx in idx_slice:
        r = ranges[idx]
        if fast_mode and np.mean(r**2) < energy_thresh*0.75:  # skip very silent ranges
            results.append((-1, 1.0, 0.0, 0, 0.0))
            continue
        res = _process_range(idx, ranges, domains_path, domain_embs_path, n_domains, range_size,
                             use_gpu=use_gpu, domains_gpu=domains_gpu)
        results.append(res)
    return results

def compress_audio(
    signal,
    framerate,
    sampwidth,
    tile_size=1024,
    emb_dim=16,
    top_k=top_k,
    ef_search=50,
    use_gpu=False,
    energy_thresh=1e-4,
    domains_tmpdir=None,
    batch_size_gpu=512,
    batch_size_cpu=128,
    fast_mode=True,
    transient_weight=1.0,
    n_mels=40,
    cpu_workers=None,
):
    """
    Compress audio using fractal domain matching with pipeline parallelism.
    CPU: candidate selection via embeddings
    GPU: exact affine solve in batches
    """
    xp = cp if (use_gpu and GPU_WORKING) else np

    range_size = max(4, tile_size // 256)
    domain_step = max(1, range_size // 4)

    # ---------------- Voiced detection ----------------
    voiced_mask = voiced_detection(
        signal,
        frame_size=range_size * 2,
        energy_threshold=energy_thresh,
    )
    weighted_signal = signal * voiced_mask
    original_len = len(weighted_signal)

    # ---------- Short / silent audio ----------
    if np.sum(weighted_signal ** 2) < 1e-8:
        return (
            [],
            np.zeros((0, range_size), dtype=np.float32),
            0,
            range_size,
            tile_size,
            domain_step,
            energy_thresh,
            original_len,
        )

    pad_len = (range_size - (original_len % range_size)) % range_size
    if pad_len:
        weighted_signal = np.pad(weighted_signal, (0, pad_len), mode="reflect")

    n_ranges = len(weighted_signal) // range_size
    if n_ranges == 0:
        return (
            [],
            np.zeros((0, range_size), dtype=np.float32),
            0,
            range_size,
            tile_size,
            domain_step,
            energy_thresh,
            original_len,
        )

    ranges = weighted_signal.reshape(n_ranges, range_size)

    # ---------------- Build domains ----------------
    domains_path = None
    emb_path = None
    domains_array = None

    try:
        domains_path, n_domains = build_domains_memmap(
            signal,
            tile_size,
            range_size,
            domain_step,
            block_size=500,
            tmpdir=domains_tmpdir,
            use_gpu=(use_gpu and GPU_WORKING),
        )

        if n_domains == 0:
            return (
                [],
                np.zeros((0, range_size), dtype=np.float32),
                0,
                range_size,
                tile_size,
                domain_step,
                energy_thresh,
                original_len,
            )

        domains_array = np.memmap(
            domains_path,
            dtype="float32",
            mode="r",
            shape=(n_domains, range_size))
        domains_xp = (
            xp.asarray(domains_array)
            if (use_gpu and GPU_WORKING)
            else domains_array
        )

        # ---------------- Embeddings ----------------
        emb_path = build_domain_embeddings(
            domains_path,
            n_domains,
            range_size,
            emb_dim=emb_dim,
            block_size=4096,
            tmpdir=domains_tmpdir,
        )

        # ---------------- ANN index ----------------
        ann_index_path = None
        if n_domains > 4096 and HNSW_AVAILABLE:
            ann_index_path = build_ann_index(
                emb_path,
                n_domains,
                emb_dim=emb_dim,
                method="hnsw",
                ef=ef_search,
            )

        # ---------------- Pipeline parallelism ----------------

        candidate_queue = mp.Queue(maxsize=64)
        results_queue = mp.Queue()

        # num_cpu_workers = mp.cpu_count() 
        if cpu_workers is None:
            cpu_workers = max(1, mp.cpu_count() // 2)
        cpu_slices = np.array_split(np.arange(n_ranges), cpu_workers)

        cpu_processes = [
            mp.Process(
                target=cpu_worker,
                kwargs=dict(
                    idx_slice=sl,
                    ranges=ranges,
                    range_embs=np.memmap(
                        emb_path,
                        dtype="float32",
                        mode="r",
                        shape=(n_ranges, emb_dim),
                    ),
                    domain_embs_path=emb_path,
                    n_domains=n_domains,
                    emb_dim=emb_dim,
                    candidate_queue=candidate_queue,
                    ann_index_path=ann_index_path,
                    energy_thresh=energy_thresh,
                    fast_mode=fast_mode,
                    batch_size=batch_size_cpu,
                ),
            )
            for sl in cpu_slices
        ]


        n_fft = max(1024, range_size)
        mel_fb = get_mel_filterbank(
            sr=framerate, n_fft=n_fft, n_mels=n_mels
        )
        transient_masks = [compute_transient_mask(r) for r in ranges]

        gpu_process = mp.Process(
            target=gpu_worker,
            kwargs={
                "candidate_queue": candidate_queue,
                "domains_path": domains_path,
                "ranges": ranges, 
                "range_size": range_size,
                "results_queue": results_queue,
                "use_gpu": use_gpu and GPU_WORKING,
                "batch_size": batch_size_gpu,
                "num_cpu_workers": cpu_workers,
                "transient_weight": transient_weight,
                "n_mels": n_mels,
                "mel_fb": mel_fb,
                "transient_masks": transient_masks,
            },
        )

        # ---------------- Start processes (ONCE) ----------------
        for p in cpu_processes:
            p.start()
        gpu_process.start()

        # ---------------- Collect results ----------------
        matches = [None] * n_ranges
        finished = 0
        while finished < n_ranges:
            idx, res = results_queue.get()
            matches[idx] = res
            finished += 1

        return (
            matches,
            np.asarray(domains_array),
            n_ranges,
            range_size,
            tile_size,
            domain_step,
            energy_thresh,
            original_len,
        )

    finally:
        # ---------------- Cleanup ----------------
        for p in getattr(locals(), "cpu_processes", []):
            if p.is_alive():
                p.join()

        if "gpu_process" in locals() and gpu_process.is_alive():
            gpu_process.join()

        try:
            if domains_path:
                os.remove(domains_path)
            if emb_path:
                os.remove(emb_path)
        except Exception:
            pass


# ---------------- save/load compressed ----------------

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

        # Write matches (use signed domain indices to allow -1 sentinel)
        for m in matches:
            b = struct.pack('<iffBf', int(m[0]), float(m[1]), float(m[2]), int(m[3]), float(m[4]))
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
            m = struct.unpack('<iffBf', b)
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

    # Handle sentinel -1 domain indices used for skipped/unvoiced ranges: mask them
    invalid_mask = domain_indices < 0
    if invalid_mask.any():
        # Use a cleaned indices array to avoid indexing errors; set invalid tiles to zero later
        domain_indices_clean = domain_indices.copy()
        domain_indices_clean[invalid_mask] = 0
        domain_indices = domain_indices_clean

    starts = xp.arange(n_ranges, dtype=xp.int32) * range_size
    scatter_idx = xp.repeat(starts, range_size) + xp.tile(xp.arange(range_size, dtype=xp.int32), n_ranges)
    counts_flat = xp.ones(n_ranges * range_size, dtype=xp.float32)
    eps = 1e-12

    for it in range(iterations):
        recon_ranges = recon.reshape(n_ranges, range_size)

        tiles = domains_xp[domain_indices]

        # For entries that had sentinel -1, ensure their tiles are all zeros
        if 'invalid_mask' in locals() and invalid_mask.any():
            # zero-out the tiles and corresponding stored parameters so they contribute nothing
            tiles = xp.array(tiles, copy=True)
            tiles[invalid_mask] = 0
            s_stored = s_stored.copy()
            o_stored = o_stored.copy()
            s_stored[invalid_mask] = 0.0
            o_stored[invalid_mask] = 0.0
            sym_flags = sym_flags.copy()
            sym_flags[invalid_mask] = False

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
        
        logger.info(f'Processed {len(matches)} ranges, domain matrix shape {domains.shape}')

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
    pc.add_argument('input', help='input WAV file or directory')
    pc.add_argument(
        'output',
        nargs='?',
        default=None,
        help='output FWAV file (required unless --batch)'
    )
    pc.add_argument('--tile', type=int, default=1024)
    pc.add_argument('--out', default=None, help='output directory (batch mode)')
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
        # ---------------- Non-batch ----------------
        if not args.batch:
            if args.output is None:
                parser.error("compress requires OUTPUT unless --batch is used")
            # args.out is literal file path in non-batch
            process_file_compress(args.input, args.output, args.tile, args.energy_thresh, args.gpu)

        # ---------------- Batch ----------------
        else:
            if args.output is not None:
                parser.error("Do not provide positional OUTPUT when using --batch; use --out instead")
            out_dir = args.out or args.input
            files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith('.wav')]
            files_to_process = []
            for f in files:
                outpath = os.path.join(out_dir, os.path.basename(f) + '.fwav')
                if not os.path.exists(outpath):
                    files_to_process.append(f)

            logger.info(f'Batch compressing {len(files_to_process)}/{len(files)} files using {args.workers} workers')

            if files_to_process:
                pool = Pool(processes=min(args.workers, len(files_to_process)))
                try:
                    jobs = [
                        pool.apply_async(
                            process_file_compress,
                            (f, os.path.join(out_dir, os.path.basename(f) + '.fwav'),
                             args.tile, args.energy_thresh, args.gpu)
                        ) for f in files_to_process
                    ]
                    results = [j.get() for j in jobs]
                finally:
                    pool.close()
                    pool.join()

                metrics_file = os.path.join(out_dir, 'compression_metrics.json')
                os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
                with open(metrics_file, 'w') as mf:
                    json.dump(results, mf, indent=2)
                logger.info(f'Wrote metrics to {metrics_file}')
            else:
                logger.info('No files to compress — all already exist.')

    elif args.cmd == 'decompress':
        # ---------------- Non-batch ----------------
        if not args.batch:
            out_file = args.out or (os.path.splitext(args.input)[0] + '_recon.wav')
            process_file_decompress(args.input, out_file, args.iter, args.eps, args.gpu)

        # ---------------- Batch ----------------
        else:
            out_dir = args.out or args.input
            files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith('.fwav')]
            files_to_process = []
            for f in files:
                outpath = os.path.join(out_dir, os.path.basename(f).replace('.fwav', '_recon.wav'))
                if not os.path.exists(outpath):
                    files_to_process.append(f)

            logger.info(f'Batch decompressing {len(files_to_process)}/{len(files)} files using {args.workers} workers')

            if files_to_process:
                pool = Pool(processes=min(args.workers, len(files_to_process)))
                try:
                    jobs = [
                        pool.apply_async(
                            process_file_decompress,
                            (f, os.path.join(out_dir, os.path.basename(f).replace('.fwav', '_recon.wav')),
                             args.iter, args.eps, args.gpu)
                        ) for f in files_to_process
                    ]
                    results = [j.get() for j in jobs]
                finally:
                    pool.close()
                    pool.join()

                metrics_file = os.path.join(out_dir, 'decompression_metrics.json')
                os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
                with open(metrics_file, 'w') as mf:
                    json.dump(results, mf, indent=2)
                logger.info(f'Wrote metrics to {metrics_file}')
            else:
                logger.info('No files to decompress — all already exist.')

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
