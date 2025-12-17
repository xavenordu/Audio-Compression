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
EMBED_K = 16

def tile_embedding(x, k=EMBED_K):
    """
    Compute shape embedding for a 1D tile.
    - DCT-II, orthonormal
    - DC excluded
    - L2 normalized
    """
    x = np.asarray(x, dtype=np.float32)
    v = dct(x, norm='ortho')
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



# ----------------------- Fractal helpers -----------------------

def frame_ranges(signal, range_size, hop=None):
    hop = hop or range_size
    total = len(signal)
    ranges = [signal[i:i+range_size] for i in range(0, total-range_size+1, hop)]
    return np.vstack(ranges) if ranges else np.empty((0, range_size), dtype=signal.dtype)

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
            out[j] = tile_embedding(tile, emb_dim)
        emb_mm[i:i+len(out)] = out

    emb_mm.flush()
    return emb_path


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


def range_candidates_from_embedding(range_block,
                                    domain_embs,
                                    emb_dim=16,
                                    top_k=64):
    """
    Fast candidate selection using embedding similarity.
    Returns top_k domain indices sorted by descending similarity.
    """
    r_emb = tile_embedding(range_block, emb_dim)

    # cosine similarity (dot product)
    scores = domain_embs @ r_emb  # (n_domains,)

    if top_k >= len(scores):
        idxs = np.argsort(-scores)
    else:
        idxs = np.argpartition(-scores, top_k)[:top_k]
        idxs = idxs[np.argsort(-scores[idxs])]

    return idxs.astype(np.int32)


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


def ann_query(range_block, index_path, top_k=64, emb_dim=EMBED_K):
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
        domains_mm = np.memmap(domains_path, dtype='float32', mode='r', shape=(n_domains_file, range_size))
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


# ---------------- CPU worker ----------------
def cpu_worker(idx_slice, ranges, domain_embs_path, n_domains, range_size, emb_dim, candidate_queue,
               ann_index_path=None, energy_thresh=1e-4, fast_mode=True, batch_size=32):
    """
    CPU worker: selects candidate domains for a batch of ranges and pushes them to the GPU queue.
    """
    # Load domain embeddings once per worker
    domain_embs = np.memmap(domain_embs_path, dtype='float32', mode='r', shape=(n_domains, emb_dim))
    
    batch_idxs = []
    batch_candidates = []
    batch_ranges = []

    # Load ANN index once per worker process (if available)
    ann_index = None
    if ann_index_path is not None and HNSW_AVAILABLE:
        try:
            ann_index = hnswlib.Index(space='ip', dim=emb_dim)
            ann_index.load_index(ann_index_path)
            # set ef for queries; can be tuned
            try:
                ann_index.set_ef(50)
            except Exception:
                pass
        except Exception:
            ann_index = None

    for idx in idx_slice:
        r = ranges[idx]
        if fast_mode and np.mean(r**2) < energy_thresh * 0.75:
            candidate_idxs = []  # empty candidate
        else:
            candidate_idxs = None
            # Try ANN query first if index loaded in this worker
            if ann_index is not None:
                try:
                    q = tile_embedding(r, k=emb_dim).astype(np.float32)
                    labels, dists = ann_index.knn_query(q.reshape(1, -1), k=32)
                    candidate_idxs = np.asarray(labels[0], dtype=np.int32)
                except Exception:
                    candidate_idxs = None

            # Fallback to linear search
            if candidate_idxs is None or (hasattr(candidate_idxs, '__len__') and len(candidate_idxs) == 0):
                candidate_idxs = range_candidates_from_embedding(r, domain_embs, top_k=32)
        
        batch_idxs.append(idx)
        batch_candidates.append(candidate_idxs)
        batch_ranges.append(r)

        # When batch is full, push to queue
        if len(batch_idxs) >= batch_size:
            candidate_queue.put(list(zip(batch_idxs, batch_candidates, batch_ranges)))
            batch_idxs, batch_candidates, batch_ranges = [], [], []

    # Push any remaining items
    if batch_idxs:
        candidate_queue.put(list(zip(batch_idxs, batch_candidates, batch_ranges)))

    # Sentinel for this worker
    candidate_queue.put(None)


# ---------------- GPU worker ----------------
def gpu_worker(candidate_queue, domains_path, range_size, results_queue, use_gpu=True, batch_size=16, num_cpu_workers=None):
    """
    Batched GPU worker: collects multiple ranges from candidate_queue,
    performs vectorized affine solves on GPU, and puts results into results_queue.
    """
    active_workers = 0
    batch = []
    xp = cp if (use_gpu and GPU_WORKING) else np

    # Load domains memmap once (determine n_domains from file size)
    domains_xp = None
    if domains_path is not None:
        try:
            file_size = os.path.getsize(domains_path)
            n_domains_file = file_size // (4 * range_size)
            domains_array = np.memmap(domains_path, dtype='float32', mode='r', shape=(n_domains_file, range_size))
            domains_xp = xp.asarray(domains_array) if (use_gpu and GPU_WORKING) else domains_array
        except Exception:
            domains_xp = None

    while True:
        try:
            item = candidate_queue.get(timeout=0.1)
        except Exception:
            item = None

        if item is None:
            active_workers += 1
            if active_workers == num_cpu_workers:
                # Process any remaining batch
                if batch:
                    _process_gpu_batch(batch, domains_xp, results_queue) # type: ignore
                break
            continue

        batch.append(item)

        if len(batch) >= batch_size:
            _process_gpu_batch(batch, domains_xp, results_queue) # type: ignore
            batch.clear()


def _process_gpu_batch(batch, domains_xp, results_queue, use_gpu=True):
    """
    Process a batch of ranges on GPU (or CPU if use_gpu=False).
    """
    if not batch:
        return

    xp = cp if (use_gpu and GPU_WORKING) else np
    # Flatten the nested batch (each item may itself be a list of tuples)
    flat = [item for sub in batch for item in (sub if isinstance(sub, (list, tuple)) and not isinstance(sub[0], (int,)) else [sub])]
    # Each flat item is expected to be (idx, candidate_idxs, range_block)
    batch_idxs = [b[0] for b in flat]
    batch_candidate_idxs = [b[1] for b in flat]
    batch_ranges = [b[2] for b in flat]

    n_ranges = len(flat)
    n_candidates_max = max((len(c) for c in batch_candidate_idxs), default=0)
    if n_candidates_max == 0:
        # All empty candidates
        for idx in batch_idxs:
            results_queue.put((idx, (-1, 1.0, 0.0, 0, 0.0)))
        return
    # Ensure domains_xp is available
    if domains_xp is None:
        for idx in batch_idxs:
            results_queue.put((idx, (-1, 1.0, 0.0, 0, 0.0)))
        return

    # Process each range in batch
    for idx, candidates, r in zip(batch_idxs, batch_candidate_idxs, batch_ranges):
        # candidates may be a numpy array; avoid ambiguous truth value
        if candidates is None or (hasattr(candidates, '__len__') and len(candidates) == 0):
            results_queue.put((idx, (-1, 1.0, 0.0, 0, 0.0)))
            continue
        # Call the existing exact_affine_gpu function
        res = exact_affine_gpu(r, candidates, domains_xp)
        results_queue.put((idx, res))

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
def exact_affine_gpu(range_block, domain_idxs, domains_xp):
    """
    Batched GPU affine solve for one range and many candidate domains.
    """
    r = cp.asarray(range_block)
    tiles = domains_xp[domain_idxs]          # shape: (num_candidates, range_size)
    tiles_m = tiles[:, ::-1]                  # mirrored tiles

    best_err = cp.inf
    best_result = None

    for tiles_cur, sym in [(tiles, 0), (tiles_m, 1)]:
        mean_d = tiles_cur.mean(axis=1)
        d_c = tiles_cur - mean_d[:, None]
        r_mean = r.mean()
        r_c = r - r_mean

        denom = cp.sum(d_c * d_c, axis=1)
        valid = denom > 1e-8

        s = cp.zeros_like(denom)
        s[valid] = cp.sum(d_c[valid] * r_c, axis=1) / denom[valid]
        o = r_mean - s * mean_d

        recon = s[:, None] * tiles_cur + o[:, None]
        err = cp.linalg.norm(recon - r, axis=1)

        i = int(cp.argmin(err))
        if err[i] < best_err:
            best_err = err[i]
            best_result = (int(domain_idxs[i]), float(s[i]), float(o[i]), sym, float(err[i]))

    return best_result

def _process_range(idx, ranges, domains_path, domain_embs_path, n_domains, range_size, emb_dim=16, use_gpu=False, domains_gpu=None):
    """
    Worker function for one range. Uses embedding shortlist and optionally GPU affine solve.
    """
    r = ranges[idx]

    if domain_embs_path is None:
        return -1, 0.0, 0.0, 0, float('inf')

    domain_embs = np.memmap(domain_embs_path, dtype='float32', mode='r', shape=(n_domains, emb_dim))
    candidate_idxs = range_candidates_from_embedding(r, domain_embs, top_k=32)
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

def compress_audio(signal, framerate, sampwidth, tile_size=1024, energy_thresh=1e-4,
                   use_gpu=False, domains_tmpdir=None, batch_size=512, fast_mode=True):
    """
    Compress audio using fractal domain matching with **pipeline parallelism**.
    CPU: candidate selection via embeddings
    GPU: exact affine solve in batches
    """
    xp = cp if (use_gpu and GPU_WORKING) else np

    range_size = max(4, tile_size // 64)
    domain_step = max(1, range_size // 2)

    # ---------------- Voiced detection ----------------
    voiced_mask = voiced_detection(signal, frame_size=range_size*2, energy_threshold=energy_thresh)
    weighted_signal = signal * voiced_mask
    original_len = len(weighted_signal)

    if np.sum(weighted_signal**2) < 1e-8:
        return [], np.zeros((0, range_size), dtype=np.float32), 0, range_size, \
               tile_size, domain_step, energy_thresh, original_len

    pad_len = (range_size - (original_len % range_size)) % range_size
    if pad_len:
        weighted_signal = np.pad(weighted_signal, (0, pad_len), mode='reflect')

    n_ranges = len(weighted_signal) // range_size
    if n_ranges == 0:
        return [], np.zeros((0, range_size), dtype=np.float32), 0, range_size, \
               tile_size, domain_step, energy_thresh, original_len

    ranges = weighted_signal.reshape(n_ranges, range_size)

    # ---------------- Build domains (memmap) ----------------
    domains_path, n_domains = build_domains_memmap(
        signal, tile_size, range_size, domain_step,
        block_size=500, tmpdir=domains_tmpdir, use_gpu=(use_gpu and GPU_WORKING)
    )
    if n_domains == 0:
        return [], np.zeros((0, range_size), dtype=np.float32), 0, range_size, \
               tile_size, domain_step, energy_thresh, original_len

    # Domain stats / clustering removed: embeddings (ANN) are used for shortlist

    # Load domains into CPU or GPU
    domains_array = np.memmap(domains_path, dtype='float32', mode='r', shape=(n_domains, range_size)) # type: ignore
    domains_xp = xp.asarray(domains_array) if (use_gpu and GPU_WORKING) else domains_array

    # ---------------- Build domain embeddings (memmap) ----------------
    emb_path = build_domain_embeddings(domains_path, n_domains, range_size, emb_dim=EMBED_K, block_size=4096, tmpdir=domains_tmpdir)

    # ---------------- Pipeline parallelism ----------------
    manager = mp.Manager()
    candidate_queue = manager.Queue(maxsize=64)  # CPU → GPU queue
    results_queue = manager.Queue()

    num_cpu_workers = mp.cpu_count()
    cpu_slices = np.array_split(np.arange(n_ranges), num_cpu_workers)

    # Start CPU workers
    # Optionally build ANN index for large domain sets (done once in parent)
    ann_index_path = None
    try:
        if n_domains > 4096 and HNSW_AVAILABLE:
            ann_index_path = build_ann_index(emb_path, n_domains, emb_dim=EMBED_K, method='hnsw')
            if ann_index_path:
                logger.info(f'[FWAVC] Built HNSW ANN index at {ann_index_path}')
    except Exception:
        ann_index_path = None

    cpu_processes = [
        mp.Process(target=cpu_worker,
                   args=(sl, ranges, emb_path, n_domains, range_size, EMBED_K, candidate_queue, ann_index_path, energy_thresh, fast_mode))
        for sl in cpu_slices
    ]

    # Start batched GPU worker
    gpu_process = mp.Process(target=gpu_worker,
                             args=(candidate_queue, domains_path, range_size, results_queue, use_gpu, batch_size, num_cpu_workers))

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

    # Cleanup
    for p in cpu_processes:
        p.join()
    gpu_process.join()

    # Convert domains to CPU array for saving
    domains_array_cpu = np.asarray(domains_array)

    # Remove temporary files
    try:
        if domains_path: os.remove(domains_path)
        if emb_path: os.remove(emb_path)
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
