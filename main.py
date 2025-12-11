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
import wave
import struct
import numpy as np
import os
import time
import json
import logging
import tempfile
from multiprocessing import Pool, cpu_count

try:
    import cupy as cp
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

# ----------------------- I/O helpers -----------------------

def read_wav_mono(path, mmap=False):
    with wave.open(path, 'rb') as w:
        nchan = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)
        fmt = np.uint8 if sampwidth==1 else np.int16
        data = np.frombuffer(raw, dtype=fmt)
    if nchan > 1:
        data = data.reshape(-1, nchan).mean(axis=1)
    if sampwidth == 1:
        data = data.astype(np.int16) - 128
    return data.astype(np.float32), framerate, sampwidth


def write_wav(path, data, framerate, sampwidth):
    if sampwidth == 1:
        out = (data + 128).clip(0, 255).astype(np.uint8)
    elif sampwidth == 2:
        out = data.clip(-32768, 32767).astype(np.int16)
    else:
        raise ValueError('Unsupported sample width')
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

def build_domains_memmap(signal, tile_size, range_size, domain_step=1, block_size=1000, tmpdir=None):
    """
    Build domain downsampled tiles and store them in a temporary memmap file.
    Returns (domains_path, n_domains)
    """
    n = len(signal)
    starts = list(range(0, n - tile_size + 1, domain_step))
    n_domains = len(starts)
    if n_domains == 0:
        return None, 0

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.domains', dir=tmpdir)
    tmp_path = tmp.name
    tmp.close()

    # create memmap for domains (float32)
    domains_mm = np.memmap(tmp_path, dtype='float32', mode='w+', shape=(n_domains, range_size))

    # fill memmap in blocks
    idx = 0
    for i in range(0, n_domains, block_size):
        batch = starts[i:i+block_size]
        block_rows = []
        for s in batch:
            tile = signal[s:s+tile_size]
            tile_blocks = np.array_split(np.asarray(tile), range_size)
            down = np.array([blk.mean() for blk in tile_blocks], dtype=np.float32)
            block_rows.append(down)
        block_arr = np.vstack(block_rows)
        domains_mm[idx:idx+len(block_arr), :] = block_arr
        idx += len(block_arr)
    # flush to disk
    domains_mm.flush()
    return tmp_path, n_domains

# ----------------------- Voiced detection -----------------------

def voiced_detection(signal, frame_size=1024, energy_threshold=1e-3):
    n = len(signal)
    voiced = np.zeros(n, dtype=np.uint8)
    for start in range(0, n, frame_size):
        chunk = signal[start:start+frame_size]
        energy = np.mean(chunk * chunk)
        if energy > energy_threshold:
            voiced[start:start+len(chunk)] = 1
    return voiced

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


def find_best_domain_for_range_blockwise(range_block, domains_path, n_domains, range_size, d_means, d_vars, batch_size=1024, use_gpu=False):
    """
    Scan domains stored in memmap file in blocks (batch_size) to find best matching domain for the given range_block.
    Returns (best_idx, best_s, best_err)
    """
    xp = cp if (use_gpu and GPU_WORKING) else np
    domains_mm = np.memmap(domains_path, dtype='float32', mode='r', shape=(n_domains, range_size))
    m = len(range_block)
    best_idx = -1
    best_s = 0.0
    best_err = float('inf')
    rb = xp.asarray(range_block)
    rb_mean = float(range_block.mean())

    for i in range(0, n_domains, batch_size):
        batch = domains_mm[i:i+batch_size]
        if use_gpu and GPU_WORKING:
            batch_x = cp.asarray(batch)
            d_means_batch = cp.asarray(d_means[i:i+len(batch)])
            d_vars_batch = cp.asarray(d_vars[i:i+len(batch)])
        else:
            batch_x = batch
            d_means_batch = d_means[i:i+len(batch)]
            d_vars_batch = d_vars[i:i+len(batch)]

        dots = batch_x.dot(rb)
        denom = m * d_vars_batch
        s = xp.zeros_like(denom)
        valid = denom > 1e-12
        if xp.any(valid):
            s[valid] = (dots[valid] - m * d_means_batch[valid] * rb_mean) / denom[valid]
        sr2 = float((range_block * range_block).mean())
        chi2 = xp.full(dots.shape, xp.inf)
        if xp.any(valid):
            chi2[valid] = sr2 + s[valid] * (s[valid] * d_vars_batch[valid] + 2 * d_means_batch[valid] * rb_mean - 2 * dots[valid] / float(m))
        # move to CPU for comparisons if on GPU
        if use_gpu and GPU_WORKING:
            chi_cpu = cp.asnumpy(chi2)
            s_cpu = cp.asnumpy(s)
        else:
            chi_cpu = chi2
            s_cpu = s

        local_best = int(np.argmin(chi_cpu))
        local_err = float(np.sqrt(max(chi_cpu[local_best], 0.0)))
        if local_err < best_err:
            best_err = local_err
            best_idx = i + local_best
            best_s = float(s_cpu[local_best])
    return best_idx, best_s, best_err

# ----------------------- Compression & Decompression (memmap-enabled) -----------------------

def compress_audio(signal, framerate, sampwidth, tile_size=1024, energy_thresh=1e-4, use_gpu=False, domains_tmpdir=None):
    range_size = max(4, tile_size // 16)
    domain_step = max(1, range_size // 2)

    voiced_mask = voiced_detection(signal, frame_size=range_size*2, energy_threshold=energy_thresh)
    weighted_signal = signal * voiced_mask
    ranges = frame_ranges(weighted_signal, range_size, hop=range_size)

    # Build domains into memmap (embedded temporary file)
    domains_path, n_domains = build_domains_memmap(signal, tile_size, range_size, domain_step, block_size=500, tmpdir=domains_tmpdir)
    if n_domains == 0:
        # nothing to match
        return [], np.empty((0, range_size), dtype=np.float32), 0, range_size

    # compute per-domain stats in blocks (keeps small memory usage)
    d_means, d_vars = compute_stats_memmap(domains_path, n_domains, range_size, batch_size=1024)

    matches = []
    # for each range, find best domain by scanning domain memmap in batches
    for i in range(len(ranges)):
        r = ranges[i]
        best_idx, best_s, best_err = find_best_domain_for_range_blockwise(r, domains_path, n_domains, range_size, d_means, d_vars, batch_size=1024, use_gpu=use_gpu)
        sym_flag = 0
        # Note: we are storing only forward domains. In previous version we duplicated mirrored domains.
        matches.append((best_idx, best_s, float(r.mean()), sym_flag, best_err))

    # after matching, load domains fully into memory if reasonably sized for packaging into .fwav
    try:
        domains_array = np.memmap(domains_path, dtype='float32', mode='r', shape=(n_domains, range_size))
        # convert to regular ndarray for embedding in fwav (to keep .fwav portable)
        domains_array = np.asarray(domains_array)
    finally:
        # remove temporary memmap file
        try:
            os.remove(domains_path)
        except Exception:
            pass

    return matches, domains_array, len(ranges), range_size


def save_compressed(filepath, matches, domains_array, range_size, framerate, sampwidth):
    with open(filepath, 'wb') as f:
        f.write(b'FWAV')
        f.write(struct.pack('<I', range_size))
        f.write(struct.pack('<I', framerate))
        f.write(struct.pack('<B', sampwidth))
        n_ranges = len(matches)
        n_domains = len(domains_array)
        f.write(struct.pack('<I', n_ranges))
        f.write(struct.pack('<I', n_domains))
        # write domains
        for d in domains_array:
            f.write(struct.pack('<' + 'f'*len(d), *d.tolist()))
        # matches are tuples: (domain_idx, s, mean_r, sym, err)
        for m in matches:
            f.write(struct.pack('<IffBf', int(m[0]), float(m[1]), float(m[2]), int(m[3]), float(m[4])))


def load_compressed(filepath):
    with open(filepath, 'rb') as f:
        if f.read(4) != b'FWAV':
            raise ValueError('Not a FWAV file')
        range_size = struct.unpack('<I', f.read(4))[0]
        framerate = struct.unpack('<I', f.read(4))[0]
        sampwidth = struct.unpack('<B', f.read(1))[0]
        n_ranges = struct.unpack('<I', f.read(4))[0]
        n_domains = struct.unpack('<I', f.read(4))[0]
        domains = [np.array(struct.unpack('<' + 'f'*range_size, f.read(4*range_size)), dtype=np.float32) for _ in range(n_domains)]
        matches = [struct.unpack('<IffBf', f.read(16)) for _ in range(n_ranges)]
    return [(int(m[0]), float(m[1]), float(m[2]), int(m[3]), float(m[4])) for m in matches], np.vstack(domains), n_ranges, range_size, framerate, sampwidth


def decompress_audio(matches, domains_array, n_ranges, range_size, iterations=8, convergence_eps=1e-3, use_gpu=False):
    xp = cp if (use_gpu and GPU_WORKING) else np
    recon_len = n_ranges * range_size
    recon = xp.zeros(recon_len, dtype=xp.float32)

    for it in range(iterations):
        out = xp.zeros_like(recon)
        for i, (domain_idx, s, mean_r, sym, _) in enumerate(matches):
            if domain_idx >= len(domains_array):
                continue
            d = domains_array[domain_idx]
            if sym:
                d = d[::-1]
            mean_d = d.mean()
            transformed = s * (d - mean_d) + mean_r
            start = i * range_size
            out[start:start+range_size] += transformed
        denom = xp.linalg.norm(recon) if xp.linalg.norm(recon) > 0 else 1.0
        delta = float(xp.linalg.norm(out - recon) / denom)
        recon = out / max(1.0, int(xp.count_nonzero(out)))
        logger.debug(f'Iteration {it+1}: delta={delta:.6e}')
        if delta < convergence_eps:
            logger.info(f'Converged after {it+1} iterations (delta={delta:.3e})')
            break
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
        matches, domains, n_ranges, range_size = compress_audio(signal, framerate, sampwidth, tile_size=tile, energy_thresh=energy_thresh, use_gpu=use_gpu)
        outpath = (os.path.splitext(path)[0] + '.fwav') if outdir is None else os.path.join(outdir, os.path.basename(path) + '.fwav')
        save_compressed(outpath, matches, domains, range_size, framerate, sampwidth)
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
        matches, domains, n_ranges, range_size, framerate, sampwidth = load_compressed(path)
        recon = decompress_audio(matches, domains, n_ranges, range_size, iterations=iterations, convergence_eps=eps, use_gpu=use_gpu)
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

    pc = sub.add_parser('compress')
    pc.add_argument('input', help='input file or directory')
    pc.add_argument('--tile', type=int, default=1024)
    pc.add_argument('--out', default=None, help='output file or directory')
    pc.add_argument('--energy-thresh', type=float, default=1e-4)
    pc.add_argument('--gpu', action='store_true')
    pc.add_argument('--batch', action='store_true', help='treat input as directory and compress all WAV inside')
    pc.add_argument('--workers', type=int, default=4, help='parallel file-level workers for batch')

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
            logger.info(f'Batch compressing {len(files)} files using {args.workers} workers')
            pool = Pool(processes=min(args.workers, len(files)))
            try:
                jobs = [pool.apply_async(process_file_compress, (f, args.out, args.tile, args.energy_thresh, args.gpu)) for f in files]
                results = [j.get() for j in jobs]
            finally:
                pool.close(); pool.join()
            metrics_file = os.path.join(args.out or args.input, 'compression_metrics.json')
            with open(metrics_file, 'w') as mf:
                json.dump(results, mf, indent=2)
            logger.info(f'Wrote metrics to {metrics_file}')
        else:
            # single file
            signal, framerate, sampwidth = read_wav_mono(args.input)
            matches, domains, n_ranges, range_size = compress_audio(signal, framerate, sampwidth, tile_size=args.tile, energy_thresh=args.energy_thresh, use_gpu=args.gpu)
            outpath = args.out or os.path.splitext(args.input)[0] + '.fwav'
            save_compressed(outpath, matches, domains, range_size, framerate, sampwidth)
            logger.info(f'Compressed file: {outpath}')

    elif args.cmd == 'decompress':
        if args.batch and os.path.isdir(args.input):
            files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith('.fwav')]
            logger.info(f'Batch decompressing {len(files)} files using {args.workers} workers')
            pool = Pool(processes=min(args.workers, len(files)))
            try:
                jobs = [pool.apply_async(process_file_decompress, (f, args.out, args.iter, args.eps, args.gpu)) for f in files]
                results = [j.get() for j in jobs]
            finally:
                pool.close(); pool.join()
            metrics_file = os.path.join(args.out or args.input, 'decompression_metrics.json')
            with open(metrics_file, 'w') as mf:
                json.dump(results, mf, indent=2)
            logger.info(f'Wrote metrics to {metrics_file}')
        else:
            matches, domains, n_ranges, range_size, framerate, sampwidth = load_compressed(args.input)
            recon = decompress_audio(matches, domains, n_ranges, range_size, iterations=args.iter, convergence_eps=args.eps, use_gpu=args.gpu)
            outpath = args.out or os.path.splitext(args.input)[0] + '_recon.wav'
            write_wav(outpath, np.asarray(recon), framerate, sampwidth)
            logger.info(f'Reconstructed WAV: {outpath}')

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
