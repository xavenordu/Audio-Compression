#!/usr/bin/env python3
"""
fractal_wav_compressor_gpu_cli.py
Upgraded fractal compressor/decompressor for WAV files with GPU support, memory mapping, convergence-based iterative decoding,
batch processing, and logging/metrics.
Provides CLI and Python API.
"""

import argparse
import wave
import struct
import numpy as np
import os
import time
import json
import logging
from multiprocessing import Pool, cpu_count

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('fwavc')

# ----------------------- I/O helpers -----------------------

def read_wav_mono(path, mmap=False):
    with wave.open(path, 'rb') as w:
        nchan = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        if mmap:
            dtype = np.uint8 if sampwidth==1 else np.int16
            # NOTE: memmap directly on WAV file is fragile (header size varies). Use normal read for portability.
            raw = w.readframes(nframes)
            data = np.frombuffer(raw, dtype=dtype)
        else:
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


def build_domain_pool(signal, tile_size, range_size, domain_step=1, block_size=1000, use_gpu=False):
    n = len(signal)
    xp = cp if use_gpu else np
    starts = list(range(0, n-tile_size+1, domain_step))
    for i in range(0, len(starts), block_size):
        batch = starts[i:i+block_size]
        block_domains = []
        for s in batch:
            tile = signal[s:s+tile_size]
            # downsample tile into `range_size` values by block-averaging
            tile_blocks = np.array_split(np.asarray(tile), range_size)
            down = xp.array([blk.mean() for blk in tile_blocks], dtype=xp.float32)
            block_domains.append(down)
        yield xp.vstack(block_domains)


def voiced_detection(signal, frame_size=1024, energy_threshold=1e-3):
    n = len(signal)
    voiced = np.zeros(n, dtype=np.uint8)
    for start in range(0, n, frame_size):
        chunk = signal[start:start+frame_size]
        energy = np.mean(chunk * chunk)
        if energy > energy_threshold:
            voiced[start:start+len(chunk)] = 1
    return voiced

# ----------------------- Matching math -----------------------

def compute_stats(blocks):
    means = blocks.mean(axis=1)
    mean_sq = (blocks*blocks).mean(axis=1)
    vars = mean_sq - means*means
    vars = np.where(vars<0,0.0,vars)
    sums = blocks.sum(axis=1)
    return means, vars, sums


def match_range_to_domains(range_block, range_mean, range_var, domains, d_means, d_vars, d_sums, xp=np):
    m = len(range_block)
    dots = domains.dot(range_block)
    denom = m * d_vars
    s = xp.zeros_like(denom)
    valid = denom > 1e-12
    if xp.any(valid):
        s[valid] = (dots[valid] - m*d_means[valid]*range_mean)/denom[valid]
    sr2 = (range_block*range_block).mean()
    chi2 = xp.full(dots.shape, xp.inf)
    if xp.any(valid):
        chi2[valid] = sr2 + s[valid]*(s[valid]*d_vars[valid] + 2*d_means[valid]*range_mean - 2*dots[valid]/float(m))
    chi = xp.sqrt(xp.maximum(chi2, 0.0))
    best_idx = int(xp.argmin(chi))
    return best_idx, float(s[best_idx]), float(range_mean), 0, float(chi[best_idx])

# ----------------------- Compression & Decompression -----------------------

def compress_audio(signal, framerate, sampwidth, tile_size=1024, energy_thresh=1e-4, use_gpu=False):
    range_size = max(4, tile_size//16)
    domain_step = max(1, range_size//2)
    voiced_mask = voiced_detection(signal, frame_size=range_size*2, energy_threshold=energy_thresh)
    weighted_signal = signal * voiced_mask
    ranges = frame_ranges(weighted_signal, range_size, hop=range_size)

    # build domains in blocks to limit peak memory usage
    domains_gen = build_domain_pool(signal, tile_size, range_size, domain_step, block_size=500, use_gpu=use_gpu)
    xp = cp if use_gpu else np
    domains_blocks = []
    for block in domains_gen:
        domains_blocks.append(block)
    if not domains_blocks:
        domains_array = xp.empty((0, range_size), dtype=xp.float32)
    else:
        domains_array = xp.vstack(domains_blocks)
    domains_mir = domains_array[:, ::-1] if len(domains_array)>0 else domains_array
    all_domains = xp.vstack([domains_array, domains_mir]) if len(domains_array)>0 else domains_array

    r_means = ranges.mean(axis=1)
    r_vars = (ranges*ranges).mean(axis=1) - r_means*r_means
    r_vars = np.where(r_vars<0,0.0,r_vars)
    d_means, d_vars, d_sums = compute_stats(all_domains)

    # matching across ranges; include range index so we can reorder after parallel execution
    def worker(i):
        r = ranges[i]
        rb_mean = r_means[i]
        rb_var = r_vars[i]
        best_idx, s, mean_r, sym, err = match_range_to_domains(r, rb_mean, rb_var, all_domains, d_means, d_vars, d_sums, xp)
        return (i, int(best_idx), float(s), float(mean_r), int(sym), float(err))

    matches_collected = []
    pool = Pool(processes=min(cpu_count(), 8))
    try:
        for res in pool.imap_unordered(worker, range(len(ranges))):
            matches_collected.append(res)
    finally:
        pool.close(); pool.join()

    # sort by range index and strip index for downstream usage
    matches_collected.sort(key=lambda x: x[0])
    matches = [(m[1], m[2], m[3], m[4], m[5]) for m in matches_collected]

    return matches, np.asarray(domains_array), ranges.shape[0], range_size


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
        for d in domains_array:
            f.write(struct.pack('<'+'f'*len(d), *d.tolist()))
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
        domains = [np.array(struct.unpack('<'+'f'*range_size, f.read(4*range_size)), dtype=np.float32) for _ in range(n_domains)]
        matches = [struct.unpack('<IffBf', f.read(16)) for _ in range(n_ranges)]
    return [(int(m[0]), float(m[1]), float(m[2]), int(m[3]), float(m[4])) for m in matches], np.vstack(domains), n_ranges, range_size, framerate, sampwidth


def decompress_audio(matches, domains_array, n_ranges, range_size, iterations=8, convergence_eps=1e-3, use_gpu=False):
    xp = cp if use_gpu else np
    recon_len = n_ranges*range_size
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
            transformed = s*(d-mean_d)+mean_r
            start = i*range_size
            out[start:start+range_size] += transformed
        denom = xp.linalg.norm(recon) if xp.linalg.norm(recon)>0 else 1.0
        delta = float(xp.linalg.norm(out-recon)/denom)
        recon = out / max(1.0, int(xp.count_nonzero(out)))
        logger.debug(f'Iteration {it+1}: delta={delta:.6e}')
        if delta < convergence_eps:
            logger.info(f'Converged after {it+1} iterations (delta={delta:.3e})')
            break
    return recon

# ----------------------- Metrics & Utilities -----------------------

def compute_snr(original, reconstructed):
    # ensure same length
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
        outpath = (os.path.splitext(path)[0]+'.fwav') if outdir is None else os.path.join(outdir, os.path.basename(path)+'.fwav')
        save_compressed(outpath, matches, domains, range_size, framerate, sampwidth)
        elapsed = time.time() - start
        in_size = os.path.getsize(path)
        out_size = os.path.getsize(outpath)
        ratio = in_size / out_size if out_size>0 else 0
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
        outpath = (os.path.splitext(path)[0]+'_recon.wav') if outdir is None else os.path.join(outdir, os.path.basename(path)+'_recon.wav')
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
            matches, domains, n_ranges, range_size = compress_audio(*read_wav_mono(args.input), tile_size=args.tile, energy_thresh=args.energy_thresh, use_gpu=args.gpu)
            outpath = args.out or os.path.splitext(args.input)[0]+'.fwav'
            save_compressed(outpath, matches, domains, range_size, read_wav_mono(args.input)[1], read_wav_mono(args.input)[2])
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
            outpath = args.out or os.path.splitext(args.input)[0]+'_recon.wav'
            write_wav(outpath, np.asarray(recon), framerate, sampwidth)
            logger.info(f'Reconstructed WAV: {outpath}')

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
