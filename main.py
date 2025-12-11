#!/usr/bin/env python3
"""
fractal_wav_compressor_gpu_cli.py
Upgraded fractal compressor/decompressor for WAV files with GPU support, memory mapping, and convergence-based iterative decoding.
Provides CLI and Python API.
"""

import argparse
import wave
import struct
import numpy as np
import os
from multiprocessing import Pool, cpu_count

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

# ----------------------- I/O helpers -----------------------

def read_wav_mono(path, mmap=False):
    with wave.open(path, 'rb') as w:
        nchan = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        if mmap:
            dtype = np.uint8 if sampwidth==1 else np.int16
            data = np.memmap(path, dtype=dtype, mode='r', offset=44)
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
            tile_blocks = xp.array_split(tile, range_size)
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
    s[valid] = (dots[valid] - m*d_means[valid]*range_mean)/denom[valid]
    sr2 = (range_block*range_block).mean()
    chi2 = xp.full(dots.shape, xp.inf)
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
    domains_gen = build_domain_pool(signal, tile_size, range_size, domain_step, block_size=500, use_gpu=use_gpu)
    domains_list = [block for block in domains_gen]
    xp = cp if use_gpu else np
    domains_array = xp.vstack(domains_list)
    domains_mir = domains_array[:, ::-1]
    all_domains = xp.vstack([domains_array, domains_mir])

    r_means = ranges.mean(axis=1)
    r_vars = (ranges*ranges).mean(axis=1) - r_means*r_means
    r_vars = np.where(r_vars<0,0.0,r_vars)
    d_means, d_vars, d_sums = compute_stats(all_domains)

    matches = []
    def worker(i):
        r = ranges[i]
        rb_mean = r_means[i]
        rb_var = r_vars[i]
        return match_range_to_domains(r, rb_mean, rb_var, all_domains, d_means, d_vars, d_sums, xp)

    pool = Pool(processes=min(cpu_count(),8))
    try:
        for res in pool.imap_unordered(worker, range(len(ranges))):
            matches.append(res)
    finally:
        pool.close(); pool.join()
    matches.sort(key=lambda x: x[0])
    return matches, domains_array, ranges.shape[0], range_size


def save_compressed(filepath, matches, domains_array, range_size, framerate, sampwidth):
    with open(filepath, 'wb') as f:
        f.write(b'FWAV')
        f.write(struct.pack('<I', range_size))
        f.write(struct.pack('<I', framerate))
        f.write(struct.pack('<B', sampwidth))
        n_domains = len(domains_array)
        n_ranges = len(matches)
        f.write(struct.pack('<I', n_ranges))
        f.write(struct.pack('<I', n_domains))
        for d in domains_array:
            f.write(struct.pack('<'+'f'*len(d), *d.tolist()))
        for m in matches:
            f.write(struct.pack('<IffB', m[0], m[1], m[2], m[3]))


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
        matches = [struct.unpack('<IffB', f.read(13)) for _ in range(n_ranges)]
    return matches, np.vstack(domains), n_ranges, range_size, framerate, sampwidth


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
            if sym: d = d[::-1]
            mean_d = d.mean()
            transformed = s*(d-mean_d)+mean_r
            start = i*range_size
            out[start:start+range_size] += transformed
        delta = xp.linalg.norm(out-recon)/xp.linalg.norm(recon) if xp.linalg.norm(recon)>0 else xp.inf
        recon = out / max(1.0, xp.count_nonzero(out))
        if delta < convergence_eps:
            break
    return recon

# ----------------------- CLI -----------------------

def main():
    parser = argparse.ArgumentParser(description='Fractal WAV compressor with GPU and convergence decoding')
    sub = parser.add_subparsers(dest='cmd')

    pc = sub.add_parser('compress')
    pc.add_argument('input')
    pc.add_argument('--tile', type=int, default=1024)
    pc.add_argument('--out', default=None)
    pc.add_argument('--energy-thresh', type=float, default=1e-4)
    pc.add_argument('--gpu', action='store_true')

    pd = sub.add_parser('decompress')
    pd.add_argument('input')
    pd.add_argument('--out', default=None)
    pd.add_argument('--iter', type=int, default=8)
    pd.add_argument('--eps', type=float, default=1e-3)
    pd.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    if args.cmd == 'compress':
        signal, framerate, sampwidth = read_wav_mono(args.input)
        matches, domains, n_ranges, range_size = compress_audio(signal, framerate, sampwidth, tile_size=args.tile, energy_thresh=args.energy_thresh, use_gpu=args.gpu)
        outpath = args.out or os.path.splitext(args.input)[0]+'.fwav'
        save_compressed(outpath, matches, domains, range_size, framerate, sampwidth)
        print('Compressed file:', outpath)

    elif args.cmd == 'decompress':
        matches, domains, n_ranges, range_size, framerate, sampwidth = load_compressed(args.input)
        recon = decompress_audio(matches, domains, n_ranges, range_size, iterations=args.iter, convergence_eps=args.eps, use_gpu=args.gpu)
        outpath = args.out or os.path.splitext(args.input)[0]+'_recon.wav'
        write_wav(outpath, recon, framerate, sampwidth)
        print('Reconstructed WAV:', outpath)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
