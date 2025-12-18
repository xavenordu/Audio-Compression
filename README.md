# Fractal WAV Compressor (FWAV)

A research‑grade **fractal audio compression system** for WAV files, progressively upgraded with **GPU acceleration**, **memory‑mapped domain storage**, **ANN‑based candidate search**, and **pipeline parallelism**.

This README documents **everything implemented so far**, step by step, explaining *what was built*, *why it exists*, and *how the system currently works end‑to‑end*.

---

## 1. Project Goal

The goal of this project is to implement a **scalable fractal audio compressor** capable of:

- Handling **long audio files** without exhausting RAM
- Leveraging **GPU acceleration** where available
- Maintaining **reasonable reconstruction quality** while experimenting with compression ratios
- Providing a **clean CLI and Python API** for experimentation

This is **not** a lossless codec. It is a **fractal / self‑similarity‑based lossy compressor**.

---

## 2. High‑Level Architecture

The system consists of four major phases:

1. **Input processing** (WAV loading, mono conversion)
2. **Compression** (domain construction, candidate search, affine matching)
3. **Serialization** (custom `.fwav` format)
4. **Decompression** (iterative fractal reconstruction)

Key design principles:

- Stream large data via **memmap** instead of RAM
- Separate **fast approximate search** from **exact matching**
- Allow CPU/GPU fallback automatically
- Keep the format **self‑contained**

---

## 3. Audio I/O Layer

### 3.1 WAV Reading

Implemented `read_wav_mono()`:

- Supports:
  - 8‑bit PCM
  - 16‑bit PCM
  - 24‑bit PCM
  - 32‑bit float WAV
- Automatically converts multi‑channel audio to **mono**
- Normalizes data into `float32`

This guarantees a **consistent internal representation**.

### 3.2 WAV Writing

Implemented `write_wav()`:

- Writes back using the **original sample width**
- Correctly handles integer and float formats
- Used during decompression output

---

## 4. GPU Detection & Safety

### 4.1 CuPy Integration

At startup:

- Attempts to import `cupy`
- Performs a **real GPU self‑test** (`cp.arange(2).sum()`)

Three modes are supported automatically:

| Mode | Behavior |
|----|----|
| GPU working | Full GPU acceleration |
| CuPy present but failing | CPU fallback |
| No GPU | CPU only |

This prevents silent failures and ensures reliability.

---

## 5. Fractal Compression Basics (Implemented)

### 5.1 Ranges

- The signal is split into **non‑overlapping ranges**
- Range size is derived dynamically:

```
range_size = max(4, tile_size // 128)
```

### 5.2 Domains

- Domains are extracted from the **original signal**
- Larger `tile_size`, overlapping by `domain_step`
- Each domain is **downsampled** to `range_size`

This preserves shape while reducing storage cost.

---

## 6. Memory‑Mapped Domain Storage (Major Upgrade)

### Problem Solved

Classic fractal compressors load **all domains into RAM**, which breaks for long audio.

### Solution Implemented

- Domains are written into a **temporary memmap file**
- Stored as:

```
(n_domains, range_size) float32
```

### Benefits

- Constant memory usage
- Scales to arbitrarily long audio
- Can be streamed to GPU when needed

This is one of the **most important architectural improvements** so far.

---

## 7. Shape Embeddings (Fast Candidate Search)

### 7.1 Tile Embedding

Each range/domain is embedded using:

- DCT‑II (orthonormal)
- DC coefficient removed
- Weighted high‑frequency emphasis
- L2 normalization

Embedding dimension:

```
EMBED_K = 32
```

### 7.2 Why Embeddings

Instead of brute‑forcing all domains:

- Compute **cosine similarity** in embedding space
- Select top‑K candidates
- Perform **exact affine matching only on those**

This reduces complexity dramatically.

---

## 8. ANN Acceleration (Optional)

If available:

- `hnswlib` is used
- Builds a persistent **HNSW index** over embeddings

Conditions:

- Enabled automatically for large domain counts
- Transparent fallback to linear search

This provides near‑constant‑time candidate selection.

---

## 9. Affine Matching

For each range/domain pair, the compressor solves:

```
R ≈ s * D + o
```

Implemented features:

- Optimal least‑squares scale (`s`) and offset (`o`)
- Mirrored domain check (symmetry)
- Error measured via L2 norm

The best match is stored as:

```
(domain_index, s, o, symmetry_flag, error)
```

---

## 10. CPU–GPU Pipeline Parallelism (Major Upgrade)

### Problem

- CPU good at search
- GPU good at math
- Naive approaches idle one or the other

### Solution

A **producer–consumer pipeline**:

#### CPU Workers

- Compute embeddings
- Perform ANN / linear candidate search
- Push candidate lists to a shared queue

#### GPU Worker

- Consumes candidate batches
- Runs vectorized affine solves
- Returns best match per range

This allows **full hardware utilization**.

---

## 11. Voiced / Silent Detection

Implemented `voiced_detection()`:

- Frame‑based energy analysis
- Hysteresis thresholding
- Smooths detection

Used to:

- Skip silent ranges
- Avoid wasting matches on noise
- Improve compression ratio

Silent ranges are encoded using a **sentinel domain index (-1)**.

---

## 12. Custom FWAV File Format

### 12.1 Header

Includes:

- Version
- Range size
- Sample rate
- Sample width
- Tile size
- Domain step
- Energy threshold
- Number of ranges
- Number of domains
- Original signal length

### 12.2 Payload

1. All domain tiles (float32)
2. All matches

### 12.3 Integrity

- SHA‑256 checksum embedded
- Verified during load

This ensures **self‑contained, robust storage**.

---

## 13. Decompression & Iterative Reconstruction

### 13.1 Initial State

- Reconstruction buffer initialized to zeros

### 13.2 Iterative Refinement

For each iteration:

- Apply stored transforms
- Average overlapping contributions
- Optionally refine scale (`s`)
- Check convergence

Stops when:

- Convergence threshold reached, or
- Max iterations exceeded

### 13.3 Stability Controls

- Scale clipping (`s_clip`)
- Optional scale damping

---

## 14. Metrics & Evaluation

Implemented utilities:

- SNR computation
- Compression ratio logging
- Timing statistics

Batch mode writes JSON metrics automatically.

---

## 15. CLI Interface

### Compress

```
python fractal_wav_compressor_gpu_cli.py compress input.wav output.fwav --gpu
```

### Decompress

```
python fractal_wav_compressor_gpu_cli.py decompress input.fwav --gpu
```

### Batch Processing

- Directory input
- Parallel workers
- Automatic skipping of existing outputs

---

## 16. What Has Been Achieved So Far

✔ Fully working fractal compressor
✔ Handles long audio via memmap
✔ CPU/GPU automatic execution
✔ ANN‑accelerated search
✔ Robust file format
✔ Iterative reconstruction
✔ Batch & metrics support

This is **well beyond a toy implementation** and forms a solid experimental platform.

---

## 17. Known Limitations (Current State)

- Compression ratios are not yet competitive
- Quality tuning is ongoing (range/domain balance)
- ANN index rebuilding cost is non‑trivial
- No psychoacoustic weighting yet

These are expected for a research‑grade system.

---

## 18. Next Logical Directions

Potential next steps:

- Multi‑resolution domains
- Per‑band fractal matching
- Adaptive range sizes
- Psychoacoustic error metrics
- Domain reuse across files

---

**Status:** Active research / experimental implementation

