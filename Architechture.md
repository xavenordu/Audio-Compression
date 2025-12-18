┌──────────────────────────────────┐
│          Input WAV File           │
│   (Mono / Multi → Mono)           │
│   8 / 16 / 24 / 32-bit PCM/float  │
└───────────────┬──────────────────┘
                │ read_wav_mono
                ▼
┌──────────────────────────────────┐
│        Signal Preprocessing        │
│ - Convert to float32              │
│ - Optional clipping (float WAV)   │
│ - Voiced detection (energy-based) │
│ - Apply voiced mask               │
│ - Preserve original_len           │
└───────────────┬──────────────────┘
                │ reshape → ranges
                ▼
┌──────────────────────────────────┐
│          Range Formation           │
│ - range_size = tile_size // K     │
│ - Pad signal (reflect) if needed  │
│ - Non-overlapping ranges          │
│ - Silent ranges → sentinel (-1)   │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│      Build Domains (MEMMAP)        │
│ - Slide tile_size over signal     │
│ - domain_step overlap             │
│ - Downsample tiles → range_size   │
│ - Block-wise processing           │
│ - Written to temp memmap file     │
│ - GPU optional (CuPy)             │
└───────────────┬──────────────────┘
                │ domains.memmap
                ▼
┌──────────────────────────────────┐
│   Build Domain Embeddings (MEMMAP) │
│ - DCT-II per domain tile          │
│ - Remove DC component             │
│ - Frequency weighting             │
│ - L2 normalization                │
│ - Stored as (n_domains, emb_dim)  │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│ Optional ANN Index Construction   │
│ - HNSW (hnswlib)                  │
│ - Built only if domain count large│
│ - Persistent on disk              │
│ - Automatic fallback              │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│  CPU–GPU Pipeline Parallelism     │
│                                  │
│  ┌──────────────┐   ┌──────────┐ │
│  │ CPU Workers  │   │ GPU Worker│ │
│  │──────────────│   │──────────│ │
│  │ - Load emb   │   │ - Load    │ │
│  │ - ANN / dot  │   │   domains │ │
│  │ - Top-K cand │──▶│ - Batched │ │
│  │ - Skip silent│   │   affine  │ │
│  │ - Queue push │   │   solves  │ │
│  └──────────────┘   └──────────┘ │
│                                  │
│ - Producer/consumer queues       │
│ - Full CPU + GPU utilization     │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│        Affine Matching            │
│ - Exact least-squares solve       │
│ - Compute scale (s) & offset (o)  │
│ - Evaluate mirrored domains       │
│ - Choose lowest L2 error          │
│ - Sentinel for skipped ranges     │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│        Assemble Matches            │
│ Each range → tuple:               │
│ (domain_idx, s, o, sym, error)    │
│ domain_idx = -1 for silence       │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│          Save .FWAV File           │
│ - Magic + version                 │
│ - range_size, tile_size           │
│ - domain_step, energy_threshold   │
│ - framerate, sampwidth            │
│ - n_ranges, n_domains             │
│ - original_len                    │
│ - SHA-256 checksum placeholder    │
│ - Domains (float32)               │
│ - Matches (binary packed)         │
│ - Final checksum written          │
└──────────────────────────────────┘

Decompression Path

┌──────────────────────────────────┐
│             Load .FWAV             │
│ - Verify magic & version          │
│ - Read metadata                   │
│ - Verify SHA-256 checksum         │
│ - Load domains & matches          │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│      Decompression Setup           │
│ - Allocate recon buffer           │
│ - Move to GPU if enabled          │
│ - Build scatter indices           │
│ - Handle sentinel (-1) domains    │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│ Iterative Fractal Reconstruction  │
│ For each iteration:               │
│ - Gather domain tiles             │
│ - Apply symmetry                  │
│ - Center tiles & ranges           │
│ - Re-estimate scale (optional)    │
│ - Clip / damp scale               │
│ - Apply affine transform          │
│ - Scatter-add contributions       │
│ - Normalize overlaps              │
│ - Convergence check (Δ norm)      │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│     Final Reconstructed Signal     │
│ - Trim to original_len            │
│ - Optional clipping [-1, 1]       │
│ - write_wav                       │
└──────────────────────────────────┘
