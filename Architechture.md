┌─────────────────────────────┐
│       Input WAV File         │
│  (Mono, 8/16/24/32-bit)     │
└─────────────┬───────────────┘
              │ read_wav_mono
              ▼
┌─────────────────────────────┐
│   Signal Preprocessing       │
│ - Convert to float32         │
│ - Optional clipping [-1,1]   │
│ - Voiced detection mask      │
│ - Weighted signal            │
└─────────────┬───────────────┘
              │ frame_ranges
              ▼
┌─────────────────────────────┐
│    Build Domains Memmap      │
│ - Tile size → range_size     │
│ - Downsample each tile       │
│ - Store as memmap            │
│ - Block-wise, GPU optional   │
└─────────────┬───────────────┘
              │ compute_stats_memmap
              ▼
┌─────────────────────────────┐
│   Domain Statistics          │
│ - Means & variances          │
│ - Used for scaling (s)       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Block-wise Domain Matching  │
│ - Split ranges into blocks   │
│ - For each range:            │
│   • Compute best domain idx  │
│   • Compute scale (s)        │
│   • Compute error (chi²)     │
│   • Check symmetry            │
│ - GPU/CPU optional           │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Assemble Matches List       │
│ - Each match:                │
│   (domain_idx, s, mean_r,    │
│    sym_flag, error)          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Save .FWAV File             │
│ - Header + metadata          │
│ - Domains (memmap → ndarray) │
│ - Matches (binary packed)    │
└─────────────────────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Load .FWAV  │
                   └─────────────┘
                          │
                          ▼
┌─────────────────────────────┐
│   Decompression             │
│ - Load domains & matches    │
│ - Convert to GPU if needed  │
│ - Precompute scatter indices │
│ - Iterate (n iterations):   │
│   • Extract tiles           │
│   • Apply symmetry          │
│   • Apply scale & mean      │
│   • Flatten for scatter     │
│   • Accumulate & normalize  │
│   • Convergence check       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Reconstructed WAV          │
│ - Optionally clip [-1,1]    │
│ - write_wav                  │
└─────────────────────────────┘
