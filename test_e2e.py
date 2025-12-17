import numpy as np
import time
from fractal import compress_audio, save_compressed, load_compressed, decompress_audio, compute_snr


def make_tone(sr=8000, dur=0.12, freq=440.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    amp = 0.5 * (2**15 - 1)
    sig = (amp * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return sig.astype(np.float32), sr, 2


def test_compress_decompress_e2e(tmp_path):
    sig, sr, sampwidth = make_tone()

    # Call multiprocessing pipeline (uses temp dir for memmap files)
    matches, domains, n_ranges, range_size, tile_size, domain_step, energy_thresh, orig_len = compress_audio(
        sig, sr, sampwidth, tile_size=128, energy_thresh=1e-4,
        use_gpu=False, domains_tmpdir=str(tmp_path), batch_size=32, fast_mode=True
    )

    # Basic sanity checks
    assert len(matches) == n_ranges
    assert domains.shape[1] == range_size

    # Save / load
    fwav = tmp_path / "test_e2e.fwav"
    save_compressed(str(fwav), matches, domains, range_size, sr, sampwidth, tile_size, domain_step, energy_thresh, len(sig))

    matches2, domains2, n_ranges2, range_size2, fr2, sw2, tile2, domain_step2, energy_threshold2, orig_len2 = load_compressed(str(fwav))

    # Decompress
    recon = decompress_audio(matches2, domains2, n_ranges2, range_size2, iterations=8, convergence_eps=1e-3, use_gpu=False, original_len=orig_len2)
    recon_np = np.asarray(recon)

    snr = compute_snr(sig, recon_np)
    # Expect at least modest quality for the tiny sine tone (tunable)
    assert snr > 4.0
