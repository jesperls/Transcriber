import numpy as np

def resample_pcm(pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return pcm
    duration = pcm.shape[0] / orig_sr
    new_len = int(duration * target_sr)
    src_indices = np.arange(pcm.shape[0])
    tgt_indices = np.linspace(0, pcm.shape[0] - 1, new_len)
    return np.interp(tgt_indices, src_indices, pcm).astype(np.int16)