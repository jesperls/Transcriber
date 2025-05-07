import numpy as np
import scipy.signal

def resample_pcm(pcm: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return pcm
    new_len = int(len(pcm) * (target_sr / orig_sr))
    # resample to new length, output float64
    resampled = scipy.signal.resample(pcm, new_len)
    # clip and convert to int16
    resampled = np.clip(resampled, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return resampled.astype(np.int16)