import os
import queue
import threading
import tempfile
import contextlib
import re
import sounddevice as sd

from utils import resample_pcm
from constants import TARGET_RATE, FRAME_MS, VAD_MODE, MAX_SILENCE_MS, PARTIAL_INTERVAL_MS
from typing import List, Tuple, Dict

import numpy as np
import webrtcvad
import nemo.collections.asr as nemo_asr
from scipy.io.wavfile import write as wav_write

class VADTranscriber(threading.Thread):
    def __init__(self, text_queue: queue.Queue, devices: List[int], model) -> None:
        super().__init__(daemon=True)
        self.text_q = text_queue
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.max_silence = int(MAX_SILENCE_MS / FRAME_MS)
        self.partial_frames = int(PARTIAL_INTERVAL_MS / FRAME_MS)
        self.model = model
        self.audio_q = queue.Queue()
        self.running = False
        self.segment = 0
        self.devices = devices

    def _audio_callback_factory(self, rate: int):
        def callback(indata: np.ndarray, frames: int, time_info: Dict, status) -> None:
            if status:
                print(f"Audio stream status: {status}")
            pcm = indata.copy().flatten()
            self.audio_q.put((pcm, rate))
        return callback

    def run(self) -> None:
        self.running = True
        with contextlib.ExitStack() as stack:
            for dev in self.devices:
                info = sd.query_devices(dev)
                dev_rate = int(info['default_samplerate'])
                blocksize = int(dev_rate * FRAME_MS / 1000)
                stack.enter_context(sd.InputStream(
                    samplerate=dev_rate,
                    channels=1,
                    dtype="int16",
                    blocksize=blocksize,
                    device=dev,
                    callback=self._audio_callback_factory(dev_rate)
                ))
            buffer, silence, triggered, frames = [], 0, False, 0
            while self.running:
                try:
                    pcm, rate = self.audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                pcm_rs = resample_pcm(pcm.astype(np.int16), rate, TARGET_RATE)
                is_speech = self.vad.is_speech(pcm_rs.tobytes(), sample_rate=TARGET_RATE)

                if not triggered and is_speech:
                    triggered = True
                    buffer = [pcm_rs]
                    silence = frames = 0
                    self.segment += 1
                elif triggered:
                    buffer.append(pcm_rs)
                    frames += 1
                    if frames >= self.partial_frames and silence == 0:
                        self._enqueue_transcription(buffer, final=False)
                        frames = 0
                    if not is_speech:
                        silence += 1
                        if silence > self.max_silence:
                            self._enqueue_transcription(buffer, final=True)
                            triggered = False
                            silence = 0
                            with self.audio_q.mutex:
                                self.audio_q.queue.clear()
                    else:
                        silence = 0

    def _enqueue_transcription(self, buffer: List[np.ndarray], final: bool) -> None:
        """Spawn a thread to process and transcribe buffered audio."""
        data = np.concatenate(buffer)
        seg_id = self.segment
        threading.Thread(target=self._transcribe, args=(data, final, seg_id), daemon=True).start()

    _spawn_transcribe = _enqueue_transcription

    def _transcribe(self, data: np.ndarray, final: bool, seg_id: int) -> None:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_write(f.name, TARGET_RATE, data)
            path = f.name
        try:
            text = self.model.transcribe([path], verbose=False)[0].text.strip()
            if self._verify_heuristics(text):
                self.text_q.put({'text': text, 'final': final, 'id': seg_id})
        except Exception as e:
            print(f"ASR error: {e}")
        finally:
            os.remove(path)

    def _verify_heuristics(self, text: str) -> bool:
        words = re.findall(r'\w+(?:-\w+)+|\w+', text)
        return len(words) >= 2

    def stop(self) -> None:
        self.running = False
