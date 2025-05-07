import os
import queue
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
import tempfile
import contextlib

from utils import resample_pcm
import constants
from typing import List, Dict

import numpy as np
import webrtcvad
from scipy.io.wavfile import write as wav_write
import sounddevice as sd

logger = logging.getLogger(__name__)

class VADTranscriber(threading.Thread):
    def __init__(self, text_queue: queue.Queue, devices: List[int], model,
                 vad_mode: int = constants.VAD_MODE,
                 frame_ms: int = constants.FRAME_MS,
                 max_silence_ms: int = constants.MAX_SILENCE_MS,
                 partial_interval_ms: int = constants.PARTIAL_INTERVAL_MS,
                 min_frames: int = constants.MIN_FRAMES,
                 max_frames: int = constants.MAX_FRAMES) -> None:
        super().__init__(daemon=True)
        self.text_q = text_queue
        self.vad = webrtcvad.Vad(vad_mode)
        self.frame_ms = frame_ms
        self.max_silence = int(max_silence_ms / frame_ms)
        self.partial_frames = int(partial_interval_ms / frame_ms)
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.model = model
        self.audio_q = queue.Queue()
        self.running = False
        self.segment = 0
        self.devices = devices
        # executor for transcribe tasks
        self.executor = ThreadPoolExecutor(max_workers=2)

    def _audio_callback_factory(self, rate: int):
        def callback(indata: np.ndarray, frames: int, time_info: Dict, status) -> None:
            if status:
                logger.warning("Audio stream status: %s", status)
            pcm = indata.copy().flatten()
            self.audio_q.put((pcm, rate))
        return callback

    def _setup_streams(self):
        stack = contextlib.ExitStack()
        for dev in self.devices:
            info = sd.query_devices(dev)
            dev_rate = int(info['default_samplerate'])
            blocksize = int(dev_rate * self.frame_ms / 1000)
            stack.enter_context(sd.InputStream(
                samplerate=dev_rate,
                channels=1,
                dtype="int16",
                blocksize=blocksize,
                device=dev,
                callback=self._audio_callback_factory(dev_rate)
            ))
        return stack

    def _init_loop_state(self):
        self.buffer = []
        self.silence = 0
        self.triggered = False
        self.frames = 0
        self.speech_frames = 0

    def _process_frame(self, pcm_rs, is_speech):
        if not self.triggered and is_speech:
            self.triggered = True
            self.buffer = [pcm_rs]
            self.silence = self.frames = 0
            self.segment += 1
        elif self.triggered:
            self.buffer.append(pcm_rs)
            self.frames += 1
            if not is_speech:
                self.silence += 1
                if self.speech_frames >= self.max_frames or self.silence > self.max_silence:
                    if self.speech_frames >= self.min_frames:
                        self._enqueue_transcription(self.buffer, final=True)
                    self._reset_loop_state()
            else:
                self.silence = 0
                self.speech_frames += 1
            if self.frames >= self.partial_frames and self.speech_frames >= self.min_frames:
                self._enqueue_transcription(self.buffer, final=False)
                self.frames = 0

    def _reset_loop_state(self):
        self.triggered = False
        self.silence = 0
        self.frames = 0
        self.speech_frames = 0
        with self.audio_q.mutex:
            self.audio_q.queue.clear()

    def run(self) -> None:
        self.running = True
        with self._setup_streams() as stack:
            self._init_loop_state()
            while self.running:
                try:
                    pcm, rate = self.audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                pcm_rs = resample_pcm(pcm.astype(np.int16), rate, constants.TARGET_RATE)
                is_speech = self.vad.is_speech(pcm_rs.tobytes(), sample_rate=constants.TARGET_RATE)
                self._process_frame(pcm_rs, is_speech)

    def _enqueue_transcription(self, buffer: List[np.ndarray], final: bool) -> None:
        """Spawn a thread to process and transcribe buffered audio."""
        data = np.concatenate(buffer)
        seg_id = self.segment
        # submit transcription to executor
        self.executor.submit(self._transcribe, data, final, seg_id)

    def _transcribe(self, data: np.ndarray, final: bool, seg_id: int) -> None:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_write(f.name, constants.TARGET_RATE, data)
            path = f.name
        try:
            text = self.model.transcribe([path], verbose=False)[0].text.strip()
            # always enqueue transcription
            self.text_q.put({'text': text, 'final': final, 'id': seg_id})
        except Exception as e:
            logger.error("ASR error: %s", e)
        finally:
            os.remove(path)

    def stop(self) -> None:
        self.running = False
