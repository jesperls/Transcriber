import os
import queue
import sounddevice as sd
import nemo.collections.asr as nemo_asr
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QDialog, QListWidget, QAbstractItemView, QDialogButtonBox, QMenu, QSpinBox, QFormLayout
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction

from constants import *
from transcriber import VADTranscriber

class InputDeviceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Audio Input")
        self.resize(400, 300)
        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        for idx, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                self.list_widget.addItem(f"{dev['name']} (#{idx})")
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_device(self) -> int | None:
        item = self.list_widget.currentItem()
        return int(item.text().split('#')[-1].rstrip(')')) if item else None

class ConfigDialog(QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QFormLayout(self)
        self.maxSpin = QSpinBox()
        self.maxSpin.setRange(1, 100)
        self.maxSpin.setValue(settings['max_lines'])
        layout.addRow("Max Lines:", self.maxSpin)
        self.histSpin = QSpinBox()
        self.histSpin.setRange(1, 500)
        self.histSpin.setValue(settings['history_lines'])
        layout.addRow("History Lines:", self.histSpin)
        self.clearSpin = QSpinBox()
        self.clearSpin.setRange(100, 10000)
        self.clearSpin.setValue(settings['clear_timeout'])
        layout.addRow("Clear Timeout (ms):", self.clearSpin)
        self.vadSpin = QSpinBox()
        self.vadSpin.setRange(0, 3)
        self.vadSpin.setValue(settings['vad_mode'])
        layout.addRow("VAD Mode:", self.vadSpin)
        self.frameSpin = QSpinBox()
        self.frameSpin.setRange(1, 1000)
        self.frameSpin.setValue(settings['frame_ms'])
        layout.addRow("Frame Ms:", self.frameSpin)
        self.maxSilSpin = QSpinBox()
        self.maxSilSpin.setRange(1, 5000)
        self.maxSilSpin.setValue(settings['max_silence_ms'])
        layout.addRow("Max Silence Ms:", self.maxSilSpin)
        self.partSpin = QSpinBox()
        self.partSpin.setRange(1, 10000)
        self.partSpin.setValue(settings['partial_interval_ms'])
        layout.addRow("Partial Interval Ms:", self.partSpin)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def getValues(self):
        return {
            'max_lines': self.maxSpin.value(),
            'history_lines': self.histSpin.value(),
            'clear_timeout': self.clearSpin.value(),
            'vad_mode': self.vadSpin.value(),
            'frame_ms': self.frameSpin.value(),
            'max_silence_ms': self.maxSilSpin.value(),
            'partial_interval_ms': self.partSpin.value()
        }

class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = {
            'max_lines': MAX_LINES,
            'history_lines': HISTORY_LINES,
            'clear_timeout': CLEAR_TIMEOUT_MS,
            'vad_mode': VAD_MODE,
            'frame_ms': FRAME_MS,
            'max_silence_ms': MAX_SILENCE_MS,
            'partial_interval_ms': PARTIAL_INTERVAL_MS
        }
        self.text_q = queue.Queue()
        self.transcriber = None
        self.current = None
        self.show_history = False
        self.devices = []
        self.history_lines = []
        self.partial_text = ""
        self._setup_ui()

        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
        default = sd.default.device
        input_dev = default[0] if isinstance(default, (list, tuple)) else default
        self._restart_transcriber([input_dev])

        self.clear_timer = QTimer(self)
        self.clear_timer.setSingleShot(True)
        self.clear_timer.timeout.connect(self._clear)
        self._start_poll()

    def _setup_ui(self):
        self.setWindowFlags(
            Qt.X11BypassWindowManagerHint | Qt.BypassWindowManagerHint |
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.8)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6,6,6,6)

        font = self.font()
        font.setPointSize(16)
        self.text = QPlainTextEdit(readOnly=True)
        self.text.setFont(font)
        self.text.setStyleSheet("background:rgba(0,0,0,0.7); color:white; border:none;")
        self.text.document().setMaximumBlockCount(self.settings['max_lines'])
        self.text.setContextMenuPolicy(Qt.CustomContextMenu)
        self.text.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.text)

        h = self.text.fontMetrics().lineSpacing() * self.settings['max_lines'] + 24
        self.resize(480, h)

    def _show_context_menu(self, pos):
        globalPos = self.text.mapToGlobal(pos)
        self._open_context_menu(globalPos)

    def _open_context_menu(self, globalPos):
        menu = QMenu(self)
        select_act = menu.addAction("Select Audio Input...")
        menu.addSeparator()
        history_act = QAction("Show History", menu)
        history_act.setCheckable(True)
        history_act.setChecked(self.show_history)
        menu.addAction(history_act)
        clear_hist_act = menu.addAction("Clear History")
        config_act = menu.addAction("Configure...")
        action = menu.exec(globalPos)
        if action == select_act:
            self._choose_inputs()
        elif action == history_act:
            self.show_history = history_act.isChecked()
            if self.show_history:
                self.clear_timer.stop()
                self.text.document().setMaximumBlockCount(self.settings['history_lines'] + 1)
                self._load_history_lines()
                self._render_history()
            else:
                self.text.document().setMaximumBlockCount(self.settings['max_lines'])
                self.clear_timer.start(self.settings['clear_timeout'])
                self._render_live()
        elif action == clear_hist_act:
            try:
                os.remove(HISTORY_FILE)
            except:
                pass
            self.history_lines = []
            self.text.clear()
        elif action == config_act:
            dlg = ConfigDialog(self, self.settings)
            if dlg.exec() == QDialog.Accepted:
                self.settings.update(dlg.getValues())
                self._apply_settings()

    def contextMenuEvent(self, event):
        self._open_context_menu(event.globalPos())
        return super().contextMenuEvent(event)

    def _apply_settings(self):
        self.text.document().setMaximumBlockCount(self.settings['max_lines'])
        h = self.text.fontMetrics().lineSpacing() * self.settings['max_lines'] + 24
        self.resize(480, h)
        self.clear_timer.setInterval(self.settings['clear_timeout'])
        self._restart_transcriber(self.devices)
        # reload history view if enabled after config changes
        if self.show_history:
            self.clear_timer.stop()
            self.text.document().setMaximumBlockCount(self.settings['history_lines'] + 1)
            self._load_history_lines()
            self._render_history()
        else:
            self.text.document().setMaximumBlockCount(self.settings['max_lines'])
            self.clear_timer.start(self.settings['clear_timeout'])
            self._render_live()

    def _load_history_lines(self):
        self.history_lines = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                self.history_lines = lines[-self.settings['history_lines']:]
            except Exception as e:
                print("Could not load history:", e)

    def _render_history(self):
        self.text.clear()
        for line in self.history_lines:
            self.text.appendPlainText(line)
        self.text.appendPlainText(self.partial_text)

    def _render_live(self):
        self.text.clear()
        num_hist = self.settings['max_lines'] - 1 if self.partial_text else self.settings['max_lines']
        hist_to_show = self.history_lines[-num_hist:] if num_hist > 0 else []
        for line in hist_to_show:
            self.text.appendPlainText(line)
        if self.partial_text:
            self.text.appendPlainText(self.partial_text)

    def _choose_inputs(self):
        dlg = InputDeviceDialog(self)
        if dlg.exec() == QDialog.Accepted:
            new_dev = dlg.selected_device()
            if new_dev is not None:
                self._restart_transcriber([new_dev])

    def _restart_transcriber(self, devices):
        if self.transcriber:
            self.transcriber.stop()
        with self.text_q.mutex:
            self.text_q.queue.clear()
        self.text.clear()
        self.current = None
        self.devices = devices
        self.history_lines = []
        self.partial_text = ""
        self.transcriber = VADTranscriber(
            self.text_q,
            devices,
            self.asr_model,
            self.settings['vad_mode'],
            self.settings['frame_ms'],
            self.settings['max_silence_ms'],
            self.settings['partial_interval_ms']
        )
        self.transcriber.start()

    def _start_poll(self):
        timer = QTimer(self)
        timer.timeout.connect(self._poll)
        timer.start(100)

    def _append_history(self, text: str) -> None:
        self.history_lines.append(text)
        if len(self.history_lines) > self.settings['history_lines']:
            self.history_lines.pop(0)
        try:
            with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
                f.write(text + '\n')
        except Exception as e:
            print(f"Could not write history: {e}")

    def _poll(self):
        updated = False
        while not self.text_q.empty():
            item = self.text_q.get()
            txt, final, seg = item['text'], item['final'], item['id']
            txt = ' '.join(txt.split())
            if final:
                self._append_history(txt)
                self.partial_text = ''
            else:
                self.partial_text = txt
                self.current = seg
            updated = True
        if updated:
            if self.show_history:
                self._render_history()
            else:
                self._render_live()
            self.clear_timer.stop()
            self.clear_timer.start(self.settings['clear_timeout'])

    def _clear(self):
        if not self.show_history:
            self.text.clear()
