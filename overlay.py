import os
import queue
import sounddevice as sd
import nemo.collections.asr as nemo_asr
from PySide6.QtWidgets import QWidget, QVBoxLayout, QPlainTextEdit, QDialog, QListWidget, QAbstractItemView, QDialogButtonBox, QMenu
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction

from constants import HISTORY_LINES, HISTORY_FILE, MAX_LINES, CLEAR_TIMEOUT_MS, MODEL_NAME
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

class Overlay(QWidget):
    def __init__(self):
        super().__init__()
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
        self.text.document().setMaximumBlockCount(MAX_LINES)
        layout.addWidget(self.text)

        h = self.text.fontMetrics().lineSpacing() * MAX_LINES + 24
        self.resize(480, h)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        select_act = menu.addAction("Select Audio Input...")
        menu.addSeparator()
        history_act = QAction("Show History", menu)
        history_act.setCheckable(True)
        history_act.setChecked(self.show_history)
        menu.addAction(history_act)

        action = menu.exec(event.globalPos())
        if action == select_act:
            self._choose_inputs()
        elif action == history_act:
            self.show_history = history_act.isChecked()
            if self.show_history:
                self.clear_timer.stop()
                self.text.document().setMaximumBlockCount(HISTORY_LINES + 1)
                self._load_history_lines()
                self._render_history()
            else:
                self.text.document().setMaximumBlockCount(MAX_LINES)
                self.clear_timer.start(CLEAR_TIMEOUT_MS)
                self._render_live()
        return super().contextMenuEvent(event)

    def _load_history_lines(self):
        self.history_lines = []
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                self.history_lines = lines[-HISTORY_LINES:]
            except Exception as e:
                print("Could not load history:", e)

    def _render_history(self):
        self.text.clear()
        for line in self.history_lines:
            self.text.appendPlainText(line)
        self.text.appendPlainText(self.partial_text)

    def _render_live(self):
        self.text.clear()
        num_hist = MAX_LINES - 1 if self.partial_text else MAX_LINES
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
        self.transcriber = VADTranscriber(self.text_q, devices, self.asr_model)
        self.transcriber.start()

    def _start_poll(self):
        timer = QTimer(self)
        timer.timeout.connect(self._poll)
        timer.start(100)

    def _poll(self):
        updated = False
        while not self.text_q.empty():
            item = self.text_q.get()
            txt, final, seg = item['text'], item['final'], item['id']
            txt = ' '.join(txt.split())
            if final:
                self.history_lines.append(txt)
                if len(self.history_lines) > HISTORY_LINES:
                    self.history_lines.pop(0)
                try:
                    with open(HISTORY_FILE, 'a', encoding='utf-8') as f:
                        f.write(txt + '\n')
                except Exception as e:
                    print("Could not write history:", e)
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
            self.clear_timer.start(CLEAR_TIMEOUT_MS)

    def _clear(self):
        if not self.show_history:
            self.text.clear()
