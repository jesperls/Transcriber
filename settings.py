import json
import os
import constants

_SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')

DEFAULT_SETTINGS = {
    'max_lines': constants.MAX_LINES,
    'history_lines': constants.HISTORY_LINES,
    'clear_timeout': constants.CLEAR_TIMEOUT_MS,
    'vad_mode': constants.VAD_MODE,
    'frame_ms': constants.FRAME_MS,
    'max_silence_ms': constants.MAX_SILENCE_MS,
    'partial_interval_ms': constants.PARTIAL_INTERVAL_MS,
    'min_frames': constants.MIN_FRAMES,
    'max_frames': constants.MAX_FRAMES,
    'appearance': {
        'font_size': 16,
        'opacity': 0.8,
        'font_family': 'Courier',
        'background_color': 'rgba(0,0,0,0.7)',
        'text_color': '#FFFFFF'
    },
    'input_device': None
}

def load_settings():
    if os.path.exists(_SETTINGS_FILE):
        try:
            with open(_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    settings = DEFAULT_SETTINGS.copy()
    save_settings(settings)
    return settings


def save_settings(settings: dict):
    try:
        with open(_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Could not save settings: {e}")
