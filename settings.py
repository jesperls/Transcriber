import json
import os
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional
import constants

_SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'settings.json')

logger = logging.getLogger(__name__)

@dataclass
class Appearance:
    font_size: int = 16
    opacity: float = 0.8
    font_family: str = 'Courier'
    background_color: str = 'rgba(0,0,0,0.7)'
    text_color: str = '#FFFFFF'

@dataclass
class Settings:
    max_lines: int = constants.MAX_LINES
    history_lines: int = constants.HISTORY_LINES
    clear_timeout: int = constants.CLEAR_TIMEOUT_MS
    vad_mode: int = constants.VAD_MODE
    frame_ms: int = constants.FRAME_MS
    max_silence_ms: int = constants.MAX_SILENCE_MS
    partial_interval_ms: int = constants.PARTIAL_INTERVAL_MS
    min_frames: int = constants.MIN_FRAMES
    max_frames: int = constants.MAX_FRAMES
    appearance: Appearance = field(default_factory=Appearance)
    input_device: Optional[int] = None

def load_settings() -> Settings:
    """Load settings from file or return defaults."""
    if os.path.exists(_SETTINGS_FILE):
        try:
            with open(_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # build Appearance and Settings
            app_data = data.get('appearance', {})
            appearance = Appearance(**app_data)
            # filter settings for Settings
            init_data = {k: v for k, v in data.items() if k != 'appearance'}
            settings = Settings(appearance=appearance, **init_data)
            return settings
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to load settings, using defaults: {e}")
    # write defaults
    settings = Settings()
    save_settings(settings)
    return settings

def save_settings(settings: Settings) -> None:
    """Save settings to file."""
    try:
        with open(_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(asdict(settings), f, indent=2)
    except OSError as e:
        logger.error(f"Could not save settings: {e}")
