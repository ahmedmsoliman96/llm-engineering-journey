import logging
import subprocess

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: str) -> float | None:
    """
    Launches a subprocess using ffprobe to extract the duration of the audio in seconds.
    Args:
        audio_path (str): The audio file path.
    Returns:
        float | None: The duration of the audio in seconds or None if it can't be
        determined or an error occurred.
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
            capture_output=True, text=True, timeout=10, check=True
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError) as e:
        logger.warning(f"could not determine audio duration: {str(e)}")
        return None


def format_duration(seconds: float) -> str:
    """
    Formats audio duration into human-readable string (e.g. '1h 20m', '40m 30s' or '20s').
    Args:
        seconds (float): Seconds to format.
    Returns:
        str: The formatted duration of the audio.
    """
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"