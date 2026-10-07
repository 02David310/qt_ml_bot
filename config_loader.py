import os
import sys

def load_config():
    """Load config.txt from same folder as the exe or script."""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller exe
        base = os.path.dirname(sys.executable)
    else:
        # Running as normal script
        base = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(base, "config.txt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.txt not found at {config_path}\n"
            "Please create it with:\n"
            "TELEGRAM_TOKEN=your_token_here\n"
            "GROQ_API_KEY=your_gemini_key_here"
        )

    config = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                config[key.strip()] = val.strip()

    return config