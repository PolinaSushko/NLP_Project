import json

def load_config(config_path = 'settings.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Try looking in the app root directory
        try:
            with open('/app/settings.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_path} not found")