import json

def load_config(config_path = 'settings.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_path} not found")