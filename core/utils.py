import json
import os

CONFIG_PATH = "config.json"

def load_config(path=CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_cha_config(path=CONFIG_PATH) -> dict:
    config = load_config(path)
    return config.get("cha", {})

def load_llm_config(path=CONFIG_PATH) -> dict:
    config = load_config(path)
    return config.get("llm", {})

def save_config(config: dict, path=CONFIG_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)