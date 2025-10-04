import os
import json
from typing import Dict, Any

DEFAULT_AI_CONFIG = 'config/ai_config.json'
DEFAULT_AI_STATE = 'config/ai_state.json'

class ModelStore:
    """
    JSON-backed parameter and state store for AI learners (online).
    """
    def __init__(self, config_path: str = DEFAULT_AI_CONFIG, state_path: str = DEFAULT_AI_STATE):
        self.config_path = config_path
        self.state_path = state_path
        self.data = {"params": {}, "stats": {}}
        self.state = {"arms": {}, "meta": {}}
        self._load()

    def _load(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.data = json.load(f)
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r') as f:
                self.state = json.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_params(self) -> Dict[str, Any]:
        return self.data.get("params", {})

    def set_params(self, p: Dict[str, Any]):
        self.data["params"] = p

    def get_state(self) -> Dict[str, Any]:
        return self.state

    def set_state(self, s: Dict[str, Any]):
        self.state = s
