import json
import os

class WeightsStore:
    def __init__(self, path):
        self.path = path
        self.weights = {}
        self.load_weights()

    def load_weights(self):
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            self.weights = {}
            return
        with open(self.path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                self.weights = {}
                return
        # Normalize loaded structure to: {str(user_id): [w_c, w_cf, w_pop]}
        normalized = {}
        if isinstance(data, dict):
            for uid, val in data.items():
                key = str(uid)
                # handle nested format {uid: {"hybrid": [...]}}
                if isinstance(val, dict) and "hybrid" in val:
                    normalized[key] = val["hybrid"]
                # already flat list of weights
                elif isinstance(val, list):
                    normalized[key] = val
                # unexpected formats: try to find a list inside
                else:
                    try:
                        # if val is something like {"0":..., ...} skip
                        if isinstance(val, (tuple,)):
                            normalized[key] = list(val)
                        else:
                            normalized[key] = val
                    except Exception:
                        normalized[key] = val
        self.weights = normalized

    def get_user_weights(self, user_id):
            """Retourne la liste des poids internes (flat list [w_c,w_cf,w_pop])."""
            return self.weights.get(str(user_id))

    def set_user_weights(self, user_id, new_weights_list):
            """new_weights_list est une liste [w_c,w_cf,w_pop]"""
            self.weights[str(user_id)] = new_weights_list

    def save_weights(self):
            # Ensure saved JSON has structure: {user_id: {"hybrid": [...]}}
            json_dict = {}
            for uid, w in self.weights.items():
                key = str(uid)
                # if someone stored nested dict accidentally, try to unwrap
                if isinstance(w, dict) and "hybrid" in w and isinstance(w["hybrid"], list):
                    json_dict[key] = {"hybrid": w["hybrid"]}
                else:
                    json_dict[key] = {"hybrid": w}
            with open(self.path, "w") as f:
                json.dump(json_dict, f, indent=4)
        
