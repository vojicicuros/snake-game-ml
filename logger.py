
import csv
import os
from datetime import datetime

class CSVLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._init()

    def _init(self):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["step_type","episode","step","return","avg50","epsilon","buffer_size","loss","eval_return","timestamp"])

    def log(self, **kwargs):
        # Accept either 'ret' or 'return' (since 'return' can't be used as kwarg in calls)
        ep_return = kwargs.get("ret", kwargs.get("return", None))
        row = {
            "step_type": kwargs.get("step_type","train"),
            "episode": kwargs.get("episode",0),
            "step": kwargs.get("step",0),
            "return": ep_return,
            "avg50": kwargs.get("avg50", None),
            "epsilon": kwargs.get("epsilon", None),
            "buffer_size": kwargs.get("buffer_size", None),
            "loss": kwargs.get("loss", None),
            "eval_return": kwargs.get("eval_return", None),
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([row[k] for k in ["step_type","episode","step","return","avg50","epsilon","buffer_size","loss","eval_return","timestamp"]])
