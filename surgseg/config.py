import numpy as np
import torch

# ----- Classes / constants -----
CLASS_NAMES = [
    "tool_shaft", "tool_clasper", "tool_wrist",
    "thread", "clamps", "suturing_needle",
    "suction_tool", "catheter", "needle_holder"
]
NUM_CLASSES = len(CLASS_NAMES)

# Background
IGNORE_LABEL = 255

# TODO: Analyze & adjust this
COVERAGE_TRAIN_PCT = {
    "tool_shaft": 12.04, "tool_clasper": 2.24, "tool_wrist": 3.59,
    "thread": 1.00, "clamps": 0.15, "suturing_needle": 0.44,
    "suction_tool": 0.51, "catheter": 0.19, "needle_holder": 0.20
}

IMG_EXTS = {".png"}
MASK_EXTS = {".png"}

def make_class_weights():
    arr = np.array([COVERAGE_TRAIN_PCT[k] for k in CLASS_NAMES], dtype=np.float32)
    inv = 1.0 / (arr + 1e-6)
    inv = inv / inv.mean()
    inv = np.clip(inv, 0.5, 4.0)
    return torch.tensor(inv, dtype=torch.float32)
