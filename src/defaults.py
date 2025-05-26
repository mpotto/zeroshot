from src.constants import *

defaults = {
    AMAZON: {
        "data": {
            "data_dir": "/mnt/ssd/ronak/datasets/wilds",
            "num_labels": 5,
        },
        "model": {
            "architecture": "bert",
            "task": SEQUENCE_CLASSIFICATION,
            "num_labels": 5,
        },
        "optim": {
            "algo": "adam",
            "lr": 2e-6,
            "cosine_decay": True,
        },
        "training": {
            "log_dir": "/mnt/ssd/ronak/logs/",
            "output_dir": "/mnt/ssd/ronak/output",
            "init_from": "scratch",
            "max_iters": 2000,
            "eval_interval": 200,
            "eval_iters": 100,
            "batch_size": 48,
            "grad_accumulation_steps": 1,
            "save_checkpoints": False,
        },
    },
}