defaults = {
    "imagenet_captions_250k": {
        "data": {
            "data_dir": "/mnt/ssd/ronak/datasets/",
            "img_embed": "vit_b32_laion2b",
            "txt_embed": "vit_b32_laion2b",
        },
        "model": {
            "architecture": "clip",
            "in_features_img": 512,
            "hidden_size_img": 256,
            "n_layers_img": 2,
            "in_features_txt": 512,
            "hidden_size_txt": 256,
            "n_layers_txt": 2,
            "out_features": 128,
        },
        "optim": {
            "algo": "adam",
            "lr": 1e-2,
            "momentum": 0.0,
            "weight_decay": 0.01,
            "cosine_decay": False,
        },
        "training": {
            "log_dir": "logs/",
            "output_dir": "/mnt/ssd/ronak/output",
            "init_from": "scratch",
            "max_iters": 5000,
            "eval_interval": 500,
            "eval_iters": 200,
            "batch_size": 512,
            "grad_accumulation_steps": 1,
        }
    },
}