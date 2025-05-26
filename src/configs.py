configs = {
    "imagenet_captions_250k": {
        "zsp_clip_hq": {
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
            "data": {
                "img_embed": "vit_b32_laion2b",
                "txt_embed": "vit_b32_datacompxl",
            },
            "optim": {
                "lr": 1e-2,
                "momentum": 0.0,
                "weight_decay": 0.01,
            },
        },
        "zsp_clip_mq": {
            "model": {
                "architecture": "clip",
                "in_features_img": 512,
                "hidden_size_img": 256,
                "n_layers_img": 2,
                "in_features_txt": 768,
                "hidden_size_txt": 256,
                "n_layers_txt": 2,
                "out_features": 128,
            },
            "data": {
                "img_embed": "vit_b32_laion2b",
                "txt_embed": "bert-base-uncased",
            },
            "optim": {
                "lr": 1e-2,
                "momentum": 0.0,
                "weight_decay": 0.01,
            },
        },
        "zsp_clip_lq": {
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
            "data": {
                "img_embed": "vit_b32_laion2b",
                "txt_embed": "gpt2-small",
            },
            "optim": {
                "lr": 1e-2,
                "momentum": 0.0,
                "weight_decay": 0.01,
            },
        },
        "zsp_vicreg_hq": {
            "model": {
                "architecture": "vicreg",
                "in_features_img": 512,
                "hidden_size_img": 256,
                "n_layers_img": 2,
                "in_features_txt": 512,
                "hidden_size_txt": 256,
                "n_layers_txt": 2,
                "out_features": 128,
                "lam": 1,
            },
            "data": {
                "img_embed": "vit_b32_laion2b",
                "txt_embed": "vit_b32_datacompxl",
            },
            "optim": {
                "lr": 3e-3,
                "momentum": 0.0,
                "weight_decay": 0.01,
            },
        },
        "zsp_vicreg_debug": {
            "model": {
                "architecture": "vicreg",
                "in_features_img": 512,
                "hidden_size_img": 256,
                "n_layers_img": 2,
                "in_features_txt": 512,
                "hidden_size_txt": 256,
                "n_layers_txt": 2,
                "out_features": 128,
                "lam": 25,
            },
            "data": {
                "img_embed": "vit_b32_laion2b",
                "txt_embed": "vit_b32_datacompxl",
            },
            "optim": {
                "lr": 3e-3,
                "momentum": 0.0,
                "weight_decay": 0.01,
            },
            "training": {
                "max_iters": 500,
                "eval_interval": 100,
                "eval_iters": 50,
            }
        },
    }
}
