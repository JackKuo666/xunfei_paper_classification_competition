{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": "/home/featurize/data/robert_pretrain/"
        },
        "token_indexers": {
            "robert": {
                "type": "pretrained_transformer",
                "model_name": "/home/featurize/data/robert_pretrain/"
            }
        },
        "max_tokens": 512
    },
    "train_data_path": "data/paper_classification/train_small.csv",
    "vocabulary":{
        "type": "from_files",
        "directory": "data/vocab.tar.gz"
    },
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "robert": {
                    "type": "pretrained_transformer",
                    "model_name": "/home/featurize/data/robert_pretrain/"
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": "/home/featurize/data/robert_pretrain/",
            "requires_grad": true,
            "dropout": 0.1
        }
    },
    "data_loader": {
        "batch_size": 16,
        "cuda_device": 0,
        "max_instances_in_memory": 1600,
        "shuffle": true
    },
    "trainer": {
        "checkpointer":{
            "type": "simple_checkpointer",
            "serialization_dir":"checkpoint",
            "save_every_num_seconds": 120
        },
        "num_epochs": 1,
        "validation_metric": "+accuracy",
        "learning_rate_scheduler": {
          "type": "slanted_triangular",
          "num_epochs": 1,
          "num_steps_per_epoch": 3088,
          "cut_frac": 0.06
        },
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5,
            "weight_decay": 0.1
        }
    }
}
