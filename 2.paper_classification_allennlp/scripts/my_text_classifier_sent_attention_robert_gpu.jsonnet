{
    "dataset_reader" : {
        "type": "sent_attention_reader",
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
    "train_data_path": "/home/featurize/data/train_split_data/train_data.csv",
    "validation_data_path": "/home/featurize/data/train_split_data/dev_data.csv",
    "vocabulary":{
        "type": "from_files",
        "directory": "/home/featurize/data/vocab.tar.gz"
    },
    "model": {
        "type": "sent_attention",
        "embedder_title": {
            "token_embedders": {
                "robert": {
                    "type": "pretrained_transformer",
                    "model_name": "/home/featurize/data/robert_pretrain/"
                }
            }
        },
        "encoder_title": {
            "type": "bert_pooler",
            "pretrained_model": "/home/featurize/data/robert_pretrain/",
            "requires_grad": true,
            "dropout": 0.1
        },
        "embedder_abstract": {
            "token_embedders": {
                "robert": {
                    "type": "pretrained_transformer",
                    "model_name": "/home/featurize/data/robert_pretrain/"
                }
            }
        },
        "encoder_abstract": {
            "type": "bert_pooler",
            "pretrained_model": "/home/featurize/data/robert_pretrain/",
            "requires_grad": true,
            "dropout": 0.1
        }
    },
    "data_loader": {
        "batch_size": 2,
        "cuda_device": 0,
        "max_instances_in_memory": 40,
        "shuffle": true
    },
    "trainer": {
        "checkpointer":{
            "type": "simple_checkpointer",
            "serialization_dir":"/home/featurize/data/checkpoint/",
            "save_every_num_seconds": 1200
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
