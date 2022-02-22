{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": "/home/featurize/data/bert_pretrain/"
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": "/home/featurize/data/bert_pretrain/"
            }
        },
        "max_tokens": 512
    },
    "train_data_path": "/home/featurize/data/train.csv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": "/home/featurize/data/bert_pretrain/"
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": "/home/featurize/data/bert_pretrain/",
            "requires_grad": true
        }
    },
    "data_loader": {
        "batch_size": 8,
        "cuda_device": 0,
        "max_instances_in_memory": 1600,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 1
    }
}
