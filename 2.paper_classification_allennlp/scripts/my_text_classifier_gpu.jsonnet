{
    "dataset_reader" : {
        "type": "classification-tsv",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "/home/featurize/data/train.csv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 512
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 512
        }
    },
    "data_loader": {
        "batch_size": 16,
        "cuda_device": 0,
        "max_instances_in_memory": 1600,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "cuda_device": 0,
        "num_epochs": 1
    }
}
