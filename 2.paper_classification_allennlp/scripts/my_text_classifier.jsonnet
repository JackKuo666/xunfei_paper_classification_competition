{
    "dataset_reader" : {
        "type": "classification-tsv",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "data/paper_classification/train_small.csv",
    "validation_data_path": "data/paper_classification/dev_small.csv",
    "vocabulary":{
        "type": "from_files",
        "directory": "data/vocab.tar.gz"
    },
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 10
        }
    },
    "data_loader": {
        "batch_size": 8,
        "max_instances_in_memory": 80,
        "cuda_device": 0,
        "shuffle": true
    },
    "trainer": {
         "checkpointer":{
            "type": "simple_checkpointer",
            "serialization_dir":"checkpoint",
            "save_every_num_seconds": 300
        },
        "optimizer": "adam",
        "num_epochs": 8
    }
}
