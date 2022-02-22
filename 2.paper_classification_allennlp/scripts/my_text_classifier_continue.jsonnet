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
        "type": "from_archive",
        "archive_file": "checkpoint_cnn_model/model.tar.gz"
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
        "num_epochs": 5
    }
}
