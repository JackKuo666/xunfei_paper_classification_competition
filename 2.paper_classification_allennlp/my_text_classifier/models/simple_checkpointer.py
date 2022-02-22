from typing import Optional, Union
import os

from allennlp.training.checkpointer import Checkpointer


@Checkpointer.register("simple_checkpointer")
class SimpleCheckpointer(Checkpointer):
    def __init__(
            self,
            serialization_dir: Union[str, os.PathLike],
            save_every_num_seconds: Optional[float] = None):
        super().__init__(serialization_dir)
        self._save_every_num_seconds = save_every_num_seconds
        self._serialization_dir = str(serialization_dir)
