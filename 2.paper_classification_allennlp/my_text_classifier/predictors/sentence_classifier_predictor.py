from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides
from typing import List, Iterator, Dict, Tuple, Any, Type, Union, Optional
from allennlp.common.util import JsonDict, sanitize


@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(Predictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        title = json_dict["title"]
        abstract = json_dict["abstract"]
        return self._dataset_reader.text_to_instance(title+abstract)

    @overrides
    def load_line(self, line: str) -> JsonDict:
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        self.line = line
        line = line.strip().split("\t")
        paperid, title, abstract = line
        return {"paperid": paperid, "title": title, "abstract": abstract}

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        if "paperid" in outputs:
            return str(outputs["paperid"]) + "," + str(outputs["categories"]) + "\n"
        else:
            # 这种情况是最后不在batch中的需要单独预测
            return str(self.line.split("\t")[0]) + "," + str(outputs["label"]) + "\n"

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        outputs = self.predict_batch_instance(instances)
        outputs = [{"paperid": i["paperid"], "categories":j["label"]} for i, j in zip(inputs, outputs)]

        return outputs