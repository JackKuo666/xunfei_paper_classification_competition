from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer


@DatasetReader.register("sent_attention_reader")
class SentAttentionReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, title: str, abstract: str, label: str = None) -> Instance:
        tokens_title = self.tokenizer.tokenize(title)
        tokens_abstract = self.tokenizer.tokenize(abstract)
        if self.max_tokens:
            tokens_title = tokens_title[: self.max_tokens]
            tokens_abstract = tokens_abstract[: self.max_tokens]
        text_field_title = TextField(tokens_title, self.token_indexers)
        text_field_abstract = TextField(tokens_abstract, self.token_indexers)
        # 4、同时加入两个就是两个inputs text了
        fields = {"title": text_field_title, "abstract": text_field_abstract}
        # 3、如果含有label的话，就加入，说明是训练；没有的话不加入，说明是测试
        if label:
            fields["label"] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for line in lines:
                line = line.strip().split("\t")
                # 1、这里判断输入的train(含有label)还是test(不含label)
                if len(line) == 4:
                    paperid, title, abstract, categories = line
                else:
                    paperid, title, abstract = line
                    categories = None
                # 2、这里判断是不是标题，如果是就略过
                if paperid == "paperid":
                    continue
                yield self.text_to_instance(title, abstract, categories)
