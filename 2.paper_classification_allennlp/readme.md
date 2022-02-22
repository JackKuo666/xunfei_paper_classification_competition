# 这个是我参加科大讯飞[学术论文分类挑战赛](https://challenge.xfyun.cn/topic/info?type=academic-paper-classification)
# 1.依赖 
python == 3.8.0

allennlp == 2.4.0

pip install allennlp -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2.训练
## 2.2 使用 allennlp train 训练 embedding+bag_of_embedding

### train
```buildoutcfg
allennlp train scripts/my_text_classifier.jsonnet --serialization-dir checkpoint --include-package my_text_classifier -f
```
### eval
```buildoutcfg
allennlp evaluate checkpoint/model.tar.gz data/movie_review/test.tsv --include-package my_text_classifier
```

### predict
```buildoutcfg
allennlp predict checkpoint/model.tar.gz data/movie_review/test.jsonl --include-package my_text_classifier --predictor sentence_classifier
```
# 1.2.build vocab
跟lazy同样道理的是 voacab的构建：由于训练数据比较大的时候，在每一次训练时从头构建vocab（需要完整遍历一边所有的数据集）是比较耗时的，所以我们这里手动构建一个vocab,然后在每次修改模型，训练模型的时候直接load就行了。

1、在jsonnet中设置：
```buildoutcfg
"test_data_path": "data/paper_classification/test.csv",
"datasets_for_vocab_creation": ["train","test"],
```
2、提前使用build-vocab命令生成vocab,这里也可以使用自己的脚本生成，然后仿照`vocab_model.tar.gz`填入就行了
```buildoutcfg
allennlp build-vocab scripts/my_text_classifier.jsonnet data/vocab_model.tar.gz --include-package my_text_classifier
```
我们的生成的位置在：`data/vocab_model.tar.gz`

3、只有生成之后才能在jsonnet中删除`"datasets_for_vocab_creation": ["train","test"],`然后再添加：
```buildoutcfg
    "vocabulary":{
        "type": "from_files",
        "directory": "data/vocab_model.tar.gz"
    },
```
这样之后的训练就会直接加载这个vocab了，不会浪费更多的时间。

## 2.3 使用 allennlp train 训练 bert embedding+bert pool
这里需要下载[bert预训练模型](https://huggingface.co/bert-base-uncased/tree/main)放到``bert_pretrain`文件夹下
### train
train bert
```buildoutcfg
allennlp train scripts/my_text_classifier_bert.jsonnet --serialization-dir checkpoint_bert --include-package my_text_classifier
```
train robert
```buildoutcfg

```
### eval
```buildoutcfg
allennlp evaluate checkpoint_bert/model.tar.gz data/movie_review/test_small.tsv --include-package my_text_classifier
```

### predict
```buildoutcfg
allennlp predict checkpoint/model.tar.gz data/paper_classification/test.csv -
-output-file data/paper_classification/predict_result.csv  --include-package my_text_classifier --predictor sentence_classifier --batch-size 8 --silent
```
注意： 这里的model 是gpu服务器上训练的，里面如果含有预训练的model 的路径，需要在本地预测的话，路径要与服务器中的一致
这里我们将bert预训练的参数放在：`F:\home\featurize\data\bert_pretrain`

## 2、如何进行从 line 读取 test； 然后batch predict；然后存储指定格式的预测结果：
### 1、从line 读取 test(而不是从json中):
需要在自定义的predictor中重写这个方法：
```py 
    @overrides
    def load_line(self, line: str) -> JsonDict:
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        return {"sentence": line}
```
### 2. batch predict
需要在自定义的predictor中重写这个方法：
```py 
    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        outputs = self.predict_batch_instance(instances)
        outputs = [{"paperid": i["sentence"].split("\t")[0], "categories":j["label"]} for i, j in zip(inputs, outputs)]

        return outputs
```
### 3.把预测结果以指定格式存储
需要在自定义的predictor中重写这个方法：
```py 
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
```
还需要再自定义模型中重写这个方法：
```
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        label = torch.argmax(output_dict["probs"], dim=1)
        label = [self.vocab.get_token_from_index(int(i), "labels") for i in label]
        output_dict["label"] = label
        return output_dict

```
### 4、batch_预测的命令：
cpu
```buildoutcfg
allennlp predict checkpoint/model.tar.gz data/paper_classification/test.csv --output-file data/paper_classification/predict_result.csv  --include-package my_text_classifier --predictor sentence_classifier --batch-size 8 --silent
```
注意：只有在predictor中重写了`predict_batch_json`方法，同时在预测命令中指定：`--batch-size 8` ，才能batch 预测。
gpu
```buildoutcfg
nohup allennlp predict /home/featurize/data/checkpoint/model/model.tar.gz /home/featurize/data/test.csv 
--output-file /home/featurize/data/predict_result.csv  --include-package my_text_classifier 
--predictor sentence_classifier --batch-size 16 --cuda-device 0 --silent &
```
注意：只有指定`--cuda-device 0` 才能在预测阶段使用gpu。



## 3、如何继续训练模型 ?
之前指定训练一个epoch，训练完成之后修改jsonnet，想接着训练更多的epoch。
分两种情况：
### 1、上次训练的checkpoint还在的情况下：
直接修改`scripts/train.jsonnet`和`checkpoint/config.jsonnet`中的`num_epochs`，然后训练命令改为：
```buildoutcfg
nohup allennlp train scripts/my_text_classifier_robert_gpu.jsonnet --serialization-dir /home/featurize/data/checkpoint/model --include-package my_text_classifier --recover &
```
注意：继续训练的命令是后面跟`--recover`
注意：因为这里是 recover ，所以千万不能跟 -f,要不然会把checkpoint 的内容删除掉

### 2、上次训练的checkpoint不存在了，只有`model.tar.gz`的情况下：
1、修改`tain.jsonnet`中的model，同时修改自己想要训练的`num_epochs`:
```
    "model": {
        "type": "from_archive",
        "archive_file": "/home/featurize/data/model.tar.gz"
    },
 ```
2、继续训练：
```
nohup allennlp train scripts/robert_continue_train.jsonnet --serialization-dir /home/featurize/data/checkpoint_continue/ --include-package my_text_classifier &
```
注意：这个的前提是之前训练的`moderl.tar.gz`中的`config.json`中的 “model” 是自定义的而不是从archive中得到的，如果同样也是从archive中得到的，那么需要修改config.json中的“model”:
```
mkdir model
# 解压命令
tar -xvf model.tar.gz -C model/ 
cd model
vi config.jsonnet
# 修改完之后压缩命令
tar -zcvf model.tar.gz config.json meta.json vocabulary weights.th
```

# 4、如何定时存储 checkpoint
由于默认情况下的`jsonnet`没有注册`Checkpointer`，所以需要我们注册一个，然后把`save_every_num_seconds` 作为传参通过`jsonnet`传进去。

1、重写一个 checkpointer,目的是可以在jsonnet中使用
```py
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
```
### 2、在jsonnet中指定`save_every_num_seconds`：
```
 "trainer": {
        "checkpointer":{
            "type": "simple_checkpointer",
            "serialization_dir":"checkpoint",
            "save_every_num_seconds": 1200
        },
 }
```

# 5、如何输入两个inputs?(如何写一个有效的dataset reader)
```py
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

```
## 2、如何输入bert 方式
```py
    def text_to_instance(self, text_a: str, text_b: str, label: str = None) -> Instance:
        # 80% of the text_a length in the training set is less than 256, 512 - 256 = 256.
        tokens_a = self.tokenizer.tokenize(text_a)[:self.max_tokens//2]
        tokens_b = self.tokenizer.tokenize(text_b)[:self.max_tokens-len(tokens_a)]
        # 4、text_a+text_b 中间是sep 同时输入 bert

        tokens = self.tokenizer.add_special_tokens(tokens_a[1:-1], tokens_b[1:-1])

        text_field = TextField(tokens, self.token_indexers)

        fields = {"text": text_field}
        # 3、如果含有label的话，就加入，说明是训练；没有的话不加入，说明是测试
        if label:
            fields["label"] = LabelField(label)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r", encoding="utf-8") as lines:
            for line in lines:
                line = line.strip().split("\t")
                # 1、这里判断输入的train(含有label)还是test(不含label)
                if len(line) == 3:
                    text_a, text_b, categories = line
                else:
                    text_a, text_b, = line
                    categories = None
                if text_a == "text_a":
                    continue
                yield self.text_to_instance(text_a, text_b, categories)

```




# todo
1、预测一下结果，然后上传                          done
2、跑一下别人的robert结果                         done
3、然后用allennlp构造robert,跑一下看看             done 
3.2、 分一下train 和 dev                         done
3.3、batch predict with line dump              done 【需要总结一下: done】
4、跑一下news_classification 那个sent attention  doing...
    train:
```buildoutcfg
allennlp train scripts/my_text_classifier_sent_attention.jsonnet -s checkpoint --include-package my_text_classifier -f
```
5、构造一下ps_2,跑一下：https://github.com/allenai/allennlp/blob/v0.6.1/tutorials/getting_started/using_as_a_library_pt1.md
6、试着将那个sent attention 改为两个输入的 done


今天：
    1、服务器跑完之后，上传新的predict 和 read tsv,进行batch predict    done 
    2、然后将train 的dev上传上去，训练robert 看看                      done
    3、然后将  sent attention 上传上去训练看看 还需要修改网络            done 效果不怎么好

todo:
    1、继续robert 训练 done 结果不太好，需要看看是因为train dev分开的不太好，造成结果拟合的不好？ 还是因为本身过拟合了？
    2、robert 是否需要 train validation 分开        todo_1: 从6个epoch开始，不在拆分开训练，看看结果是否和分开的结果好 done：比分开的好
    3、robert 的输入是否需要 segment id 分开的方式     todo_3: 找一下看看有没有 done bert_reader  结果没有一个input 好  【结论:因为roberta本身没有nsp任务】
        ```
            allennlp train scripts/my_text_classifier_bert_2_input.jsonnet -s checkpoint --include-package my_text_classifier -f
        ```
    4、尝试新的sent attention 方式
    5、模型在 allennlp上fine tune 的方法    
    6、试一下xlnet 
    7、ITPT 继续预训练 【基于robert加入test先进行预训练一下，然后再分类训练，结果可能会更高点】  todo_2: 模型在这上面的基础训练 done:训练了5个epoch; train loss:1.545

    todo: 1、bert 看看能不能调用robert 预训练模型，然后bert 进行 2 input  【不能】
          2、如果不能，看看 bert 进行 2 train test 预训练，然后再 2 input  【done】
          3、尝试基于bert 进行多模型集成 或者两个loss 加和试试

    小结：目前为止，最好的模型是8月5号gpu 只train没有使用dev, roberta 模型训练9个epoch,结果是0.8037
        接下来基于robert的尝试有：1、train_dev，9个epoch，结果是0.7846
                              2、sent attention，最好的结果是0.7609
                              3、先在train,test上训练语言模型，然后再title+abstract | title[sep]abstrct 两种方式输入，最好结果0.7979
                                【这里后来发现roberta没有nsp任务，所以输入title[sep]abstrct没有效果】
        基于bert的尝试有：
                        1、 输入 title+abstract , 5个epoch, 结果是 0.76
                        2、 sent attention, 结果是0.77
                        3、 先在train,test上训练语言模型，然后再输入 title[sep]abstrct， 最好结果0.7898

    综上：1、roberta 没有 nsp 任务，所以加不加 title[sep]abstrct 都不影响结果
         2、bert 加入 train test 的预训练语言模型；以及换成 title[sep]abstrct 输入都有提升
         3、robert 由于没有train dev划分，直接训练结果比较好，所以误导了自己，后来都没有使用train dev的方式训练，然后重点都放在了 增加预训练和更多的epoch上了，
            其实这里方向错了，应该是更改更多的结构，尝试不同的模型，因为通过train dev的方式可以看到该结构的模型最好的validation 的结果是3个epoch，0.80左右，
            那么接下来可以换网络结果，尝试不同的输出，或者数据增强，或者集成学习了，只有在train dev 上有明显的提升的结构，接下来才能在此基础上进行全部train