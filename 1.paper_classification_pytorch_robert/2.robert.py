import gc
import os

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pylab import rcParams
from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置显卡
# 配置

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 25, 20
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# train = pd.read_csv('data/train_small.csv', sep='\t')
# test = pd.read_csv('data/test_small.csv', sep='\t')
train = pd.read_csv('data/train.csv', sep='\t')
test = pd.read_csv('data/test.csv', sep='\t')
sub = pd.read_csv('data/sample_submit.csv')
# print(test[:5])

# 拼接title与abstract
train['text'] = train['title'] + ' ' + train['abstract']
test['text'] = test['title'] + ' ' + test['abstract']

label_id2cate = dict(enumerate(train.categories.unique()))
label_cate2id = {value: key for key, value in label_id2cate.items()}

train['label'] = train['categories'].map(label_cate2id)

train = train[['text', 'label']]
train_y = train["label"]

train_df = train[['text', 'label']][:45000]
eval_df = train[['text', 'label']][45000:]
# train_df = train[['text', 'label']][:70]
# eval_df = train[['text', 'label']][70:]

model_args = ClassificationArgs()
model_args.max_seq_length = 20
model_args.train_batch_size = 8
model_args.num_train_epochs = 1
model_args.fp16 = False
model_args.evaluate_during_training = False
model_args.overwrite_output_dir = True


model_type = 'roberta'
model_name = 'roberta-base'
print("training {}.........".format(model_name))
model_args.cache_dir = './caches' + '/' + model_name.split('/')[-1]
model_args.output_dir = './outputs' + '/' + model_name.split('/')[-1]

model = ClassificationModel(
    model_type,
    model_name,
    num_labels=39,
    use_cuda=True,
    args=model_args)

model.train_model(train_df, eval_df=eval_df)
result, _, _ = model.eval_model(eval_df, acc=accuracy_score)
print(result)

data = []
for index, row in test.iterrows():
    data.append(str(row['text']))
predictions, raw_outputs = model.predict(data)
sub = pd.read_csv('data/sample_submit.csv')
sub['categories'] = predictions
sub['categories'] = sub['categories'].map(label_id2cate)
sub.to_csv('result/submit_{}.csv'.format(model_name), index=False)
del model
gc.collect()