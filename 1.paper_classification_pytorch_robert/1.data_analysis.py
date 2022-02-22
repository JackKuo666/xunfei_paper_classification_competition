import pandas as pd

train = pd.read_csv('data/train.csv', sep='\t')
print(train[:5])
test = pd.read_csv('data/test.csv', sep='\t')
sub = pd.read_csv('data/sample_submit.csv')
print(test[:5])
# 拼接title与abstract
train['text'] = train['title'] + ' ' + train['abstract']
test['text'] = test['title'] + ' ' + test['abstract']

print(train.categories.unique())
exit()
label_id2cate = dict(enumerate(train.categories.unique()))
label_cate2id = {value: key for key, value in label_id2cate.items()}
