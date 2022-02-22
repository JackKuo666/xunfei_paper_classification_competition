import warnings
import pandas as pd

warnings.filterwarnings('ignore')

train_data = pd.read_csv('data/paper_classification/train.csv', sep='\t')
test_data = pd.read_csv('data/paper_classification/test.csv', sep='\t')
train_data['text'] = train_data['title'] + '.' + train_data['abstract']
test_data['text'] = test_data['title'] + '.' + test_data['abstract']
data = pd.concat([train_data, test_data])
data['text'] = data['text'].apply(lambda x: x.replace('\n', ''))

text = '\n'.join(data.text.tolist())

with open('data/paper_classification/for_IPTP_train_text.txt', 'w') as f:
    f.write(text)