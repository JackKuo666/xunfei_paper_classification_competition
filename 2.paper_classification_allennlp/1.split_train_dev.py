import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def all_data2fold(data_file, train_dir, test_dir):
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    print(f.columns)
    print(f.head())

    # print(f["title"].shape)
    # print(f["abstract"].shape)
    # print(f["categories"].shape)

    f["text"] = f["paperid"] + "\t" + f["title"] + "\t" + f["abstract"]
    x = f["text"]
    y = f["categories"]

    # print(f["text"].shape)
    # print(pd.value_counts(y))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # print(x_train.shape)
    # print(x_test.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    # print(pd.value_counts(y_train))

    with open(train_dir, "w", encoding="utf-8") as fout:
        for text, label in zip(x_train.tolist(), y_train.tolist()):
            fout.writelines(text + "\t" + label + "\n")
    logging.info("save  " + train_dir +" done !")

    with open(test_dir, "w", encoding="utf-8") as fout:
        for text, label in zip(x_test.tolist(), y_test.tolist()):
            fout.writelines(text + "\t" + label + "\n")
    logging.info("save  " + test_dir +" done !")

    return


if __name__ == "__main__":
    data_file = 'data/paper_classification/train.csv'
    train_dir = "data/paper_classification/train_split_data/train_data.csv"
    test_dir = "data/paper_classification/train_split_data/dev_data.csv"
    all_data2fold(data_file, train_dir, test_dir)
