import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--train-path', default='data/dataset_train_preprocessed.csv')
    parser.add_argument('--test-path', default='data/dataset_test_preprocessed.csv')

    return parser.parse_args()


def preprocess(test_size: float, train_path: str, test_path: str):
    train_dataset = pd.read_csv('data/dataset.csv')
    train_X = train_dataset.loc[:, train_dataset.columns != 'price_range']

    x = train_X.values
    min_max_scaler = preprocessing.MinMaxScaler()
    scaler = min_max_scaler.fit(x)
    x_scaled = scaler.transform(x)

    df_train = pd.DataFrame(x_scaled)
    df_train['price_range'] = train_dataset['price_range']
    df_train, df_test = train_test_split(df_train, test_size=test_size, random_state=101)

    df_train.to_csv(train_path)
    df_test.to_csv(test_path)


if __name__ == '__main__':
    args = parse_args()
    preprocess(args.test_size, args.train_path, args.test_path)