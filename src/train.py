import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-path', default='data/dataset_train_preprocessed.csv')
    parser.add_argument('--model-path', default='data/model.pkl')
    parser.add_argument('--n-estimators', type=int, default=100)

    return parser.parse_args()


def train(train_path: str, model_path: str, n_estimators: int):
    train_dataset = pd.read_csv(train_path)
    X = train_dataset.drop('price_range',axis=1)
    y = train_dataset['price_range']

    model = RandomForestClassifier(n_estimators=n_estimators).fit(X,y)

    pickle.dump(model, open(model_path, 'wb'))


if __name__ == '__main__':
    args = parse_args()
    train(args.train_path, args.model_path, args.n_estimators)