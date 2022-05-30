import pandas as pd
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test-path', default='data/dataset_test_preprocessed.csv')
    parser.add_argument('--model-path', default='data/model.pkl')
    parser.add_argument('--evaluate-path', default='data/evaluate.csv')

    return parser.parse_args()


def evaluate(test_path: str, model_path: str, evaluate_path: str):
    loaded_model = pickle.load(open(model_path, 'rb'))

    train_dataset=pd.read_csv(test_path)
    X=train_dataset.drop('price_range',axis=1)
    y=train_dataset['price_range']
    result = loaded_model.score(X, y)

    with open(evaluate_path, 'w') as f:
        f.write(f'Score\n{result}')


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.test_path, args.model_path, args.evaluate_path)