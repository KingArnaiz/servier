from servier.parsing import parse_predict_args
from servier.train import make_predictions

if __name__ == '__main__':
    args = parse_predict_args()
    make_predictions(args)
