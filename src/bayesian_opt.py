import pandas as pd
import numpy as np
import config
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from skopt import gp_minimize
from skopt import space


def optimize(params, param_names, x, y):
    """Takes all arguments from search space and traning features/target
        Initializes the models by setting the chosen param and runs CV
    Args:
        params [dict]: convert params to dict
        param_names [list]: make a list of param names
        x [float]: feature values
        y [int]: target values as binary
    Returns:
        [float]: Returns an accuracy score after 5 Folds
    """
    # set the parameters as dictionaries
    params = dict(zip(param_names, params))

    # initiate RandomForestClassifie and K-fold (5)
    model = RandomForestClassifier(**params)
    kf = StratifiedKFold(n_splits=5)
    acc = []

    # loop over kfolds
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        pred = model.predict(xtest)

        # append mean-accuracy to empty list
        fold_accuracy = accuracy_score(ytest, pred)
        acc.append(fold_accuracy)
    # return negative acc to find max optimization
    return -np.mean(acc)


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    features = df[
        ["RSI", "50MA", "200MA", "14-high", "14-low", "%K", "SC", "MACD", "Signal_MACD"]
    ].values
    target = df["Target"].values

    # define the range of input values to test the Bayes_op to create prop-distribution
    param_space = [
        space.Integer(50, 300, name="n_estimators"),
        space.Integer(4, 24, name="max_depth"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        space.Integer(5, 30, name="min_samples_split"),
        space.Integer(4, 24, name="min_samples_leaf"),
        space.Real(0.01, 1, prior="uniform", name="max_features"),
    ]
    param_names = [
        "n_estimators",
        "max_depth",
        "criterion",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
    ]
    # define the loss function to minimize (acc will be negative)
    optimization_function = partial(
        optimize, param_names=param_names, x=features, y=target
    )
    # initiate gp_minimize for Bayesian Optimization to select the best input values
    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=10,
        n_random_starts=10,
        verbose=10,
    )
    print(dict(zip(param_names, result.x)))
