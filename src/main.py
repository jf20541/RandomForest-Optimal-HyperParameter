import pandas as pd
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split


def optimization_parameters():
    """
    Initiated RandomGridSearchCV and GridSearchCV
    to seek the optimal hyperparameters for RandomForest
    Evalauted model's performance based on optimal parameters
    """
    param_distributions = {
        "n_estimators": list(range(50, 300, 50)),
        "max_features": ["auto", "log2"],
        "max_depth": list(range(1, 21, 2)),
        "min_samples_leaf": list(range(4, 22, 2)),
        "min_samples_split": list(range(5, 30, 5)),
        "criterion": ["gini", "entropy"],
    }
    param_grid = {
        "n_estimators": list(range(50, 300, 50)),
        "max_depth": list(range(1, 21, 2)),
        "min_samples_leaf": list(range(4, 22, 2)),
        "min_samples_split": list(range(5, 30, 5)),
        "criterion": ["gini", "entropy"],
    }

    rfc = RandomForestClassifier()

    # 5 * 10 * 9 * 5 * 2 = 4500 iterations
    # will take a lot of time
    model = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5,
    )

    model = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=param_distributions,
        n_iter=20,
        scoring='accuracy',
        verbose=10,
        n_jobs=1,
        cv=5,
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"RandomForestClassifier with GridSearchCV: {acc:0.2f}%")
    print("Best parameters set:")

    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    features = df[
        ["RSI", "50MA", "200MA", "14-high", "14-low", "%K", "SC", "MACD", "Signal_MACD"]
    ].values
    target = df["Target"].values
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20
    )
    optimization_parameters()