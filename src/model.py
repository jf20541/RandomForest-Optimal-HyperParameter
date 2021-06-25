import pandas as pd
import config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, plot_roc_curve
import matplotlib.pyplot as plt


def train():
    """
    Initiating RandomForest classifier with defined parameters
    Evaluate the models with accuracy and classification_report
    """
    global model
    model = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=10,
    )

    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    acc = accuracy_score(y_test, pred) * 100

    # Evalute RandomForest Classifier
    report = classification_report(
        y_true=y_test,
        y_pred=pred,
        target_names=[1, 0],
        output_dict=True,
    )
    eval_model = pd.DataFrame(report)
    print(f"Random Forest Classifier Accuracy Score: {acc:0.2f}%")
    print(eval_model)


def roc_curve():
    roc = plot_roc_curve(model, x_test, y_test, alpha=0.8, name="ROC Curve", lw=1)
    plt.title("ROC Curve RandomForest Classifier with Defined Parameters")
    plt.savefig(config.MODEL_ROC)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    features = df[
        ["RSI", "50MA", "200MA", "14-high", "14-low", "%K", "SC", "MACD", "Signal_MACD"]
    ].values
    target = df["Target"].values
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20
    )
    train()
    roc_curve()
