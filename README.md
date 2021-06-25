# RandomForestHyperParameter

## Objective
Supervised Learning (Classification Problem), Using ensembled model which uses ```(n_estimators)``` to improve performance, avoids over-fitting and finds optimal hyper-parameters using GridSearchCV, RandomSearchCV and Bayesian Optimization using Gaussian Process. Calculated 9 market indicators based on S&P 500 Index (SPY) OHLC Price to predict the Target Binary values (0,1) as negative and positive values for the Adjusted Closing Price Change, respectively.

## Model
Random Forest: A meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

Bayes Optimization uses probability to find the min function. It finds the input value to a function which outputs the lowest value. The model often tends to performance better and use less processing power.

## Parameters
- n_estimators [int]: number of decision trees in the Random Forest model
- criterion {“gini”, “entropy”}: measures the quality of split (information gain) 
- max_depth [int]: max depth of the tree
- min_samples_split [int]: min number of samples required to split an internal node
- min_samples_leaf [int]: min number of samples required to be at a leaf node
- max_features {“auto”, “log2”, "sqrt"}: number of features to consider when looking for the best split

## Metric
![](https://latex.codecogs.com/gif.latex?Accuracy%3D%5Cfrac%7Btp&plus;tn%7D%7B%28tp%20&plus;%20tn%29&plus;%28fp-fn%29%29%7D)\
tp = True Positive\
tn = True Negative\
fp = False Positive\
fn = False Negative

## Output
```bash
Bayesian Optimization Gaussian Process

Current minimum: -0.7096 = 70.96%
{'n_estimators': 131, 'max_depth': 16, 'criterion': 'gini', 'min_samples_split': 14, 'min_samples_leaf': 22, 'max_features': 0.9502885825330616}
```
```bash
RandomForestClassifier with RandomSearchCV: 75.00%

Best parameters set:
  n_estimators: 150
  criterion: entropy
  max_depth: 20
  max_features: log2
  min_samples_leaf: 16
  min_samples_split: 15
```
```bash
Defined Parameters 

Random Forest Classifier Accuracy Score: 70.75%
  n_estimators:100
  criterion: gini
  max_depth: 10
```
```bash
RandomForestClassifier with GridSearchCV: 74.36%

Best parameters set:
  n_estimators:100
  criterion: gini
  max_depth: 15
  max_features: log2
  min_samples_leaf: 12
  min_samples_split: 5
```

### Code
Created 5 modules
- `main.py`: Initiated RandomSearchCV and GridSearchCV for optimal parameters
- `train.py`: Initiated Bayesian Optimization using Gaussian Process and initiate StratefiedKFold
- `model.py`: Initiated RandomForest Classifier with defined parameters and evaluated
- `grid_dispatcher.py`: Defined dictionary with keys that are names of model and values are models themselves
- `features.py`: Calculated RSI, MACD, MA50, MA200, 14-low, 14-high, Stochastic Oscillator, Signal MACD, %K 
- `config.py`: Defined file paths as global variable

### Run
In a terminal or command window, navigate to the top-level project directory `RandomForestHyperParameter/` (that contains this README) and run the following command:
```bash
pip install --upgrade pip && pip install -r requirements.txt
``` 

## Data
```bash
Target           int64

Features: 
RSI            float64
50MA           float64
200MA          float64
14-high        float64
14-low         float64
%K             float64
SC             float64
MACD           float64
Signal_MACD    float64
```
## Sources
https://scikit-optimize.github.io/stable/modules/classes.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
https://latex.codecogs.com/eqneditor/editor.php
