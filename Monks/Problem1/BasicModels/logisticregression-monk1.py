import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import os.path as path

def getMonk1TrainDatasetPath():
    return path.join(path.abspath(path.dirname(__file__)), "..", "..", "Datasets", "monks-1train.csv")

def getMonk1TestDatasetPath():
    return path.join(path.abspath(path.dirname(__file__)), "..", "..", "Datasets", "monks-1test.csv")

trainDataset = pd.read_csv(getMonk1TrainDatasetPath())
testDataset = pd.read_csv(getMonk1TestDatasetPath())

X_train = trainDataset.iloc[:, 1:7]
Y_train = trainDataset.iloc[:, 0]

X_test = testDataset.iloc[:, 1:7]
Y_test = testDataset.iloc[:, 0]

# Training
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

logreg_search = GridSearchCV(
    LogisticRegression(),
    param_grid=param_grid,
    cv=kf,
    scoring='accuracy',
    verbose=4,
    n_jobs=-1
)

logreg_search.fit(X_train, Y_train)

print(f"For Logistic Regression in regards to Monk 1 Training, best parameters are {logreg_search.best_params_} with a score of {logreg_search.best_score_:0.5f}") #0.66767

best_model_logreg = logreg_search.best_estimator_

predictions_train_logreg = best_model_logreg.predict(X_train)

cm_logreg_train = confusion_matrix(Y_train, predictions_train_logreg)
disp_logreg_train = ConfusionMatrixDisplay(confusion_matrix=cm_logreg_train)
disp_logreg_train.plot()
plt.title('Confusion Matrix Training - Logistic Regression')
plt.show()

# Testing
test_accuracy_logreg = best_model_logreg.score(X_test, Y_test)

predictions_test_logreg = best_model_logreg.predict(X_test)
print(f"For Logistic Regression in regards to Monk 1 Testing, accuracy score is {test_accuracy_logreg:0.5f}") #0.66589

cm_logreg_test = confusion_matrix(Y_test, predictions_test_logreg)
disp_logreg_test = ConfusionMatrixDisplay(confusion_matrix=cm_logreg_test)
disp_logreg_test.plot()
plt.title('Confusion Matrix Test Monk 1 Logistic Regression')
plt.savefig(path.join(path.dirname(__file__), "confusion-matrixes", "LogReg_CM.png"))