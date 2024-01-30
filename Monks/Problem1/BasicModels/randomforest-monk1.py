import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import os.path as path

def getMonk1TrainDatasetPath ():
    return path.join(path.abspath(path.dirname(__file__)), "..", "..", "Datasets", "monks-1train.csv")

def getMonk1TestDatasetPath ():
    return path.join(path.abspath(path.dirname(__file__)), "..", ".." ,"Datasets", "monks-1test.csv")

trainDataset= pd.read_csv(getMonk1TrainDatasetPath())
testDataset= pd.read_csv(getMonk1TestDatasetPath())

X_train = trainDataset.iloc[:, 1:7]
Y_train = trainDataset.iloc[:, 0]

X_test = testDataset.iloc[:, 1:7]
Y_test = testDataset.iloc[:, 0]


#Training 
param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [ 3, 5, 7, 10]
            }

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rf_search = GridSearchCV(
    RandomForestClassifier(n_jobs=-1),
    param_grid = param_grid,
    cv = kf,
    scoring = 'accuracy',
    verbose = 4,
    n_jobs = -1
)

rf_search.fit(X_train, Y_train)

print(f"For Random Forest' in regards to Monk 1 Training, best parameters are {rf_search.best_params_} with a score of {rf_search.best_score_:0.5f}")

best_model = rf_search.best_estimator_

predictions_train = best_model.predict(X_train)

cm = confusion_matrix(Y_train, predictions_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix Training')
plt.show()


#Testing
test_accuracy = best_model.score(X_test, Y_test)

predictions_test = best_model.predict(X_test)
print(f"For Random Forest in regards to Monk 1 Testing, accuracy score is {test_accuracy:0.5f}")
mse = mean_squared_error(Y_test, predictions_test)
print(f"For Random Forest in regards to Monk 1 Testing, MSE is {mse:0.5f}")


cmTest = confusion_matrix(Y_test, predictions_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cmTest)
disp.plot()
plt.title('Confusion Matrix Test Monk 1 Random Forest')
plt.savefig(path.join(path.dirname(__file__), "confusion-matrixes","RF_CM.png"))