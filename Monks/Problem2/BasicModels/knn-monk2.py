import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os.path as path

def getMonk2TrainDatasetPath ():
    return path.join(path.abspath(path.dirname(__file__)), "..", "..","Datasets", "monks-2train.csv")

def getMonk2TestDatasetPath ():
    return path.join(path.abspath(path.dirname(__file__)), "..","..", "Datasets", "monks-2test.csv")

trainDataset= pd.read_csv(getMonk2TrainDatasetPath())
testDataset= pd.read_csv(getMonk2TestDatasetPath())

X_train = trainDataset.iloc[:, 1:7]
Y_train = trainDataset.iloc[:, 0]

X_test = testDataset.iloc[:, 1:7]
Y_test = testDataset.iloc[:, 0]


#Training 
param_grid = {
                'n_neighbors': list(range(1, 20, 2)),
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            }

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid = param_grid,
    cv = kf,
    scoring = 'accuracy',
    verbose = 4,
    n_jobs = -1
)

knn_search.fit(X_train, Y_train)

print(f"For KNN in regards to Monk 2 Training, best parameters are {knn_search.best_params_} with a score of {knn_search.best_score_:0.5f}") #0.66649

best_model = knn_search.best_estimator_

predictions_train = best_model.predict(X_train)

cm = confusion_matrix(Y_train, predictions_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix Training')
plt.show()


#Testing
test_accuracy = best_model.score(X_test, Y_test)

predictions_test = best_model.predict(X_test)
print(f"For KNN in regards to Monk 2 Testing, accuracy score is {test_accuracy:0.5f}")  #0.69838

cmTest = confusion_matrix(Y_test, predictions_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cmTest)
disp.plot()
plt.title('Confusion Matrix Test Monk 2 KNN')
plt.savefig(path.join(path.dirname(__file__), "confusion-matrixes","KNN_CM.png"))