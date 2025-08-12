import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10000):
        self.w = np.zeros(input_size)
        self.b = 0
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x > 0 else 0

    def predict(self, x):
        z = np.dot(self.w, x) + self.b
        return self.activation(z)

    def predict_batch(self, X):
        # Predict for multiple samples
        return np.array([self.predict(xi) for xi in X])

    def fit(self, X, y):
        for _ in range(self.epochs):
            indices = np.arange(len(y))
            np.random.shuffle(indices)
            for i in indices:
                xi, yi = X[i], y[i]
                y_pred = self.predict(xi)
                error = yi - y_pred
                self.w += error * self.lr * xi
                self.b += error * self.lr

file = pd.read_csv('student_results.csv')

X = file[['Math', 'Science', 'English', 'History', 'Computer']].values
y = file['Final_Result'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

kf = KFold(n_splits=10, shuffle=True, random_state=70)

accuracies = []
precisions = []
recalls = []

for train_index, test_index in kf.split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    p = Perceptron(input_size=5, learning_rate=0.05, epochs=1000)
    p.fit(x_train, y_train)
    y_pred = p.predict_batch(x_test)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, zero_division=0))
    recalls.append(recall_score(y_test, y_pred, zero_division=0))

print("10-Fold Cross Validation Results:")
print("Average Accuracy:", np.mean(accuracies))
print("Average Precision:", np.mean(precisions))
print("Average Recall:", np.mean(recalls))


new_student = np.array([50, 60, 70, 80, 90])
new_student_scaled = scaler.transform(new_student.reshape(1, -1))[0]

prediction = p.predict(new_student_scaled)
print("Pass" if prediction == 1 else "Fail")
