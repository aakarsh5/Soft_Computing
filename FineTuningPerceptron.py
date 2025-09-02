import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

class MyPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x > 0 else 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            for i in indices:
                xi, yi = X[i], y[i]
                y_pred = self.predict_single(xi)
                error = yi - y_pred
                self.w += error * self.learning_rate * xi
                self.b += error * self.learning_rate
        return self

    def predict_single(self, xi):
        z = np.dot(self.w, xi) + self.b
        return self.activation(z)

    def predict(self, X):
        return np.array([self.predict_single(xi) for xi in X])

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

# Load Data
file = pd.read_csv("student_results.csv")
X = file[['Math', 'Science', 'English', 'History', 'Computer']].values
y = file['Final_Result'].values  

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GridSearchCV
param_grid = {
    'learning_rate': [0.001, 0.01],
    'epochs': [500, 1000]
}

grid = GridSearchCV(MyPerceptron(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
print("Test Accuracy:", grid.best_estimator_.score(X_test, y_test))

# Predictions on test data
y_pred = grid.best_estimator_.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Precision, Recall, F1-score
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

# Predict New Student
new_student = np.array([50, 60, 70, 80, 90])
new_student_scaled = scaler.transform(new_student.reshape(1, -1))

prediction = grid.best_estimator_.predict(new_student_scaled)[0]
print("\nNew Student Prediction:", "Pass" if prediction == 1 else "Fail")
