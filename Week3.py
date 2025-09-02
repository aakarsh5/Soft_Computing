import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------- Load MNIST Online ----------------
print("Loading MNIST dataset from OpenML...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0   # normalize pixel values
y = y.astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Custom Perceptron (manual epochs with partial_fit) ----------------
print("\nTraining Custom Perceptron...")
classes = np.unique(y_train)
custom_perceptron = Perceptron(max_iter=1, warm_start=True, tol=None)  # allow manual epochs
val_acc = []
start = time.time()
for epoch in range(10):  # simulate epochs
    custom_perceptron.partial_fit(X_train, y_train, classes=classes)
    y_val_pred = custom_perceptron.predict(X_test)
    val_acc.append(accuracy_score(y_test, y_val_pred))
time_custom = time.time() - start
y_pred_custom = custom_perceptron.predict(X_test)

# ---------------- Built-in Perceptron ----------------
print("Training Built-in Perceptron...")
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)  # increased max_iter
start = time.time()
perceptron.fit(X_train, y_train)
time_perceptron = time.time() - start
y_pred_perceptron = perceptron.predict(X_test)

# ---------------- MLP Classifier ----------------
print("Training MLP Classifier...")
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=50, 
                    early_stopping=True, random_state=42, verbose=False)  # longer training + early stopping
start = time.time()
mlp.fit(X_train, y_train)
time_mlp = time.time() - start
y_pred_mlp = mlp.predict(X_test)

# ---------------- SVM (subset for speed) ----------------
print("Training SVM (subset of 10000 samples for speed)...")
svm = SVC(kernel='linear')
start = time.time()
svm.fit(X_train[:10000], y_train[:10000])
time_svm = time.time() - start
y_pred_svm = svm.predict(X_test)

# ---------------- Evaluation Function ----------------
def evaluate(y_true, y_pred, train_time, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, Training Time: {train_time:.2f}s")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    return acc, prec, rec, train_time

# ---------------- Collect Results ----------------
results = {}
results['Custom Perceptron'] = evaluate(y_test, y_pred_custom, time_custom, "Custom Perceptron")
results['Built-in Perceptron'] = evaluate(y_test, y_pred_perceptron, time_perceptron, "Built-in Perceptron")
results['MLP'] = evaluate(y_test, y_pred_mlp, time_mlp, "MLP")
results['SVM'] = evaluate(y_test, y_pred_svm, time_svm, "SVM")

# ---------------- Comparison Table ----------------
df_results = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "Training Time (s)"]).T
print("\n=== Comparison Table ===")
print(df_results)

# ---------------- Plot Epoch vs Validation Accuracy ----------------
plt.plot(range(1, 11), val_acc, marker='o', label="Custom Perceptron")
plt.xlabel("Epochs")
plt.ylabel("Validation Accuracy")
plt.title("Epochs vs Validation Accuracy (Custom Perceptron)")
plt.legend()
plt.show()

