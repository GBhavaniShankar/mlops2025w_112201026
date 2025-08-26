from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

print("Loading the MNIST dataset (handwritten digits)...")
digits = datasets.load_digits()
X = digits.data
y = digits.target

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(set(y))}")

print("\nSplitting into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")

print("\nApplying feature scaling...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Scaling done.")

print("\nTraining Logistic Regression model...")
log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train, y_train)
print("Training complete.")

print("\nMaking predictions with Logistic Regression...")
y_pred = log_reg.predict(X_test)
print("Predicted labels (first 20):", y_pred[:20])

print("\nEvaluating Logistic Regression...")
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nTraining a Decision Tree Classifier for comparison...")
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

tree_acc = accuracy_score(y_test, tree_pred)
print(f"Decision Tree Accuracy: {tree_acc:.2f}")

print("\nVisualizing some test images with predictions...")

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(digits.images[i], cmap="gray")  # original 8x8 image
    ax.set_title(f"True: {y[i]}\nPred: {log_reg.predict([X[i]])[0]}")
    ax.axis("off")

plt.tight_layout()
plt.show()