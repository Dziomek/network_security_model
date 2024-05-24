import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sampling_strategy import x_resampled, y_resampled
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import data, label_encoder

# Normalize data
scaler = StandardScaler()
x_resampled = scaler.fit_transform(x_resampled)

# Podział danych na zestaw treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=955818, shuffle=True, stratify=y_resampled)

# One-hot encoding
num_classes = 15
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Define Neural Network model
model = Sequential()

# Define layers
# Hidden layer 1 (also Input Layer)
model.add(Dense(256, input_shape=(x_train.shape[1],), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# Hidden layer 2
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# Hidden layer 3
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# Hidden layer 4
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# Output layer
model.add(Dense(num_classes, activation='softmax'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trenowanie modelu
start_time = time.time()
print("Training started...")

history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test), verbose=2)

# Timer end
end_time = time.time()
training_time = end_time - start_time

print("Training finished")
print(f"Czas treningu: {training_time:.2f} sekund")

# Prognozy na zbiorze testowym przy użyciu wytrenowanego modelu
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("Accuracy on test set:", accuracy_score(y_test_classes, y_pred_classes))
print(classification_report(y_test_classes, y_pred_classes))

# Confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
# Since neural networks are typically black boxes, we use permutation feature importance
importances = []

def compute_importance(model, X, y):
    baseline = model.evaluate(X, y, verbose=0)
    importance = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        save_column = X[:, i].copy()
        np.random.shuffle(X[:, i])
        m = model.evaluate(X, y, verbose=0)
        X[:, i] = save_column
        importance[i] = baseline[0] - m[0]
    return importance

importances = compute_importance(model, x_test, y_test_classes)

# Plot feature importance
feature_names = data.drop('Label', axis=1).columns
importance_df = pd.DataFrame(importances, index=feature_names, columns=['Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=importance_df.Importance, y=importance_df.index)
plt.title('Feature Importance')
plt.show()