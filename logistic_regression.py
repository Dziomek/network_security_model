from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import time
from sampling_strategy import x_resampled, y_resampled
import seaborn as sns
import matplotlib.pyplot as plt

# Podział danych na zbiory treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=12421511, shuffle=True, stratify=y_resampled)

# Konwersja wszystkich kolumn na typ float
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Inicjalizacja klasyfikatora LogisticRegression z ustalonymi hiperparametrami
log_reg = LogisticRegression(random_state=12521521, max_iter=100)

# Trening modelu
start_time = time.time()
print("Training started...")

log_reg.fit(x_train, y_train)

# Timer end
end_time = time.time()
training_time = end_time - start_time

print("Training finished")
print(f"Czas treningu: {training_time:.2f} sekund")
print(f"Czas rozpoczęcia: {time.ctime(start_time)}")
print(f"Czas zakończenia: {time.ctime(end_time)}")

# Prognozy na zbiorze testowym przy użyciu wytrenowanego modelu
y_pred = log_reg.predict(x_test)

print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Obliczanie i wyświetlanie macierzy pomyłek
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=log_reg.classes_, yticklabels=log_reg.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
