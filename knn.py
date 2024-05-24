import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sampling_strategy import x_resampled, y_resampled
import seaborn as sns
import matplotlib.pyplot as plt

# Podział danych zresamplowanych na zestaw treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=955818, shuffle=True, stratify=y_resampled)

# Konwersja wszystkich kolumn na typ float
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Definiowanie siatki hiperparametrów
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Inicjalizacja klasyfikatora KNeighborsClassifier
knn = KNeighborsClassifier()

# Inicjalizacja GridSearchCV z 5-krotną walidacją krzyżową
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Dopasowanie GridSearchCV do danych treningowych
start_time = time.time()
print("Grid search started...")

grid_search.fit(x_train, y_train)

# Timer end
end_time = time.time()
grid_search_time = end_time - start_time

print("Grid search finished")
print(f"Time: {grid_search_time:.2f} s")

# Wyświetlenie najlepszych hiperparametrów
best_params = grid_search.best_params_
print("Best parameters found by grid search:")
print(best_params)

# Wytrenowanie modelu KNN z najlepszymi hiperparametrami na pełnym zestawie treningowym
best_knn = grid_search.best_estimator_

# Prognozy na zbiorze testowym przy użyciu najlepszego modelu KNN
y_pred_best_knn = best_knn.predict(x_test)

print("Accuracy on test set with best KNN:", accuracy_score(y_test, y_pred_best_knn))
print(classification_report(y_test, y_pred_best_knn))

# Obliczanie i wyświetlanie macierzy pomyłek dla najlepszego modelu KNN
conf_matrix_best_knn = confusion_matrix(y_test, y_pred_best_knn)

# Wizualizacja macierzy pomyłek dla najlepszego modelu KNN
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_best_knn, annot=True, fmt='d', cmap='Blues', xticklabels=best_knn.classes_, yticklabels=best_knn.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Best KNN')
plt.show()