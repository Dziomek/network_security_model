import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sampling_strategy import x_resampled, y_resampled
import seaborn as sns
import matplotlib.pyplot as plt

# Podział danych zresamplowanych na zestaw treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.5, random_state=955818, shuffle=True, stratify=y_resampled)

# Konwersja wszystkich kolumn na typ float
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Inicjalizacja klasyfikatora RandomForestClassifier z domyślnymi hiperparametrami
rf = RandomForestClassifier(random_state=585665, min_samples_split=5)

# Dopasowanie GridSearchCV do danych treningowych
start_time = time.time()
print("Training started...")

rf.fit(x_train, y_train)

# Timer end
end_time = time.time()
training_time = end_time - start_time

print("Training finished")
print(f"Czas treningu: {training_time:.2f} sekund")

# Prognozy na zbiorze testowym przy użyciu wytrenowanego modelu
y_pred = rf.predict(x_test)

print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Obliczanie i wyświetlanie macierzy pomyłek
conf_matrix = confusion_matrix(y_test, y_pred)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Znaczenie cech
feature_importances = rf.feature_importances_

# Tworzenie DataFrame z cechami i ich znaczeniami
features_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': feature_importances})

# Sortowanie według znaczenia cech
features_df = features_df.sort_values(by='Importance', ascending=False)

# Wyświetlenie najważniejszych cech
print("Feature Importances:")
print(features_df)

# Wizualizacja znaczenia cech
plt.figure(figsize=(10, 7))
sns.barplot(x=features_df['Importance'], y=features_df['Feature'])
plt.title('Feature Importances')
plt.show()