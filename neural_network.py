import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt
from sampling_strategy import x_resampled, y_resampled

# Podział danych zresamplowanych na zestaw treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.5, random_state=955818, shuffle=True, stratify=y_resampled)

# Konwersja wszystkich kolumn na typ float
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Konwersja etykiet do postaci binarnej (one-hot encoding)
num_classes = 15  # Adjusted number of classes to 15
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Inicjalizacja modelu sieci neuronowej
model = Sequential()

# Dodawanie warstw do modelu
model.add(Dense(128, input_shape=(x_train.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trenowanie modelu
start_time = time.time()
print("Training started...")

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=2)

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

# Obliczanie i wyświetlanie macierzy pomyłek
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Wizualizacja historii treningu
plt.figure(figsize=(10, 7))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Model Accuracy')
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Model Loss')
plt.show()
