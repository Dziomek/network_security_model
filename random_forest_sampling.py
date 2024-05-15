import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Słownik ścieżek do plików
file_paths = [
    'data/friday_ddos.csv',
    'data/wednesday_dos.csv',
    'data/tuesday_patators.csv',
    'data/friday_port_scan.csv',
    'data/monday_normal.csv',
    'data/thursday_infiltration.csv',
    'data/thursday_brf_xss_sqlin.csv',
    'data/friday_bot.csv'
]

# Wczytanie danych i czyszczenie nazw kolumn
data_frames = [pd.read_csv(path).rename(columns=lambda x: x.strip()) for path in file_paths]

# Agregacja danych
data = pd.concat(data_frames, ignore_index=True)

# Usunięcie niepotrzebnych kolumn
data = data.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], axis=1)

# Sprawdzenie duplikatów przed przetwarzaniem
print(f"Liczba duplikatów przed przetwarzaniem: {data.duplicated().sum()}")

# Kodowanie etykiet (zamiana na liczby całkowite), bo domyślnie są zapisane plaintextem
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])

# Liczność klas przed próbkowaniem
print("Liczność klas przed próbkowaniem:")
print(data['Label'].value_counts())

# Podział na cechy i etykiety
x = data.drop('Label', axis=1)  # cechy
y = data['Label']  # etykiety

# Usunięcie nieskończonych wartości
x.replace([np.inf, -np.inf], np.nan, inplace=True)

# Sprawdzenie i ograniczenie zbyt dużych wartości
# Zakładamy, że wartości powinny być ograniczone do sensownego zakresu, np. percentyle 99%
for column in x.columns:
    upper_limit = np.nanpercentile(x[column], 99)
    x[column] = np.where(x[column] > upper_limit, upper_limit, x[column])

# Tworzenie słownika z docelową licznością klas
target_counts = {0: 60000}
for label in np.unique(y):
    if label != 0:
        target_counts[label] = 12000

# Tworzenie pipeline'ów dla imputacji, undersamplingu i oversamplingu
resample_strategy = {k: v for k, v in target_counts.items() if v < y.value_counts()[k]}
oversample_strategy = {k: v for k, v in target_counts.items() if v > y.value_counts()[k]}

pipeline = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),  # dodanie imputacji
    ('undersample', RandomUnderSampler(sampling_strategy=resample_strategy)),
    ('oversample', SMOTE(sampling_strategy=oversample_strategy, random_state=42))
])

# Zastosowanie pipeline'ów do danych
x_resampled, y_resampled = pipeline.fit_resample(x, y)

# Liczność klas po próbkowaniu
print("Liczność klas po próbkowaniu:")
print(pd.Series(y_resampled).value_counts())

# Podział na zestaw treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=50, shuffle=True, stratify=y_resampled)

# Konwersja wszystkich kolumn na typ float
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Konwersja x_train i x_test z numpy.ndarray na pandas DataFrame
x_train_df = pd.DataFrame(x_train, columns=[f"feature_{i}" for i in range(x_train.shape[1])])
x_test_df = pd.DataFrame(x_test, columns=[f"feature_{i}" for i in range(x_test.shape[1])])

# Sprawdzenie duplikatów w zbiorze treningowym
dup_train = x_train_df.duplicated().sum()
print(f"Liczba duplikatów w zbiorze treningowym: {dup_train}")

# Sprawdzenie duplikatów w zbiorze testowym
dup_test = x_test_df.duplicated().sum()
print(f"Liczba duplikatów w zbiorze testowym: {dup_test}")

# Timer start
start_time = time.time()
print("Rozpoczęcie treningu modelu...")

# Tworzenie i trenowanie modelu lasu losowego
model = RandomForestClassifier(n_estimators=100, random_state=10)
model.fit(x_train, y_train)  # Trening modelu

# Timer end
end_time = time.time()
training_time = end_time - start_time

print("Trening zakończony.")
print(f"Czas treningu: {training_time:.2f} sekund")
print(f"Czas rozpoczęcia: {time.ctime(start_time)}")
print(f"Czas zakończenia: {time.ctime(end_time)}")

# Ocena modelu
y_pred = model.predict(x_test)
print("Dokładność:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Tworzenie DataFrame z prawdziwymi i przewidywanymi etykietami
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=y_test.index)

# # Wyszukanie błędnie sklasyfikowanych rekordów
# errors_index = results[results['Actual'] != results['Predicted']].index

# # Wybieranie błędnie sklasyfikowanych rekordów z oryginalnego zbioru danych
# error_records = data.loc[errors_index]

# # Zamiana liczbowych etykiet z powrotem na oryginalne etykiety
# error_records['Actual Labels'] = [label_mapping[label] for label in results.loc[errors_index, 'Actual']]
# error_records['Predicted Labels'] = [label_mapping[label] for label in results.loc[errors_index, 'Predicted']]

# # Zapisywanie błędnie sklasyfikowanych rekordów do pliku CSV
# error_records.to_csv('wrong_results.csv', index=True)
