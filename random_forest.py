import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

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
print(len(data))

# Usunięcie niepotrzebnych kolumn
data = data.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], axis=1)

# Kodowanie etykiet (zamiana na liczby całkowite), bo domyślnie są zapisane plaintextem
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])

# Zapisujemy mapowanie odwrotne dla etykiet (do odczytywania potem błędnych wyników w CSV)
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
print(label_mapping)

# Podział na cechy i etykiety
x = data.drop('Label', axis=1) # cechy
y = data['Label'] # etykiety

# Podział na zestaw treningowy i testowy
# shuffle służy do losowego mieszania danych podczas dzielenia na zbiór treningowy/testowy 
# random_state wybiera losowy seed, w zależności od niego wyniki są różne
# stratify zapewnia sprawiedliwy podział danych! Wymaga podania etykiet (y)
# chodzi o to, żeby w zbiorze treningowym i testowym była podobna propocja danych tego samego typu (o tych samych etykietach)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50, shuffle=True, stratify=y)

# Zamiana wartości nieskończonych na NaN
x_train.replace([np.inf, -np.inf], np.nan, inplace=True)
x_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Sprawdzenie, czy występują wartości NaN i ich liczba
print("Liczba wartości NaN w danych treningowych:", x_train.isna().sum().sum())
print("Liczba wartości NaN w danych testowych:", x_test.isna().sum().sum())

# Imputacja - wypełnienie brakujących wartości, np. medianą lub średnią
# mediana chyba lepsza niż średnia ze względu na wartości odstające
x_train.fillna(x_train.median(), inplace=True)
x_test.fillna(x_test.median(), inplace=True)

# Konwersja wszystkich kolumn na typ float
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

# Timer start
start_time = time.time()
print("Rozpoczęcie treningu modelu...")

# Tworzenie i trenowanie modelu lasu losowego
model = RandomForestClassifier(n_estimators=2, random_state=10)
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

# Wyszukanie błędnie sklasyfikowanych rekordów
errors_index = results[results['Actual'] != results['Predicted']].index

# Wybieranie błędnie sklasyfikowanych rekordów z oryginalnego zbioru danych
error_records = data.loc[errors_index]

# Zamiana liczbowych etykiet z powrotem na oryginalne etykiety
error_records['Actual Labels'] = [label_mapping[label] for label in results.loc[errors_index, 'Actual']]
error_records['Predicted Labels'] = [label_mapping[label] for label in results.loc[errors_index, 'Predicted']]

# Zapisywanie błędnie sklasyfikowanych rekordów do pliku CSV
error_records.to_csv('wrong_results.csv', index=True)
