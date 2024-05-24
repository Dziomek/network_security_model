import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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

# Załadowanie i agregacja danych, usunięcie zbędnych kolumn
data_frames = [pd.read_csv(path).rename(columns=lambda x: x.strip()) for path in file_paths]
data = pd.concat(data_frames, ignore_index=True)
data = data.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], axis=1)

features_array = data.columns.tolist()
features_array = features_array[:-1]

# Kodowanie etykiet oraz podział na cechy i etykiety
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])
x = data.drop('Label', axis=1)  # cechy
y = data['Label']  # etykiety

print("Number of classes before sampling: ")
print(data['Label'].value_counts())

x.replace([np.inf, -np.inf], np.nan, inplace=True)

# Postępowanie z elementami odstającymi
for column in x.columns:
    upper_limit = np.nanpercentile(x[column], 99)
    x[column] = np.where(x[column] > upper_limit, upper_limit, x[column])

# Strategia próbkowania
target_counts = {
    0: 300000,
    8: 10 * y.value_counts()[8],        # 1000% liczności klasy 8
    9: 10 * y.value_counts()[9],        # 1000% liczności klasy 9
    13: 10 * y.value_counts()[13]       # 1000% liczności klasy 13
}

# Tworzenie pipeline'ów dla imputacji, undersamplingu i oversamplingu
resample_strategy = {k: v for k, v in target_counts.items() if v < y.value_counts()[k]}
oversample_strategy = {k: v for k, v in target_counts.items() if v > y.value_counts()[k]}

pipeline = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('undersample', RandomUnderSampler(sampling_strategy=resample_strategy)),
    ('oversample', SMOTE(sampling_strategy=oversample_strategy, random_state=121214)),
])

# Zastosowanie pipeline'ów do danych
x_resampled, y_resampled = pipeline.fit_resample(x, y)

# Liczność klas po próbkowaniu
print("Number of classes after sampling: ")
print(pd.Series(y_resampled).value_counts())

# Podział danych zresamplowanych na zestaw treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.5, random_state=12421511, shuffle=True, stratify=y_resampled)

# Stworzenie i trenowanie modelu SVM z dobranymi hiperparametrami
svm_model = SVC(C=0.1, gamma=0.1, kernel='linear', random_state=4518656)
start_time = time.time()
print("Training strted...")

svm_model.fit(x_train, y_train)

# Timer end
end_time = time.time()
training_time = end_time - start_time

print("Training finished")
print(f"Czas treningu: {training_time:.2f} sekund")
print(f"Czas rozpoczęcia: {time.ctime(start_time)}")
print(f"Czas zakończenia: {time.ctime(end_time)}")

y_pred = svm_model.predict(x_test)
print("Dokładność:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
