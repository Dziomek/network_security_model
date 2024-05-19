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
import matplotlib.pyplot as plt
import seaborn as sns

file_paths = [
    'data_labelled/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    'data_labelled/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'data_labelled/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'data_labelled/Monday-WorkingHours.pcap_ISCX.csv',
    'data_labelled/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'data_labelled/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'data_labelled/Tuesday-WorkingHours.pcap_ISCX.csv',
    'data_labelled/Wednesday-workingHours.pcap_ISCX.csv'
]

#Załadowanie i agregacja danych, usunięcie zbędnych kolumn
data_frames = [pd.read_csv(path).rename(columns=lambda x: x.strip()) for path in file_paths]
data = pd.concat(data_frames, ignore_index=True)
data = data.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Source Port', 'Destination Port'], axis=1)

features_array = data.columns.tolist()
features_array = features_array[:-1]

#Kodowanie etykiet oraz podział na cechy i etykiety
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])
x = data.drop('Label', axis=1)  # cechy
y = data['Label']  # etykiety

print("Liczność klas przed próbkowaniem:")
print(data['Label'].value_counts())

x.replace([np.inf, -np.inf], np.nan, inplace=True)

#Postępowanie z elementami odstającymi
for column in x.columns:
    upper_limit = np.nanpercentile(x[column], 99)
    x[column] = np.where(x[column] > upper_limit, upper_limit, x[column])

#Strategia próbkowania
target_counts = {0: 60000}
for label in np.unique(y):
    if label != 0:
        target_counts[label] = 12000

# Tworzenie pipeline'ów dla imputacji, undersamplingu i oversamplingu
resample_strategy = {k: v for k, v in target_counts.items() if v < y.value_counts()[k]}
oversample_strategy = {k: v for k, v in target_counts.items() if v > y.value_counts()[k]}

pipeline = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('undersample', RandomUnderSampler(sampling_strategy=resample_strategy)),
    ('oversample', SMOTE(sampling_strategy=oversample_strategy, random_state=121214))
])

# Zastosowanie pipeline'ów do danych
x_resampled, y_resampled = pipeline.fit_resample(x, y)

# Liczność klas po próbkowaniu
print("Liczność klas po próbkowaniu:")
print(pd.Series(y_resampled).value_counts())

# Podział na zestaw treningowy i testowy
# x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=50, shuffle=True, stratify=y_resampled)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=12421511, shuffle=True, stratify=y)

# Konwersja wszystkich kolumn na typ float
x_train = x_train.astype(np.float32)
print(x_train)
x_test = x_test.astype(np.float32)
# Konwersja x_train i x_test z numpy.ndarray na pandas DataFrame
x_train_df = pd.DataFrame(x_train, columns=features_array)
x_test_df = pd.DataFrame(x_test, columns=features_array)

start_time = time.time()
print("Rozpoczęcie treningu modelu...")

model = RandomForestClassifier(n_estimators=100, random_state=2425252)
model.fit(x_resampled, y_resampled)

# Timer end
end_time = time.time()
training_time = end_time - start_time

print("Trening zakończony.")
print(f"Czas treningu: {training_time:.2f} sekund")
print(f"Czas rozpoczęcia: {time.ctime(start_time)}")
print(f"Czas zakończenia: {time.ctime(end_time)}")

y_pred = model.predict(x_test)
print("Dokładność:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Po trenowaniu modelu, uzyskaj ważności cech
feature_importances = model.feature_importances_
print(len(feature_importances), len(x_train_df.columns))
# Przygotuj DataFrame do wizualizacji
features_df = pd.DataFrame({
    'Feature': x_train_df.columns,  # x_train_df to DataFrame Twoich cech
    'Importance': feature_importances
})

# Sortuj cechy według ważności
features_df = features_df.sort_values(by='Importance', ascending=False)

# Wizualizacja ważności cech
top_features_df = features_df.head(10)

# Wizualizacja ważności 10 najważniejszych cech
plt.figure(figsize=(10, 8))
sns.barplot(data=top_features_df, x='Importance', y='Feature')
plt.title('10 najważniejszych cech w modelu Random Forest')
plt.xlabel('Waga')
plt.ylabel('Cecha')
plt.show()
