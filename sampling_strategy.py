from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from data_preprocessing import x, y
import numpy as np
import pandas as pd

# Sampling strategy
target_counts = {
    0: 300000,
    8: 10 * y.value_counts()[8],
    9: 10 * y.value_counts()[9],        
    13: 10 * y.value_counts()[13]       
}
resample_strategy = {k: v for k, v in target_counts.items() if v < y.value_counts()[k]}
oversample_strategy = {k: v for k, v in target_counts.items() if v > y.value_counts()[k]}
# Create pipelines for imputing, undersampling and oversampling
pipeline = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
    ('undersample', RandomUnderSampler(sampling_strategy=resample_strategy, random_state=7518525)),
    ('oversample', SMOTE(sampling_strategy=oversample_strategy, random_state=121214))
])
# Fit pipeline to data (sample)
x_resampled, y_resampled = pipeline.fit_resample(x, y)
# Debugging: Number of each class records after sampling
class_counts = pd.Series(y_resampled).value_counts()
# Sumaryczna liczba wszystkich danych
total_records_resampled = class_counts.sum()
print(f"Total number of records after resampling: {total_records_resampled}")
print("Number of each class records after sampling: ")
print(class_counts)