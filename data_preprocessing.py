import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the data
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
data_frames = [pd.read_csv(path).rename(columns=lambda x: x.strip()) for path in file_paths]

# Aggregation
data = pd.concat(data_frames, ignore_index=True)

# Delete unnecessary columns
data = data.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], axis=1)
# Encode all the labels, divide into features and labels
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])
x = data.drop('Label', axis=1)  # features
y = data['Label']  # labels
# Count inf and -inf values before any processing
inf_values = np.isinf(data).sum().sum()
neg_inf_values = np.isneginf(data).sum().sum()
# Output the counts
print(f"Number of inf values: {inf_values}")
print(f"Number of -inf values: {neg_inf_values}")
# Replace inf and -inf values with NaN to handle this data later
x.replace([np.inf, -np.inf], np.nan, inplace=True)
# Replace outliers values with value of 99 percentile of each column
for column in x.columns:
    upper_limit = np.nanpercentile(x[column], 99)
    x[column] = np.where(x[column] > upper_limit, upper_limit, x[column])
# Debugging
total_records = data.shape[0]
print(f"Total number of records: {total_records}")
print("Number of each class records before sampling: ")
print(data['Label'].value_counts())


