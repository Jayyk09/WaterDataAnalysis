import pandas as pandas
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler


path = "cityofchicago-beach-water-quality-automated-sensors/data.csv"

def get_data(path):
    data = pandas.read_csv(path)
    return data

data = get_data(path)

data_info = data.info()
data_head = data.head()

#clean the data by removing the rows with missing values

data['Measurement Timestamp'] = pandas.to_datetime(data['Measurement Timestamp'], errors='coerce')
data_cleaned = data.dropna(subset=['Measurement Timestamp'])

#fill the rest of the missing data with median values
for col in ['Transducer Depth', 'Wave Height', 'Wave Period', 'Turbidity']:
    data_cleaned[col].fillna(data_cleaned[col].median(), inplace=True)

#filter out wave height less than 0
data_cleaned = data_cleaned[data_cleaned['Wave Height'] > 0]

# data analysis and visualization
features = data_cleaned[['Beach Name', 'Water Temperature', 'Wave Height', 'Turbidity']]

label_encoder = LabelEncoder()
features['Beach Name'] = label_encoder.fit_transform(features['Beach Name'])

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features[['Beach Name', 'Water Temperature', 'Wave Height', 'Turbidity']])

# Apply K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
features['Cluster'] = kmeans.fit_predict(X_scaled)

# View the clusters
print(features[['Beach Name', 'Cluster']])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features[['Beach Name', 'Water Temperature', 'Wave Height', 'Turbidity']])
