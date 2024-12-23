import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

structured_data = pd.read_csv('./data/sensor_data.csv', sep=',')

def load_spectral_data(sample_id):
    filename = f'./data/spectral_data/s{int(sample_id)}.csv'
    try:
        spectral_data = pd.read_csv(filename, sep=';', header=None, on_bad_lines='skip', usecols=range(1, 553))
        pixels = spectral_data.iloc[0].values
        reflection = pd.to_numeric(spectral_data.iloc[1].values, errors='coerce')
        absorption = pd.to_numeric(spectral_data.iloc[2].values, errors='coerce')
        return pixels, reflection, absorption
    except Exception as e:
        print(f"Error reading spectral data for sample ID {sample_id}: {e}")
        return None, None, None

combined_data = []

for index, row in structured_data.iterrows():
    sample_id = row['Sample ID']
    pixels, reflection, absorption = load_spectral_data(sample_id)
    
    if reflection is not None and absorption is not None:
        reflection_mean = reflection.mean()
        reflection_std = reflection.std()
        reflection_min = reflection.min()
        reflection_max = reflection.max()

        absorption_mean = absorption.mean()
        absorption_std = absorption.std()
        absorption_min = absorption.min()
        absorption_max = absorption.max()
        
        combined_data.append({
            'Sample ID': sample_id,
            'Tempreture': row['Tempreture'],
            'Moisture': row['Moisture'],
            'ph': row['ph'],
            'Quality': row['Quality'],
            'Reflection_Mean': reflection_mean,
            'Reflection_STD': reflection_std,
            'Reflection_Min': reflection_min,
            'Reflection_Max': reflection_max,
            'Absorption_Mean': absorption_mean,
            'Absorption_STD': absorption_std,
            'Absorption_Min': absorption_min,
            'Absorption_Max': absorption_max
        })

combined_df = pd.DataFrame(combined_data)
print(combined_df)

label_encoder = LabelEncoder()
combined_df['Quality'] = label_encoder.fit_transform(combined_df['Quality'])

print(combined_df['Quality'].value_counts())

X = combined_df.drop(columns=['Sample ID', 'Quality'])
y = combined_df['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_, labels=label_encoder.transform(label_encoder.classes_)))
