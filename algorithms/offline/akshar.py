import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, mean_squared_error


df = pd.read_csv("/home/mgoyani/CORL/updated_data_with_height.csv")

df.dropna(subset=['behavior_statement'], inplace=True)

continuous_columns = [
    'heart_caloriesOut', 'heart_max', 'heart_min', 'heart_avg',
    'restingHeartRate', 'spo2_avg', 'spo2_min', 'spo2_max',
    'sleep_duration', 'sleep_efficiency', 'minutesAsleep',
    'minutesAwake', 'timeInBed', 'BMI_index', 'stress_level'
]

scaler = MinMaxScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

df = df[df['gender'].isin(['Male', 'Female'])]
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(int)

le = LabelEncoder()
df['behavior_statement'] = le.fit_transform(df['behavior_statement'])

X = df.drop(columns=['behavior_statement', 'behavior_status', 'start_time', 'stop_time', 'user_id'])


def calculate_risk(row):
    if row['stress_level'] > 0.7 and row['heart_max'] > 0.8:
        return 'high'
    elif row['stress_level'] > 0.4:
        return 'moderate'
    else:
        return 'low'

df['risk_level'] = df.apply(calculate_risk, axis=1)
df['risk_level'] = le.fit_transform(df['risk_level'])

X_train, X_val, y_risk_train, y_risk_val = train_test_split(
    X, df['risk_level'], test_size=0.2, random_state=42
)

input_dim = X_train.shape[1]

# Define MLP for Risk Prediction
mlp_input = layers.Input(shape=(input_dim,))
dense1 = layers.Dense(64, activation='relu')(mlp_input)
dense2 = layers.Dense(32, activation='relu')(dense1)
output = layers.Dense(3, activation='softmax')(dense2)

risk_model = models.Model(inputs=mlp_input, outputs=output)
risk_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


risk_model.fit(X_train, y_risk_train, epochs=20, batch_size=32, validation_data=(X_val, y_risk_val))

y_pred = np.argmax(risk_model.predict(X_val), axis=1)
print(classification_report(y_risk_val, y_pred))

import joblib

joblib.dump(risk_model, "/home/mgoyani/CORL/risk_model.pkl")
loaded_model_pkl = joblib.load("/home/mgoyani/CORL/risk_model.pkl")
#
# sample_input = np.array([0.5, 0.6, 0.4, 0.7, 0.5, 0.8, 0.7, 0.6, 0.6, 0.7, 0.5, 0.6, 0.7, 0.5, 0.8])
#
#
# predicted_risk = np.argmax(loaded_model_pkl.predict(sample_input), axis=1)
# print(f"Predicted Risk Level: {predicted_risk}")
#

