import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import joblib
from flask import Flask, request, jsonify

# Inisialisasi Flask
app = Flask(__name__)

# Baca data dari file CSV
data = pd.read_csv('train.csv')  

data_olah = data[['femaleres', 'age', 'married', 'children', 'hhsize', 'edu',
"day_of_week", "saved_mpesa", "received_mpesa", "given_mpesa", "ent_wagelabor", "ent_ownfarm", "ent_business", "ent_nonagbusiness", "depressed"
]]

def check_outliers(data_olah, mean, std, threshold=3):
    z_scores = (data_olah - mean)/std
    return np.abs(z_scores)> threshold
r_outliers = set()
for i in data_olah:
    mean = np.mean(data_olah[i])
    std = np.std(data_olah[i])
    for j in range(data_olah.shape[0]):
        r = data_olah.iloc[j][i]
        otl = check_outliers(r, mean, std)
        if otl:
            r_outliers.add(j)
data_olah = data_olah.drop(r_outliers, axis=0)
print(len(r_outliers))
print(data_olah.shape)
data_olah.head

# Ambil fitur (data_x) dan target (data_y)
data_x = data_olah[['femaleres', 'age', 'married', 'children', 'hhsize', 'edu',
"day_of_week", "saved_mpesa", "received_mpesa", "given_mpesa", "ent_wagelabor", "ent_ownfarm", "ent_business", "ent_nonagbusiness"
]]
data_y = data_olah['depressed']

# Skalakan fitur menggunakan MinMaxScaler
scaler = MinMaxScaler()
data_x_scaled = scaler.fit_transform(data_x)

# Inisialisasi model Naive Bayes
nb_model = GaussianNB()
nb_model.fit(data_x_scaled, data_y)

# Simpan model dan scaler ke file
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Tambahkan rute utama
@app.route('/')
def home():
    return "API Depresi Flask aktif. Gunakan endpoint /predict untuk prediksi."

@app.route('/predict', methods=['POST'])
def predict():
    # Dapatkan data dari permintaan
    data = request.json['data']
    data = np.array(data).reshape(1, -1)

    # Muat model dan scaler dari file
    nb_model = joblib.load('nb_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Skalakan data baru
    data_scaled = scaler.transform(data)

    # Prediksi probabilitas
    predicted_proba = nb_model.predict_proba(data_scaled)

    # Kembalikan hasil sebagai JSON
    return jsonify({
        'predicted_proba_class_0': predicted_proba[0][0],
        'predicted_proba_class_1': predicted_proba[0][1]
    })

if __name__ == '__main__':
    app.run(debug=True)
