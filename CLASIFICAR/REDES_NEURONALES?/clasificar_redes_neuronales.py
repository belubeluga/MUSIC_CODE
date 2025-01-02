import os
import librosa
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

path = "/Users/belengotz/Desktop/Data/genres_original"
print("Path to dataset files:", path)

#EXTRAER CARACT.
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None
    
    # Extraer características del audio
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    rmse = librosa.feature.rms(y=y).mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfccs = [np.mean(coeff) for coeff in mfcc]  # Promedio de cada MFCC
    
    # Retorna las características como un vector
    return [
        chroma_stft, rmse, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, *mfccs
    ]

#según el dataset
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

#EXTRACCIÓN CARACT. Y ETIQUETAS
def process_genre(genre):
    genre_path = os.path.join(path, genre)
    features = []
    labels = []
    
    # Verificar que genre_path es un directorio
    if not os.path.isdir(genre_path):
        print(f"Error: {genre_path} no es un directorio válido.")
        return features, labels
    
    for file in os.listdir(genre_path):
        if file.endswith('.wav'):
            file_path = os.path.join(genre_path, file)
            print(f"Extrayendo características de {file_path}")
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(genre)
    return features, labels


# Paralelizar la extracción de características
results = Parallel(n_jobs=-1)(delayed(process_genre)(genre) for genre in genres)

# Unir los resultados
X = []
y = []
for features, labels in results:
    X.extend(features)
    y.extend(labels)

if len(X) == 0 or len(y) == 0:
    print("Error: No se extrajeron características de los audios. Verifica la estructura del dataset.")
    exit()

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Codificar las etiquetas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo de red neuronal
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(genres), activation='softmax'))

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("Accuracy:", accuracy_score(y_test_labels, y_pred_labels))
print("Classification Report:")
print(classification_report(y_test_labels, y_pred_labels, target_names=genres))

# Función para clasificar un archivo de audio específico
def classify_audio_nn(file_path, model, scaler, label_encoder):
    feature = extract_features(file_path)
    if feature is not None:
        feature = np.array(feature).reshape(1, -1)
        feature = scaler.transform(feature)  # Normalizar
        prediction = model.predict(feature)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_label[0]
    else:
        return None

# Clasificar el archivo de audio específico
audio_file = "/Users/belengotz/Desktop/MUSIC_CODE/CLASIFICAR/NAVIDADcancion.wav"
predicted_genre = classify_audio_nn(audio_file, model, scaler, label_encoder)
if predicted_genre:
    print(f"El género predicho para {audio_file} es: {predicted_genre}")
else:
    print(f"No se pudo clasificar el archivo {audio_file}")