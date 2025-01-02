import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import ffmpeg
from pydub import AudioSegment

# Ruta al dataset
path = "/Users/belengotz/Desktop/Data/genres_original"
print("Path to dataset files:", path)

# Función para extraer características del audio
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

# Lista de géneros según el dataset
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Extracción de características y etiquetas
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

# Verificar que X y y no están vacíos
if len(X) == 0 or len(y) == 0:
    print("Error: No se extrajeron características de los audios. Verifica la estructura del dataset.")
    exit()

# Convertir listas a arrays de numpy
X = np.array(X)
y = np.array(y)

# Normalizar las características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Entrenar el clasificador
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predecir en los datos de prueba
y_pred = classifier.predict(X_test)

# Evaluar el modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))



# Función para clasificar un archivo de audio específico
def classify_audio(file_path):
    feature = extract_features(file_path)
    if feature is not None:
        feature = np.array(feature).reshape(1, -1)
        feature = scaler.transform(feature)
        prediction = classifier.predict(feature)
        return prediction[0]
    else:
        return None

# Clasificar el archivo de audio específico
