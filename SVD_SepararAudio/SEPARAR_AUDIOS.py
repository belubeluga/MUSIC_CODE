import librosa
import numpy as np
import soundfile as sf

# Cargar el archivo de audio
y, sr = librosa.load("/Users/belengotz/Desktop/The Beatles - Hey Jude - TheBeatlesVEVO (youtube).mp3", sr=None)

# Aplicar STFT para obtener una matriz de tiempo-frecuencia
S = librosa.stft(y)

# Separar en componentes armónico (voz y sostenido) y percutivo (instrumentos)
S_harmonic, S_percussive = librosa.decompose.hpss(S)

# Crear máscaras binarizadas para enfatizar cada componente
mask_harmonic = np.abs(S_harmonic) > np.abs(S_percussive)
mask_percussive = np.abs(S_percussive) > np.abs(S_harmonic)

# Aplicar la máscara a cada componente
S_harmonic_cleaned = S_harmonic * mask_harmonic
S_percussive_cleaned = S_percussive * mask_percussive


# Normalizar para evitar saturación
S_harmonic_cleaned /= np.max(np.abs(S_harmonic_cleaned))
S_percussive_cleaned /= np.max(np.abs(S_percussive_cleaned))

# Reconstruir la señal de audio para la voz y los instrumentos
y_vox = librosa.istft(S_harmonic_cleaned)
y_instr = librosa.istft(S_percussive_cleaned)

# Aplicar un factor de ganancia (ajustable)
gain_factor = 5  # Incrementa o disminuye este valor para ajustar el volumen
y_vox = np.clip(y_vox * gain_factor, -1, 1)  # Limitar para evitar clipping
y_instr = np.clip(y_instr * gain_factor, -1, 1)

# Guardar los audios resultantes con volumen ajustado
sf.write("voz_separada_mejorada_con_volumen.wav", y_vox, sr)
sf.write("instrumentos_separados_mejorada_con_volumen.wav", y_instr, sr)