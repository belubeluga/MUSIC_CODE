import librosa
import numpy as np
import soundfile as sf

# Cargar el archivo de audio
y, sr = librosa.load("/Users/belengotz/Desktop/The Beatles - Hey Jude - TheBeatlesVEVO (youtube).mp3", sr=None)

# Aplicar STFT para obtener una matriz de tiempo-frecuencia
S = librosa.stft(y)

# Separar en componentes armónico (voz) y percutivo (instrumentos)
S_harmonic, S_percussive = librosa.decompose.hpss(S)

# Tomar la magnitud del componente armónico para aplicar SVD
S_magnitude = np.abs(S_harmonic)

# Aplicar SVD a la matriz de magnitudes del componente armónico
U, Sigma, Vt = np.linalg.svd(S_magnitude, full_matrices=False)

# Seleccionar los primeros componentes para reconstruir la voz
k = 5  # Ajusta el número de componentes para aislar la voz
S_vox_magnitude = U[:, :k] @ np.diag(Sigma[:k]) @ Vt[:k, :]

# Reconstruir la señal del componente de voz con la fase original
S_vox = S_vox_magnitude * np.exp(1j * np.angle(S_harmonic))

# Reconstruir las señales de audio de voz y percusión usando la inversa de STFT
y_vox = librosa.istft(S_vox)
y_instr = librosa.istft(S_percussive)

# Aplicar un factor de ganancia (ajustable) para incrementar el volumen
gain_factor = 5  # Ajusta este valor para cambiar el volumen
y_vox = np.clip(y_vox * gain_factor, -1, 1)
y_instr = np.clip(y_instr * gain_factor, -1, 1)

# Guardar los audios resultantes con volumen ajustado
sf.write("voz_separada_svd_con_volumen.wav", y_vox, sr)
sf.write("instrumentos_svd_separados_con_volumen.wav", y_instr, sr)
