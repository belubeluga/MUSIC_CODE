import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from pydub import AudioSegment
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write
""" ᯓ★ """

# Cargar el audio
audio_path = "/Users/belengotz/Desktop/MUSIC_CODE/bluenote-pentatonic-electric-guitar-melody_120bpm_A_major.wav"
y, sr = librosa.load(audio_path, sr=None)  # y: señal de audio, sr: frecuencia de muestreo



def distortion(y, gain=2.0):
    """
    ─ ⊹ ⊱ The Soundtrack to the Revolution ⊰ ⊹ ─
    
    Referencias: guitarras eléctricas o sintetizadores a lo Led Zeppelin, Metallica o Nirvana 🤘
    
    Aplica un efecto de distorsión a una señal de audio.

    PARÁMETROS:
        y (np.ndarray): Señal de audio original.
        gain (float): Factor de amplificación para generar la distorsión.

    RETORNO:
        np.ndarray: Señal de audio distorsionada.
    """
    y_distorted = gain * y
    y_distorted = np.clip(y_distorted, -1.0, 1.0) # clip para evitar SATURACIÓN EXTREMA
    return y_distorted

def reverb(y, sr, room_size=0.9, decay=0.9, wet_level=0.1):
    """
    ─ ⊹ ⊱ REVERB ⊰ ⊹ ─
    
    Referencias: atmósferas al estilo de U2 o efectos como los de Radiohead.
    (todavía en proceso)

    Añade reverberación a una señal de audio.

    Parámetros:
        y (np.ndarray): Señal de audio original.
        sr (int): Tasa de muestreo de la señal.
        room_size (float): Duración del impulso en segundos (tamaño de la sala).
        decay (float): Factor de decaimiento del impulso.
        wet_level (float): Proporción de mezcla de la reverberación.

    Retorno:
        np.ndarray: Señal de audio con reverberación aplicada.
    """
    impulse_duration = int(sr * room_size)
    impulse = np.power(np.linspace(1, decay, impulse_duration), 2)
    impulse *= np.hanning(impulse_duration) # suavizar con ventana de Hann
    impulse /= np.max(np.abs(impulse)) # normalizo
    
    y_reverb = convolve(y, impulse, mode='full')[:len(y)]
    output = (1 - wet_level) * y + wet_level * y_reverb
    output /= np.max(np.abs(output)) # normalizo

    return output

def echo(y, sr, delay=0.4, decay=0.5):
    """
    ─ ⊹ ⊱ ECHO echo echo echo ⊰ ⊹ ─

    Referencias: ambientes envolventes y texturas como las de "Echoes" de Pink Floyd, "With or Without You" de U2.

    Añade un efecto de eco a una señal de audio.

    Parámetros:
        y (np.ndarray): Señal de audio original.
        sr (int): Tasa de muestreo de la señal.
        delay (float): Tiempo de retraso del eco en segundos.
        decay (float): Factor de atenuación del eco.

    Retorno:
        np.ndarray: Señal de audio con el efecto de eco aplicado.
    """
    delay_samples = int(sr * delay)
    echo_signal = np.zeros(len(y) + delay_samples)
    echo_signal[:len(y)] = y
    echo_signal[delay_samples:] += decay * y
    return echo_signal[:len(y)]


def lowpass_filter(y, sr, cutoff=3000, order=6):
    """
    ─ ⊹ ⊱ MR CLEAN ⊰ ⊹ ─

    Referencias: afinar detalles, los Beatles o Fleetwood Mac.

    Aplica un filtro pasa-bajo a la señal para eliminar el ruido no deseado (de alta frecuencia).
    
    Parámetros:
        y (np.ndarray): Señal de audio original.
        sr (int): Tasa de muestreo de la señal.
        cutoff (float): Frecuencia de corte del filtro en Hz.
        order (int): Orden del filtro.
    
    Retorno:
        np.ndarray: Señal filtrada.
    """
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y_filtered = lfilter(b, a, y)
    y_filtered /= np.max(np.abs(y_filtered)) # noramlizo
    return y_filtered


y_distorted = distortion(y)
y_reverb = reverb(y, sr)
y_echo = echo(y, sr)
y_cleaned = lowpass_filter(y, sr)


# para visualizar
colors = ["#1e304d", "#4d4870", "#85689e", "#d38fd3", "#f8baed"]
#print(plt.style.available)
#plt.style.use("dark_background")

plt.figure(figsize=(12, 8))
plt.subplot(5, 1, 1)
plt.title("Original") #fontdict={'fontsize': 24, 'fontweight': 'bold', 'fontfamily': 'Roboto Slab'}
librosa.display.waveshow(y, sr=sr, color=colors[0])

plt.subplot(5, 1, 2)
plt.title("Distorsión")
librosa.display.waveshow(y_distorted, sr=sr, color=colors[1])

plt.subplot(5, 1, 3)
plt.title("Reverberación")
librosa.display.waveshow(y_reverb, sr=sr, color=colors[2])

plt.subplot(5, 1, 4)
plt.title("Eco")
librosa.display.waveshow(y_echo, sr=sr, color=colors[3])

plt.subplot(5, 1, 5)
plt.title("Limpieza (Filtro Pasabajos)")
librosa.display.waveshow(y_cleaned, sr=sr, color=colors[4])

plt.tight_layout()
plt.show()




# para guardar audio procesado
write("distorted_audio.wav", sr, (y_distorted * 32767).astype(np.int16))
write("reverb_audio.wav", sr, (y_reverb * 32767).astype(np.int16))
write("echo_audio.wav", sr, (y_echo * 32767).astype(np.int16))
write("cleaned_audio.wav", sr, (y_cleaned * 32767).astype(np.int16))
