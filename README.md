01:43 ━━━━●───── 03:50

=======
 ⇆ㅤ ㅤ◁ㅤ ❚❚ ㅤ▷ ㅤㅤ↻

### ABOUT ☕︎
Endless creative explorations

In my journey to blend music and mathematics—my two greatest passions—this repository hosts my personal experiments. From ignorance I explore, try, fail, and occasionally succeed in coding tools and techniques that mix music, linear algebra, signal processing, numerical methods, and machine learning.

### ᯓ★ CONTENT
• **Classification** – Algorithms that identify musical genres or patterns.
• **SVD** – Audio separation and compression.
• **Filters** – Equalizers and audio-effects design.

### ᯓ★ OBJECTIVE
Explore the infinite possibilities where music meets science. Whether it is compressing signals, classifying genres, or designing filters that reshape sonic perception, this repository is conceived as a creative laboratory for innovation and learning.

### ᯓ★ TECHY STUFF
Language: Python  
Libraries: `numpy`, `scipy`, `librosa`, `matplotlib`, `scikit-learn`, `tensorflow`, `pydub`, `soundfile`, `joblib`  
Workspace: Jupyter Notebooks for interactive visualization & experimentation.

---

## 🗂️ Repository Map & How-To-Run

| Folder / File | What it does | How to try it |
|--------------|-------------|---------------|
| `CLASIFICAR/` | Traditional machine-learning genre classifiers.  | `python CLASIFICAR/clasificar.py` (Random Forest). Make sure `path` inside the script points to the GTZAN dataset or any folder with `*.wav` files organised by genre. |
| `CLASIFICAR/REDES_NEURONALES?/` | Neural-network based genre classifier using Keras/TensorFlow. | `python CLASIFICAR/REDES_NEURONALES?/clasificar_redes_neuronales.py` after setting the same `path` variable. |
| `SVD_SepararAudio/` | Vocal / instruments separation via Short-Time Fourier Transform (STFT) + Singular Value Decomposition (SVD). Generates separate `.wav` files with adjustable gain. | 1) Place target audio inside the folder. 2) Edit the filename in `SVDmusica.py`. 3) `python SVD_SepararAudio/SVDmusica.py`. |
| `FILTROS/EQ/` | A toolbox of creative audio effects: **distortion**, **reverb**, **echo**, and **low-pass cleaning**. Demonstrated in `EQ1.py`, it loads a guitar riff and writes processed variants. | `python FILTROS/EQ/EQ1.py`; tweak parameters such as `gain`, `room_size`, `delay`, or filter `cutoff` to taste. |
| Notebook `SVDmusicaGRAFICO.ipynb` | Visual explanation of the SVD separation process with plots. | Open with Jupyter Lab / VSCode and run all cells. |

### Running the examples
1. Create a Python virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install numpy scipy librosa matplotlib scikit-learn tensorflow pydub soundfile joblib
   ```
3. Execute the desired script as indicated in the table above.

### Contributing / Feedback
Pull-requests, issues, and ideas are welcome! If you find a bug or have an improvement in mind, feel free to open an issue.

---

> “Mathematics is the music of reason, and music is the mathematics of emotion.”

```
_░▒███████
░██▓▒░░▒▓██
██▓▒░__░▒▓██___██████
██▓▒░____░▓███▓__░▒▓██
██▓▒░___░▓██▓_____░▒▓██
██▓▒░_______________░▒▓██
_██▓▒░______________░▒▓██
__██▓▒░____________░▒▓██
___██▓▒░__________░▒▓██
____██▓▒░________░▒▓██
_____██▓▒░_____░▒▓██
______██▓▒░__░▒▓██
_______█▓▒░░▒▓██
_________░▒▓██
_______░▒▓██
_____░▒▓██


▒▒▒▒▒▒▒▒▒▒ 100% ᴄᴏᴍᴘʟᴇᴛᴇ!

```




## -------------------------------------------

### SOBRE ☕︎
Exploraciones Creativas sin fin alguno

En la búsqueda de fusionar la música y las matemáticas, mis dos pasiones principales, presento en este código mis experimentos creativos como motivación personal para no dejar la carrera e ir a dedicarme solamente a la música :) 
Desde mi ignorancia exploro, intento desarrollar cosas y varias veces fracaso, pero muy cada tanto logro codear herramientas y técnicas que combinan música, álgebra lineal, procesamiento de señales, métodos numéricos y aprendizaje automático.

▶︎·၊၊||၊|။||||၊၊||၊|။||||၊၊||၊|။||||။·၊၊||၊|။||||။||၊|။||||၊၊||၊|။||||၊၊||၊|။||||။·၊၊||၊|။||||။||၊|။||||၊၊||၊|။||||၊၊||၊|။|

### ᯓ★ CONTENIDO 
Clasificación: Algoritmos para identificar géneros o patrones musicales.
SVD: Separación y compresión de audio.
Filtros: Creación de ecualizadores y efectos de sonido.

### ᯓ★ OBJETIVO
Explorar las posibilidades infinitas donde la música y la ciencia se encuentran. 
Ya sea comprimiendo señales, clasificando géneros o diseñando filtros que alteren la percepción sonora, este repositorio está pensado como un laboratorio creativo para innovar y aprender.

### ᯓ★ TECHY STUFF
Lenguaje: Python
Bibliotecas: numpy, scipy, librosa, matplotlib, sklearn, tensorflow
Frameworks: Jupyter Notebooks para la visualización y experimentación interactiva.


