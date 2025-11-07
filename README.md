# 🎵 Music Genre Classification App

An interactive **Music Genre Classification** desktop application built using **Tkinter** and **TensorFlow**.  
It allows users to select a music file (e.g., `.wav`), process it using **Librosa**, and predict its genre using a pre-trained deep learning model (`.h5` file).

---

## 🧠 Overview

This project combines **deep learning**, **audio signal processing**, and **GUI design**:

- Extracts **mel-spectrogram** features using `librosa`
- Classifies songs into genres using a **Convolutional Neural Network (CNN)**
- Provides an easy-to-use interface using **Tkinter**
- Visualizes waveforms and mel-spectrograms in real-time

---

## 📁 Project Structure

| File / Folder                      | Description                                 |
| ---------------------------------- | ------------------------------------------- |
| `app.py`                           | Main GUI application (Tkinter + TensorFlow) |
| `music_genre_classification.ipynb` | Notebook for training/testing the model     |
| `best_model.h5`                    | Pre-trained deep learning model             |
| `requirements.txt`                 | Python dependencies                         |
| `README.md`                        | Project documentation (this file)           |

---


## 🎶 Dataset Information

This project uses the **GTZAN Music Genre Dataset**, a popular benchmark dataset for music classification.  
It contains **1000 audio tracks**, each 30 seconds long, across **10 genres**:

`Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock`

You can download the dataset from Kaggle here:  
🔗 [GTZAN Music Genre Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

After downloading, place the dataset in your preferred folder and update the path in your Jupyter Notebook if you want to retrain the model.

---

## ⚙️ Installation Guide

### 🧩 Prerequisites

- **Python 3.10.9** (recommended)  
  Check your version:

```bash
  py --version
```

If you don’t have Python 3.11, download it from:
🔗 https://www.python.org/downloads/

pip (Python package manager)
Check if installed:

```bash
py -3.11 -m pip --version

```
- 🪄 Step 1 — Clone the Repository

If you have Git installed:
``` bash
git clone https://github.com/<your-username>/music-genre-classification.git
cd music-genre-classification
```
Or manually download the ZIP from GitHub and extract it.

- 🧱 Step 2 — Install Dependencies

There are two ways to install dependencies.
Option 1 — Quick Setup (Global Install)
Easiest for users who just want to try the app.
``` bash
py -3.10 -m pip install -r requirements.txt
```
``` bash
py -3.10 app.py
```
Option 2 — Recommended Setup (Virtual Environment)
Keeps dependencies isolated and avoids version conflicts.

``` bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
py app.py
```
To exit the environment later:
``` bash
deactivate
```

- 🧠 Step 3 — Verify Installation

To make sure everything is installed properly:
``` bash
py -3.11 -m pip list
```
---

## 🚀 How to Run the Application
- 🪄 Run the GUI App

Simply run:
``` bash
py -3.11 app.py
```

The application window will open.
From there:

- Click Browse to select a music file (.wav or .mp3).
- The app will extract audio features.
- Click Predict Genre to see the predicted result.

---