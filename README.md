# 👁️ EOG Movement Prediction
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)  
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen.svg)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)  
## 📌 Overview
This project focuses on predicting **eye movements (Left / Right)** using **Electrooculography (EOG) signals**.  
It applies signal preprocessing techniques and machine learning models to classify eye movements, providing a step toward assistive technologies for patients (e.g., ALS patients) who rely on eye movement for communication.

---

## 🚀 Features
- Preprocessing pipeline:
  - Mean removal  
  - Bandpass filtering  
  - Normalization  
  - Downsampling  
  - Wavelet feature extraction  
- Machine Learning Models:
  - **KNN Classifier**
  - **Random Forest Classifier**  
- Deployment using **Streamlit** (Interactive web app).
- File upload support for testing custom EOG signal data.
- Visualization of preprocessing and accuracy results.

---

## 📊 Model Performance
| Model             | Accuracy |
|-------------------|----------|
| Random Forest     | ~0.90    |
| KNN Classifier    | ~0.80    |

---

## 🧠 Tech Stack
- Python 🐍  
- NumPy, Pandas  
- SciPy, PyWavelets  
- Scikit-learn  
- Matplotlib, Seaborn  
- Streamlit  


