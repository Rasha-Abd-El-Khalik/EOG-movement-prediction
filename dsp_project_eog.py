

# pip install PyWavelets

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import butter, filtfilt, resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# # Read the file
# with open("train_right.txt", "r") as file:
#     lines = file.readlines()

# # Add a new column (e.g., append a value to each line)
# new_lines = [line.strip() + "\tright\n" for line in lines]

# # Write the updated content back to the file
# with open("train_right.txt", "w") as file:
#     file.writelines(new_lines)



# # Read the file
# with open("train_left.txt", "r") as file:
#     lines = file.readlines()

# # Add a new column (e.g., append a value to each line)
# new_lines = [line.strip() + "\tleft\n" for line in lines]

# # Write the updated content back to the file
# with open("train_left.txt", "w") as file:
#     file.writelines(new_lines)


# file_paths = ['train_right.txt', 'train_left.txt']  # Example file paths
# output_file = 'Train_Signals.txt'  # Name of the new file

# # Open the output file in write mode
# with open(output_file, 'w') as outfile:
#     for file_path in file_paths:
#         # Open each file and write its contents to the new file
#         with open(file_path, 'r') as infile:
#             outfile.write(infile.read())  # Write the content of the file
# print(f"file created: {output_file}")

# # Read the file
# with open("test_right.txt", "r") as file:
#     lines = file.readlines()

# # Add a new column (e.g., append a value to each line)
# new_lines = [line.strip() + "\tright\n" for line in lines]

# # Write the updated content back to the file
# with open("test_right.txt", "w") as file:
#     file.writelines(new_lines)



# # Read the file
# with open("test_left.txt", "r") as file:
#     lines = file.readlines()

# # Add a new column (e.g., append a value to each line)
# new_lines = [line.strip() + "\tleft\n" for line in lines]

# # Write the updated content back to the file
# with open("test_left.txt", "w") as file:
#     file.writelines(new_lines)



# file_paths = ['test_right.txt', 'test_left.txt']  # Example file paths
# output_file = 'Test_Signals.txt'  # Name of the new file

# # Open the output file in write mode
# with open(output_file, 'w') as outfile:
#     for file_path in file_paths:
#         # Open each file and write its contents to the new file
#         with open(file_path, 'r') as infile:
#             outfile.write(infile.read())  # Write the content of the file
# print(f"file created: {output_file}")

def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        features = []
        labels = []
        for line in file:
            parts = line.split()  # Split the line into parts
            row = list(map(int, parts[:-1]))  # Convert all but the last column to integers
            label = parts[-1]  # The last column is the label (string)
            features.append(row)
            labels.append(label)

    return features, labels


file_train = r"C:\EOG_Project\DSP project\Train_Signals.txt"
file_test  = r"C:\EOG_Project\DSP project\Test_Signals.txt"



features_Train ,label_Train = read_file_to_list(file_train)
features_Test , label_Test= read_file_to_list(file_test)

# print(len(features_Test))
# print(len(features_Train))
# print(len(label_Test))
# print(len(label_Train))

def remove_mean(signal):
    mean_removed_signal = signal - np.mean(signal)
    return mean_removed_signal

def bandpass_filter(sampling_rate, signal, low_cutoff=0.5, high_cutoff=20.0):
    nyquist = sampling_rate * 0.5
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def normalize(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val)  # Scale between 0 and 1
    return normalized_signal


def downsample(signal, factor):
    downsampled_signal = signal[::factor]
    return downsampled_signal

def wavelet_feature_extraction(signal, wavelet='db2', level=2):
    if len(signal) == 0:
        raise ValueError("Input signal is empty.")
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    approximation = coeffs[0]
    details = coeffs[1:]
    # print(approximation[5])
    return approximation



def process(signal,sampling_rate=176):
    mean_removed_data = [remove_mean(np.array(row)) for row in signal]
    # print(len(mean_removed_data))
    filtered_data = [bandpass_filter(176, row) for row in mean_removed_data]
    # print(len(filtered_data))
    normalized_data = [normalize(row) for row in filtered_data]
    # print(len(normalized_data))
    downsampled_data = [downsample(row, 4) for row in normalized_data]
    # print(len(downsampled_data))
    wavelet_features = wavelet_feature_extraction(downsampled_data)
    # print(len(wavelet_features))
    # print(wavelet_features[5])
    # plt.figure(figsize=(12, 6))
    # plt.title("Preprocessing")
    # plt.plot(signal[0],label="General Signal")
    # plt.plot(mean_removed_data[0],label="after mean removal")
    # plt.plot(filtered_data[0],label="filtered")
    # plt.plot(normalized_data[0],label="normalized")
    # plt.plot(downsampled_data[0],label="down_sampled")
    # plt.legend()
    # plt.show()
    return wavelet_features


wavelet_train=process(features_Train)
wavelet_test= process(features_Test)

# # print("wavelet_train shape:", np.shape(wavelet_train))
# # print("wavelet_test shape:", np.shape(wavelet_test))
# # print("label_Train shape:", np.shape(label_Train))
# # print("label_Test shape:", np.shape(label_Test))
# print(len(wavelet_train))
# print(len(wavelet_test))
# print(len(label_Train))
# print(len(label_Test))

X_train, X_test, y_train, y_test = wavelet_train,wavelet_test,label_Train,label_Test

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def knn_classification(X_train, X_test, y_train, y_test, n_neighbors=3):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)


    # print(f"Accuracy: {accuracy * 100:.2f}%")
    # print("Confusion Matrix:")
    # print(cm)
    # print("Classification Report:")
    # print(report)

    return knn, y_pred


X_train = np.array(wavelet_train).reshape(len(wavelet_train), -1)
X_test = np.array(wavelet_test).reshape(len(wavelet_test), -1)

y_train = np.ravel(label_Train)
y_test = np.ravel(label_Test)

knn_model, predictions = knn_classification(X_train, X_test, y_train, y_test, n_neighbors=5)

from sklearn.ensemble import RandomForestClassifier

acc1=[]
acc2=[]
for i in range (2,22):
  dtc1=RandomForestClassifier(max_depth=i,criterion="entropy",random_state=42)
  dtc1.fit(X_train,y_train)
  dtc2=RandomForestClassifier(max_depth=i,criterion="gini",random_state=42)
  dtc2.fit(X_train,y_train)
  acc1.append(dtc1.score(X_test, y_test))
  acc2.append(dtc2.score(X_test, y_test))

# plt.plot(range(2,22),acc1,label="entropy")
# plt.plot(range(2,22),acc2,label="gini")
# plt.legend()

random_forest_model = RandomForestClassifier(criterion="gini", max_depth=19,random_state=40)
random_forest_model.fit(X_train, y_train)
y_pred = random_forest_model.predict(X_test)
accuracy_RF = accuracy_score(y_test, y_pred)
# print(f"Accuracy of the RandomForestClassifier model: {accuracy_RF:.2f}")

import seaborn as sns
from sklearn.metrics import accuracy_score

accuracy_KNN = accuracy_score(y_test, predictions)

models = pd.DataFrame({
    'Model': ['Classification Knn', 'Random Forest'],
    'Accuracy_score': [accuracy_KNN, accuracy_RF]
})
sns.barplot(x='Accuracy_score', y='Model', data=models)
models.sort_values(by='Accuracy_score', ascending=False)




