import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model

# 資料路徑
DATA_PATH = 'C:\\Users\\USER\\Desktop\\flask-templete\\train'
LABEL = 'A'
test_audio_path = 'C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio'
feature_dim_1 = 13  # MFCC features
feature_dim_2 = 50  # Modify this based on the correct number of timesteps
channel = 1
epochs = 2500
batch_size = 10
verbose = 1
num_classes = 1  # 只有一個類別 'A'

def get_labels(path=DATA_PATH):  
    labels = [LABEL]
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)  # 回傳['A'] [0] [[1. 0. 0.]]

def cepstral_mean_subtraction(mfccs):
    mean = np.mean(mfccs, axis=1, keepdims=True)
    cms_mfccs = mfccs - mean
    return cms_mfccs

def wav2mfcc(file_path, max_len=50):  # Modify max_len based on the correct number of timesteps
    wave, sr = librosa.load(file_path, mono=True, sr=44100)
    mfccs = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=feature_dim_1)
    mfcc = cepstral_mean_subtraction(mfccs)
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def save_data_to_array(path=DATA_PATH, max_len=50):
    labels, _, _ = get_labels(path)
    mfcc_vectors = []
    wavfiles = [os.path.join(path, wavfile) for wavfile in os.listdir(path)]
    for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(LABEL)):
        mfcc = wav2mfcc(wavfile, max_len=max_len)
        mfcc_vectors.append(mfcc)
    np.save(LABEL + '.npy', np.array(mfcc_vectors))

def get_train_test(split_ratio=0.6, random_state=42):
    labels, indices, _ = get_labels(DATA_PATH)
    X = np.load(LABEL + '.npy')
    y = np.zeros(X.shape[0])
    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    prediction = model.predict(sample_reshaped)
    accuracy = prediction[0][0]
    return get_labels()[0][0], accuracy

# 主程式
def main():
    save_data_to_array(max_len=feature_dim_2)

    X_train, X_test, y_train, y_test = get_train_test()

    X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
    X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)

    model = get_model()
    history = model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
    model.save('MFCC.h5')
    model2 = load_model('MFCC.h5')
    
    # 查看訓練結果
    print("Training Accuracy: ", history.history['accuracy'][-1])
    print("Validation Accuracy: ", history.history['val_accuracy'][-1])
    print("Training Loss: ", history.history['loss'][-1])
    print("Validation Loss: ", history.history['val_loss'][-1])
    
    # 混淆矩陣
    y_pred = model2.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
    y_true = y_test.astype(int)
    
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # 顯示分類報告
    print(classification_report(y_true, y_pred_classes))
    
    predicted_label, accuracy = predict('C:\\Users\\USER\\Desktop\\flask-templete\\static\\audio\\B.wav', model=model2)
    print(f"Predicted label: {predicted_label}, Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()