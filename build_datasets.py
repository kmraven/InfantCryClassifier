import librosa
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from consts import *


def main():
    '''
    # データ整形
    # 具体的には、音声の長さを揃えて(末尾を0埋め)、メルスペクトログラムにする
    # torchvision.datasets.ImageFolder の形式に沿ってデータセットを構築
    '''
    train_dir = os.path.join(DATASETS_PATH_SIMPLE, 'train')
    val_dir = os.path.join(DATASETS_PATH_SIMPLE, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    len_list = []
    for current, _, files in os.walk(RAWDATA_PATH):
        for file in files:
            if file.lower().endswith('.wav'):
                y, _ = librosa.load(os.path.join(current, file), sr=None)
                len_list.append(len(y))
    target_size = np.max(len_list)
    print(f'target size: {target_size}')

    for class_name in os.listdir(RAWDATA_PATH):
        class_path = os.path.join(RAWDATA_PATH, class_name)
        if os.path.isdir(class_path):
            train_class_dir = os.path.join(train_dir, class_name)
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(val_class_dir, exist_ok=True)
            
            wav_files = os.listdir(class_path)
            train_files, val_files = train_test_split(wav_files, test_size=0.2, random_state=42)

            for save_dir, files in {train_class_dir: train_files, val_class_dir: val_files}.items():
                for file in files:
                    if file.lower().endswith('.wav'):
                        y, sr = librosa.load(os.path.join(class_path, file), sr=None)
                        original_size = len(y)
                        if original_size < target_size:
                            formatted_y = np.zeros(target_size, dtype=y.dtype)
                            formatted_y[:original_size] = y
                        else:
                            formatted_y = y
                        
                        mel_spectrogram = librosa.feature.melspectrogram(y=formatted_y, sr=sr)
                        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
                        mel_spectrogram = cv2.normalize(mel_spectrogram, None, 0, 255, cv2.NORM_MINMAX)
                        mel_spectrogram = mel_spectrogram.astype(np.uint8)
                        cv2.imwrite(os.path.join(save_dir, '.'.join(file.split('.')[:-1]) + '.png'), mel_spectrogram)


if __name__ == "__main__":
    main()
