import librosa
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
import itertools
import random
import psutil
from consts import *


def aug_with_noise(y):
    noise_amplitude = 0.1 * np.std(y)
    noise = noise_amplitude * np.random.randn(len(y))
    return y + noise


def aug_with_gain(y):
    max_amplitude = np.max(np.abs(y))
    safe_gain = 1.0 / max_amplitude if max_amplitude > 0 else 1.0
    gain = np.random.uniform(0, safe_gain)
    return y * gain


def aug_with_shift(y):
    threshold = np.max(np.abs(y)) / 10
    silent_indices = np.where(np.abs(y) < threshold)[0]

    shift_direction = np.random.choice(['left', 'right'])
    diffs = np.diff(silent_indices)
    if shift_direction == 'left':
        end_of_segments = np.where(diffs != 1)[0]
    else:
        end_of_segments = np.where(diffs != 1)[-1]

    if len(end_of_segments) > 0:
        max_index = silent_indices[end_of_segments[0]]
    elif len(end_of_segments) == 0:
        max_index = 0
    else:
        max_index = silent_indices[-1]

    if shift_direction == 'left':
        shift_amount = np.random.randint(1, max_index + 1)
        y_shifted = np.roll(y, -shift_amount)
        y_shifted[-shift_amount:] = 0
    else:
        shift_amount = np.random.randint(1, max_index + 1)
        y_shifted = np.roll(y, shift_amount)
        y_shifted[:shift_amount] = 0

    return y_shifted


def save_mel_spectrogram(y, save_file_name, aug_funcs):
    for aug_func in aug_funcs:
        y = aug_func(y)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = cv2.normalize(mel_spectrogram, None, 0, 255, cv2.NORM_MINMAX)
    mel_spectrogram = mel_spectrogram.astype(np.uint8)
    cv2.imwrite(save_file_name, mel_spectrogram)


def main():
    '''
    # データ整形
    # 具体的には、音声の長さを揃えて(末尾を0埋め)、メルスペクトログラムにする
    # torchvision.datasets.ImageFolder の形式に沿ってデータセットを構築
    '''
    train_dir = os.path.join(DATASETS_PATH, 'train')
    val_dir = os.path.join(DATASETS_PATH, 'val')
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

    aug_funcs = [aug_with_noise, aug_with_gain, aug_with_shift]
    all_combi_of_aug_funcs = []
    for r in range(1, len(aug_funcs) + 1):
        combinations_r = list(itertools.combinations(aug_funcs, r))
        all_combi_of_aug_funcs.extend(combinations_r)

    TRAIN_DATA_PER_CLASS = 5000
    wav_file_counts = {}
    for class_name in os.listdir(RAWDATA_PATH):
        class_name_path = os.path.join(RAWDATA_PATH, class_name)
        if os.path.isdir(class_name_path):
            wav_count = sum(1 for file in os.listdir(class_name_path) if file.lower().endswith('.wav'))
            wav_file_counts[class_name] = wav_count
    aug_ratio_dict = {label: TRAIN_DATA_PER_CLASS // num for label, num in wav_file_counts.items()}

    with ThreadPoolExecutor(max_workers=psutil.cpu_count(logical=False) - 2) as executor:
        for class_name in os.listdir(RAWDATA_PATH):
            class_path = os.path.join(RAWDATA_PATH, class_name)
            if not os.path.isdir(class_path):
                continue

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
                        
                        if save_dir == train_class_dir:
                            for i in range(aug_ratio_dict[class_name]):
                                executor.submit(
                                    save_mel_spectrogram,
                                    formatted_y,
                                    os.path.join(save_dir, '.'.join(file.split('.')[:-1]) + f'_{i}' + '.png'),
                                    random.choice(all_combi_of_aug_funcs)
                                )
                        else:
                            executor.submit(
                                save_mel_spectrogram,
                                formatted_y,
                                os.path.join(save_dir, '.'.join(file.split('.')[:-1]) + '.png'),
                                [lambda x: x, ]
                            )


if __name__ == "__main__":
    main()
