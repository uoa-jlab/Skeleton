import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import torch
from torch.utils.data import Dataset

class PressureSkeletonDataset(Dataset):
    def __init__(self, pressure_data, skeleton_data, decoder_input):
        self.pressure_data = torch.FloatTensor(pressure_data)
        self.skeleton_data = torch.FloatTensor(skeleton_data)
        self.decoder_input = torch.FloatTensor(decoder_input)

    def __len__(self):
        return len(self.pressure_data)

    def __getitem__(self, idx):
        return self.pressure_data[idx], self.skeleton_data[idx], self.decoder_input[idx]


def load_and_combine_data(file_pairs, seq_length, window_size):
    """複数のデータセットを読み込んで結合する"""
    all_skeleton_data = []
    all_pressure_left = []
    all_pressure_right = []
    all_decoder_input = []

    all_skeleton_label = []
    for skeleton_file, left_file, right_file in file_pairs:
        skeleton = pd.read_csv(skeleton_file)
        left = pd.read_csv(left_file, dtype=float, low_memory=False)
        right = pd.read_csv(right_file, dtype=float, low_memory=False)
        # データ長を揃える
        min_length = min(len(skeleton), len(left), len(right))

        num_joints_points = skeleton.shape[1]
        decoder_input = np.zeros((min_length, seq_length, num_joints_points))
        for i in range(min_length):
            for j in range(1, seq_length + 1):
                if i - j < 0:
                    continue
                else:
                    decoder_input[i, seq_length - j] = skeleton.iloc[i - j]

        skeleton_label = np.zeros((min_length, window_size, num_joints_points))
        for i in range(min_length):
            for j in range(window_size):
                if i + j < min_length:
                    skeleton_label[i, j] = skeleton.iloc[i + j]

        all_skeleton_data.append(skeleton.iloc[:min_length])
        all_pressure_left.append(left.iloc[:min_length])
        all_pressure_right.append(right.iloc[:min_length])
        all_decoder_input.append(decoder_input)
        all_skeleton_label.append(skeleton_label)

    return (pd.concat(all_skeleton_data, ignore_index=True),
            pd.concat(all_pressure_left, ignore_index=True),
            pd.concat(all_pressure_right, ignore_index=True),
            np.concatenate(all_decoder_input),
            np.concatenate(all_skeleton_label))


def preprocess_pressure_data(left_data, right_data):
    """圧力、回転、加速度データの前処理"""

    # 左足データから各種センサー値を抽出
    left_pressure = left_data.iloc[:, :35]  # 圧力センサーの列を適切に指定
    left_rotation = left_data.iloc[:, 35:38]  # 回転データの列を適切に指定
    left_accel = left_data.iloc[:, 38:41]  # 加速度データの列を適切に指定

    # 右足データから各種センサー値を抽出
    right_pressure = right_data.iloc[:, :35]  # 圧力センサーの列を適切に指定
    right_rotation = right_data.iloc[:, 35:38]  # 回転データの列を適切に指定
    right_accel = right_data.iloc[:, 38:41]  # 加速度データの列を適切に指定

    # データの結合(按列（属性）相拼接)
    pressure_combined = pd.concat([left_pressure, right_pressure], axis=1)
    rotation_combined = pd.concat([left_rotation, right_rotation], axis=1)
    accel_combined = pd.concat([left_accel, right_accel], axis=1)

    # NaN値を補正
    pressure_combined = pressure_combined.fillna(0.0)
    rotation_combined = rotation_combined.fillna(0.0)
    accel_combined = accel_combined.fillna(0.0)

    print("Checking pressure data for NaN or Inf...")
    print("Pressure NaN count:", pressure_combined.isna().sum().sum())
    print("Pressure Inf count:", np.isinf(pressure_combined).sum().sum())

    # 移動平均フィルタの適用
    window_size = 3
    pressure_combined = pressure_combined.rolling(window=window_size, center=True).mean()
    rotation_combined = rotation_combined.rolling(window=window_size, center=True).mean()
    accel_combined = accel_combined.rolling(window=window_size, center=True).mean()

    # NaN値を前後の値で補間
    pressure_combined = pressure_combined.bfill().ffill()
    rotation_combined = rotation_combined.bfill().ffill()
    accel_combined = accel_combined.bfill().ffill()

    # 正規化と標準化のスケーラー初期化
    pressure_normalizer = MinMaxScaler()
    rotation_normalizer = MinMaxScaler()
    accel_normalizer = MinMaxScaler()

    pressure_standardizer = StandardScaler(with_mean=True, with_std=True)
    rotation_standardizer = StandardScaler(with_mean=True, with_std=True)
    accel_standardizer = StandardScaler(with_mean=True, with_std=True)

    # データの正規化と標準化
    pressure_processed = pressure_standardizer.fit_transform(
        pressure_normalizer.fit_transform(pressure_combined)
    )
    rotation_processed = rotation_standardizer.fit_transform(
        rotation_normalizer.fit_transform(rotation_combined)
    )
    accel_processed = accel_standardizer.fit_transform(
        accel_normalizer.fit_transform(accel_combined)
    )
    # すべての特徴量を結合
    input_features = np.concatenate([
        pressure_processed,
        rotation_processed,
        accel_processed,
    ], axis=1)

    return input_features, {
        'pressure': {
            'normalizer': pressure_normalizer,
            'standardizer': pressure_standardizer
        },
        'rotation': {
            'normalizer': rotation_normalizer,
            'standardizer': rotation_standardizer
        },
        'accel': {
            'normalizer': accel_normalizer,
            'standardizer': accel_standardizer
        }
    }