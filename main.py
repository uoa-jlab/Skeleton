import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
def main():
    # データの読み込み
    data_pairs = [
        #
        # 第三回収集データ
        #
        # # 立ちっぱなし
        ('./data/20241115test3/Opti-track/Take 2024-11-15 03.20.00 PM.csv',
         './data/20241115test3/insoleSensor/20241115_152500_left.csv',
         './data/20241115test3/insoleSensor/20241115_152500_right.csv'),
        # お辞儀
        ('./data/20241115test3/Opti-track/Take 2024-11-15 03.26.00 PM.csv',
         './data/20241115test3/insoleSensor/20241115_153100_left.csv',
         './data/20241115test3/insoleSensor/20241115_153100_right.csv'),
        # 体の横の傾け
        ('./data/20241115test3/Opti-track/Take 2024-11-15 03.32.00 PM.csv',
         './data/20241115test3/insoleSensor/20241115_153700_left.csv',
         './data/20241115test3/insoleSensor/20241115_153700_right.csv'),
        # 立つ座る
        ('./data/20241115test3/Opti-track/Take 2024-11-15 03.38.00 PM.csv',
         './data/20241115test3/insoleSensor/20241115_154300_left.csv',
         './data/20241115test3/insoleSensor/20241115_154300_right.csv'),
        # スクワット
        ('./data/20241115test3/Opti-track/Take 2024-11-15 03.44.00 PM.csv',
         './data/20241115test3/insoleSensor/20241115_154900_left.csv',
         './data/20241115test3/insoleSensor/20241115_154900_right.csv'),
        # 総合(test3)
        # ('./data/20241115test3/Opti-track/Take 2024-11-15 03.50.00 PM.csv',
        # './data/20241115test3/insoleSensor/20241115_155500_left.csv', 
        # './data/20241115test3/insoleSensor/20241115_155500_right.csv'),

        # 釘宮くん
        ('./data/20241212test4/Opti-track/Take 2024-12-12 03.06.59 PM.csv',
         './data/20241212test4/insoleSensor/20241212_152700_left.csv',
         './data/20241212test4/insoleSensor/20241212_152700_right.csv'),
        # 百田くん
        ('./data/20241212test4/Opti-track/Take 2024-12-12 03.45.00 PM.csv',
         './data/20241212test4/insoleSensor/20241212_160501_left.csv',
         './data/20241212test4/insoleSensor/20241212_160501_right.csv'),
        # # # # 渡辺(me)
        ('./data/20241212test4/Opti-track/Take 2024-12-12 04.28.00 PM.csv',
         './data/20241212test4/insoleSensor/20241212_164800_left.csv',
         './data/20241212test4/insoleSensor/20241212_164800_right.csv'),
        # にるぱむさん
        ('./data/20241212test4/Opti-track/Take 2024-12-12 05.17.59 PM.csv',
         './data/20241212test4/insoleSensor/20241212_173800_left.csv',
         './data/20241212test4/insoleSensor/20241212_173800_right.csv')
    ]

    # データの読み込みと結合
    seq_length = 3
    window_size = 1
    skeleton_data, pressure_data_left, pressure_data_right, decoder_input, skeleton_label = load_and_combine_data(
        data_pairs, seq_length, window_size)

    # numpy配列に変換
    skeleton_data = skeleton_data.to_numpy()
    decoder_input = decoder_input

    # 圧力、回転、加速度データの前処理
    input_features, sensor_scalers = preprocess_pressure_data(
        pressure_data_left,
        pressure_data_right
    )
    print(input_features.shape)

    # データの分割
    train_input, val_input, train_skeleton, val_skeleton, train_decoder_input, val_decoder_input = train_test_split(
        input_features,
        skeleton_label,
        decoder_input,
        test_size=0.2,
        random_state=42
    )
    print(train_decoder_input.shape)
    print(val_decoder_input.shape)

    print(train_decoder_input[0])
    # デバイスの設定
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルのパラメータ設定
    input_dim = input_features.shape[1]  # 圧力+回転+加速度の合計次元数
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_joints = skeleton_data.shape[1] // 3  # 3D座標なので3で割る
    dropout = 0.1
    batch_size = 128

    # データローダーの設定
    train_dataset = PressureSkeletonDataset(train_input, train_skeleton, train_decoder_input)
    val_dataset = PressureSkeletonDataset(val_input, val_skeleton, val_decoder_input)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Checking final training and validation data...")
    print("Train input NaN count:", np.isnan(train_input).sum(), "Inf count:", np.isinf(train_input).sum())
    print("Train skeleton NaN count:", np.isnan(train_skeleton).sum(), "Inf count:", np.isinf(train_skeleton).sum())

    # モデルの初期化
    model = EnhancedSkeletonTransformer(
        input_dim=input_features.shape[1],  # input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_joints=num_joints,
        num_dims=3,
        dropout=dropout,
        seq_length=seq_length,
        window_size=window_size
    ).to(device)

    # 損失関数、オプティマイザ、スケジューラの設定
    # criterion = torch.nn.MSELoss()  # 必要に応じてカスタム損失関数に変更可能
    criterion = EnhancedSkeletonLoss_WithAngleConstrains(alpha=1.0, beta=0.1, gamma=0.5, window_size=window_size)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.001,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # トレーニング実行
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=300,
        save_path='./weight/best_skeleton_model.pth',
        device=device,
    )

    # モデルの保存
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'sensor_scalers': sensor_scalers,
        # 'skeleton_skaler': skeleton_scaler,
        'model_config': {
            'input_dim': input_dim,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_joints': num_joints
        }
    }
    torch.save(final_checkpoint, './weight/final_skeleton_model.pth')


if __name__ == "__main__":
    main()