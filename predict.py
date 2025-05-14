def preprocess_pressure_data(left_data, right_data):
    """圧力、回転、加速度データの前処理"""
    # 左足データから各種センサー値を抽出
    left_pressure = left_data.iloc[:, :35]
    left_rotation = left_data.iloc[:, 35:38]
    left_accel = left_data.iloc[:, 38:41]

    # 右足データから各種センサー値を抽出
    right_pressure = right_data.iloc[:, :35]
    right_rotation = right_data.iloc[:, 35:38]
    right_accel = right_data.iloc[:, 38:41]

    # データの結合
    pressure_combined = pd.concat([left_pressure, right_pressure], axis=1)
    rotation_combined = pd.concat([left_rotation, right_rotation], axis=1)
    accel_combined = pd.concat([left_accel, right_accel], axis=1)

    # NaN値を補正
    pressure_combined = pressure_combined.ffill().bfill()
    rotation_combined = rotation_combined.ffill().bfill()
    accel_combined = accel_combined.ffill().bfill()

    # 移動平均フィルタの適用
    window_size = 3
    pressure_combined = pressure_combined.rolling(window=window_size, center=True).mean()
    rotation_combined = rotation_combined.rolling(window=window_size, center=True).mean()
    accel_combined = accel_combined.rolling(window=window_size, center=True).mean()

    # NaN値を補間
    pressure_combined = pressure_combined.ffill().bfill()
    rotation_combined = rotation_combined.ffill().bfill()
    accel_combined = accel_combined.ffill().bfill()

    # 正規化と標準化
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

    # 1次微分と2次微分の計算
    pressure_grad1 = np.gradient(pressure_processed, axis=0)
    pressure_grad2 = np.gradient(pressure_grad1, axis=0)

    rotation_grad1 = np.gradient(rotation_processed, axis=0)
    rotation_grad2 = np.gradient(rotation_grad1, axis=0)

    accel_grad1 = np.gradient(accel_processed, axis=0)
    accel_grad2 = np.gradient(accel_grad1, axis=0)

    # すべての特徴量を結合（246次元になるはず）
    input_features = np.concatenate([
        pressure_processed,  # 原特徴量
        # pressure_grad1,     # 1次微分
        # pressure_grad2,     # 2次微分
        rotation_processed,
        # rotation_grad1,
        # rotation_grad2,
        accel_processed,
        # accel_grad1,
        # accel_grad2
    ], axis=1)

    return input_features


def load_and_preprocess_data(file_pairs):
    predictions_all = []

    for skeleton_file, left_file, right_file in file_pairs:
        skeleton_data = pd.read_csv(skeleton_file)
        pressure_data_left = pd.read_csv(left_file)
        pressure_data_right = pd.read_csv(right_file)

        input_features = preprocess_pressure_data(pressure_data_left, pressure_data_right)
        min_length = min(len(skeleton_data), len(input_features))

        input_features = input_features.iloc[:min_length]
        skeleton_data = skeleton_data.iloc[:min_length]

        predictions_all.append((input_features, skeleton_data))

    return predictions_all


def predict_skeleton():
    try:
        # データの読み込みと前処理
        skeleton_data = pd.read_csv('./data/20241115test3/Opti-track/Take 2024-11-15 03.50.00 PM.csv')
        pressure_data_left = pd.read_csv('./data/20241115test3/insoleSensor/20241115_155500_left.csv', skiprows=1)
        pressure_data_right = pd.read_csv('./data/20241115test3/insoleSensor/20241115_155500_right.csv', skiprows=1)

        # 入力データの前処理
        input_features = preprocess_pressure_data(pressure_data_left, pressure_data_right)
        min_length = min(input_features.shape[0], skeleton_data.shape[0])

        # 入力の次元数を取得
        seq_length = 3
        window_size = 1
        m = 1
        input_dim = input_features.shape[1]
        num_joints = skeleton_data.shape[1] // 3

        # デバイスの設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        skeleton_data = torch.FloatTensor(np.array(skeleton_data)[:min_length]).to(device)

        # モデルの初期化（固定パラメータを使用）
        model = EnhancedSkeletonTransformer(
            input_dim=input_dim,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_joints=num_joints,
            num_dims=3,
            dropout=0.1,
            seq_length=seq_length,
            window_size=window_size
        ).to(device)

        # チェックポイントの読み込み（weights_only=Trueを追加）
        checkpoint = torch.load('./weight/best_skeleton_model.pth', map_location=device, weights_only=True)

        # モデルの重みを読み込み
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")

        action = torch.zeros((min_length + 10, window_size, 63)).to(device)
        num = np.zeros(min_length + 10)
        print(action.shape)
        # 予測の実行
        print("Making predictions...")
        predictions = torch.zeros(min_length, 63).to(device)
        with torch.no_grad():
            skeleton_last = torch.zeros((seq_length, 63))
            skeleton_last = skeleton_last.unsqueeze(0).to(device)
            for i in range(min_length):
                input_tensor = torch.FloatTensor(input_features)[i].to(device)
                input_tensor = input_tensor.unsqueeze(0).to(device)
                # if i%200==0:
                #    skeleton_last=torch.zeros_like(skeleton_last)

                skeleton_predict_seq = model(input_tensor, skeleton_last)
                skeleton_predict_seq = skeleton_predict_seq.squeeze(0)
                skeleton_predict = torch.zeros(63).to(device)
                for j in range(window_size):
                    action[i + j, int(num[i + j])] = skeleton_predict_seq[j, :]
                    num[i + j] += 1
                    # print(f"j={j},i+j={i+j},num[i+j}]={int(num[i+j])}")
                weights = compute_exponential_weights(int(num[i]), m).to(device)
                for j in range(int(num[i])):
                    skeleton_predict += weights[j] * action[i, int(num[i]) - 1 - j]
                predictions[i] = skeleton_predict
                for j in range(seq_length - 1):
                    skeleton_last[0, j] = skeleton_last[0, j + 1]
                skeleton_last[0, seq_length - 1] = skeleton_predict
        '''
        # 予測の実行
        print("Making predictions...")
        predictions=torch.zeros(min_length,63).to(device)
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_features).to(device)
            input_tensor=input_tensor.to(device)
            print(input_tensor.shape,skeleton_data.shape)
            predictions=model(input_tensor,skeleton_data)
        '''
        print(f"Prediction shape: {predictions.shape}")
        criterion = EnhancedSkeletonLoss(alpha=1.0, beta=0.1, gamma=0.1, window_size=1)
        criterion1 = EnhancedSkeletonLoss_WithAngleConstrains(alpha=1.0, beta=0.1, gamma=0.1, window_size=1)
        predictionsa = predictions.unsqueeze(1)
        skeleton_dataa = skeleton_data.unsqueeze(1)
        loss = criterion(predictionsa, skeleton_dataa)
        print(f"Loss: {loss}")
        loss1 = criterion1(predictionsa, skeleton_dataa)
        print(f"Loss: {loss1}")
        predictions = predictions.cpu().numpy()
        return predictions

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise


def save_predictions(predictions, output_file='./output/predicted_skeleton.csv'):
    try:
        # 予測結果をデータフレームに変換
        num_joints = predictions.shape[1] // 3
        columns = []
        for i in range(num_joints):
            columns.extend([f'X.{i * 2 + 1}', f'Y.{i * 2 + 1}', f'Z.{i * 2 + 1}'])

        df_predictions = pd.DataFrame(predictions, columns=columns)
        df_predictions.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

    except Exception as e:
        print(f"Error saving predictions: {str(e)}")
        raise


def main():
    try:
        print("Starting prediction process...")
        predictions = predict_skeleton()

        print("\nSaving predictions...")
        save_predictions(predictions)
        print(predictions)

        print("Prediction process completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()