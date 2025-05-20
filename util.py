def add_gaussian_noise(data, noise_factor=0.05):
    """给数据添加高斯噪声，默认噪声方差为原始数据的5%"""
    std = torch.std(data, dim=0, keepdim=True)  # 计算原始数据的标准差
    noise = torch.randn_like(data) * (std * noise_factor)  # 生成噪声
    return data + noise


def positional_encoding(seq_len, d_model):
    # 初始化一个全零矩阵，形状为 (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)

    # position: [0, 1, 2, ..., seq_len-1]，形状为 (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

    # div_term: 指数衰减因子，用于控制正弦和余弦的频率
    # torch.arange(0, d_model, 2): 取嵌入维度中偶数位置的索引
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

    # 偶数位置使用正弦函数，奇数位置使用余弦函数
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def add_positional_encoding(x):
    batch_size, seq_len, d_model = x.shape

    # 构建位置编码矩阵，形状 (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model, device=x.device, dtype=x.dtype)

    # 位置向量 (seq_len, 1)
    position = torch.arange(0, seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)

    # 频率因子（对偶数索引维度计算）
    div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device, dtype=x.dtype) * (-math.log(10000.0) / d_model))

    # 对偶数维度使用 sin, 奇数维度使用 cos
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # 将位置编码添加到输入张量上，pe.unsqueeze(0) 的形状为 (1, seq_len, d_model)
    x = x + pe.unsqueeze(0)
    return x


class PressureSkeletonDataset(Dataset):
    def __init__(self, pressure_data, skeleton_data, decoder_input):
        self.pressure_data = torch.FloatTensor(pressure_data)
        self.skeleton_data = torch.FloatTensor(skeleton_data)
        self.decoder_input = torch.FloatTensor(decoder_input)

    def __len__(self):
        return len(self.pressure_data)

    def __getitem__(self, idx):
        return self.pressure_data[idx], self.skeleton_data[idx], self.decoder_input[idx]


class EnhancedSkeletonTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_joints, num_dims=3, dropout=0.1, seq_length=3,
                 window_size=5):
        super().__init__()

        # クラス属性としてnum_jointsを保存
        self.num_joints = num_joints
        self.num_dims = num_dims
        self.seq_length = seq_length
        self.window_size = window_size

        # 入力の特徴抽出を強化
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, d_model)
        )

        # より深いTransformerネットワーク
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.decoder_feature_extractor = nn.Sequential(
            nn.Linear(num_joints * num_dims, d_model)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_encoder_layers
        )
        self.predict = nn.Sequential(
            nn.Linear(d_model * seq_length, d_model * window_size)
        )

        # 出力層の強化
        self.output_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_joints * num_dims)
        )

        self.constraint = SkeletonConstraints(num_joints)
        # スケール係数（学習可能パラメータ）
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, decoder_input):
        batch_size = x.shape[0]

        # 特徴抽出
        features = self.feature_extractor(x)
        features = features.unsqueeze(1)
        decoder_input = self.decoder_feature_extractor(decoder_input)
        decoder_input = add_positional_encoding(decoder_input)

        # Transformer処理
        transformer_output = self.transformer_encoder(features)
        transformer_output = self.transformer_decoder(decoder_input, transformer_output)
        # predict = transformer_output[:,transformer_output.shape[1]-1,:]

        predict = transformer_output.reshape(batch_size, -1)
        predict_next = self.predict(predict)
        predict_next = predict_next.reshape(batch_size, self.window_size, -1)

        # 出力生成とスケーリング
        output = self.output_decoder(predict_next)
        # output = self.constraint(output)
        output = output * self.output_scale  # 出力のスケーリング

        return output


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_path, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for pressure, skeleton, decoder_input in train_loader:
            # データをGPUに移動
            pressure = pressure.to(device)
            skeleton = skeleton.to(device)
            decoder_input = decoder_input.to(device)

            optimizer.zero_grad()

            pressure = add_gaussian_noise(pressure, noise_factor=0.1)
            decoder_input = add_gaussian_noise(decoder_input, noise_factor=0.1)
            if torch.rand(1).item() < 0.95:
                decoder_input=torch.zeros_like(decoder_input)
            outputs = model(pressure, decoder_input)
            loss = criterion(outputs, skeleton)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for pressure, skeleton, decoder_input in val_loader:
                # データをGPUに移動
                pressure = pressure.to(device)
                skeleton = skeleton.to(device)
                decoder_input = decoder_input.to(device)

                outputs = model(pressure, decoder_input)
                loss = criterion(outputs, skeleton)
                val_loss += loss.item()

        # 平均損失の計算
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # スケジューラのステップ
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch + 1}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Learning Rate: {current_lr:.6f}')

        # モデルの保存
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, save_path)
            print(f'Model saved at epoch {epoch + 1}')

        print('-' * 60)


# class SkeletonLoss(nn.Module):
#     def __init__(self, joint_connections):
#         super().__init__()
#         self.joint_connections = joint_connections

#     def forward(self, pred, target):
#         # MSE損失
#         mse_loss = F.mse_loss(pred, target)

#         # 骨格の長さの一貫性を保つための損失
#         bone_length_loss = self.calculate_bone_length_loss(pred, target)

#         # 関節角度の制約に関する損失
#         angle_loss = self.calculate_angle_loss(pred)

#         return mse_loss + 0.1 * bone_length_loss + 0.1 * angle_loss


def load_model(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

    return model, optimizer, scheduler, epoch, best_val_loss


# モデルの推論
def predict(model, pressure_data):
    model.eval()
    with torch.no_grad():
        pressure_tensor = torch.FloatTensor(pressure_data)
        predictions = model(pressure_tensor)
    return predictions.numpy()


class SkeletonConstraints(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self.num_joints = num_joints

        # 定义关节层级（父子关系）
        self.joint_hierarchy = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 背骨
            (5, 6), (6, 7), (7, 8),  # 腕和肩（示例）
            (9, 10), (10, 11), (11, 12), (5, 9),  # 另一侧的腕和肩（示例）
            (13, 14), (14, 15), (15, 16),  # 腿的一侧（示例）
            (17, 18), (18, 19), (19, 20), (13, 17)  # 另一侧的腿（示例）
        ]

        # 每条骨骼的标准长度，长度与joint_hierarchy的数量保持一致
        self.bone_lengths = nn.Parameter(torch.ones(len(self.joint_hierarchy)), requires_grad=False)

    def forward(self, skeleton_pred):
        """
        skeleton_pred: 张量，形状 (batch_size, num_joints*3)
        """
        batch_size = skeleton_pred.shape[0]
        skeleton_3d = skeleton_pred.view(batch_size, self.num_joints, 3)

        # 初始化约束后的骨架，将根关节位置复制过来（假定索引 0 是根关节）
        constrained_skeleton = torch.zeros_like(skeleton_3d)
        constrained_skeleton[:, 0] = skeleton_3d[:, 0]

        # 遍历关节层级，逐步传播骨骼约束
        for idx, (parent, child) in enumerate(self.joint_hierarchy):
            # 计算原始骨架中父子关节之间的向量
            bone_vector = skeleton_3d[:, child] - skeleton_3d[:, parent]
            # 计算骨骼向量的长度，并防止除零
            bone_length = torch.norm(bone_vector, dim=1, keepdim=True)
            bone_length = torch.clamp(bone_length, min=1e-6)
            # 使用该骨骼的标准长度
            desired_length = self.bone_lengths[idx]
            # 对向量归一化后乘以期望长度
            normalized_bone = bone_vector * (desired_length / bone_length)
            # 利用父关节在约束骨架中的位置确定子关节的位置
            constrained_skeleton[:, child] = constrained_skeleton[:, parent] + normalized_bone

        return constrained_skeleton.view(batch_size, -1)

def compute_exponential_weights(k, m):
    """计算指数衰减权重 w_i = exp(-m * i)，并归一化"""
    indices = torch.arange(k)  # 生成 i = 0, 1, ..., k-1
    weights = torch.exp(-m * indices)  # 计算 w_i
    return weights / weights.sum()  # 归一化，使得所有权重之和为 1
