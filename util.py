import math
import torch
def add_gaussian_noise(data, noise_factor=0.1):
    std = torch.std(data, dim=0, keepdim=True)  # 计算原始数据的标准差
    noise = torch.randn_like(data) * (std * noise_factor)  # 生成噪声
    return data + noise
def add_positional_encoding(x):
    """
    x: Tensor of shape (batch_size, seq_len, d_model)
    """
    batch_size, seq_len, d_model = x.shape

    # 初始化位置编码矩阵 (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model, device=x.device)

    position = torch.arange(seq_len, device=x.device).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device) * (-math.log(10000.0) / d_model))  # (d_model/2)

    # 处理嵌入维度为奇数的情况
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:(d_model//2)])  # 防止越界

    # 添加维度以便广播： (1, seq_len, d_model)
    pe = pe.unsqueeze(0)

    # 加到输入上 (batch_size, seq_len, d_model)
    return x + pe


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
            decoder_input = add_positional_encoding(decoder_input)

            pressure = add_gaussian_noise(pressure, noise_factor=0.1)
            decoder_input = add_gaussian_noise(decoder_input, noise_factor=0.1)
            if torch.rand(1).item() < 0.95:
                decoder_input = torch.zeros_like(decoder_input)
            outputs = model(pressure, decoder_input)
            loss = criterion(outputs, skeleton)

            optimizer.zero_grad()
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

def compute_exponential_weights(k, m):
    """计算指数衰减权重 w_i = exp(-m * i)，并归一化"""
    indices = torch.arange(k)  # 生成 i = 0, 1, ..., k-1
    weights = torch.exp(-m * indices)  # 计算 w_i
    return weights / weights.sum()  # 归一化，使得所有权重之和为 1

def load_model(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

    return model, optimizer, scheduler, epoch, best_val_loss