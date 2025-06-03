import torch
import torch.nn as nn

from util import add_positional_encoding


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
        # スケール係数（学習可能パラメータ）
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, decoder_input):
        batch_size = x.shape[0]

        # 特徴抽出
        features = self.feature_extractor(x)
        features = features.unsqueeze(1)
        decoder_input = self.decoder_feature_extractor(decoder_input)
        #decoder_input = add_positional_encoding(decoder_input)

        # Transformer処理
        transformer_output = self.transformer_encoder(features)
        transformer_output = self.transformer_decoder(decoder_input, transformer_output)
        # predict = transformer_output[:,transformer_output.shape[1]-1,:]

        predict = transformer_output.reshape(batch_size, -1)
        predict_next = self.predict(predict)
        predict_next = predict_next.reshape(batch_size, self.window_size, -1)

        # 出力生成とスケーリング
        output = self.output_decoder(predict_next)
        output = output * self.output_scale  # 出力のスケーリング

        return output