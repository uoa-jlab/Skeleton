import torch
import torch.nn as nn
import torch.nn.functional as F
class EnhancedSkeletonLoss_WithAngleConstrains(nn.Module):
    def __init__(self, alpha=1.0, gamma=0.5, window_size=5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.window_size = window_size

    def forward(self, pred, target):
        # MSE損失
        # 假设 pred 和 target 的原始形状为 [batch_size, window_size * num_joints * 3]
        batch_size = int(pred.shape[0])
        num_joints = int(pred.shape[2] // 3)

        # 重塑为 [batch_size, window_size, num_joints, 3]
        pred_reshaped = pred.view(batch_size, self.window_size, num_joints, 3)
        target_reshaped = target.view(batch_size, self.window_size, num_joints, 3)

        # 定义关节点权重，默认全部为 1.0
        joint_weights = torch.tensor(0.2, device=pred.device) * torch.ones(num_joints, device=pred.device)
        # 对背骨的关节点（索引 0～4）赋予较高权重
        for idx in [0, 1, 2, 3, 4]:
            joint_weights[idx] = 2.0
        # 对两个腿的关节点（例如：一侧 13～16，另一侧 17～20）赋予较高权重
        for idx in [13, 14, 15, 16, 17, 18, 19, 20]:
            joint_weights[idx] = 2.0

        # 计算加权均方误差
        # 先计算每个坐标的平方误差，形状为 [batch_size, window_size, num_joints, 3]
        squared_diff = (pred_reshaped - target_reshaped) ** 2
        # 对坐标求和得到每个关节的误差，形状为 [batch_size, window_size, num_joints]
        squared_diff = squared_diff.sum(dim=-1)
        # 将关节点的权重扩展到 [1, 1, num_joints] 后相乘
        weighted_squared_diff = squared_diff * joint_weights.view(1, 1, num_joints)
        # 平均得到加权的均方误差
        mse_loss = weighted_squared_diff.mean()

        eps = 1e-6
        angle_loss = 0.0
        angle_pairs = [
            ((0, 1), (1, 2)),
            ((1, 2), (2, 3)),
            ((2, 3), (3, 4)),
            ((13, 17), (17, 18)),
            ((13, 17), (13, 14)),
            ((17, 18), (18, 19)),
            ((18, 19), (19, 20)),
            ((13, 14), (14, 15)),
            ((14, 15), (15, 16))
        ]
        for (bone1, bone2) in angle_pairs:
            # 预测向量计算
            pred_vec1 = pred_reshaped[:, :, bone1[1], :] - pred_reshaped[:, :, bone1[0], :]
            pred_vec2 = pred_reshaped[:, :, bone2[1], :] - pred_reshaped[:, :, bone2[0], :]
            dot_pred = (pred_vec1 * pred_vec2).sum(dim=-1)
            norm_pred1 = torch.norm(pred_vec1, dim=-1)
            norm_pred2 = torch.norm(pred_vec2, dim=-1)
            cos_pred = dot_pred / (norm_pred1 * norm_pred2 + eps)
            cos_pred = torch.clamp(cos_pred, -1.0, 1.0)

            # 目标向量计算
            target_vec1 = target_reshaped[:, :, bone1[1], :] - target_reshaped[:, :, bone1[0], :]
            target_vec2 = target_reshaped[:, :, bone2[1], :] - target_reshaped[:, :, bone2[0], :]
            dot_target = (target_vec1 * target_vec2).sum(dim=-1)
            norm_target1 = torch.norm(target_vec1, dim=-1)
            norm_target2 = torch.norm(target_vec2, dim=-1)
            cos_target = dot_target / (norm_target1 * norm_target2 + eps)
            cos_target = torch.clamp(cos_target, -1.0, 1.0)

            # 直接比较余弦值的差异
            angle_loss += F.mse_loss(cos_pred, cos_target)
        angle_loss = angle_loss / len(angle_pairs)

        return self.alpha * mse_loss + self.gamma * angle_loss


class EnhancedSkeletonLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # MSE損失
        mse_loss = F.mse_loss(pred, target)
        batch_size = int(pred.shape[0])
        # 変化量の損失
        motion_loss = F.mse_loss(
            pred[1:] - pred[:-1],
            target[1:] - target[:-1]
        )
        return self.alpha * mse_loss / batch_size
