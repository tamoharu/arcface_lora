import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, alpha=1, beta=1):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        # LoRAのパラメータ
        self.A = nn.Parameter(torch.randn(output_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, input_dim))
        # 重みの初期化
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, x, original_weight):
        # LoRAの適用
        delta_weight = self.alpha * torch.matmul(self.A, self.B)
        return F.linear(x, original_weight + self.beta * delta_weight)


class ModifiedIResNet50(nn.Module):
    def __init__(self, original_model, num_classes, rank):
        super(ModifiedIResNet50, self).__init__()
        # 元のモデルの層をコピー（最終層を除く）
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
        # 最終全結合層を取得
        # original_modelの構造に応じて、最終全結合層を正しく参照するコードを書く
        self.original_fc = original_model.fc  # 例: original_modelがfc属性を持っている場合
        
        # LoRA層の追加
        self.lora_fc = LoRALayer(self.original_fc.in_features, num_classes, rank)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        # 元の重みをLoRA層に渡す
        x = self.lora_fc(x, self.original_fc.weight)
        return x