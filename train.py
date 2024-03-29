from backbone.lora import ModifiedIResNet50
import torch
from insightface.dataset import get_dataloader  # 仮の関数名、実際の関数名に置き換えてください
from insightface.losses import CombinedMarginLoss
import torch.optim as optim
from backbone.iresnet import iresnet50


def train():
    # パラメータの設定
    num_classes = 10
    rank = 64
    learning_rate = 1e-10
    epochs = 10  # トレーニングのエポック数
    batch_size = 10  # バッチサイズ
    local_rank = 0  # ローカルランク（単一GPUの場合は0）
    num_workers = 0  # データローディングに使用するワーカー数

    # 状態辞書をロード
    model = torch.load('./models/arcface_w600k_r50.pth')
    if torch.cuda.is_available():
        model.to('cuda')

    # ModifiedIResNet50のインスタンスを作成
    # model = ModifiedIResNet50(model, num_classes, rank)

    # データローダーの準備
    train_loader = get_dataloader(
        root_dir='./prepared_data',  # データセットのディレクトリ
        local_rank=local_rank,
        batch_size=batch_size,
        num_workers=num_workers
    )
    # 損失関数の定義
    criterion = CombinedMarginLoss(s=64, m1=1.0, m2=0.5, m3=0.0)

    # オプティマイザーの設定（LoRA層のみを更新）
    # optimizer = optim.Adam(model.lora_fc.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    loss = None  # デフォルト値をNoneに設定
    for epoch in range(epochs):
        for data, target in train_loader:
            # データとターゲットをCUDAに移動
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion.forward(output, target)
            loss = loss.clamp_min_(1e-30).log_().mean() * (-1)
            print(loss)
            loss.backward()
            optimizer.step()

        # エポックの終わりに損失を表示
        if loss is not None:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        else:
            print(f'Epoch {epoch+1}, No data processed.')

    # トレーニングループの後にモデルを保存
    model_save_path = './models/trained_model.pth'  # 保存先のパス
    torch.save(model, model_save_path)
    print(f'Model saved to {model_save_path}')


if __name__ == "__main__":
    train()