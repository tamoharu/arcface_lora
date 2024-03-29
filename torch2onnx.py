import torch


model_path = "../models/arcface_w600k_r50.pth"
model = torch.load(model_path)
model.eval()
dummy_input = torch.randn(1, 3, 112, 112)
torch.onnx.export(model,               # ロードしたモデル
                  dummy_input,         # モデルの入力（ダミー）
                  "model.onnx",        # 出力ファイルのパス
                  export_params=True,  # モデルの学習済みパラメータを含める
                #   opset_version=11,    # 使用するONNXのバージョン
                #   do_constant_folding=True,  # 定数畳み込みの最適化を行う
                  input_names=['input'],   # 入力の名前
                  output_names=['output'],  # 出力の名前
                #   dynamic_axes={'input': {0: 'batch_size'},  # バッチサイズを動的に
                #                 'output': {0: 'batch_size'}})
)