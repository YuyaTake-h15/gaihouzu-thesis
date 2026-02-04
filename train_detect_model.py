#yoloモデル学習コード
from ultralytics import YOLO

# 使用するモデル（小さくて速いyolov8nを使用。必要に応じて yolov8s, yolov8m などに変更可）
model = YOLO("yolo8n.pt")

# 学習設定
data_yaml_path = r"C:/Users/TakedaYuya/Landmark_Gaihouzu_new/data.yaml"
epochs = 100
batch_size = 16
img_size = 256  # 解像度（必要に応じて変更）

# モデルの訓練
model.train(
    data=data_yaml_path,
    epochs=epochs,
    batch=batch_size,
    imgsz=img_size,
    project="Gaihouzu_Symbol",  # 出力先フォルダ
    name="map_symbol_detect",
    exist_ok=True  # 既存フォルダに上書きしてもOK
)
