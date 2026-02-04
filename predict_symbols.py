#記号判別モデルNo.1

import os
import cv2
import csv
from ultralytics import YOLO
import numpy as np
import torch 

# --- 1. 基本設定 ---
TILE_SIZE = 256
STRIDE = 192
BASE_DIR = 'C:/Users/TakedaYuya/Landmark_Gaihouzu_new'
MODEL_PATH = r'Gaihouzu\detect_landmark\weights\best.pt'
ORIGINAL_IMAGE_DIR = os.path.join(BASE_DIR, 'input')
TILE_IMAGE_DIR = os.path.join(BASE_DIR, 'input_split')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CONF_THRESHOLD = 0.25

# ★追加設定: 重複削除のしきい値
IOU_THRESHOLD = 0.5  # 重なりが50%以上なら「同じ物体」とみなす

# --- 2. 関数定義 ---

def create_tile_offset_lookup(original_image_dir, tile_size, stride):
    # (前回と同じため省略しますが、そのまま使ってください)
    print(f"'{original_image_dir}' をスキャンしてオフセット対応表を作成中...")
    offset_lookup = {}
    for fname in os.listdir(original_image_dir):
        if not (fname.endswith('.jpg') or fname.endswith('.png')): continue
        img_path = os.path.join(original_image_dir, fname)
        try:
            image = cv2.imread(img_path)
            if image is None: continue
            h, w = image.shape[:2]
        except: continue
        basename = os.path.splitext(fname)[0]
        tile_id = 0
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                tile_h = min(y + tile_size, h) - y
                tile_w = min(x + tile_size, w) - x
                if tile_h < tile_size // 2 or tile_w < tile_size // 2: continue
                tile_name_key = f"{basename}_{tile_id}"
                offset_lookup[tile_name_key] = (x, y)
                tile_id += 1
    return offset_lookup

def convert_tile_to_full_coords(tile_basename, tile_bbox, offset_lookup):
    if tile_basename not in offset_lookup: return None
    offset_x, offset_y = offset_lookup[tile_basename]
    xmin, ymin, xmax, ymax = tile_bbox
    return (offset_x + xmin, offset_y + ymin, offset_x + xmax, offset_y + ymax)

# ★追加関数: IoU計算
def calculate_iou(box1, box2):
    # box = [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    if union == 0: return 0
    return intersection / union

# ★追加関数: NMS (重複削除)
def non_max_suppression_custom(detections, iou_threshold):
    if not detections: return []
    
    # 信頼度が高い順にソート
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    keep = []
    
    while detections:
        # 一番信頼度が高いものを「残すリスト(keep)」に入れる
        current = detections.pop(0)
        keep.append(current)
        
        # 残りのリストから、今選んだものと「場所が被っている」ものを削除する
        non_overlapping = []
        for det in detections:
            # 違う画像ファイルなら比較しない
            if det['original_image'] != current['original_image']:
                non_overlapping.append(det)
                continue
            
            # クラスが違うなら比較しない（神社と学校が重なっていても消さない）
            if det['label'] != current['label']:
                non_overlapping.append(det)
                continue

            # IoUを計算
            iou = calculate_iou(current['full_bbox'], det['full_bbox'])
            
            # 重なりが小さければ残す（重なりが大きければ削除＝リストに入れない）
            if iou < iou_threshold:
                non_overlapping.append(det)
        
        detections = non_overlapping
    
    return keep

# --- 3. メイン実行 ---

# (モデル読み込みなどは同じ)
model = YOLO(MODEL_PATH)
tile_offset_map = create_tile_offset_lookup(ORIGINAL_IMAGE_DIR, TILE_SIZE, STRIDE)

print(f"\n--- 推論実行 ---")
tile_image_paths = [os.path.join(TILE_IMAGE_DIR, f) for f in os.listdir(TILE_IMAGE_DIR) if f.endswith(('.jpg', '.png'))]
if not tile_image_paths: exit()

results_list = model(tile_image_paths, conf=CONF_THRESHOLD, verbose=False) # verbose=Falseでログ抑制

# 全体座標への変換
all_detections_raw = []

for results in results_list:
    tile_full_path = results.path
    tile_fname = os.path.basename(tile_full_path)
    tile_basename = os.path.splitext(tile_fname)[0]
    
    # 元画像名の抽出ロジック(安全策)
    parts = tile_basename.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        original_image_name = '_'.join(parts[:-1])
    else:
        original_image_name = tile_basename

    boxes = results.boxes
    for box in boxes:
        bbox_tile = box.xyxy[0].cpu().tolist()
        bbox_full = convert_tile_to_full_coords(tile_basename, bbox_tile, tile_offset_map)
        
        if bbox_full:
            class_id = int(box.cls[0].cpu())
            all_detections_raw.append({
                "original_image": original_image_name,
                "label": model.names[class_id],
                "confidence": float(box.conf[0].cpu()),
                "full_bbox": bbox_full,
                "source_tile": tile_fname
            })

print(f"\n変換完了: 重複削除前 {len(all_detections_raw)} 件")

# --- ★ステップ4.5: 重複削除 (NMS) の実行 ---
print("--- ステップ4.5: 重複削除処理を実行中 ---")
final_detections = non_max_suppression_custom(all_detections_raw, IOU_THRESHOLD)

removed_count = len(all_detections_raw) - len(final_detections)
print(f"重複削除完了: {removed_count} 件を削除しました。")
print(f"最終検出数: {len(final_detections)} 件")

# --- ステップ5: 保存 ---
output_csv_path = os.path.join(OUTPUT_DIR, 'all_detections_full_coords_clean.csv')
headers = ["original_image", "label", "confidence", "full_Xmin", "full_Ymin", "full_Xmax", "full_Ymax", "source_tile"]

with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for det in final_detections:
        bbox = det['full_bbox']
        row = [
            det['original_image'], det['label'], det['confidence'],
            bbox[0], bbox[1], bbox[2], bbox[3], det['source_tile']
        ]
        writer.writerow(row)

print(f"保存完了: {output_csv_path}")