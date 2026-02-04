#元画像タイル分割用
import os
import cv2
import numpy as np

# --- 1. 設定 ---
tile_size = 256  # 分割サイズ
stride = 192   # 重なり (256 - 64 = 192)
base_input_dir = 'C:/Users/TakedaYuya/Landmark_Gaihouzu_new'

# --- 2. inputディレクトリのタイル化処理 ---
# ( C:\...\input を C:\...\input_split にタイル化します )

input_image_dir = os.path.join(base_input_dir, 'input')
output_image_dir = os.path.join(base_input_dir, 'input_split')
os.makedirs(output_image_dir, exist_ok=True)

print(f"--- 'input' フォルダのタイル化を開始 ---")
print(f"読み込み元: {input_image_dir}")
print(f"保存先: {output_image_dir}")

total_files = 0
total_tiles = 0

for fname in os.listdir(input_image_dir):
    if not (fname.endswith('.jpg') or fname.endswith('.png')):
        continue
    
    # NI_52_11_8.jpg などのファイルを処理
    print(f"\n処理中のファイル: {fname}")
    img_path = os.path.join(input_image_dir, fname)
    image = cv2.imread(img_path)
    if image is None:
        print(f"  画像が読み込めませんでした: {img_path}")
        continue

    h, w = image.shape[:2]
    basename = os.path.splitext(fname)[0]
    total_files += 1
    tile_id = 0

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = image[y:min(y + tile_size, h), x:min(x + tile_size, w)]
            tile_h, tile_w = tile.shape[:2]
            if tile_h < tile_size // 2 or tile_w < tile_size // 2:
                continue  # あまりにも小さい端のタイルはスキップ

            # (例) NI_52_11_8_0.jpg, NI_52_11_8_1.jpg ... という名前で保存
            out_img_path = os.path.join(output_image_dir, f"{basename}_{tile_id}.jpg")
            cv2.imwrite(out_img_path, tile)
            tile_id += 1
    
    print(f"  -> {tile_id} 枚のタイルを生成しました。")
    total_tiles += tile_id

print(f"\n--- タイル化完了 ---")
print(f"合計 {total_files} 個のファイルを処理し、{total_tiles} 枚のタイルを 'input_split' に保存しました。")