import os
import cv2
import torch
import numpy as np
import sys
import math
import pandas as pd

# ==========================================
# 1. 設定
# ==========================================
BASE_DIR = r'C:/Users/TakedaYuya/Landmark_Gaihouzu_new'
INPUT_IMG_PATH = os.path.join(BASE_DIR, 'input/NI-52-11-14.jpg')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
CHECKPOINT_PATH = r'C:\Users\TakedaYuya\Landmark_Gaihouzu_new\DexiNed-master\checkpoints\10_model.pth'
YOLO_CSV_PATH = r'C:\Users\TakedaYuya\Landmark_Gaihouzu_new\output\all_detections_full_coords_clean.csv'

TILE_SIZE = 512
STRIDE = 512

# ==========================================
# 2. モデル読み込み準備
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from model import DexiNed
    print(" model.py から DexiNed を読み込みました")
except ImportError:
    print(f" エラー: 'model.py' が見つかりません。")
    sys.exit()

# ==========================================
# 3. 関数定義
# ==========================================
def infer_one_tile(model, tile_img, device):
    """1枚のタイルを受け取り、エッジ画像を返す"""
    h, w = tile_img.shape[:2]
    img_tensor = torch.from_numpy(tile_img).float()
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor -= torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        preds = model(img_tensor)
        if isinstance(preds, list): pred_fuse = preds[-1]
        else: pred_fuse = preds
            
    pred_fuse = torch.sigmoid(pred_fuse).cpu().numpy()
    if pred_fuse.ndim == 4: pred_fuse = pred_fuse[0, 0]
    elif pred_fuse.ndim == 3: pred_fuse = pred_fuse[0]
    
    result = (pred_fuse * 255).astype(np.uint8)
    return result

def mask_symbols_with_yolo(img, csv_path):
    """
    YOLOの結果を使って、記号部分を周囲の色でインペインティング（修復）する
    これにより「四角い枠」が出るのを防ぐ
    """
    if not os.path.exists(csv_path):
        print(f" YOLO結果CSVが見つかりません: {csv_path}")
        return img
    
    print(f"🧹 インペインティング処理で地図記号を消去します... ({csv_path})")
    
    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception as e:
        print(f" CSV読み込みエラー: {e}")
        return img

    # マスク画像を作成（黒背景）
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    count = 0
    margin = 3 # 記号より少し広めに指定して、記号全体をカバーする

    for index, row in df.iterrows():
        try:
            if isinstance(row[3], str):
                try: float(row[3])
                except ValueError: continue

            # 座標取得 (3:xmin, 4:ymin, 5:xmax, 6:ymax)
            xmin = int(float(row[3]))
            ymin = int(float(row[4]))
            xmax = int(float(row[5]))
            ymax = int(float(row[6]))
            
            # 画像範囲制限
            xmin = max(0, xmin - margin)
            ymin = max(0, ymin - margin)
            xmax = min(w, xmax + margin)
            ymax = min(h, ymax + margin)
            
            # ★修正点: マスク画像の該当箇所を「白」にする
            # 記号の場所だけを白く塗ったマスクを作る
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
            
            count += 1
        except Exception:
            continue

    if count == 0:
        print(" マスク対象が見つかりませんでした。")
        return img

    print(f" 合計 {count} 箇所の記号を修復対象としてセットしました。")
    print(" インペインティング実行中... (これには数秒〜数分かかります)")
    
    # ★インペインティング実行
    # cv2.INPAINT_TELEA: Fast Marching Methodに基づく手法（高速で自然）
    # radius=3: 周囲3ピクセルの色を参照して修復
    inpainted_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    return inpainted_img

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 画像ロード
    if not os.path.exists(INPUT_IMG_PATH):
        print(f" 画像なし: {INPUT_IMG_PATH}")
        return
    img_org = cv2.imread(INPUT_IMG_PATH)
    full_h, full_w = img_org.shape[:2]
    
    # ==========================================
    # ★追加: インペインティング処理
    # ==========================================
    # 記号を「周囲の色」で埋めて消す
    target_img = mask_symbols_with_yolo(img_org, YOLO_CSV_PATH)
    
    # 確認用保存
    debug_path = os.path.join(OUTPUT_DIR, 'debug_inpainted_input.jpg')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cv2.imwrite(debug_path, target_img)
    print(f" インペインティング結果を確認用に保存しました: {debug_path}")
    # ==========================================

    # 2. モデルロード
    model = DexiNed().to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        print(" 重みファイルなし")
        return
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # 3. 推論
    full_edge_map = np.zeros((full_h, full_w), dtype=np.uint8)
    total_tiles = math.ceil(full_h / STRIDE) * math.ceil(full_w / STRIDE)
    count = 0

    print(f" タイル処理開始 (Size={TILE_SIZE}, Stride={STRIDE}, Total={total_tiles} tiles)")

    for y in range(0, full_h, STRIDE):
        for x in range(0, full_w, STRIDE):
            count += 1
            print(f"\r Processing tile {count}/{total_tiles}...", end="")

            y_end = min(y + TILE_SIZE, full_h)
            x_end = min(x + TILE_SIZE, full_w)
            
            tile = target_img[y:y_end, x:x_end]
            h_crop, w_crop = tile.shape[:2]
            
            pad_h = TILE_SIZE - h_crop
            pad_w = TILE_SIZE - w_crop
            
            if pad_h > 0 or pad_w > 0:
                tile_padded = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
            else:
                tile_padded = tile

            edge_tile = infer_one_tile(model, tile_padded, device)
            valid_edge = edge_tile[0:h_crop, 0:w_crop]
            full_edge_map[y:y_end, x:x_end] = valid_edge

    print("\n 完了")
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'dexined_edge.png'), full_edge_map)

if __name__ == '__main__':
    main()