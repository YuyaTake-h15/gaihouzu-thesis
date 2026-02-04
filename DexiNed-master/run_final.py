import os
import cv2
import torch
import numpy as np
import sys

# ==========================================
# 1. 設定
# ==========================================
# ★重要: あなたの環境に合わせてパスを設定
BASE_DIR = r'C:/Users/TakedaYuya/Landmark_Gaihouzu_new'
INPUT_IMG_PATH = os.path.join(BASE_DIR, 'input/NI_52_11_8.jpg')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
# 重みファイル（絶対パスで指定）
CHECKPOINT_PATH = r'C:\Users\TakedaYuya\Landmark_Gaihouzu_new\DexiNed-master\checkpoints\10_model.pth'

# ==========================================
# 2. 正規の model.py を読み込む準備
# ==========================================
# 現在のフォルダパスをシステムに追加して、model.py をインポートできるようにする
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from model import DexiNed
    print("✅ model.py から DexiNed を読み込みました")
except ImportError:
    print(f"❌ エラー: 同じフォルダに 'model.py' が見つかりません。")
    print(f"   現在の場所: {current_dir}")
    sys.exit()

# ==========================================
# 3. 実行処理
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- 画像ロード ---
    if not os.path.exists(INPUT_IMG_PATH):
        print(f"❌ エラー: 入力画像がありません: {INPUT_IMG_PATH}")
        return
    
    img_org = cv2.imread(INPUT_IMG_PATH)
    if img_org is None:
        print("❌ エラー: 画像ファイルが壊れているか読み込めません")
        return
        
    h, w = img_org.shape[:2]
    print(f"Original Size: {w}x{h}")

    # --- リサイズ (メモリ対策) ---
    # DexiNedは画像サイズが大きいとCPU/GPUメモリを大量に食うため、一旦縮小します
    # 精度を上げたい場合は (1024, 1024) などにしてください
    process_size = (512, 512) 
    img_resized = cv2.resize(img_org, process_size)
    
    # テンソル化 (DexiNedの仕様に合わせる)
    img_tensor = torch.from_numpy(img_resized).float()
    
    # 配列の並び替え (H,W,C) -> (C,H,W)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # 正規化（平均値を引く）
    img_tensor -= torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1)
    img_tensor = img_tensor.to(device)

    # --- モデルロード ---
    model = DexiNed().to(device)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ エラー: 重みファイルが見つかりません: {CHECKPOINT_PATH}")
        return

    try:
        # 重みのロード
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        model.eval()
        print("✅ モデルの重みを正常にロードしました")
    except Exception as e:
        print("❌ モデルロード中にエラーが発生しました:")
        print(e)
        return

    # --- 推論実行 ---
    print("⏳ 推論中... (CPUの場合は数分かかることがあります)")
    with torch.no_grad():
        preds = model(img_tensor)
        # DexiNedは複数のスケールのリストを返します。最後(-1)または融合結果を使うのが一般的
        # model.pyの仕様によっては preds がリストの場合とテンソルの場合があります
        if isinstance(preds, list):
            pred_fuse = preds[-1] # リストの最後を取得
        else:
            pred_fuse = preds
            
    # --- 結果の保存 ---
    # シグモイド関数で 0.0~1.0 の確率に変換
    pred_fuse = torch.sigmoid(pred_fuse).cpu().numpy()
    
    # バッチ次元などを削除して2次元画像データにする
    if pred_fuse.ndim == 4:
        pred_fuse = pred_fuse[0, 0]
    elif pred_fuse.ndim == 3:
        pred_fuse = pred_fuse[0]

    # 元のサイズに戻す
    pred_fuse = cv2.resize(pred_fuse, (w, h))
    
    # 0-255の整数に変換
    result = (pred_fuse * 255).astype(np.uint8)
    
    # 色を反転（白地に黒線にしたい場合）
    # result = cv2.bitwise_not(result) 
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'dexined_edge.png')
    cv2.imwrite(out_path, result)
    print(f"🎉 成功！エッジ画像を保存しました: {out_path}")

if __name__ == '__main__':
    main()