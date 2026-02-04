#地図を半分にリサイズする
import cv2
import os

# --- 1. 設定 ---
# あなたの環境に合わせたパス
BASE_DIR = 'C:/Users/TakedaYuya/Landmark_Gaihouzu_new'
#INPUT_IMAGE_DIR = os.path.join(BASE_DIR, 'img')
#OUTPUT_IMAGE_DIR = os.path.join(BASE_DIR, 'img_resized')

INPUT_IMAGE_DIR = os.path.join(BASE_DIR, 'gaihouzu_another')
OUTPUT_IMAGE_DIR = os.path.join(BASE_DIR, 'gaihouzu_resized')

# 処理したいJPGファイル名
#INPUT_FILENAME = "NI_52_11_8_full.jpg" # (★必要に応じてこのファイル名を変更)
#INPUT_FILENAME = "NI-52-12-14.jpeg" 
#INPUT_FILENAME = "NI-52-11-13.jpeg"

# 出力ファイル名 (例: NI_52_11_8_half_size.jpg)
OUTPUT_FILENAME = f"{os.path.splitext(INPUT_FILENAME)[0]}_half_size.jpg"

# 入力と出力のフルパスを構築
img_path = os.path.join(INPUT_IMAGE_DIR, INPUT_FILENAME)
out_path = os.path.join(OUTPUT_IMAGE_DIR, OUTPUT_FILENAME)

print(f"--- 画像リサイズ開始 ---")
print(f"入力画像: {img_path}")

# --- 2. 画像の読み込み ---
image = cv2.imread(img_path)

if image is None:
    print(f"エラー: 画像が読み込めませんでした: {img_path}")
else:
    # --- 3. 元のサイズを取得 ---
    # image.shape は (高さ, 幅, 色チャネル) の順
    original_height, original_width = image.shape[:2]
    print(f"元のサイズ: 幅={original_width}, 高さ={original_height}")

    # --- 4. 新しいサイズ（半分）を計算 ---
    # 整数除算 (//) を使ってピクセル数を整数にします
    new_width = original_width // 2
    new_height = original_height // 2
    print(f"新しいサイズ: 幅={new_width}, 高さ={new_height}")

    # --- 5. リサイズ実行 ---
    # cv2.INTER_AREA は画像を「縮小」する際に推奨される補間方法です
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # --- 6. 保存 ---
    try:
        # outputフォルダがなければ作成
        os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
        # 新しい画像を保存
        cv2.imwrite(out_path, resized_image)
        print(f"リサイズした画像を保存しました: {out_path}")
    except Exception as e:
        print(f"エラー: 画像の保存に失敗しました。 {e}")

print("--- 処理完了 ---")