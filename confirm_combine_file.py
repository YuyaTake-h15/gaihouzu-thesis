import os
import cv2
import csv
import random # 色をランダムに生成するため
"""
座標変更結果を元画像にドットで描画するスクリプト
---------------------------------------------------------
機能:
1. ファイル設定
2. 地物の正解データと比較データの読み込み
3. 正解データと比較データの距離差を計算
4. 距離差が指定の距離未満だとTrue、以上だとFalseとして現存判定を行う
5. 結果をCSVファイルとして保存
"""
# --- 1. 基本設定 ---
# （あなたの環境に合わせて確認してください）

BASE_DIR = 'C:/Users/TakedaYuya/Landmark_Gaihouzu_new'
ORIGINAL_IMAGE_DIR = os.path.join(BASE_DIR, 'input') # 元の巨大画像が入っているフォルダ
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')        # 検出結果CSVがあるフォルダ
VISUALIZE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'visualize_full_coords') # 描画結果の保存先

# 検出結果のCSVファイルパス
DETECTIONS_CSV_PATH = os.path.join(OUTPUT_DIR, 'all_detections_full_coords_clean.csv')

# ドットの描画設定
DOT_RADIUS = 5  # ドットの半径（ピクセル）
DOT_THICKNESS = -1 # -1 で塗りつぶし

# --- ★★★ ここを修正しました ★★★ ---
# ラベルごとの色を定義 (YAMLファイルのクラス名に一致させる)
# names: ['shrine', 'temple', 'rice_field','tea_field','mulberry_field','school','orchard']
LABEL_COLORS = {
    'shrine': (0, 0, 255),         # 神社 (赤)
    'temple': (0, 255, 0),         # 寺 (緑)
    'rice_field': (0, 255, 255),   # 田 (シアン)
    'tea_field': (255, 255, 0),    # 茶畑 (黄)
    'mulberry_field': (255, 0, 255), # 桑畑 (マゼンタ)
    'school': (255, 0, 0),         # 学校 (青)
    'orchard': (128, 128, 128)     # 果樹園 (灰色)
    # 登録されていないラベルはランダムな色になります
}
# --- ★★★ 修正ここまで ★★★ ---

# --- 2. 補助関数 ---

def get_color_for_label(label):
    """
    ラベル名に対応する色を返す。未登録ならランダムに生成。
    """
    if label in LABEL_COLORS:
        return LABEL_COLORS[label]
    else:
        # ランダムな色を生成 (BGRフォーマット)
        print(f"警告: ラベル '{label}' がLABEL_COLORSに未登録です。ランダムな色を割り当てます。")
        # 一度生成した色は保存して、同じラベルには同じ色を使う
        random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        LABEL_COLORS[label] = random_color
        return random_color

# --- 3. メイン実行 ---

os.makedirs(VISUALIZE_OUTPUT_DIR, exist_ok=True)
print(f"描画結果は '{VISUALIZE_OUTPUT_DIR}' に保存されます。")

if not os.path.exists(DETECTIONS_CSV_PATH):
    print(f"エラー: 検出結果ファイル '{DETECTIONS_CSV_PATH}' が見つかりません。")
    print("YOLO推論スクリプトでCSVを保存していることを確認してください。")
    exit()

print(f"\n--- 検出結果CSV '{DETECTIONS_CSV_PATH}' を読み込み中 ---")

# 検出結果を元画像名ごとにグループ化するための辞書
detections_by_image = {}

try:
    with open(DETECTIONS_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f) # ヘッダーをキーとして辞書形式で読み込む
        for row in reader:
            original_image_name = row['original_image']
            
            # BBox座標を数値に変換
            xmin = float(row['full_Xmin'])
            ymin = float(row['full_Ymin'])
            xmax = float(row['full_Xmax'])
            ymax = float(row['full_Ymax'])
            
            # 中心座標を計算
            center_x = int((xmin + xmax) / 2)
            center_y = int((ymin + ymax) / 2)
            
            # ラベルと中心座標を保存
            if original_image_name not in detections_by_image:
                detections_by_image[original_image_name] = []
            detections_by_image[original_image_name].append({
                'label': row['label'],
                'center_x': center_x,
                'center_y': center_y
            })
    print(f"{len(detections_by_image)} 件の元画像に対して検出結果を読み込みました。")

except Exception as e:
    print(f"エラー: CSVファイルの読み込み中に問題が発生しました。 {e}")
    exit()

print("\n--- 元画像への描画を開始 ---")

processed_count = 0
for original_img_name, detections in detections_by_image.items():
    # 元画像のパスを構築
    img_path_jpg = os.path.join(ORIGINAL_IMAGE_DIR, f"{original_img_name}.jpg")
    img_path_png = os.path.join(ORIGINAL_IMAGE_DIR, f"{original_img_name}.png")

    image = None
    if os.path.exists(img_path_jpg):
        image = cv2.imread(img_path_jpg)
    elif os.path.exists(img_path_png):
        image = cv2.imread(img_path_png)
    else:
        print(f"警告: 元画像 '{original_img_name}.jpg/.png' が '{ORIGINAL_IMAGE_DIR}' に見つかりません。スキップします。")
        continue

    if image is None:
        print(f"警告: 元画像 '{original_img_name}' を読み込めませんでした。スキップします。")
        continue

    print(f"元画像 '{original_img_name}' に {len(detections)} 個の検出点を描画中...")

    for det in detections:
        label = det['label']
        center_x = det['center_x']
        center_y = det['center_y']
        
        # ラベルに応じた色を取得
        color = get_color_for_label(label)
        
        # ドット（丸）を描画
        cv2.circle(image, (center_x, center_y), DOT_RADIUS, color, DOT_THICKNESS)
        
        # (オプション) ラベル名をテキストで描画する場合
        # cv2.putText(image, label, (center_x + DOT_RADIUS, center_y), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 描画した画像を保存
    output_img_path = os.path.join(VISUALIZE_OUTPUT_DIR, f"{original_img_name}_detections.jpg")
    cv2.imwrite(output_img_path, image)
    processed_count += 1

print(f"\n--- 処理完了 ---")
print(f"合計 {processed_count} 件の元画像に検出結果を描画し、'{VISUALIZE_OUTPUT_DIR}' に保存しました。")