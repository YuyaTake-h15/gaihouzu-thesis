import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pyproj import Transformer
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
# =========================================================
# 1. 設定：環境に合わせて変更してください
# =========================================================

# ファイルパス設定
POINTS_FILE_PATH = "C:/Users/TakedaYuya/Landmark_Gaihouzu_new/NI-52-11-14.jpg.points"  # QGISのGCPファイル
YOLO_CSV_PATH = "C:/Users/TakedaYuya/Landmark_Gaihouzu_new/output/all_detections_full_coords_clean.csv"   # YOLOのCSVファイル

# 出力ファイル名
OUTPUT_SYMBOLS_PATH = "final_symbols_wgs84.csv"   # シンボルの緯度経度
OUTPUT_CORNERS_PATH = "image_bounds_wgs84.csv"    # 四隅の緯度経度

# 【重要】画像のサイズ（ピクセル）を入力してください
# ※画像のプロパティで「幅」と「高さ」を確認して書き換えてください
IMG_WIDTH = 7237   # 例: 幅 (px)
IMG_HEIGHT = 5566   # 例: 高さ (px)

# =========================================================
# 2. 関数：GCP読み込みとモデル学習
# =========================================================
def train_model_from_gcp(filepath):
    print(f" GCPファイルを読み込んでいます: {filepath}")
    
    # ポイントファイルを読み込み (エンコーディング対応)
    try:
        df = pd.read_csv(filepath, comment='#', encoding='shift_jis')
        df.columns = [c.strip() for c in df.columns]
        df = df[df['enable'] == 1].copy()
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, comment='#', encoding='cp932')
        df.columns = [c.strip() for c in df.columns]
        df = df[df['enable'] == 1].copy()
    except Exception:
        # ヘッダーがない場合の予備対応
        try:
            df = pd.read_csv(filepath, header=None, comment='#', encoding='shift_jis')
        except:
            df = pd.read_csv(filepath, header=None, comment='#', encoding='cp932')
            
        df.columns = ['mapX', 'mapY', 'sourceX', 'sourceY', 'enable', 'dX', 'dY', 'residual']
        df = df[df['enable'] == 1].copy()

    # 座標変換: EPSG:3857(メートル) -> EPSG:4326(緯度経度)
    # QGISのGCPがWebメルカトルで保存されている場合を想定
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(df["mapX"].values, df["mapY"].values)
    
    # 学習データ作成
    # ソース画像上のピクセル座標(sourceX, sourceY) から 緯度経度(lat, lon) を予測する
    X = df[["sourceX", "sourceY"]].values
    y_lon = lon
    y_lat = lat
    
    # 多項式特徴量 (degree=2: 画像の歪みを考慮するため2次関数を使用)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    # モデル学習
    model_lon = LinearRegression().fit(X_poly, y_lon)
    model_lat = LinearRegression().fit(X_poly, y_lat)
    
    print(f" モデル学習完了 (使用GCP点数: {len(df)})")
    return model_lon, model_lat, poly

# =========================================================
# 3. メイン処理
# =========================================================

def main():
    # --- Step 1: モデルの作成 ---
    model_lon, model_lat, poly = train_model_from_gcp(POINTS_FILE_PATH)

    # =========================================================
    # Step 2: YOLO検出シンボルの座標変換
    # =========================================================
    print(f"\n YOLOデータの変換を開始します...")
    df_yolo = pd.read_csv(YOLO_CSV_PATH)

    # 中心座標の計算
    df_yolo['px'] = (df_yolo['full_Xmin'] + df_yolo['full_Xmax']) / 2
    df_yolo['py_original'] = (df_yolo['full_Ymin'] + df_yolo['full_Ymax']) / 2

    # ★Y軸の反転処理 (GCPとの整合性を取るため)
    df_yolo['py'] = df_yolo['py_original'] * -1

    # 座標変換
    target_coords = df_yolo[['px', 'py']].values
    target_poly = poly.transform(target_coords)

    df_yolo['latitude'] = model_lat.predict(target_poly)
    df_yolo['longitude'] = model_lon.predict(target_poly)

    # 必要なカラムを出力
    output_columns = [
        'label', 'confidence', 'latitude', 'longitude', 
        'original_image', 'px', 'py_original'
    ]
    df_output = df_yolo[output_columns]
    df_output.to_csv(OUTPUT_SYMBOLS_PATH, index=False, encoding='utf-8-sig')
    print(f" シンボル座標を保存しました: {OUTPUT_SYMBOLS_PATH}")

    # =========================================================
    # Step 3: 画像四隅(外枠)の座標変換
    # =========================================================
    print(f"\n 四隅の範囲計算を開始します (画像サイズ: {IMG_WIDTH}x{IMG_HEIGHT})")

    # 四隅の定義 (左上 -> 右上 -> 右下 -> 左下)
    corners_data = [
        {'name': 'Top-Left',     'px': 1134, 'py_original': 607},
        {'name': 'Top-Right',    'px': 6653, 'py_original': 621},
        {'name': 'Bottom-Right', 'px': 6628, 'py_original': 5018},
        {'name': 'Bottom-Left',  'px': 1113, 'py_original': 4998}
    ]

    df_corners = pd.DataFrame(corners_data)

    # ★ここでも同じくY軸反転処理を行う
    df_corners['py'] = df_corners['py_original'] * -1

    # 座標変換
    target_corners = df_corners[['px', 'py']].values
    corners_poly = poly.transform(target_corners)

    df_corners['latitude'] = model_lat.predict(corners_poly)
    df_corners['longitude'] = model_lon.predict(corners_poly)

    # CSV保存
    output_cols_corners = ['name', 'latitude', 'longitude', 'px', 'py_original']
    df_corners[output_cols_corners].to_csv(OUTPUT_CORNERS_PATH, index=False, encoding='utf-8-sig')
    print(f" 四隅座標を保存しました: {OUTPUT_CORNERS_PATH}")

    # =========================================================
    # 結果の表示 (Vector取得コードへのコピペ用)
    # =========================================================
    print("\n" + "="*60)
    print(" 地理院Vector取得コード用の POLYGON_WGS84 リスト")
    print("   (以下のリストをコピーして、Vector取得コードに貼り付けてください)")
    print("="*60)
    print("POLYGON_WGS84 = [")
    for _, row in df_corners.iterrows():
        print(f"    ({row['latitude']:.6f}, {row['longitude']:.6f}), # {row['name']}")
    print("]")
    print("="*60)

if __name__ == "__main__":
    main()