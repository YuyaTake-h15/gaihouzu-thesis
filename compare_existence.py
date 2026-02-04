import pandas as pd
import numpy as np
import os
"""
神社・寺・学校などの比較結果を出力するスクリプト
---------------------------------------------------------
機能:
1. ファイル設定
2. 地物の正解データと比較データの読み込み
3. 正解データと比較データの距離差を計算
4. 距離差が指定の距離未満だとTrue、以上だとFalseとして現存判定を行う
5. 結果をCSVファイルとして保存
"""
# =========================================================
# 1. 設定
# =========================================================

# 入力ファイルパス
PRED_FILE = "C:/Users/TakedaYuya/Landmark_Gaihouzu_new/final_symbols_wgs84.csv"        # 外邦図の検出結果 (Step 2の出力)
GT_SHRINE_TEMPLE = "C:/Users/TakedaYuya/Landmark_Gaihouzu_new/shrine_temple_fixed.csv" # 寺社の正解データ (Step 3の出力)
GT_SCHOOL = "C:/Users/TakedaYuya/Landmark_Gaihouzu_new/ans_data/school_only.csv"      # 学校の正解データ (今回指定のCSV)

# 出力ディレクトリ
OUTPUT_DIR = "comparison_results"

# 判定のしきい値 (メートル)
# この距離以内なら「現存 (True)」とみなす
EXISTENCE_THRESHOLD_METERS = 100.0

# クラス対応表 (YOLOの英語ラベル : 正解データの検索キーワードリスト)
# ※学校データは「小学校」「中学校」「高等学校」「大学」などを検索対象にします
CLASS_MAPPING = {
    "school": ["小学校", "中学校", "高等学校", "中等教育学校", "大学", "高等専門学校", "特別支援学校"],
    "shrine": ["神社"],
    "temple": ["寺", "寺院"]
}

# =========================================================
# 2. 関数定義
# =========================================================

def haversine_np(lat1, lon1, lat2, lon2):
    """
    2点間の距離(メートル)を計算する (NumPyベクトル化対応)
    """
    R = 6371000.0  # 地球の半径 (m)
    
    # 緯度経度をラジアンに変換
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def load_ground_truth_school(filepath):
    """学校データを読み込む (指定フォーマット: 学校名,種別,経度,緯度)"""
    try:
        # 1行目がヘッダーである前提で読み込み
        df = pd.read_csv(filepath)
        
        # カラム名を統一するためのマッピング
        rename_map = {
            "学校名": "name",
            "種別": "category",
            "緯度": "lat",
            "経度": "lon"
        }
        
        # 必要なカラムが揃っているか確認しつつリネーム
        df_renamed = df.rename(columns=rename_map)
        
        required_cols = ["name", "category", "lat", "lon"]
        if not all(col in df_renamed.columns for col in required_cols):
            print(f" 学校データのカラム名が一致しません。含まれているカラム: {df.columns.tolist()}")
            return pd.DataFrame()

        return df_renamed[required_cols].copy()

    except Exception as e:
        print(f" 学校データの読み込みエラー: {e}")
        return pd.DataFrame()

def load_ground_truth_shrine_temple(filepath):
    """寺社データを読み込む"""
    try:
        df = pd.read_csv(filepath)
        # カラム名を統一: name, category, lat, lon
        rename_map = {"名称": "name", "種別": "category", "緯度": "lat", "経度": "lon"}
        df = df.rename(columns=rename_map)
        return df[['name', 'category', 'lat', 'lon']].copy()
    except Exception as e:
        print(f" 寺社データの読み込みエラー: {filepath} ({e})")
        return pd.DataFrame()

# =========================================================
# 3. メイン処理
# =========================================================

def main():
    print(" 比較・現存判定処理を開始します...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 外邦図(YOLO)データの読み込み
    try:
        df_pred = pd.read_csv(PRED_FILE)
        print(f" 外邦図データ読み込み完了: {len(df_pred)}件")
    except FileNotFoundError:
        print(f" ファイルが見つかりません: {PRED_FILE}")
        return

    # 2. 正解データの読み込みと統合
    print(" 正解データをロード中...")
    df_schools = load_ground_truth_school(GT_SCHOOL)
    df_shrine_temple = load_ground_truth_shrine_temple(GT_SHRINE_TEMPLE)
    
    # 全正解データを1つのDataFrameにまとめる
    df_gt_all = pd.concat([df_schools, df_shrine_temple], ignore_index=True)
    
    if df_gt_all.empty:
        print(" 正解データが1件も読み込めませんでした。終了します。")
        return
    
    print(f" 正解データ統合完了: {len(df_gt_all)}件 (学校: {len(df_schools)}, 寺社: {len(df_shrine_temple)})")

    # 3. クラスごとに比較処理
    for yolo_label, gt_keywords in CLASS_MAPPING.items():
        print(f"\n---  {yolo_label} の比較処理 ---")

        # (1) 検出データの抽出
        df_p = df_pred[df_pred['label'] == yolo_label].copy()
        if df_p.empty:
            print(f"  -> 検出データなし。スキップします。")
            continue

        # (2) 正解データの抽出（キーワード部分一致）
        mask = df_gt_all['category'].apply(lambda x: any(k in str(x) for k in gt_keywords))
        df_g = df_gt_all[mask].copy()

        if df_g.empty:
            print(f"  -> 対応する正解データがありません。スキップします。")
            continue

        # (3) 最近傍探索
        results = []
        
        # 正解データの配列化 (高速化のため)
        gt_lats = df_g['lat'].values
        gt_lons = df_g['lon'].values
        gt_names = df_g['name'].values
        gt_cats = df_g['category'].values

        for idx, row in df_p.iterrows():
            p_lat = row['latitude']
            p_lon = row['longitude']
            
            # 全正解データとの距離を計算
            dists = haversine_np(p_lat, p_lon, gt_lats, gt_lons)
            
            # 最も近い正解データを探す
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            
            # 現存判定 (指定距離以内か？)
            is_existing = min_dist <= EXISTENCE_THRESHOLD_METERS

            # 結果格納
            # ★ここで画像内座標(px, py_original)を追加しました
            results.append({
                "外邦図_地図記号": row['label'],
                "外邦図_画像X(px)": row['px'],              # 追加
                "外邦図_画像Y(py)": row['py_original'],     # 追加
                "外邦図_緯度": p_lat,
                "外邦図_経度": p_lon,
                "現存結果": is_existing,     # True / False
                "正解データ_種別": gt_cats[min_idx],
                "正解データ_固有名": gt_names[min_idx] if pd.notna(gt_names[min_idx]) else None,
                "正解データ_緯度": gt_lats[min_idx],
                "正解データ_経度": gt_lons[min_idx],
                "距離差_メートル": round(min_dist, 2)
            })

        # (4) CSV保存
        if results:
            out_df = pd.DataFrame(results)
            
            out_filename = os.path.join(OUTPUT_DIR, f"result_{yolo_label}.csv")
            out_df.to_csv(out_filename, index=False, encoding='utf-8-sig')
            
            true_count = out_df['現存結果'].sum()
            print(f"  保存完了: {out_filename}")
            print(f"    総数: {len(out_df)} 件")
            print(f"    現存判定(True): {true_count} 件 (判定基準: {EXISTENCE_THRESHOLD_METERS}m以内)")
        else:
            print("  -> 結果生成なし。")

if __name__ == "__main__":
    main()