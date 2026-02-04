#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
土地利用変化解析システム (Land Use Change Analysis System)
---------------------------------------------------------
機能:
1. DexiNedのエッジ画像から「面(ポリゴン)」を抽出
2. YOLOの結果を用いてポリゴンに「属性(桑畑・田など)」を付与
3. image_bounds_wgs84.csv から有効範囲(四隅)を読み込み、枠外を無視
4. GCPファイルから学習したモデルで緯度経度を算出
5. ★追加: 現代の水域データを取得し、海上の誤検出ポリゴンを削除
6. 現代データと比較し、変化判定と面積計算を行う
7. 比較結果CSV(学校・寺社)を読み込み、現存状況をドットで描画する
"""

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.ops import transform as shapely_transform
from geopy.distance import geodesic
import os
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pyproj import Transformer
import json
from shapely.geometry import mapping
import requests
import math
from mapbox_vector_tile import decode

# ==========================================
# 1. 設定・ファイルパス
# ==========================================

# ベースディレクトリ
BASE_DIR = r"C:/Users/TakedaYuya/Landmark_Gaihouzu_new"

# 入力ファイル
IMAGE_PATH = os.path.join(BASE_DIR, "input/NI_52_11_8.jpg")
EDGE_PATH = os.path.join(BASE_DIR, "output/dexined_edge.png")
YOLO_CSV = os.path.join(BASE_DIR, "final_symbols_wgs84.csv")
MODERN_CSV = os.path.join(BASE_DIR, "agriculture_fixed.csv")
GCP_POINTS_FILE = os.path.join(BASE_DIR, "NI_52_11_8_full_half_size_23.jpg.points")
BOUNDS_CSV = os.path.join(BASE_DIR, "image_bounds_wgs84.csv")

# 比較結果CSVのパス
COMPARISON_DIR = os.path.join(BASE_DIR, "comparison_results")
RESULT_SCHOOL = os.path.join(COMPARISON_DIR, "result_school.csv")
RESULT_SHRINE = os.path.join(COMPARISON_DIR, "result_shrine.csv")
RESULT_TEMPLE = os.path.join(COMPARISON_DIR, "result_temple.csv")

# 出力ファイル
OUTPUT_IMAGE = os.path.join(BASE_DIR, "final_landuse_change.jpg")

# 投影座標系 (面積計算用: JGD2011 / Japan Plane Rectangular CS II)
METRIC_EPSG = "EPSG:6670"

# 判定基準 (メートル)
EXISTENCE_THRESHOLD = 100.0 

# ★追加: ベクトルタイル設定 (水域マスク用)
TILE_BASE_URL = "https://cyberjapandata.gsi.go.jp/xyz/experimental_bvmap"
Z_WATER = 16  # 水域取得用ズームレベル

# 色設定 (BGR形式)
COLORS = {
    "shrine":         (255, 0, 255),
    "temple":         (255, 0, 255),
    "rice_field":     (0, 255, 255),   # 黄
    "tea_field":      (0, 100, 0),     # 濃緑
    "mulberry_field": (0, 200, 0),     # 緑
    "school":         (255, 0, 0),     # 青
    "orchard":        (0, 165, 255),   # オレンジ
    "CHANGED":        (0, 0, 255),     # 赤 (農地消失)
    "UNKNOWN":        (200, 200, 200),
    
    "dot_school":     (255, 0, 0),     # 青
    "dot_shrine":     (203, 192, 255), # ピンク
    "dot_temple":     (128, 0, 128),   # 紫
    "dot_lost":       (0, 0, 255)      # 赤
}

LABEL_MAPPING = {
    "rice_field":     ["田"],
    "mulberry_field": ["畑", "果樹園", "茶畑"], 
    "tea_field":      ["茶畑"],
    "orchard":        ["果樹園"],
    "shrine": [], "temple": [], "school": []
}
AGRICULTURE_LABELS = ["rice_field", "mulberry_field", "tea_field", "orchard"]

# グローバル変数
MODEL_LON = None
MODEL_LAT = None
POLY_FEATURE = None
VALID_PIXEL_POLYGON = None 
MAP_BOUNDS = {}            

# ==========================================
# 2. 有効範囲の読み込み (Step 0)
# ==========================================
def load_map_bounds(csv_path):
    global VALID_PIXEL_POLYGON, MAP_BOUNDS
    if not os.path.exists(csv_path):
        print(f" エラー: 範囲定義ファイルが見つかりません ({csv_path})")
        return False

    print(f" 有効範囲定義を読み込み中... ({csv_path})")
    try:
        df = pd.read_csv(csv_path)
        order = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        pixel_points = []
        for name in order:
            row = df[df['name'] == name]
            if not row.empty:
                px = row.iloc[0]['px']
                py = row.iloc[0]['py_original']
                pixel_points.append((px, py))
        
        if len(pixel_points) == 4:
            VALID_PIXEL_POLYGON = Polygon(pixel_points)
            print("   画像上の有効エリア(ピクセル)を設定しました")
        else:
            return False

        lats = df['latitude'].values
        lons = df['longitude'].values
        MAP_BOUNDS = {
            "lat_max": np.max(lats), "lat_min": np.min(lats),
            "lon_max": np.max(lons), "lon_min": np.min(lons)
        }
        return True
    except Exception as e:
        print(f" 範囲読み込みエラー: {e}")
        return False

# ==========================================
# 3. 座標変換モデル構築
# ==========================================
def train_coordinate_model(filepath):
    if not os.path.exists(filepath):
        print(f" エラー: GCPファイルが見つかりません ({filepath})")
        return False

    try:
        df = pd.read_csv(filepath, comment='#', encoding='shift_jis')
        df.columns = [c.strip() for c in df.columns]
        if 'enable' in df.columns: df = df[df['enable'] == 1].copy()
    except:
        try:
            df = pd.read_csv(filepath, header=None, comment='#', encoding='cp932')
            df = df.iloc[:, :5] 
            df.columns = ['mapX', 'mapY', 'sourceX', 'sourceY', 'enable']
            df = df[df['enable'] == 1].copy()
        except Exception:
            return False

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(df["mapX"].values, df["mapY"].values)
    X = df[["sourceX", "sourceY"]].values
    
    global POLY_FEATURE, MODEL_LON, MODEL_LAT
    POLY_FEATURE = PolynomialFeatures(degree=2)
    X_poly = POLY_FEATURE.fit_transform(X)
    MODEL_LON = LinearRegression().fit(X_poly, lon)
    MODEL_LAT = LinearRegression().fit(X_poly, lat)
    
    return True

def pixel_to_latlon(px, py):
    if MODEL_LON is None: return 0, 0
    target_py = py * -1 
    target = np.array([[px, target_py]])
    target_poly = POLY_FEATURE.transform(target)
    lat = MODEL_LAT.predict(target_poly)[0]
    lon = MODEL_LON.predict(target_poly)[0]
    return lat, lon

# ==========================================
# ★追加: ベクトルタイル用ユーティリティ
# ==========================================
def deg_to_tile(lat_deg, lon_deg, z):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** z
    x_tile = int(math.floor((lon_deg + 180.0) / 360.0 * n))
    y_tile = int(math.floor((1.0 - (math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)) / 2.0 * n))
    return x_tile, y_tile

def tile_to_wgs84(z, x, y, local_x, local_y, extent):
    world_size = 40075016.68557849
    origin_shift = world_size / 2.0
    tile_size_meters = world_size / (2 ** z)
    tile_origin_mx = x * tile_size_meters - origin_shift
    tile_origin_my = origin_shift - y * tile_size_meters
    mx = tile_origin_mx + (local_x / float(extent)) * tile_size_meters
    my = tile_origin_my - (local_y / float(extent)) * tile_size_meters
    lon = (mx / origin_shift) * 180.0
    lat = 180.0 / math.pi * (2.0 * math.atan(math.exp((my / origin_shift) * math.pi)) - math.pi / 2.0)
    return lat, lon

def get_modern_water_polygons(bounds_dict):
    """
    国土地理院ベクトルタイルから「水域(海・湖)」のポリゴンを取得する
    """
    print(" 現代の水域データ(マスク用)を取得中...")
    
    # 検索範囲
    lat_min, lat_max = bounds_dict["lat_min"], bounds_dict["lat_max"]
    lon_min, lon_max = bounds_dict["lon_min"], bounds_dict["lon_max"]
    
    x1, y1 = deg_to_tile(lat_min, lon_min, Z_WATER)
    x2, y2 = deg_to_tile(lat_max, lon_max, Z_WATER)
    
    water_polys = []

    for x in range(min(x1, x2), max(x1, x2) + 1):
        for y in range(min(y1, y2), max(y1, y2) + 1):
            url = f"{TILE_BASE_URL}/{Z_WATER}/{x}/{y}.pbf"
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code != 200: continue
                decoded = decode(resp.content)
                
                if "waterarea" not in decoded: continue
                
                actual_extent = decoded["waterarea"].get('extent', 4096)
                
                for feat in decoded["waterarea"].get("features", []):
                    geom = feat.get("geometry", {})
                    if geom.get("type") not in ["Polygon", "MultiPolygon"]: continue
                    
                    wgs84_coords = []
                    coords_raw = geom.get("coordinates")[0] 
                    
                    for ring in coords_raw: 
                         if isinstance(ring[0], int): # [x, y]
                            local_x, local_y = ring
                            local_y = actual_extent - local_y
                            lat, lon = tile_to_wgs84(Z_WATER, x, y, local_x, local_y, actual_extent)
                            wgs84_coords.append((lon, lat)) 
                         else: # [[x,y]...]
                             for p in ring:
                                local_x, local_y = p
                                local_y = actual_extent - local_y
                                lat, lon = tile_to_wgs84(Z_WATER, x, y, local_x, local_y, actual_extent)
                                wgs84_coords.append((lon, lat))

                    if len(wgs84_coords) > 2:
                        water_polys.append(Polygon(wgs84_coords))
                        
            except Exception:
                continue
                
    print(f"   -> {len(water_polys)} 個の水域ポリゴンを取得しました。")
    return water_polys

def filter_sea_polygons(polygons, water_polys):
    """
    現代の水域と重なるポリゴンを削除する
    """
    print(" 海域ポリゴンの除去フィルタを実行中...")
    if not water_polys: return polygons
    
    filtered_polygons = []
    removed_count = 0
    
    for poly in polygons:
        # 重心座標を使用
        cx, cy = poly["center"]
        
        # 既にlat/lonがあるか確認、なければ計算
        if "lat" in poly:
            lat, lon = poly["lat"], poly["lon"]
        else:
            lat, lon = pixel_to_latlon(cx, cy)
            poly["lat"] = lat
            poly["lon"] = lon
            
        center_point = Point(lon, lat)
        
        is_sea = False
        for w_poly in water_polys:
            if w_poly.contains(center_point):
                is_sea = True
                break
        
        if is_sea:
            removed_count += 1
        else:
            filtered_polygons.append(poly)
            
    print(f"   -> {removed_count} 個の海域ポリゴンを削除しました。")
    return filtered_polygons

# ==========================================
# 4. データ処理・解析関数
# ==========================================

def extract_polygons_from_edge(edge_path):
    print(" 領域抽出とフィルタリングを実行中...")
    img = cv2.imread(edge_path, 0)
    if img is None: return []

    h, w = img.shape[:2]
    image_size = h * w
    _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=3) 
    contours, _ = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    MIN_AREA = 200
    MAX_AREA = image_size * 0.05 
    
    count_out_of_bounds = 0 

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if MIN_AREA < area < MAX_AREA:
            pts = cnt.reshape(-1, 2)
            if len(pts) >= 3:
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0) 

                if poly.is_valid:
                    if VALID_PIXEL_POLYGON is not None:
                        if not poly.intersects(VALID_PIXEL_POLYGON):
                            count_out_of_bounds += 1
                            continue
                        
                        intersection = poly.intersection(VALID_PIXEL_POLYGON)
                        intersection_area = intersection.area
                        
                        if intersection_area < (poly.area * 0.8):
                            count_out_of_bounds += 1
                            continue

                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else: cx, cy = 0, 0
                    
                    polygons.append({
                        "contour": cnt, "shapely": poly, "center": (cx, cy),
                        "yolo_labels": [], "final_label": None, "is_changed": False,
                        "area_m2": 0.0
                    })
    
    print(f"   -> 抽出完了: {len(polygons)} 個 (範囲外除外: {count_out_of_bounds} 個)")
    return polygons

def assign_attributes(polygons, yolo_csv):
    print("属性付与中...")
    if not os.path.exists(yolo_csv): return polygons

    df = pd.read_csv(yolo_csv)
    points = []
    
    for _, row in df.iterrows():
        px, py = row['px'], row['py_original']
        pt = Point(px, py)
        if VALID_PIXEL_POLYGON is not None:
            if not VALID_PIXEL_POLYGON.contains(pt): continue

        points.append({"pt": pt, "label": row['label']})

    for poly in polygons:
        poly_shape = poly["shapely"]
        for p_data in points:
            if poly_shape.contains(p_data["pt"]):
                poly["yolo_labels"].append(p_data["label"])
        
        if poly["yolo_labels"]:
            labels = poly["yolo_labels"]
            poly["final_label"] = max(set(labels), key=labels.count)
        else:
            poly["final_label"] = None
    return polygons

def check_change(polygons, modern_csv):
    print("現代データとの比較中...")
    if os.path.exists(modern_csv): modern_df = pd.read_csv(modern_csv)
    else: modern_df = pd.DataFrame()

    valid_list = []
    for poly in polygons:
        # 重心はフィルタリング時に既に計算されているはずだが念のため
        cx, cy = poly["center"]
        if "lat" not in poly:
            lat, lon = pixel_to_latlon(cx, cy)
            poly["lat"] = lat
            poly["lon"] = lon
        else:
            lat, lon = poly["lat"], poly["lon"]
        
        if MAP_BOUNDS:
            if not (MAP_BOUNDS["lat_min"] <= lat <= MAP_BOUNDS["lat_max"] and
                    MAP_BOUNDS["lon_min"] <= lon <= MAP_BOUNDS["lon_max"]):
                continue
        
        valid_list.append(poly)

        label = poly["final_label"]
        if label is None:
            poly["is_changed"] = False
            continue

        target_types = LABEL_MAPPING.get(label)
        if not target_types:
            poly["is_changed"] = False
            continue
            
        if not modern_df.empty:
            subset = modern_df[
                (modern_df["種別"].isin(target_types)) &
                (modern_df["緯度"] > lat - 0.005) & (modern_df["緯度"] < lat + 0.005) &
                (modern_df["経度"] > lon - 0.005) & (modern_df["経度"] < lon + 0.005)
            ]
            min_dist = float('inf')
            for _, row in subset.iterrows():
                d = geodesic((lat, lon), (row["緯度"], row["経度"])).meters
                if d < min_dist: min_dist = d
            
            if min_dist <= EXISTENCE_THRESHOLD: poly["is_changed"] = False
            else: poly["is_changed"] = True

    return valid_list

def fill_gaps_by_proximity(polygons):
    print(" 近傍補間による穴埋め処理を実行中 (範囲拡大版)...")
    
    TARGET_LABELS = ["rice_field", "mulberry_field", "tea_field", "orchard"]
    SEARCH_BUFFER = 25.0
    changed_count = 0

    labeled_polygons = [p for p in polygons if p["final_label"] in TARGET_LABELS]
    
    for poly in polygons:
        if poly["final_label"] is not None:
            continue
            
        current_shape = poly["shapely"]
        search_area = current_shape.buffer(SEARCH_BUFFER)
        neighbor_labels = []
        
        for neighbor in labeled_polygons:
            if search_area.intersects(neighbor["shapely"]):
                intersection = search_area.intersection(neighbor["shapely"])
                if intersection.area > 50:
                    neighbor_labels.append(neighbor["final_label"])
        
        if not neighbor_labels:
            continue
            
        most_common_label = max(set(neighbor_labels), key=neighbor_labels.count)
        
        poly["final_label"] = most_common_label
        poly["yolo_labels"].append(most_common_label + "(fill)")
        changed_count += 1

    print(f"   -> {changed_count} 個の未分類領域を周囲の属性で補完しました。(範囲: {SEARCH_BUFFER}px)")
    return polygons

def calculate_agricultural_area(polygons):
    print("\n 農地面積の集計とGeoJSON出力を開始します...")
    
    project = Transformer.from_crs("EPSG:4326", METRIC_EPSG, always_xy=True).transform

    total_agri_area = 0.0
    lost_area = 0.0
    count_agri = 0
    
    features = []

    for poly in polygons:
        label = poly["final_label"]
        is_agri = label in AGRICULTURE_LABELS
        
        wgs84_coords = []
        if poly["shapely"].is_empty:
            continue
            
        for (px, py) in list(poly["shapely"].exterior.coords):
            lat, lon = pixel_to_latlon(px, py)
            wgs84_coords.append((lon, lat))
        
        poly_wgs84 = Polygon(wgs84_coords)
        if not poly_wgs84.is_valid:
            poly_wgs84 = poly_wgs84.buffer(0) 
        
        poly_metric = shapely_transform(project, poly_wgs84)
        area_m2 = poly_metric.area
        poly["area_m2"] = area_m2
        
        if is_agri:
            count_agri += 1
            total_agri_area += area_m2
            if poly["is_changed"]:
                lost_area += area_m2

        feature = {
            "type": "Feature",
            "properties": {
                "label": str(label),
                "is_changed": bool(poly["is_changed"]),
                "python_area_m2": float(area_m2),
                "is_agri": is_agri
            },
            "geometry": mapping(poly_wgs84) 
        }
        features.append(feature)

    total_ha = total_agri_area / 10000
    lost_ha = lost_area / 10000
    loss_rate = (lost_ha / total_ha * 100) if total_ha > 0 else 0

    print("-" * 50)
    print(f" 【農地面積集計結果】")
    print(f"  ■ 推定総農地面積: {total_ha:.2f} ha")
    print(f"   減少率        : {loss_rate:.1f} %")
    print("-" * 50)

    geojson_path = os.path.join(BASE_DIR, "output_polygons.geojson")
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)
    
    print(f" 検証用ファイル保存完了: {geojson_path}")
    print("   -> QGISでこのファイルを開き、'python_area_m2'列を確認してください。")

# ==========================================
# 5. ドット描画関数
# ==========================================
def draw_comparison_dots(img):
    print(" 施設変化のドットを描画中...")
    overlay = img.copy()
    
    targets = [
        (RESULT_SCHOOL, COLORS["dot_school"], "学校"),
        (RESULT_SHRINE, COLORS["dot_shrine"], "神社"),
        (RESULT_TEMPLE, COLORS["dot_temple"], "寺院")
    ]
    
    for csv_path, exist_color, label_name in targets:
        if not os.path.exists(csv_path):
            continue
            
        try:
            df = pd.read_csv(csv_path)
            if not {'外邦図_画像X(px)', '外邦図_画像Y(py)', '現存結果'}.issubset(df.columns):
                print(f"   {label_name}: CSVのカラムが足りません。スキップします。")
                continue
                
            count_exist = 0
            count_lost = 0
            
            for _, row in df.iterrows():
                px = int(float(row['外邦図_画像X(px)']))
                py = int(float(row['外邦図_画像Y(py)']))
                is_exist = row['現存結果'] 
                
                radius = 12
                
                if is_exist:
                    cv2.circle(img, (px, py), radius, exist_color, -1)
                    cv2.circle(img, (px, py), radius, (255, 255, 255), 2)
                    count_exist += 1
                else:
                    cv2.circle(overlay, (px, py), radius, COLORS["dot_lost"], -1)
                    cv2.circle(img, (px, py), radius, COLORS["dot_lost"], 2)
                    count_lost += 1
            
            print(f"   {label_name}: 現存 {count_exist} / 消失 {count_lost}")
            
        except Exception as e:
            print(f"   {label_name} 描画エラー: {e}")

    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
def draw_result(polygons):
    print(" 結果画像の描画中...")
    if os.path.exists(IMAGE_PATH):
        img = cv2.imread(IMAGE_PATH)
    else:
        img = np.ones((4000, 4000, 3), np.uint8) * 255
        
    overlay = img.copy()
    
    for poly in polygons:
        label = poly["final_label"]
        cnt = poly["contour"]
        
        if label is None:
            cv2.drawContours(img, [cnt], -1, (255, 0, 0), 2)
        else:
            is_changed = poly.get("is_changed", False)
            if is_changed: color = COLORS["CHANGED"]
            else: color = COLORS.get(label, COLORS["UNKNOWN"])
            
            cv2.drawContours(overlay, [cnt], -1, color, -1)
            cv2.drawContours(img, [cnt], -1, color, 2)
            
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    draw_comparison_dots(img)
    
    cv2.imwrite(OUTPUT_IMAGE, img)
    print(f" 保存完了: {OUTPUT_IMAGE}")
"""
def main():
    if not load_map_bounds(BOUNDS_CSV): return
    if not train_coordinate_model(GCP_POINTS_FILE): return
    
    polygons = extract_polygons_from_edge(EDGE_PATH)
    if not polygons: return
    
    # 1. YOLO属性付与
    valid_polygons = assign_attributes(polygons, YOLO_CSV)
    
    # ==========================================
    # ★追加: 海域フィルタリング
    # ==========================================
    # 1. 現代の水域ポリゴンを取得 (GCP範囲に基づいて自動取得)
    water_polys = get_modern_water_polygons(MAP_BOUNDS)
    
    # 2. 抽出したポリゴンが海と被っていたら削除
    #    (緯度経度変換もこの中で行われます)
    land_polygons = filter_sea_polygons(valid_polygons, water_polys)
    # ==========================================

    # 3. 穴埋め処理 (海を除去した後のデータに対して行う)
    filled_polygons = fill_gaps_by_proximity(land_polygons)
    
    # 4. 変化判定
    final_polygons = check_change(filled_polygons, MODERN_CSV)
    
    calculate_agricultural_area(final_polygons)
    draw_result(final_polygons)
"""
def main():
    if not load_map_bounds(BOUNDS_CSV): return
    if not train_coordinate_model(GCP_POINTS_FILE): return
    
    polygons = extract_polygons_from_edge(EDGE_PATH)
    if not polygons: return
    
    # 1. YOLO属性付与
    valid_polygons = assign_attributes(polygons, YOLO_CSV)
    
    # ==========================================
    # ★修正: 海域フィルタリングを無効化 (元に戻す)
    # ==========================================
    # water_polys = get_modern_water_polygons(MAP_BOUNDS)
    # land_polygons = filter_sea_polygons(valid_polygons, water_polys)
    # ==========================================

    water_polys = get_modern_water_polygons(MAP_BOUNDS)
    land_polygons = filter_sea_polygons(valid_polygons, water_polys)

    # 2. 穴埋め処理 
    # ★ここを land_polygons ではなく valid_polygons に戻すのが重要！
    filled_polygons = fill_gaps_by_proximity(valid_polygons)
    
    # 3. 変化判定
    final_polygons = check_change(filled_polygons, MODERN_CSV)
    
    calculate_agricultural_area(final_polygons)
    draw_result(final_polygons)

if __name__ == "__main__":
    main()