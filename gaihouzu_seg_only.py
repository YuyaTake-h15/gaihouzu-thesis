#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
外邦図 セグメンテーション・属性付与ツール (修正版)
---------------------------------------------------------
修正点:
- 神社、寺院、学校などの「点地物」はポリゴン化せず、アイコン(ドット)として描画
- 農地（田、畑など）のみをポリゴンとして塗りつぶし・面積計算を行う
"""

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon, mapping
from shapely.ops import transform as shapely_transform
import os
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pyproj import Transformer
import json

# ==========================================
# 1. 設定・ファイルパス
# ==========================================

BASE_DIR = r"C:/Users/TakedaYuya/Landmark_Gaihouzu_new"

IMAGE_PATH = os.path.join(BASE_DIR, "input/NI_52_11_8.jpg")
EDGE_PATH = os.path.join(BASE_DIR, "output/dexined_edge.png")
YOLO_CSV = os.path.join(BASE_DIR, "final_symbols_wgs84.csv")
GCP_POINTS_FILE = os.path.join(BASE_DIR, "NI_52_11_8_full_half_size_23.jpg.points")
BOUNDS_CSV = os.path.join(BASE_DIR, "image_bounds_wgs84.csv")

OUTPUT_IMAGE = os.path.join(BASE_DIR, "result_segmentation_only.jpg")
OUTPUT_GEOJSON = os.path.join(BASE_DIR, "result_polygons.geojson")

METRIC_EPSG = "EPSG:6670"

# 色設定 (BGR形式)
COLORS = {
    # ポリゴン用 (農地)
    "rice_field":     (0, 255, 255),   # 黄
    "tea_field":      (0, 100, 0),     # 濃緑
    "mulberry_field": (0, 200, 0),     # 緑
    "orchard":        (0, 165, 255),   # オレンジ
    "UNKNOWN":        (200, 200, 200), # グレー
    
    # ドット用 (施設)
    "school":         (255, 0, 0),     # 青
    "shrine":         (255, 0, 255),   # ピンク
    "temple":         (128, 0, 128)    # 紫
}

LABEL_MAPPING = {
    "rice_field":     ["田"],
    "mulberry_field": ["畑", "果樹園", "茶畑"], 
    "tea_field":      ["茶畑"],
    "orchard":        ["果樹園"]
}

# ★分類定義の分離
# ポリゴンとして扱うもの（農地）
AGRICULTURE_LABELS = ["rice_field", "mulberry_field", "tea_field", "orchard"]
# 点として扱うもの（施設）
FACILITY_LABELS = ["shrine", "temple", "school"]

# グローバル変数
MODEL_LON = None
MODEL_LAT = None
POLY_FEATURE = None
VALID_PIXEL_POLYGON = None 

# ==========================================
# 2. 座標変換・範囲読み込み
# ==========================================
def load_map_bounds(csv_path):
    global VALID_PIXEL_POLYGON
    if not os.path.exists(csv_path):
        print(f"警告: 範囲定義ファイルが見つかりません ({csv_path}) - 全範囲を処理します")
        return True 

    try:
        df = pd.read_csv(csv_path)
        order = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        pixel_points = []
        for name in order:
            row = df[df['name'] == name]
            if not row.empty:
                pixel_points.append((row.iloc[0]['px'], row.iloc[0]['py_original']))
        
        if len(pixel_points) == 4:
            VALID_PIXEL_POLYGON = Polygon(pixel_points)
            print("有効範囲(ROI)を設定しました")
        return True
    except Exception as e:
        print(f"範囲読み込みエラー: {e}")
        return False

def train_coordinate_model(filepath):
    if not os.path.exists(filepath):
        print(f"警告: GCPファイルが見つかりません ({filepath}) - 緯度経度は計算されません")
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
        except: return False

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
# 3. 画像処理・属性付与
# ==========================================

def extract_polygons_from_edge(edge_path):
    print("領域抽出を実行中...")
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
    count_out = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            pts = cnt.reshape(-1, 2)
            if len(pts) >= 3:
                poly = Polygon(pts)
                if not poly.is_valid: poly = poly.buffer(0) 

                if poly.is_valid:
                    # 有効範囲チェック
                    if VALID_PIXEL_POLYGON is not None:
                        if not poly.intersects(VALID_PIXEL_POLYGON):
                            count_out += 1
                            continue
                        if poly.intersection(VALID_PIXEL_POLYGON).area < (poly.area * 0.8):
                            count_out += 1
                            continue

                    # 重心計算
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else: cx, cy = 0, 0
                    
                    polygons.append({
                        "contour": cnt, "shapely": poly, "center": (cx, cy),
                        "yolo_labels": [], "final_label": None, "area_m2": 0.0
                    })
    
    print(f"  -> 抽出完了: {len(polygons)} 個 (範囲外: {count_out} 個)")
    return polygons

def assign_attributes(polygons, yolo_csv):
    """
    ポリゴンに属性を付与する。
    ★修正: 農地ラベル(AGRICULTURE_LABELS)のみをポリゴンに適用する。
    """
    print("属性付与を実行中...")
    if not os.path.exists(yolo_csv): return polygons

    df = pd.read_csv(yolo_csv)
    points = []
    
    # YOLOの座標を読み込み
    for _, row in df.iterrows():
        label = row['label']
        
        # ★ここで農地ラベル以外はポリゴン属性用としては無視する
        if label not in AGRICULTURE_LABELS:
            continue

        px, py = row['px'], row['py_original']
        pt = Point(px, py)
        if VALID_PIXEL_POLYGON and not VALID_PIXEL_POLYGON.contains(pt): continue
        points.append({"pt": pt, "label": label})

    # ポリゴン内に含まれるYOLO点をカウント
    for poly in polygons:
        poly_shape = poly["shapely"]
        for p_data in points:
            if poly_shape.contains(p_data["pt"]):
                poly["yolo_labels"].append(p_data["label"])
        
        # 最頻値を属性として採用
        if poly["yolo_labels"]:
            labels = poly["yolo_labels"]
            poly["final_label"] = max(set(labels), key=labels.count)
        else:
            poly["final_label"] = None
            
    return polygons

def fill_gaps_by_proximity(polygons):
    print("近傍補間による属性穴埋めを実行中...")
    labeled_polygons = [p for p in polygons if p["final_label"] in AGRICULTURE_LABELS]
    changed_count = 0
    
    for poly in polygons:
        if poly["final_label"] is not None: continue
        
        # 周囲25pxを探す
        search_area = poly["shapely"].buffer(25.0)
        neighbor_labels = []
        
        for neighbor in labeled_polygons:
            if search_area.intersects(neighbor["shapely"]):
                if search_area.intersection(neighbor["shapely"]).area > 50:
                    neighbor_labels.append(neighbor["final_label"])
        
        if neighbor_labels:
            poly["final_label"] = max(set(neighbor_labels), key=neighbor_labels.count)
            changed_count += 1

    print(f"  -> {changed_count} 個の領域を補完しました")
    return polygons

def calculate_stats_and_save(polygons):
    print("\n面積集計とGeoJSON保存を開始します...")
    
    # 投影変換器 (WGS84 -> 平面直角座標系)
    has_gcp = (MODEL_LON is not None)
    project = None
    if has_gcp:
        project = Transformer.from_crs("EPSG:4326", METRIC_EPSG, always_xy=True).transform

    stats = {label: 0.0 for label in AGRICULTURE_LABELS}
    features = []

    for poly in polygons:
        label = poly["final_label"]
        shapely_geom = poly["shapely"]
        
        # MultiPolygon対策
        if shapely_geom.geom_type != 'Polygon': continue

        # 座標変換 (Pixel -> LatLon)
        wgs84_coords = []
        if has_gcp:
            for (px, py) in list(shapely_geom.exterior.coords):
                lat, lon = pixel_to_latlon(px, py)
                wgs84_coords.append((lon, lat))
            
            if len(wgs84_coords) < 3: continue
            poly_wgs84 = Polygon(wgs84_coords)
            
            # 面積計算 (LatLon -> Metric)
            if not poly_wgs84.is_valid: poly_wgs84 = poly_wgs84.buffer(0)
            poly_metric = shapely_transform(project, poly_wgs84)
            area_m2 = poly_metric.area
            poly["area_m2"] = area_m2
            
            # 集計 (農地のみ)
            if label in stats:
                stats[label] += area_m2
            
            # GeoJSON用データ
            features.append({
                "type": "Feature",
                "properties": {
                    "label": str(label),
                    "area_m2": float(area_m2)
                },
                "geometry": mapping(poly_wgs84)
            })

    # 結果表示
    print("-" * 40)
    print("【明治期 農地面積集計結果】")
    total_all = 0
    for label, area_m2 in stats.items():
        area_ha = area_m2 / 10000
        total_all += area_ha
        jp_label = LABEL_MAPPING.get(label, [label])[0] # 日本語変換
        print(f"  ■ {jp_label:<5} ({label}): {area_ha:.2f} ha")
    
    print("-" * 40)
    print(f"  ★ 合計面積: {total_all:.2f} ha")
    print("-" * 40)

    # GeoJSON保存
    if features:
        with open(OUTPUT_GEOJSON, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False, indent=2)
        print(f"GeoJSON保存完了: {OUTPUT_GEOJSON}")

def draw_result(polygons):
    print("結果画像を描画中...")
    if os.path.exists(IMAGE_PATH):
        img = cv2.imread(IMAGE_PATH)
    else:
        img = np.ones((4000, 4000, 3), np.uint8) * 255
        
    overlay = img.copy()
    
    # 1. ポリゴンの描画 (農地のみ)
    for poly in polygons:
        label = poly["final_label"]
        cnt = poly["contour"]
        
        if label:
            # 属性ごとの色で塗りつぶし
            color = COLORS.get(label, COLORS["UNKNOWN"])
            cv2.drawContours(overlay, [cnt], -1, color, -1)
            cv2.drawContours(img, [cnt], -1, color, 2)
        else:
            # 未分類は青枠のみ
            cv2.drawContours(img, [cnt], -1, (255, 0, 0), 1)

    # 2. 点地物（施設）の描画
    # YOLOのCSVを再度読み込んで、点として描画する
    if os.path.exists(YOLO_CSV):
        df = pd.read_csv(YOLO_CSV)
        for _, row in df.iterrows():
            label = row['label']
            if label in FACILITY_LABELS: # 施設のみ描画
                px, py = int(row['px']), int(row['py_original'])
                
                # 有効範囲チェック
                if VALID_PIXEL_POLYGON is not None:
                    if not VALID_PIXEL_POLYGON.contains(Point(px, py)): continue

                color = COLORS.get(label, (0, 0, 255))
                radius = 12
                
                cv2.circle(overlay, (px, py), radius, color, -1) # 塗りつぶし円
                cv2.circle(img, (px, py), radius, (255, 255, 255), 2) # 白枠
    
    # 半透明合成
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    
    cv2.imwrite(OUTPUT_IMAGE, img)
    print(f"保存完了: {OUTPUT_IMAGE}")

def main():
    # 1. 準備
    load_map_bounds(BOUNDS_CSV)
    train_coordinate_model(GCP_POINTS_FILE)
    
    # 2. セグメンテーション (DexiNedエッジからポリゴン化)
    polygons = extract_polygons_from_edge(EDGE_PATH)
    if not polygons: return
    
    # 3. 属性付与 (YOLO結果との突合 - ★農地のみ)
    valid_polygons = assign_attributes(polygons, YOLO_CSV)
    
    # 4. 穴埋め (未分類ポリゴンの推測)
    filled_polygons = fill_gaps_by_proximity(valid_polygons)
    
    # 5. 集計と保存 (比較なし、純粋な面積計算)
    calculate_stats_and_save(filled_polygons)
    
    # 6. 描画 (農地は面、施設は点として描画)
    draw_result(filled_polygons)

if __name__ == "__main__":
    main()