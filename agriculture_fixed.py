#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import requests
import pandas as pd
import math
from mapbox_vector_tile import decode
# =========================================================
# 1. 設定・定数定義
# =========================================================
# 検索範囲（熊本周辺の指定座標：前回のコードと同じ）
"""
POLYGON_WGS84 = [
    (32.83871593133105, 130.5007713887726),
    (32.836904272987695, 130.75053262359825),
    (32.67059532207984, 130.75114184578075),
    (32.67080341916152, 130.49969781341173)
]
#佐賀周辺
POLYGON_WGS84 = [
    (33.336782, 130.250292), # Top-Left
    (33.337196, 130.500642), # Top-Right
    (33.170073, 130.500884), # Bottom-Right
    (33.169882, 130.249663), # Bottom-Left
]
POLYGON_WGS84 = [
    (33.336974, 130.000898), # Top-Left
    (33.337326, 130.250825), # Top-Right
    (33.170264, 130.251518), # Bottom-Right
    (33.170090, 130.000619), # Bottom-Left
]
"""
POLYGON_WGS84 = [
    (33.170449, 130.000696), # Top-Left
    (33.170861, 130.251476), # Top-Right
    (33.003993, 130.252493), # Bottom-Right
    (33.003569, 130.000564), # Bottom-Left
]

# ベクトルタイル設定
Z = 16  # 推奨ズームレベル
TILE_BASE_URL = "https://cyberjapandata.gsi.go.jp/xyz/experimental_bvmap"
# 抽出対象のコード設定 (ftCode)
FTCODE_RICE = 6311     # 田
FTCODE_FIELD = 6312    # 畑
FTCODE_TEA = 6313      # 茶畑
FTCODE_ORCHARD = 6314  # 果樹園
# ターゲットリスト
TARGET_FTCODES = (FTCODE_RICE, FTCODE_FIELD, FTCODE_TEA, FTCODE_ORCHARD)
# 対象レイヤー (農地の記号は主に symbol レイヤーに含まれます)
TARGET_LAYERS = ["symbol", "label"]
# =========================================================
# 2. ユーティリティ関数
# =========================================================
def deg_to_tile(lat_deg, lon_deg, z):
    """緯度経度からタイル座標(X, Y)を計算"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** z
    x_tile = int(math.floor((lon_deg + 180.0) / 360.0 * n))
    y_tile = int(math.floor((1.0 - (math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)) / 2.0 * n))
    return x_tile, y_tile
def tile_to_wgs84(z, x, y, local_x, local_y, extent):
    """
    タイル座標とタイル内画素座標(local_x, local_y)を緯度経度に変換
    Webメルカトル(EPSG:3857)のメートル座標を経由して計算
    """
    world_size = 40075016.68557849
    origin_shift = world_size / 2.0
    tile_size_meters = world_size / (2 ** z)
    tile_origin_mx = x * tile_size_meters - origin_shift
    tile_origin_my = origin_shift - y * tile_size_meters
    # local_y は手動反転済みとして計算
    mx = tile_origin_mx + (local_x / float(extent)) * tile_size_meters
    my = tile_origin_my - (local_y / float(extent)) * tile_size_meters
    lon = (mx / origin_shift) * 180.0
    lat = 180.0 / math.pi * (2.0 * math.atan(math.exp((my / origin_shift) * math.pi)) - math.pi / 2.0)
    return lat, lon
def point_in_polygon(lat, lon, polygon):
    """Ray castingアルゴリズムによるポリゴン内判定"""
    inside = False
    n = len(polygon)
    for i in range(n):
        lat1, lon1 = polygon[i]
        lat2, lon2 = polygon[(i + 1) % n]
        if ((lon1 <= lon < lon2) or (lon2 <= lon < lon1)):
            if lon1 == lon2:
                continue
            t = (lon - lon1) / (lon2 - lon1)
            lat_cross = lat1 + t * (lat2 - lat1)
            if lat < lat_cross:
                inside = not inside
    return inside
# =========================================================
# 3. メイン処理
# =========================================================
def extract_agriculture():
    print("農業記号(田・畑・茶畑・果樹園)の抽出を開始します...")
    # 1. 検索範囲のタイルインデックスを計算
    lats = [p[0] for p in POLYGON_WGS84]
    lons = [p[1] for p in POLYGON_WGS84]
    x1, y1 = deg_to_tile(min(lats), min(lons), Z)
    x2, y2 = deg_to_tile(max(lats), max(lons), Z)
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    tile_list = [(Z, x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]
    print(f" 検索範囲: ZL{Z}, タイル数: {len(tile_list)}枚")
    results = []
    # 2. 各タイルを処理
    for idx, (z, x, y) in enumerate(tile_list, start=1):
        if idx % 50 == 0 or idx == 1 or idx == len(tile_list):
            print(f"[{idx}/{len(tile_list)}] 処理中... (Tile: {z}/{x}/{y})")
        try:
            url = f"{TILE_BASE_URL}/{z}/{x}/{y}.pbf"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            decoded = decode(resp.content)
        except Exception as e:
            print(f"Error at {z}/{x}/{y}: {e}")
            continue
        found_in_tile = False
        # 3. 指定レイヤー内の地物を走査
        for layer_name in TARGET_LAYERS:
            if layer_name not in decoded:
                continue
            actual_extent = decoded[layer_name].get('extent', 4096)
            for feat in decoded[layer_name].get("features", []):
                props = feat.get("properties", {})
                ftCode = props.get("ftCode")
                # 対象のコードでなければスキップ
                if ftCode not in TARGET_FTCODES:
                    continue
                # カテゴリ判定
                if ftCode == FTCODE_RICE:
                    category = "田"
                elif ftCode == FTCODE_FIELD:
                    category = "畑"
                elif ftCode == FTCODE_TEA:
                    category = "茶畑"
                elif ftCode == FTCODE_ORCHARD:
                    category = "果樹園"
                else:
                    continue
                geom = feat.get("geometry", {})
                if geom.get("type") != "Point":
                    continue
                coords = geom.get("coordinates")
                if not coords:
                    continue
                local_x, local_y = coords
                # Y軸反転
                local_y = actual_extent - local_y
                lat, lon = tile_to_wgs84(z, x, y, local_x, local_y, extent=actual_extent)
                # ポリゴン内判定
                if not point_in_polygon(lat, lon, POLYGON_WGS84):
                    continue
                # 名前属性はない場合が多いが、あれば取得
                name = props.get("knj") or props.get("name") or ""
                results.append({
                    "名称": name,
                    "種別": category,
                    "緯度": lat,
                    "経度": lon,
                    "ftCode": ftCode,
                    "source_layer": layer_name
                })
                found_in_tile = True
        time.sleep(0.05)
    # 4. 結果の保存
    if not results:
        print("\n:警告: 指定範囲内にデータが見つかりませんでした。")
        return
    df = pd.DataFrame(results)
    # 重複削除
    df = df.drop_duplicates(subset=["緯度", "経度", "種別"])
    output_file = "agriculture_fixed.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print("\n" + "="*50)
    print(f":チェックマーク_緑: 処理完了！")
    print(f"   総件数: {len(df)} 件")
    print(f"   保存先: {output_file}")
    print("="*50)
    # 内訳を表示
    print(df["種別"].value_counts())
if __name__ == "__main__":
    extract_agriculture()