import cv2
import sys
import os
#四隅をクリックして座標を取得するスクリプト
# --- 設定 ---
# 記録するクリックの回数
NUM_CLICKS_NEEDED = 4
# ----------------

# クリックした座標を保存するリスト
clicked_points = []

def click_event(event, x, y, flags, params):
    """
    マウスクリック時のコールバック関数
    """
    global clicked_points
    
    # 左クリックが押された時
    if event == cv2.EVENT_LBUTTONDOWN:
        # 座標をリストに追加
        clicked_points.append({'x': x, 'y': y})
        
        # 画像に印を描画
        img_display = params['image'].copy()
        for i, point in enumerate(clicked_points):
            cv2.circle(img_display, (point['x'], point['y']), 10, (0, 0, 255), 3)
            cv2.putText(img_display, f"{i+1}", (point['x'] + 15, point['y'] + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow("Image", img_display)
        
        print(f"クリック {len(clicked_points)} 回目: (X={x}, Y={y}) を記録しました。")

        # 4回クリックしたら終了
        if len(clicked_points) >= NUM_CLICKS_NEEDED:
            print("\n--- 4点の座標を取得しました ---")
            print("ウィンドウを閉じて、以下の出力を georeference_and_poi.py にコピーしてください：\n")
            
            # georeference_and_poi.py 用のフォーマットで出力
            print("IMAGE_CORNER_PIXELS = {")
            print(f"    'top_left':     {{'x': {clicked_points[0]['x']}, 'y': {clicked_points[0]['y']}}},  # 1. 左上")
            print(f"    'top_right':    {{'x': {clicked_points[1]['x']}, 'y': {clicked_points[1]['y']}}},  # 2. 右上")
            print(f"    'bottom_left':  {{'x': {clicked_points[2]['x']}, 'y': {clicked_points[2]['y']}}},  # 3. 左下")
            print(f"    'bottom_right': {{'x': {clicked_points[3]['x']}, 'y': {clicked_points[3]['y']}}}   # 4. 右下")
            print("}")
            
            cv2.waitKey(3000) # 3秒待ってから
            cv2.destroyAllWindows() # ウィンドウを閉じる

def main():
    if len(sys.argv) < 2:
        print("エラー: 画像ファイルのパスを指定してください。")
        print("使い方: python get_pixel_coords.py <画像ファイルのパス>")
        print(r"例: python get_pixel_coords.py C:\Users\...\input\NI_52_11_8.jpg")
        return

    img_path = sys.argv[1]
    
    if not os.path.exists(img_path):
        print(f"エラー: ファイルが見つかりません: {img_path}")
        return

    # 画像を読み込む
    image = cv2.imread(img_path)
    if image is None:
        print("エラー: 画像を読み込めませんでした。")
        return

    print(f"画像 {img_path} を開きました。")
    print("地図の図郭（内枠）の四隅を、以下の順番でクリックしてください：")
    print("  1. 左上 (Top-Left)")
    print("  2. 右上 (Top-Right)")
    print("  3. 左下 (Bottom-Left)")
    print("  4. 右下 (Bottom-Right)")
    print("\n注意: ウィンドウが画面外にはみ出ている場合は、ウィンドウを移動させてください。")

    # ウィンドウを作成し、コールバック関数を設定
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL) # サイズ変更可能なウィンドウ
    cv2.setMouseCallback("Image", click_event, {'image': image})
    
    # 最初の画像を表示
    cv2.imshow("Image", image)
    
    print("\nウィンドウ上でクリック待機中... (ウィンドウを閉じるには 'q' を押してください)")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        # 'q' が押されたらループを抜ける
        if key == ord('q'):
            break
        # 4回クリックされたら自動で終了
        if len(clicked_points) >= NUM_CLICKS_NEEDED:
            break
            
    cv2.destroyAllWindows()
    print("処理を終了しました。")

if __name__ == "__main__":
    main()