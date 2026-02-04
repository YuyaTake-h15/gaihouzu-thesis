import pandas as pd
import os

# ==========================================
# 設定
# ==========================================
# さきほど保存したCSVのパス
CSV_PATH = "C:/Users/TakedaYuya/Landmark_Gaihouzu_new/manual_ground_truth_NI-52-11-9.csv"

def main():
    if not os.path.exists(CSV_PATH):
        print(f"エラー: ファイルが見つかりません ({CSV_PATH})")
        print("まずは手動カウントツールを実行して保存してください。")
        return

    # CSVを読み込む
    try:
        df = pd.read_csv(CSV_PATH)
        
        # 'label'列が存在するか確認
        if 'label' not in df.columns:
            print("エラー: CSVに 'label' 列がありません。")
            return

        print("-" * 30)
        print(f" 📂 読み込みファイル: {CSV_PATH}")
        print("-" * 30)

        # 集計実行 (value_countsで一発です)
        counts = df['label'].value_counts()

        # 結果を表示
        print("【地物別カウント結果】")
        for label, count in counts.items():
            # 英語ラベルを日本語に変換して表示
            jp_label = "不明"
            if label == "shrine": jp_label = "神社"
            elif label == "temple": jp_label = "寺院"
            elif label == "school": jp_label = "学校"
            
            print(f"  ■ {jp_label} ({label}): {count} 件")

        print("-" * 30)
        print(f"  ★ 合計: {len(df)} 件")
        print("-" * 30)

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    main()