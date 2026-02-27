# 使用 seaborn 繪製效能分析圖
import matplotlib
matplotlib.use('TkAgg')
# 用以解決 Mac 視窗無法彈出的問題
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 分別讀取 未改良前 與 改良後 模型的實驗紀錄
data_model = pd.read_csv('讀取紀錄模型輸出檔案') # 未改良前資料
data_intro = pd.read_csv('讀取紀錄模型輸出檔案') # 模型改良後紀錄

# 將原始資料展開成 Pandas DataFrame 格式
def get_df(data, type_name):
    rows = []
    groups = ['Depth', 'SAM', 'DeepLabV3+']
    for i in range(len(data['Model'])):
        for j in range(len(data['Inference_Time'][i])):
            rows.append({
                'Model_Group': groups[i],               # 模型群組
                'Model': data['Model'][i],              # 具體模型名稱
                'Type': type_name,                      # 資料類型：Original 或 Improved
                'Time': data['Inference_Time'][i][j],   # 推論時間 (X軸)
                'IoU': data['IoU'][i][j],               # 準確度指標 (Y軸)
                'F_score': data['F_score'][i][j]        # 氣泡大小參考
            })
    return pd.DataFrame(rows)

# 合併原始與改良後的資料集
df_all = pd.concat([get_df(data_model, 'Original'), get_df(data_intro, 'Improved')])

# === 繪圖設定 ===
sns.set_theme(style="whitegrid")
# 支援中文顯示（針對 Mac 環境）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
fig, ax = plt.subplots(figsize=(11, 7))

# 繪製氣泡圖：
# x: 時間, y: IoU, hue: 顏色區分群組, style: 形狀區分前後, size: 氣泡大小反映 F_score
scatter = sns.scatterplot(
    data=df_all, x='Time', y='IoU', hue='Model_Group', style='Type',
    size='F_score', sizes=(50, 400), alpha=0.6, palette='Set1', ax=ax # 設定氣泡最小與最大尺寸
)

# 計算並繪製平均位移連線（箭頭）
groups = ['Depth', 'SAM', 'DeepLabV3+']
colors = sns.color_palette('Set1', n_colors=3)

for idx, group in enumerate(groups):
    # 分別取出該群組在「原始」與「改良」下的所有數據
    orig = df_all[(df_all['Model_Group'] == group) & (df_all['Type'] == 'Original')]
    impr = df_all[(df_all['Model_Group'] == group) & (df_all['Type'] == 'Improved')]

    # 計算平均中心點 (Centroid)
    o_c = (orig['Time'].mean(), orig['IoU'].mean()) # 原始平均座標
    i_c = (impr['Time'].mean(), impr['IoU'].mean()) # 改良後平均座標

    # 繪製虛線箭頭：從原始指向改良，視覺化效能提升方向
    ax.annotate("", xy=i_c, xytext=o_c,
                arrowprops=dict(arrowstyle="->", color=colors[idx], lw=2, linestyle='--', alpha=0.8))

    # 在箭頭上方標註 IoU 的變化百分比
    iou_change = ((i_c[1] - o_c[1]) / o_c[1]) * 100
    ax.text(i_c[0], i_c[1] + 0.015, f"{iou_change:+.1f}% IoU",
            color=colors[idx], fontsize=9, fontweight='bold', ha='center')

# 座標軸與標籤優化
plt.title('影像切割模型效能位移分析 (改良前後對比)', fontsize=15, pad=20)
plt.xlabel('推論耗時 Inference Time (Seconds)', fontsize=12)
plt.ylabel('平均準確度 IoU', fontsize=12)

# 根據數據調整顯示範圍
plt.xlim(-1, 35)  # 考慮到耗時約為 0～31 秒
plt.ylim(0, 0.6)  # IoU 最高約 0.6

# 繪製 10 秒門檻參考線：通常用於區分「即時/近即時」推論的邊界
plt.axvline(x=10, color='red', linestyle='--', linewidth=2, alpha=0.6)

# 圖例設定：將圖例移至繪圖區右側避免遮擋氣泡
plt.legend(title="Model_Group / F-score / Type", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()

plt.savefig('corrected_performance_analysis.png', dpi=500)
plt.show()