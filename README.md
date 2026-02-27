# 結合深度估計之影像切割優化
**Key Words**： Computer Vision, Data Analysis, Depth Anything V2, Performance Optimization

## 📌 專案總覽（Project Overview）
本專案基於我的碩士畢業研究，意旨為解決電子商務領域中「高品質去背」與「自動化效率」之間的矛盾。透過引入 Depth Anything V2 的深度估計（Depth Estimation）技術，補足傳統語意分割模型在複雜場景下的侷限。

## 🚀 核心技術（Tech Stack）
- 語言/框架： Python, PyTorch
- 深度學習模型： Depth Anything V2, DeepLabV3+, Segment Anything (SAM)
- 數據處理與分析： Pandas, Matplotlib, Seaborn
- 實驗數據集： COCO Dataset (val2017)

## 📝 專案描述
【Situation：情境】

在電商產業中，商品圖去背是高頻需求。然而，現有的自動化模型面臨兩大痛點：
- 語意依賴： 傳統模型（如 DeepLab）若沒看過該類物件，效果極差。
- 效率低下： 高精度模型（如 SAM）在完全自動化模式下推論極慢，難以支撐大規模批次處理。

【Task：任務】

- 開發一套「非語意依賴」的自動化切割系統，利用「空間景深」而非「顏色/類別」來區分前景與背景，目標是在 10 秒內 完成高品質的去背處理。
- 利用「空間景深」完成初步去背後，引進 語意依賴 模型，觀察最終去背成果及效率。

【Action：行動】
- 數據預處理 (ETL)： 針對 COCO 數據集進行隨機抽樣實驗，建立自動化影像處理 Pipeline。
- 系統開發： 整合 Depth Anything V2 提取 空間深度特徵，並提出「梯度輪廓強化策略」，結合形態學運算（Morphological Operations），用以修正邊界鋸齒與遮罩洩漏。
- 多模態融合： 實作數學交集運算 $MF=(MD⋅(1−B))∩MS$，結合語意資訊與深度資訊。
- 對照實驗： 針對 500 張樣本影像進行五輪壓力測試，量化 IoU、F-score 與推論時間。

【Result：結果】
- 穩定度提升： 在無標註或複雜場景中，深度導向法表現顯著優於 SAM (完全自動化模式)，在複雜場景下的 IoU（交集佔聯集比）表現更具競爭力。
- 效率優化： 相較於高精度 SAM 的推論時間 31.5s，深度導向法的推論時間降至 9.26s，效率提升 240%，在維持高品質輸出的同時，大幅提升處理產能。
- 商業洞察： 透過數據分析發現 DeepLabV3+ 在特定語意場景的優勢，確立了「多模型協作」策略，可針對不同業務場景（即時 vs. 高精度）動態切換模型。

## Mathematical Logic
- 交並比（Intersection over Union，簡稱 IoU）：

  在目標檢測（Object Detection）和語意分割中，用來衡量兩個邊框（Bounding Box）或區域重疊程度的一個指標。

  IoU 計算的是兩個集合的交集（Intersection）除以它們的聯集（Union）：
  
    <p align="center">
      $$IoU = \frac{∣A∪B∣}{∣A∩B∣}$$
    </p>

## 🪧 量化指標（Quantitative Metrics）
| 模型方法 | 平均處理時間 (s) | IoU (指標) | F-score |
| --- | --- | --- | --- |
|本研究 (Depth-based) | 9.258 |	穩定 (穩定性高) |	0.466+ |
|DeepLabV3+ |	1.144 |	依賴語意類別 | 0.788 (受限)|
|Segment Anything (SAM)	| 31.519 | 低 (自動模式) | 低 (自動模式)|

## 📊 實驗結果視覺化
<p align="center">
<img width="500" alt="Github-畢業論文所需圖表-seaborn" src="https://github.com/user-attachments/assets/6869def9-44eb-44e9-961d-8d246b80ae77" />
</p>

- 由散佈圖可以清晰發現：本研究提出的方法（Depth-based）推論時間在 10 秒內，提供了比 SAM 更穩定的 IoU 表現，是兼顧成本與效果的最佳商業選擇。
- 而將深度資訊與結合語意資訊（DeepLabV3+）的策略，雖然 IoU 表現略微降低，但仍比 SAM 擁有更高效且穩定的 IoU 表現。

## 💥 Bad Case Analysis (異常值分析)
根據 `analysis/` 資料夾中的圖片可以發現：
雖然模型整體表現好，但在背景與主體深度極度接近時，仍存有部分圖像去除不完全之挑戰。這展現了我對數據的誠實度與分析問題的能力。
