import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("TkAgg")  # 或 "QtAgg"
# 用以解決 Mac 視窗無法彈出的問題

# === 評估指標（IoU、F-score） ===
def compute_iou(pred_mask, true_mask):
    # 計算 交併比 (Intersection over Union, IoU)，常用於衡量預測區域與實際區域的重疊程度
    # 計算交集：兩個遮罩同時為 True 的像素數量
    intersection = np.logical_and(pred_mask, true_mask).sum()
    # 計算聯集：兩個遮罩中至少有一個為 True 的像素總數
    union = np.logical_or(pred_mask, true_mask).sum()
    # 避免除以零：若聯集為 0，則 IoU 為 0，否則計算交比併
    return intersection / union if union > 0 else 0

def compute_fscore(pred_mask, true_mask):
    # 計算 F-score，結合精確率 (Precision) 與召回率 (Recall) 的調和平均數
    # TP (True Positive)：預測為正且實際為正
    tp = np.logical_and(pred_mask, true_mask).sum()
    # FP (False Positive)：預測為正但實際為負 (誤報)
    fp = np.logical_and(pred_mask, ~true_mask).sum()
    # FN (False Negative)：預測為負但實際為正 (漏報)
    fn = np.logical_and(~pred_mask, true_mask).sum()
    # 計算精確率 (1e-8 是為了防止除以零的小偏移量)＆召回率，並回傳 F-score 公式
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)