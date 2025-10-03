# correlation_matrix.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# ===============================
# 讀取資料
# ===============================
file_path = "train_fold1.csv"  # 這裡換成你的檔案路徑
df = pd.read_csv(file_path)

# 需要的特徵
features = [
    "TonguePale", "TipSideRed", "Spot", "Ecchymosis",
    "Crack", "Toothmark", "FurThick", "FurYellow"
]
df_sel = df[features]

# ===============================
# 正規化數據 (0~1)
# ===============================
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df_sel), columns=features)

# ===============================
# 計算關聯性矩陣
# ===============================
corr_matrix = df_norm.corr(method="pearson")

# ===============================
# 統計每個特徵的「總數」 (值為 1 的次數)
# ===============================
totals = df_sel.sum()

# ===============================
# 輸出
# ===============================
print("8x8 正規化後的關聯性矩陣：")
print(corr_matrix)
print("\n每個特徵的總數：")
print(totals)

# ===============================
# 畫熱力圖
# ===============================
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, 
    annot=True, cmap="coolwarm", fmt=".2f", 
    cbar=True, square=True
)
plt.title("Normalized Feature Correlation Matrix (8x8)", fontsize=14)

# 在圖下方加上每個特徵的總數
caption = " | ".join([f"{f}: {int(totals[f])}" for f in features])
plt.figtext(0.5, -0.05, f"Totals (sum of 1's): {caption}", wrap=True, ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("correlation_matrix_heatmap.png", dpi=300)
plt.show()

# 也存成 CSV
corr_matrix.to_csv("correlation_matrix_normalized.csv")
