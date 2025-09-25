import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 讀取 CSV
csv_path = "train_fold1.csv"   # 如果檔案不在同資料夾，請填完整路徑
df = pd.read_csv(csv_path)

# 指定八個舌苔欄位
tongue_cols = [
    "TonguePale", "TipSideRed", "Spot", "Ecchymosis",
    "Crack", "Toothmark", "FurThick", "FurYellow"
]

# 計算相關係數矩陣
# 你的欄位是 0/1 → 用 Pearson 就可以，相當於二元的 Phi 係數
corr = df[tongue_cols].corr(method="pearson")

# 繪製相關係數熱圖
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="YlGnBu", vmin=-1, vmax=1)
plt.title("Correlation of Tongue-Coating Labels", fontsize=14)
plt.tight_layout()
plt.show()
