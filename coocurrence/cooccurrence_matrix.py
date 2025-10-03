# cooccurrence_matrix.py
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURES = [
    "TonguePale", "TipSideRed", "Spot", "Ecchymosis",
    "Crack", "Toothmark", "FurThick", "FurYellow"
]

def ensure_binary(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = (df[c].astype(float) != 0).astype(int)
    return df

def plot_heatmap(base_counts, norm_matrix, title, subtitle_note, out_png, dpi=300):
    features = base_counts.index.tolist()
    n = len(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm_matrix.values, aspect="equal")  # 預設 colormap

    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticklabels(features)

    # 註記：count 與 (ratio)；遇到無法定義的分母→顯示 "-"
    for i in range(n):
        for j in range(n):
            count = base_counts.iat[i, j]
            val = norm_matrix.iat[i, j]
            text = f"{int(count)}\n({val:.2f})" if pd.notna(val) else f"{int(count)}\n(-)"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    # 輕格線
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized ratio", rotation=270, labelpad=15)

    ax.set_title(title)
    fig.text(0.5, 0.01, subtitle_note, ha="center", fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="train_fold1.csv", help="輸入 CSV 路徑")
    ap.add_argument("--outdir", default=".", help="輸出資料夾")
    ap.add_argument("--dpi", type=int, default=300, help="圖檔 DPI")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 讀取/二元化
    df = pd.read_csv(args.csv)
    df_sel = ensure_binary(df[FEATURES], FEATURES)

    # 共現計數
    X = df_sel.values
    co_mat = pd.DataFrame(X.T @ X, index=FEATURES, columns=FEATURES)

    # 列/欄正規化（以對角線為分母）
    diag = np.diag(co_mat.values).astype(float)
    n_feat = len(FEATURES)

    # Row-normalized: P(j|i)
    row_norm_vals = co_mat.values.astype(float).copy()
    for i in range(n_feat):
        if diag[i] > 0:
            row_norm_vals[i, :] /= diag[i]
        else:
            row_norm_vals[i, :] = np.nan
    row_norm = pd.DataFrame(row_norm_vals, index=FEATURES, columns=FEATURES)
    for i in range(n_feat):
        if diag[i] > 0:
            row_norm.iat[i, i] = 1.0

    # Column-normalized: P(i|j)
    col_norm_vals = co_mat.values.astype(float).copy()
    for j in range(n_feat):
        if diag[j] > 0:
            col_norm_vals[:, j] /= diag[j]
        else:
            col_norm_vals[:, j] = np.nan
    col_norm = pd.DataFrame(col_norm_vals, index=FEATURES, columns=FEATURES)
    for j in range(n_feat):
        if diag[j] > 0:
            col_norm.iat[j, j] = 1.0

    # 輸出 CSV
    co_mat.to_csv(os.path.join(args.outdir, "cooccurrence_matrix.csv"))
    row_norm.to_csv(os.path.join(args.outdir, "cooccurrence_row_norm.csv"))
    col_norm.to_csv(os.path.join(args.outdir, "cooccurrence_col_norm.csv"))

    # 圖註：各特徵總次數（= 對角線）
    totals = pd.Series(diag, index=FEATURES).astype(int)
    totals_caption = " | ".join([f"{f}: {int(totals[f])}" for f in FEATURES])

    # 畫圖
    plot_heatmap(
        co_mat, row_norm,
        "Row-normalized Co-occurrence (P(j | i))",
        f"Row divisor = diagonal count of each row feature; Totals: {totals_caption}",
        os.path.join(args.outdir, "cooccurrence_row_norm_heatmap.png"),
        dpi=args.dpi
    )
    plot_heatmap(
        co_mat, col_norm,
        "Column-normalized Co-occurrence (P(i | j))",
        f"Column divisor = diagonal count of each column feature; Totals: {totals_caption}",
        os.path.join(args.outdir, "cooccurrence_col_norm_heatmap.png"),
        dpi=args.dpi
    )

if __name__ == "__main__":
    main()
