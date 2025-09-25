# run_kfold.py — 一鍵跑完多折，免去 PowerShell 續行痛苦
import argparse, sys, subprocess, re
from pathlib import Path

def detect_folds(csv_dir: Path):
    # 掃描 train_foldX.csv / val_foldX.csv，取交集
    trains = {int(m.group(1)) for p in csv_dir.glob("train_fold*.csv")
              if (m := re.search(r"train_fold(\d+)\.csv$", p.name))}
    vals   = {int(m.group(1)) for p in csv_dir.glob("val_fold*.csv")
              if (m := re.search(r"val_fold(\d+)\.csv$", p.name))}
    folds = sorted(trains & vals)
    if not folds:
        raise SystemExit(f"在 {csv_dir} 找不到 train_fold*.csv / val_fold*.csv 的交集")
    return folds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="專案根目錄（包含 train.py）")
    ap.add_argument("--csv_dir", required=False, help="CSV 目錄，預設 <root>/data")
    ap.add_argument("--image_root", required=False, help="圖片根目錄，預設 <root>/images")
    ap.add_argument("--folds", nargs="*", type=int, help="要跑哪些折，例如: --folds 1 3 5；不填自動偵測")
    ap.add_argument("--labels", required=True,
                    help="以逗號分隔的標籤欄位，例如 TonguePale,TipSideRed,...")
    # 常用訓練參數（都可省略用預設）
    ap.add_argument("--backbone", default="swin_base_patch4_window7_224")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--head_min_pos", type=int, default=50)
    ap.add_argument("--httn_ensembles", type=int, default=1)
    ap.add_argument("--lambda_start", type=float, default=0.3)
    ap.add_argument("--lambda_target", type=float, default=0.7)
    ap.add_argument("--lambda_warmup", type=int, default=5)
    ap.add_argument("--use_attention", action="store_true")
    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--python", default=sys.executable, help="Python 可執行檔，預設當前解譯器")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    csv_dir = Path(args.csv_dir).resolve() if args.csv_dir else root / "data"
    image_root = Path(args.image_root).resolve() if args.image_root else root / "images"
    train_py = root / "train.py"
    if not train_py.exists():
        raise SystemExit(f"找不到 {train_py}")

    folds = args.folds or detect_folds(csv_dir)
    print(f"將執行折數：{folds}")

    for k in folds:
        train_csv = csv_dir / f"train_fold{k}.csv"
        val_csv   = csv_dir / f"val_fold{k}.csv"
        save_dir  = root / f"work_dir_httn_fold{k}"
        cmd = [
            args.python, str(train_py),
            "--train_csv",  str(train_csv),
            "--val_csv",    str(val_csv),
            "--image_root", str(image_root),
            "--save_dir",   str(save_dir),
            "--backbone",   args.backbone,
            "--epochs",     str(args.epochs),
            "--batch",      str(args.batch),
            "--num_workers",str(args.num_workers),
            "--lr",         str(args.lr),
            "--wd",         str(args.wd),
            "--img_size",   str(args.img_size),
            "--head_min_pos", str(args.head_min_pos),
            "--httn_ensembles", str(args.httn_ensembles),
            "--lambda_start", str(args.lambda_start),
            "--lambda_target", str(args.lambda_target),
            "--lambda_warmup", str(args.lambda_warmup),
            "--labels",     args.labels
        ]
        if args.use_attention: cmd.append("--use_attention")
        if args.use_amp:       cmd.append("--use_amp")

        print("\n=== 執行 fold", k, "===")
        print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        # Windows 對空白路徑敏感，subprocess 以 list 形式傳遞最安全
        ret = subprocess.call(cmd)
        if ret != 0:
            raise SystemExit(f"fold {k} 失敗（return code={ret}）")

    print("\n全部折數完成 ✅")

if __name__ == "__main__":
    main()
