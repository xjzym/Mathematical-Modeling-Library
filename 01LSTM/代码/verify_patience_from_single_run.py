#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具：用“一次长的训练”推导出整条 patience 扫描（省掉重复训练）

原理
----
early stopping 只决定“在哪里停”，不改变训练轨迹：同一随机种子、同一数据顺序下，
patience=10 的运行轨迹**包含** patience=6/7/8/9 的停止点。于是：
  · `select=last`：patience=k 的结果 == 第（最后一次改进轮次 + k）轮的权重/指标
  · `select=best`：所有 patience 的结果完全等价（都取运行最优轮次）
因此只要在训练时记录每轮验证指标（并可选保存每轮 checkpoint），
一次 patience=max 的训练即可得到 k = 1..max 的全部结果。

本脚本对 hidden ∈ {32,64,128} 各跑一次 patience=max，派生 k=6..10，
并与 `out/grid_results.csv` 中**单独训练**的结果逐项对账。

用法：
  python verify_patience_from_single_run.py --hiddens 32 64 128 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lstm_312 import OUT_DIR, build_panel, make_samples, train_one, torch  # noqa: E402


def derive_stop_epochs(history: list[dict], patiences: list[int]) -> dict[int, int]:
    """按 train_one 的早停规则回放历史：bad 累计到 patience 即停 → {patience: 停止轮次}。"""
    out: dict[int, int] = {}
    best, bad = float("inf"), 0
    for h in history:
        ep, score = h["epoch"], h["val_score"]
        if score < best - 1e-9:
            best, bad = score, 0
        else:
            bad += 1
        for k in patiences:
            if k not in out and bad >= k:
                out[k] = ep
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="单次训练派生整条 patience 扫描")
    ap.add_argument("--hiddens", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--patiences", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    ap.add_argument("--max-patience", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    panel = build_panel()
    samples = make_samples(panel)
    grid = pd.read_csv(OUT_DIR / "grid_results.csv")

    rows = []
    for hidden in args.hiddens:
        for seed in args.seeds:
            r = train_one(panel, samples, seed=seed, hidden=hidden, patience=args.max_patience,
                          epochs=200, lr=1e-3, batch=16, dropout=0.2, weight_decay=1e-5,
                          variant="main", select="last")
            base = pd.DataFrame(r["_history"]).set_index("epoch")
            stop = derive_stop_epochs(r["_history"], args.patiences)
            for k in args.patiences:
                ep = stop.get(k)
                if ep is None:          # 该 patience 未触发（只会发生在 k > max_patience 时）
                    continue
                ref = grid[(grid["hidden"] == hidden) & (grid["patience"] == k) & (grid["seed"] == seed)]
                if ref.empty:
                    continue
                rows.append({
                    "hidden": hidden, "seed": seed, "patience": k, "派生停止轮次": ep,
                    "派生_WMAE_Gold": base.loc[ep, "val_WMAE_gold"],
                    "派生_WMAE_Total": base.loc[ep, "val_WMAE_total"],
                    "单独训练_WMAE_Gold": float(ref["val_WMAE_true_gold"].iloc[0]),
                    "单独训练_WMAE_Total": float(ref["val_WMAE_true_total"].iloc[0]),
                    "单独训练轮数": int(ref["epochs_run"].iloc[0]),
                })

    df = pd.DataFrame(rows)
    df["Gold差值"] = (df["派生_WMAE_Gold"] - df["单独训练_WMAE_Gold"]).abs()
    df["Total差值"] = (df["派生_WMAE_Total"] - df["单独训练_WMAE_Total"]).abs()
    max_diff = float(max(df["Gold差值"].max(), df["Total差值"].max()))
    exact = int(((df["Gold差值"] < 1e-9) & (df["Total差值"] < 1e-9)).sum())

    print(df.head(10).to_string(index=False))
    print(f"\n对比组数 = {len(df)}（来自 {len(args.hiddens)}×{len(args.seeds)} 次训练）")
    print(f"逐位完全一致(<1e-9) = {exact}/{len(df)}；最大绝对差 = {max_diff:.3e}")
    print(f"判定：{'一致（差异仅来自 CPU 多线程浮点归约）' if max_diff < 1e-4 else '不一致 ✗'}")

    agg = (df.groupby(["hidden", "patience"])[["派生_WMAE_Gold", "派生_WMAE_Total"]]
           .mean().round(3).reset_index())
    print("\n由单次训练派生的完整网格（5 种子均值）——应与 out/grid_summary.csv 一致：")
    print(agg.to_string(index=False))

    df.to_csv(OUT_DIR / "derived_patience_sweep.csv", index=False, encoding="utf-8-sig")
    agg.to_csv(OUT_DIR / "derived_grid_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n已写出: {OUT_DIR/'derived_patience_sweep.csv'} , {OUT_DIR/'derived_grid_summary.csv'}")
    return 0 if max_diff < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
