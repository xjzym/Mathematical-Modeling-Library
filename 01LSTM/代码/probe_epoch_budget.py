#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
探测：如果“patience”被当作**很小的训练轮数预算**（而不是标准早停容忍度），
WMAE 会不会像论文图 3 那样剧烈跳动？

做法：固定 hidden_size ∈ {32,64,128}，把训练轮数预算设为 2..10 轮（等价于“最多只训这么多轮”，
早停阈值设得很大以保证跑满预算），记录验证集 WMAE，看跨预算的极差有多大。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lstm_312 import OUT_DIR, build_panel, make_samples, train_one, torch  # noqa: E402


def main() -> int:
    torch.set_num_threads(4)
    panel = build_panel()
    samples = make_samples(panel)
    budgets = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rows = []
    for hidden in (32, 64, 128):
        for ep in budgets:
            for seed in range(5):
                r = train_one(panel, samples, seed=seed, hidden=hidden, patience=10 ** 6,
                              epochs=ep, lr=1e-3, batch=16, dropout=0.2, weight_decay=1e-5,
                              variant="main", select="last")
                rows.append({"hidden": hidden, "epoch_budget": ep, "seed": seed,
                             "WMAE_Gold": r["val_WMAE_true_gold"],
                             "WMAE_Total": r["val_WMAE_true_total"]})
                print(f"hidden={hidden} budget={ep} seed={seed} "
                      f"G={r['val_WMAE_true_gold']:.3f} T={r['val_WMAE_true_total']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "probe_epoch_budget.csv", index=False, encoding="utf-8-sig")
    agg = df.groupby(["hidden", "epoch_budget"])[["WMAE_Gold", "WMAE_Total"]].mean().round(3)
    print("\n各训练预算下的验证 WMAE（5 种子均值）:")
    print(agg.to_string())

    for metric in ("WMAE_Gold", "WMAE_Total"):
        means = agg[metric].groupby(level=0)
        rng = (means.max() - means.min())
        print(f"\n{metric}: 每个 hidden 内“跨预算”均值极差 = "
              + ", ".join(f"{h}:{v:.2f}" for h, v in rng.items()))
        print(f"{metric}: 全部 (hidden × 预算) 的均值极差 = {agg[metric].max() - agg[metric].min():.2f}")
        print(f"{metric}: 单次运行(min~max) = {df[metric].min():.2f} ~ {df[metric].max():.2f}")
    print("\n对照：论文图 3 的极差 WMAE_Gold ≈ 3.22, WMAE_Total ≈ 5.28")
    return 0


if __name__ == "__main__":
    sys.exit(main())
