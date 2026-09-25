#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成交付包：把 LSTM（3.1.2 节）的成果整理成下游 XGBoost-Bootstrap（3.1.3 节）可直接使用的数据。

交付内容（写入 ../交付数据/data/）：
  1. lstm_trend_history.csv        每个 (国家, 目标届) 的“一步预测趋势值”（跨 5 个种子取均值±std）
  2. lstm_trend_2028.csv           每个国家的 2028 年趋势值（跨 5 个种子取均值±std）+ host 概率
  3. base_panel_country_year.csv   82 国 × 19 届（1952–2024）的基础面板：真实奖牌、Host、Prep、alpha_c
  4. host_years_by_country.csv     各国主办年份（含 2028/2032 未来主办）——生成主场效应特征用
  5. lstm_run_config.json          交付模型的配置与数据口径
  6. lstm_metrics.csv              交付配置（32×6, main 变体）的 5 种子评估指标
  7. manifest.json                 文件清单 + 行数 + SHA256 校验和

用法：python export_delivery.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lstm_312 import (DATA_DIR, HERE, MIN_EDITIONS, N_STEPS, OUT_DIR, PREP_WINDOW,  # noqa: E402
                      PRED_YEAR, START_YEAR, VAL_TARGETS, build_panel, load_hosts)

DELIVER_DIR = HERE.parent / "交付数据"
DATA_OUT = DELIVER_DIR / "data"

DELIVERED = {           # 交付的 LSTM 配置（与第一轮主实验一致）
    "hidden_size": 32,
    "patience": 6,
    "variant": "main",
    "weight_selection": "best",      # 恢复验证集最优权重
    "seeds": [0, 1, 2, 3, 4],
    "epochs_max": 200,
    "lr": 1e-3,
    "batch": 16,
    "dropout": 0.2,
    "weight_decay": 1e-5,
    "lambda_medal": 0.7,
    "lambda_host": 0.3,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- 1) 历史趋势值（跨种子聚合） ----------------
    hist = pd.read_csv(OUT_DIR / "trend_history.csv")
    hist = hist[hist["variant"] == DELIVERED["variant"]]
    g = hist.groupby(["Country", "TargetEdition"], as_index=False).agg(
        Gold_trend=("Gold_pred", "mean"), Gold_trend_std=("Gold_pred", "std"),
        Total_trend=("Total_pred", "mean"), Total_trend_std=("Total_pred", "std"),
        Host_prob=("Host_prob", "mean"),
        Gold_true=("Gold_true", "first"), Total_true=("Total_true", "first"),
        IsHost_true=("Host_true", "first"), Split=("Split", "first"), n_seeds=("seed", "count"),
    ).rename(columns={"TargetEdition": "Year"})
    hist_out = g[["Country", "Year", "Gold_trend", "Gold_trend_std", "Total_trend", "Total_trend_std",
                  "Host_prob", "Gold_true", "Total_true", "IsHost_true", "Split", "n_seeds"]]
    hist_out = hist_out.sort_values(["Country", "Year"]).reset_index(drop=True)
    hist_out.to_csv(DATA_OUT / "lstm_trend_history.csv", index=False, encoding="utf-8-sig")

    # ---------------- 2) 2028 趋势值 ----------------
    p28 = pd.read_csv(OUT_DIR / "trend_2028_by_seed.csv")
    p28 = p28[p28["variant"] == DELIVERED["variant"]]
    a28 = p28.groupby("Country", as_index=False).agg(
        Gold_trend_2028=("Gold_trend_2028", "mean"),
        Gold_trend_2028_std=("Gold_trend_2028", "std"),
        Total_trend_2028=("Total_trend_2028", "mean"),
        Total_trend_2028_std=("Total_trend_2028", "std"),
        Host_prob_2028=("Host_prob_2028", "mean"),
    )
    panel = build_panel()
    is_host_2028 = {c: int(PRED_YEAR in ys) for c, ys in panel.host_years.items()}
    a28["IsHost2028"] = a28["Country"].map(is_host_2028).fillna(0).astype(int)
    a28 = a28.sort_values("Total_trend_2028", ascending=False).reset_index(drop=True)
    a28.to_csv(DATA_OUT / "lstm_trend_2028.csv", index=False, encoding="utf-8-sig")

    # ---------------- 3) 基础面板（真实值 + Host/Prep/alpha_c） ----------------
    rows = []
    for i, c in enumerate(panel.countries):
        for j, y in enumerate(panel.editions):
            rows.append({
                "Country": c, "Year": int(y),
                "Gold": float(panel.gold[i, j]), "Total": float(panel.total[i, j]),
                "IsHost": int(panel.host[i, j]), "Prep": int(panel.prep[i, j]),
                "Alpha_c": round(float(panel.alpha[i, j]), 6),
            })
    base = pd.DataFrame(rows)
    base.to_csv(DATA_OUT / "base_panel_country_year.csv", index=False, encoding="utf-8-sig")

    # ---------------- 4) 主办年份表 ----------------
    hosts = load_hosts()
    hy = []
    for c in panel.countries:
        ys = panel.host_years.get(c, [])
        future = [y for y in ys if y > 2024]
        hy.append({"Country": c, "HostYears": ";".join(map(str, ys)),
                   "NextHostYear": (min(future) if future else ""),
                   "IsFutureHost": int(bool(future))})
    pd.DataFrame(hy).to_csv(DATA_OUT / "host_years_by_country.csv", index=False, encoding="utf-8-sig")

    # ---------------- 5) 配置 ----------------
    cfg = {
        "交付对象": "2025 MCM Problem C / Team #2500759 —— 论文 3.1.2 节 LSTM 趋势建模",
        "模型": DELIVERED,
        "数据口径": {
            "源数据目录": str(DATA_DIR),
            "序列起点": START_YEAR,
            "届次数": len(panel.editions),
            "届次列表": [int(y) for y in panel.editions],
            "国家数": len(panel.countries),
            "国家纳入门槛(1952 后有奖牌记录的届数)": MIN_EDITIONS,
            "历史政治实体合并": "苏联/独联体/ROC→俄罗斯；东德/西德/德国联队→德国；捷克斯洛伐克→捷克；南斯拉夫/南联盟/塞黑→塞尔维亚；福尔摩沙/台湾→中华台北",
            "样本对齐": "窗口=连续 3 届 → 预测下一届（t→t+1，无信息泄漏）",
            "验证集": list(VAL_TARGETS),
            "预测目标届": PRED_YEAR,
            "备战窗口(年)": PREP_WINDOW,
            "alpha_c 计算": "因果版：只用目标届之前的主办届与历史成绩；从未主办则为 0",
            "目标标准化": "Gold/Total 按训练窗口的均值/标准差标准化后计算 MSE，指标在原始尺度上报",
        },
        "样本量": {
            "总样本": int(len(hist) // DELIVERED["seeds"].__len__() * DELIVERED["seeds"].__len__()),
            "训练(目标届<=2016)": int((hist["Split"] == "train").sum()),
            "验证(目标届 2020/2024)": int((hist["Split"] == "val").sum()),
            "预测(2028)": int(len(a28)),
        },
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "生成脚本": "lstm_312/export_delivery.py",
    }
    (DATA_OUT / "lstm_run_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 6) 指标 ----------------
    ms = pd.read_csv(OUT_DIR / "metrics_summary.csv")
    ms = ms[ms["variant"] == DELIVERED["variant"]]
    ms.to_csv(DATA_OUT / "lstm_metrics.csv", index=False, encoding="utf-8-sig")

    # ---------------- 7) manifest ----------------
    manifest = {"生成时间": cfg["生成时间"], "文件": []}
    for f in sorted(DATA_OUT.glob("*")):
        if f.name == "manifest.json":
            continue
        n = None
        if f.suffix == ".csv":
            n = int(len(pd.read_csv(f)))
        manifest["文件"].append({"文件名": f.name, "行数": n, "字节": f.stat().st_size,
                                 "sha256": sha256(f)})
    (DATA_OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    # ---------------- 汇总打印 ----------------
    print("=== 交付数据已生成 ===")
    print(f"目录: {DATA_OUT}")
    for it in manifest["文件"]:
        print(f"  {it['文件名']:<32} 行数={str(it['行数']):>6}  {it['字节']:>9,} B  sha256={it['sha256'][:12]}…")
    print("\n--- lstm_trend_history.csv 前 3 行 ---")
    print(hist_out.head(3).to_string(index=False))
    print("\n--- lstm_trend_2028.csv 前 5 行 ---")
    print(a28.head(5).to_string(index=False))
    print("\n--- base_panel_country_year.csv 前 3 行（美国 1996 主办年附近）---")
    print(base[(base["Country"] == "United States") & (base["Year"].between(1992, 2000))].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
