#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2025 MCM Problem C / Team #2500759 —— 论文 3.1.2 节「LSTM：基于时间的趋势建模」复现

实现内容（与论文公式编号一一对应）：
  式(1) 主场优势系数 alpha_c        —— 因果版：只用目标届之前的历史主办届，避免信息泄漏
  式(2) 备战期效应 Prep_t           —— [t_host-8, t_host] 内为 1（8 年 = 两个奥运周期）
  式(3) 输入向量 x_t                —— [Gold, Total, Host, Prep, alpha_c]
  式(4) 双通道 LSTM + 拼接 + 全连接  —— medal 通道(2维) / host 通道(3维)，n_steps=3
  式(5) 分层损失 L = 0.7*L_medal + 0.3*L_host（MSE + BCE）
  式(6) WMAE 指标（w_c = 真实当届金牌数）

样本对齐（复现决策，见《LSTM_3.1.2_模型解析_复现第1步》）：
  窗口 = 连续 3 届(t-2,t-1,t) → 预测第 t+1 届（避免原文“同期预测”造成的信息泄漏）
  验证集 = 目标届 ∈ {2020, 2024}（按时间留出最后两届）
  训练集 = 目标届 <= 2016；预测届 = 2028

用法：
  python lstm_312.py --variant all --seeds 0 1 2 3 4 --hidden 32 --patience 6
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _setup_cjk_font() -> bool:
    """让 matplotlib 能显示中文标题（Windows 常见字体优先）。"""
    for name in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Source Han Sans SC"]:
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            matplotlib.rcParams["font.sans-serif"] = [name] + list(matplotlib.rcParams["font.sans-serif"])
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    return False


CJK_OK = _setup_cjk_font()

# --------------------------------------------------------------------------------------
# 路径与常量
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "mcm2025C" / "2025_Problem_C_Data" / "2025_Problem_C_Data"
OUT_DIR = HERE / "out"
FIG_DIR = HERE / "figs"

START_YEAR = 1952          # 序列起点（1952 年后国家口径相对稳定）
MIN_EDITIONS = 5           # 国家纳入门槛：1952 后至少 5 届有奖牌记录
N_STEPS = 3                # n_steps = 3（最近三届）
VAL_TARGETS = (2020, 2024)  # 验证集：按时间留出最后两届
PRED_YEAR = 2028           # 预测目标届
LAMBDA_MEDAL, LAMBDA_HOST = 0.7, 0.3   # 式(5a)
PREP_WINDOW = 8            # 式(2)：8 年 = 两个奥运周期

# 历史政治实体合并（用户确认：1952 年后 + 合并历史政治实体）
MERGE_MAP = {
    "Soviet Union": "Russia",
    "Unified Team": "Russia",
    "ROC": "Russia",
    "United Team of Germany": "Germany",
    "West Germany": "Germany",
    "East Germany": "Germany",
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia",
    "FR Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Formosa": "Chinese Taipei",
    "Taiwan": "Chinese Taipei",
}
# 非国家代表队：不进入国家面板
DROP_TEAMS = {
    "Independent Olympic Participants",
    "Independent Olympic Athletes",
    "Refugee Olympic Team",
    "Mixed team",
    "Australasia",
    "British West Indies",
    "Netherlands Antilles",
}
# hosts.csv 里的名称 → 奖牌表中的 NOC 名称
HOST_NAME_FIX = {"United Kingdom": "Great Britain"}

VARIANTS = {
    "main": "标准 BCE：p_hat = sigmoid(host 头输出)",
    "A": "host 头 bias 初始化为 log(0.1/0.9) ≈ -2.197（对应原文的 0.1 先验）",
    "B": "标签平滑 eps=0.1（正类目标 0.9、负类目标 0.1）",
}


# --------------------------------------------------------------------------------------
# 数据准备
# --------------------------------------------------------------------------------------
def _norm_name(s: str) -> str:
    return str(s).strip()


def load_medal_panel() -> pd.DataFrame:
    """读取奖牌表 → 清洗 NOC 名称 → 合并历史政治实体（同一届多实体求和）。"""
    df = pd.read_csv(DATA_DIR / "summerOly_medal_counts.csv")
    df["Country"] = df["NOC"].map(_norm_name)
    df = df[~df["Country"].isin(DROP_TEAMS)].copy()
    df["Country"] = df["Country"].replace(MERGE_MAP)
    df["Year"] = df["Year"].astype(int)
    df["Gold"] = df["Gold"].astype(float)
    df["Total"] = df["Total"].astype(float)
    g = df.groupby(["Country", "Year"], as_index=False)[["Gold", "Total"]].sum()
    return g.sort_values(["Country", "Year"]).reset_index(drop=True)


def load_hosts() -> dict[int, str]:
    """读取东道主表 → {年份: 合并后的国家名}（过滤停办届）。"""
    df = pd.read_csv(DATA_DIR / "summerOly_hosts.csv")
    out: dict[int, str] = {}
    for _, r in df.iterrows():
        year = int(r["Year"])
        host = _norm_name(r["Host"])
        if "Cancelled" in host or host == "" or host.lower() == "nan":
            continue
        country = _norm_name(host.split(",")[-1].split("(")[0])   # 去掉 “(postponed to 2021 …)” 之类括注
        country = HOST_NAME_FIX.get(country, country)
        country = MERGE_MAP.get(country, country)
        out[year] = country
    return out


@dataclass
class Panel:
    editions: list[int]
    countries: list[str]
    gold: np.ndarray            # (C, E)
    total: np.ndarray           # (C, E)
    host: np.ndarray            # (C, E) 0/1
    prep: np.ndarray            # (C, E) 0/1
    alpha: np.ndarray           # (C, E) 因果 alpha_c（用于“目标届=editions[e]”的样本）
    alpha_pred: np.ndarray      # (C,)  因果 alpha_c（用于目标届 = PRED_YEAR 的预测样本）
    host_years: dict[str, list[int]] = field(default_factory=dict)


def build_panel() -> Panel:
    medals = load_medal_panel()
    hosts_by_year = load_hosts()

    # 国家筛选：1952 年后至少 MIN_EDITIONS 届有奖牌记录
    recent = medals[medals["Year"] >= START_YEAR]
    cnt = recent.groupby("Country")["Year"].nunique()
    countries = sorted(cnt[cnt >= MIN_EDITIONS].index.tolist())

    editions = sorted(recent["Year"].unique().tolist())
    eidx = {y: i for i, y in enumerate(editions)}
    cidx = {c: i for i, c in enumerate(countries)}

    C, E = len(countries), len(editions)
    gold = np.zeros((C, E), dtype=np.float64)
    total = np.zeros((C, E), dtype=np.float64)
    for _, r in recent.iterrows():
        c, y = r["Country"], int(r["Year"])
        if c in cidx:
            gold[cidx[c], eidx[y]] = r["Gold"]
            total[cidx[c], eidx[y]] = r["Total"]

    # 东道主年份（按合并后的国家名）
    host_years: dict[str, list[int]] = {c: [] for c in countries}
    for y, c in hosts_by_year.items():
        if c in host_years:
            host_years[c].append(y)
    for c in host_years:
        host_years[c] = sorted(host_years[c])

    host = np.zeros((C, E), dtype=np.float64)
    prep = np.zeros((C, E), dtype=np.float64)
    for c, i in cidx.items():
        for y, j in eidx.items():
            if y in host_years[c]:
                host[i, j] = 1.0
            if any(h - PREP_WINDOW <= y <= h for h in host_years[c]):
                prep[i, j] = 1.0

    # 式(1) 因果版 alpha_c：对“目标届 = editions[j]”的样本，只用 < 该届 的历史
    # 平均项 = 该国全部有奖牌记录届（全历史，1896 起）中 < 目标届 的平均金牌
    # 主办项 = 该国主办届中 < 目标届 的平均金牌；若从未主办过则 alpha = 0
    all_gold = medals.groupby(["Country", "Year"])["Gold"].sum().reset_index()
    all_host_years = {c: [y for y, hc in hosts_by_year.items() if hc == c] for c in countries}

    alpha = np.zeros((C, E), dtype=np.float64)
    for c, i in cidx.items():
        sub = all_gold[all_gold["Country"] == c]
        hy = all_host_years[c]
        for j, y in enumerate(editions):
            alpha[i, j] = _causal_alpha(sub, hy, y)
    alpha_pred = np.zeros(C, dtype=np.float64)
    for c, i in cidx.items():
        sub = all_gold[all_gold["Country"] == c]
        alpha_pred[i] = _causal_alpha(sub, all_host_years[c], PRED_YEAR)

    return Panel(editions, countries, gold, total, host, prep, alpha, alpha_pred, host_years)


def _causal_alpha(sub: pd.DataFrame, host_years: list[int], target_year: int) -> float:
    """式(1) 的因果实现：只用 target_year 之前的历史。从未主办过 → 0。"""
    past = sub[sub["Year"] < target_year]["Gold"]
    hp = [h for h in host_years if h < target_year]
    if len(past) == 0 or len(hp) == 0:
        return 0.0
    avg = float(past.mean())
    host_avg = float(np.mean([float(sub[sub["Year"] == h]["Gold"].sum()) for h in hp]))
    return (host_avg - avg) / avg if avg > 0 else 0.0


@dataclass
class Samples:
    """X_medal:(N,3,2) X_host:(N,3,3) Y:(N,2) HostY:(N,) 以及索引信息"""
    x_medal: np.ndarray
    x_host: np.ndarray
    y: np.ndarray
    host_y: np.ndarray
    country: np.ndarray
    target_edition: np.ndarray
    gold_true: np.ndarray
    total_true: np.ndarray


def make_samples(panel: Panel, align: str = "next") -> Samples:
    """窗口 = 连续 3 届 → 预测下一届（t → t+1）；另加 2028 的纯预测样本（无标签）。

    align="same" 用于模拟原文式(4c)的同届对齐（窗口含目标届，存在信息泄漏），仅作对照实验。
    """
    C, E = panel.gold.shape
    xm, xh, yy, hy, cc, te, gt, tt = [], [], [], [], [], [], [], []

    def add(i, win, tgt_year, alpha_val, g_true, t_true, host_label):
        xm.append(np.stack([panel.gold[i, win], panel.total[i, win]], axis=1))
        xh.append(np.stack([panel.host[i, win], panel.prep[i, win],
                            np.full(N_STEPS, alpha_val)], axis=1))
        yy.append([g_true, t_true])
        hy.append(host_label)
        cc.append(i)
        te.append(tgt_year)
        gt.append(g_true)
        tt.append(t_true)

    # 有标签样本（训练 + 验证）
    last_j = E - 1 if align == "same" else E - 2
    for j in range(N_STEPS - 1, last_j + 1):
        win = slice(j - N_STEPS + 1, j + 1)
        tgt = j if align == "same" else j + 1
        for i in range(C):
            add(i, win, panel.editions[tgt], panel.alpha[i, tgt],
                panel.gold[i, tgt], panel.total[i, tgt], panel.host[i, tgt])

    # 2028 纯预测样本：窗口 = 最后三届（2016, 2020, 2024）
    win = slice(E - N_STEPS, E)
    for i in range(C):
        host_2028 = 1.0 if PRED_YEAR in panel.host_years[panel.countries[i]] else 0.0
        add(i, win, PRED_YEAR, panel.alpha_pred[i], np.nan, np.nan, host_2028)

    return Samples(
        x_medal=np.asarray(xm, dtype=np.float32),
        x_host=np.asarray(xh, dtype=np.float32),
        y=np.asarray(yy, dtype=np.float32),
        host_y=np.asarray(hy, dtype=np.float32),
        country=np.asarray(cc, dtype=np.int64),
        target_edition=np.asarray(te, dtype=np.int64),
        gold_true=np.asarray(gt, dtype=np.float64),
        total_true=np.asarray(tt, dtype=np.float64),
    )


# --------------------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------------------
class DualChannelLSTM(nn.Module):
    """式(4)：双通道 LSTM + 拼接 + 全连接；另加 host 分类头用于式(5c)。"""

    def __init__(self, hidden: int = 32, dropout: float = 0.2, host_bias_init: float | None = None):
        super().__init__()
        self.medal_lstm = nn.LSTM(input_size=2, hidden_size=hidden, batch_first=True)
        self.host_lstm = nn.LSTM(input_size=3, hidden_size=hidden, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2)
        )
        self.host_head = nn.Linear(hidden, 1)
        if host_bias_init is not None:
            with torch.no_grad():
                self.host_head.bias.fill_(host_bias_init)

    def forward(self, x_medal, x_host):
        h_medal, _ = self.medal_lstm(x_medal)
        h_host, _ = self.host_lstm(x_host)
        m = h_medal[:, -1, :]
        h = h_host[:, -1, :]
        y_hat = self.fc(torch.cat([m, h], dim=1))          # 式(4c)
        p_hat = torch.sigmoid(self.host_head(h)).squeeze(-1)  # 式(5c) 的概率输出
        return y_hat, p_hat


# --------------------------------------------------------------------------------------
# 指标（式 6）
# --------------------------------------------------------------------------------------
def wmae(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """式(6)：WMAE = Σ w|y-ŷ| / Σ w ；w 全为 0 时返回 nan。"""
    s = float(np.sum(w))
    if s <= 0:
        return float("nan")
    return float(np.sum(w * np.abs(y_true - y_pred)) / s)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, gold_true: np.ndarray) -> dict:
    err = np.abs(y_true - y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    out = {
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "MAE": float(np.mean(err)),
        "MAPE": float(np.mean(err / np.maximum(np.abs(y_true), 1.0))) * 100.0,
        "WMAE_true": wmae(y_true, y_pred, gold_true),          # w = 真实当届金牌数（用户选择）
        "WMAE_plus1": wmae(y_true, y_pred, gold_true + 1.0),   # 拉普拉斯平滑对照
    }
    return out


# --------------------------------------------------------------------------------------
# 训练 / 评估
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one(panel: Panel, samples: Samples, *, seed: int, hidden: int, patience: int,
              epochs: int, lr: float, batch: int, dropout: float, weight_decay: float,
              variant: str, select: str = "best", device: str = "cpu") -> dict:
    set_seed(seed)
    target_ed = np.asarray(samples.target_edition)  # 已是“年份”

    tr = target_ed <= max(y for y in panel.editions if y not in VAL_TARGETS)
    va = np.isin(target_ed, VAL_TARGETS)
    pr = target_ed == PRED_YEAR

    # ---- 标准化统计量（只用训练样本，避免泄漏）----
    med_tr = samples.x_medal[tr]                             # (n,3,2)
    y_mean = med_tr.reshape(-1, 2).mean(axis=0)
    y_std = med_tr.reshape(-1, 2).std(axis=0)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)

    def scale_medal(a):
        return (a - y_mean) / y_std

    def scale_y(a):
        return (a - y_mean) / y_std

    def unscale_y(a):
        return a * y_std + y_mean

    xm = scale_medal(samples.x_medal).astype(np.float32)
    yy = scale_y(samples.y).astype(np.float32)

    # host 通道第 3 维 alpha_c 标准化（其量纲远超 0/1 变量；原文未说明，此处做数值稳定化处理）
    x_host = samples.x_host.astype(np.float64).copy()
    a_tr = x_host[tr, :, 2]
    a_mean, a_std = float(a_tr.mean()), float(a_tr.std())
    a_std = a_std if a_std > 1e-6 else 1.0
    x_host[:, :, 2] = (x_host[:, :, 2] - a_mean) / a_std
    x_host = x_host.astype(np.float32)

    def loader(mask, shuffle):
        ds = TensorDataset(
            torch.from_numpy(xm[mask]), torch.from_numpy(x_host[mask]),
            torch.from_numpy(yy[mask]), torch.from_numpy(samples.host_y[mask]),
        )
        return DataLoader(ds, batch_size=batch, shuffle=shuffle)

    tl, vl = loader(tr, True), loader(va, False)

    host_bias_init = math.log(0.1 / 0.9) if variant == "A" else None
    model = DualChannelLSTM(hidden=hidden, dropout=dropout, host_bias_init=host_bias_init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCELoss()
    eps = 0.1 if variant == "B" else 0.0     # 标签平滑

    history = []
    best = {"score": float("inf"), "epoch": -1, "state": None}
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb_m, xb_h, yb, hb in tl:
            opt.zero_grad()
            y_hat, p_hat = model(xb_m, xb_h)
            l_medal = torch.mean((y_hat - yb) ** 2)                   # 式(5b)（标准化尺度）
            target = hb * (1 - eps) + (1 - hb) * eps if eps > 0 else hb
            l_host = bce(p_hat.clamp(1e-6, 1 - 1e-6), target)          # 式(5c)
            loss = LAMBDA_MEDAL * l_medal + LAMBDA_HOST * l_host       # 式(5a)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(hb)

        # ---- 验证 ----
        model.eval()
        with torch.no_grad():
            vp, vh, vy, vhb = [], [], [], []
            for xb_m, xb_h, yb, hb in vl:
                y_hat, p_hat = model(xb_m, xb_h)
                vp.append(y_hat.numpy()); vh.append(p_hat.numpy())
                vy.append(yb.numpy()); vhb.append(hb.numpy())
            vp = unscale_y(np.concatenate(vp)); vy = unscale_y(np.concatenate(vy))
            vh = np.concatenate(vh); vhb = np.concatenate(vhb)
        m_gold = metrics(vy[:, 0], vp[:, 0], samples.gold_true[va])
        m_tot = metrics(vy[:, 1], vp[:, 1], samples.gold_true[va])
        score = np.nanmean([m_gold["WMAE_true"], m_tot["WMAE_true"]])
        if not np.isfinite(score):                      # 权重全 0 的极端情况 → 退回 MSE
            score = float(np.mean((vy - vp) ** 2))
        history.append({
            "epoch": ep, "train_loss": tot / max(1, int(tr.sum())),
            "val_MSE_gold": m_gold["MSE"], "val_MSE_total": m_tot["MSE"],
            "val_WMAE_gold": m_gold["WMAE_true"], "val_WMAE_total": m_tot["WMAE_true"],
            "val_score": score,
        })
        if score < best["score"] - 1e-9:
            best = {"score": score, "epoch": ep,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience:                          # 早停（patience 由网格给定）
                break

    # 权重选择协议：
    #   best —— 恢复验证集最优权重（patience 只影响“何时停”，不影响选出哪个权重）
    #   last —— 保留早停时刻的权重（每个 (hidden, patience) 都是不同模型，对应论文图 3 的形态）
    if select == "best":
        model.load_state_dict(best["state"])
    model.eval()

    def predict(mask):
        with torch.no_grad():
            dl = loader(mask, False)
            yp, pp = [], []
            for xb_m, xb_h, _, _ in dl:
                y_hat, p_hat = model(xb_m, xb_h)
                yp.append(y_hat.numpy()); pp.append(p_hat.numpy())
        return unscale_y(np.concatenate(yp)), np.concatenate(pp)

    res = {"variant": variant, "seed": seed, "hidden": hidden, "patience": patience,
           "epochs_run": len(history), "best_epoch": best["epoch"],
           "best_val_score": best["score"], "n_train": int(tr.sum()), "n_val": int(va.sum())}

    for tag, mask in (("train", tr), ("val", va)):
        yp, pp = predict(mask)
        yt = samples.y[mask].astype(np.float64)
        gt = samples.gold_true[mask]
        for name, col in (("gold", 0), ("total", 1)):
            for k, v in metrics(yt[:, col], yp[:, col], gt).items():
                res[f"{tag}_{k}_{name}"] = v
        # host 分类指标
        hb = samples.host_y[mask]
        pred_h = (pp >= 0.5).astype(float)
        tp = float(np.sum((pred_h == 1) & (hb == 1))); fp = float(np.sum((pred_h == 1) & (hb == 0)))
        fn = float(np.sum((pred_h == 0) & (hb == 1)))
        res[f"{tag}_host_acc"] = float(np.mean(pred_h == hb))
        res[f"{tag}_host_f1"] = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float("nan")
        res[f"{tag}_host_auc"] = _auc(hb, pp)

    # 2028 预测
    if pr.sum() > 0:
        yp, pp = predict(pr)
        res["_pred2028"] = {
            "country_idx": samples.country[pr].tolist(),
            "gold": yp[:, 0].tolist(), "total": yp[:, 1].tolist(),
            "host_prob": pp.tolist(),
        }
    res["_history"] = history
    # 历史一步预测（供下游 XGBoost 使用；不含 2028 纯预测样本）
    mask_hist = target_ed != PRED_YEAR
    yp_all, pp_all = predict(mask_hist)
    res["_hist_pred"] = {
        "country_idx": samples.country[mask_hist].tolist(),
        "target_edition": target_ed[mask_hist].tolist(),
        "gold_true": samples.gold_true[mask_hist].tolist(),
        "total_true": samples.total_true[mask_hist].tolist(),
        "gold_pred": yp_all[:, 0].tolist(), "total_pred": yp_all[:, 1].tolist(),
        "host_prob": pp_all.tolist(), "host_true": samples.host_y[mask_hist].tolist(),
        "split": np.where(tr[mask_hist], "train", np.where(va[mask_hist], "val", "other")).tolist(),
    }
    return res


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = float(np.sum(y == 1)); n0 = float(np.sum(y == 0))
    return float((np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="3.1.2 节 LSTM 复现")
    ap.add_argument("--variant", default="all", choices=["main", "A", "B", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--grid", action="store_true",
                    help="调参网格模式：hidden ∈ {32,64,128} × patience ∈ {6,7,8,9,10}（复现论文图 3、图 4）")
    ap.add_argument("--hiddens", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--patiences", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    ap.add_argument("--threads", type=int, default=4, help="torch CPU 线程数（小模型下线程过多反而更慢）")
    ap.add_argument("--select", default="best", choices=["best", "last"],
                    help="权重选择协议：best=恢复验证最优；last=保留早停时刻权重（论文图 3 形态）")
    ap.add_argument("--tag", default="", help="输出文件名后缀，便于并列保存不同协议的网格结果")
    ap.add_argument("--align", default="next", choices=["next", "same"], help="样本对齐：next=t→t+1（默认，无泄漏）；same=同届对齐（模拟原文，含泄漏，仅作对照）")
    ap.add_argument("--replot", action="store_true", help="只读取 grid_summary<tag>.csv 重绘网格图，不重新训练")
    args = ap.parse_args(argv)
    if args.replot:
        summ = pd.read_csv(OUT_DIR / f"grid_summary{args.tag}.csv")
        make_grid_plots(summ, args.tag)
        print(f"已重绘: {FIG_DIR}/fig3_param_tuning{args.tag}.png , {FIG_DIR}/fig4_hidden_size{args.tag}.png  (CJK字体={CJK_OK})")
        return 0
    torch.set_num_threads(max(1, args.threads))

    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    panel = build_panel()
    samples = make_samples(panel, align=args.align)
    target_ed = np.asarray(samples.target_edition)  # 已是“年份”
    log = []
    log.append(f"数据目录: {DATA_DIR}")
    log.append(f"届次({len(panel.editions)}): {panel.editions}")
    log.append(f"国家数(>={MIN_EDITIONS}届): {len(panel.countries)}")
    log.append(f"样本总数: {len(target_ed)}  训练(<=2016): {(target_ed <= 2016).sum()}  "
               f"验证(2020,2024): {np.isin(target_ed, VAL_TARGETS).sum()}  预测(2028): {(target_ed == PRED_YEAR).sum()}")
    log.append(f"样本对齐: {args.align}")
    log.append(f"配置: hidden={args.hidden} patience={args.patience} lr={args.lr} batch={args.batch} "
               f"dropout={args.dropout} wd={args.weight_decay} epochs<={args.epochs}")
    host_ct = int((samples.host_y == 1).sum())
    log.append(f"host 标签为 1 的样本数: {host_ct} / {len(target_ed)} = {host_ct/len(target_ed):.2%}")
    log.append(f"  其中训练集 host 正样本 = {int(((target_ed <= 2016) & (samples.host_y == 1)).sum())}，"
               f"验证集 host 正样本 = {int((np.isin(target_ed, VAL_TARGETS) & (samples.host_y == 1)).sum())}")
    top_alpha = sorted(zip(panel.countries, panel.alpha_pred.tolist()), key=lambda t: -t[1])[:6]
    log.append("alpha_c(2028) 前几名: " + ", ".join(f"{c}={a:.3f}" for c, a in top_alpha))
    print("\n".join(log))

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    all_res = []
    if args.grid:
        return run_grid(panel, samples, args, log)
    for v in variants:
        for s in args.seeds:
            r = train_one(panel, samples, seed=s, hidden=args.hidden, patience=args.patience,
                          epochs=args.epochs, lr=args.lr, batch=args.batch, dropout=args.dropout,
                          weight_decay=args.weight_decay, variant=v, select=args.select)
            all_res.append(r)
            print(f"[{v}] seed={s} best_ep={r['best_epoch']:>3} run={r['epochs_run']:>3} "
                  f"val_WMAE_gold={r['val_WMAE_true_gold']:.4f} val_WMAE_total={r['val_WMAE_true_total']:.4f} "
                  f"val_MAE_gold={r['val_MAE_gold']:.4f} val_MSE_total={r['val_MSE_total']:.3f} "
                  f"host_f1={r['val_host_f1']:.3f}", flush=True)

    # ---------------- 结果落盘 ----------------
    flat = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_res]
    pd.DataFrame(flat).to_csv(OUT_DIR / "metrics_by_run.csv", index=False, encoding="utf-8-sig")

    keys = [k for k in flat[0] if k not in ("variant", "seed", "hidden", "patience")]
    summ = []
    for v in variants:
        sub = pd.DataFrame([r for r in flat if r["variant"] == v])
        row = {"variant": v}
        for k in keys:
            row[f"{k}_mean"] = float(sub[k].mean())
            row[f"{k}_std"] = float(sub[k].std(ddof=0))
        summ.append(row)
    pd.DataFrame(summ).to_csv(OUT_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    # 历史一步预测（逐 run 落盘，供下游 XGBoost 使用）
    rows = []
    for r in all_res:
        d = r["_hist_pred"]
        rows.append(pd.DataFrame({
            "variant": r["variant"], "seed": r["seed"],
            "Country": [panel.countries[i] for i in d["country_idx"]],
            "TargetEdition": d["target_edition"],
            "Gold_true": d["gold_true"], "Total_true": d["total_true"],
            "Gold_pred": d["gold_pred"], "Total_pred": d["total_pred"],
            "Host_true": d["host_true"], "Host_prob": d["host_prob"], "Split": d["split"],
        }))
    hist_df = pd.concat(rows, ignore_index=True)
    hist_df.to_csv(OUT_DIR / "trend_history.csv", index=False, encoding="utf-8-sig")

    # 2028 趋势值
    rows = []
    for r in all_res:
        d = r["_pred2028"]
        rows.append(pd.DataFrame({
            "variant": r["variant"], "seed": r["seed"],
            "Country": [panel.countries[i] for i in d["country_idx"]],
            "Gold_trend_2028": d["gold"], "Total_trend_2028": d["total"],
            "Host_prob_2028": d["host_prob"],
        }))
    p28 = pd.concat(rows, ignore_index=True)
    p28.to_csv(OUT_DIR / "trend_2028_by_seed.csv", index=False, encoding="utf-8-sig")
    ref_variant = "main" if (p28["variant"] == "main").any() else variants[0]
    agg = (p28[p28["variant"] == ref_variant].groupby("Country")
           .agg(Gold_trend_2028=("Gold_trend_2028", "mean"),
                Gold_trend_std=("Gold_trend_2028", "std"),
                Total_trend_2028=("Total_trend_2028", "mean"),
                Total_trend_std=("Total_trend_2028", "std"),
                Host_prob_2028=("Host_prob_2028", "mean"))
           .reset_index().sort_values("Gold_trend_2028", ascending=False))
    agg.to_csv(OUT_DIR / f"trend_2028_{ref_variant}.csv", index=False, encoding="utf-8-sig")

    (OUT_DIR / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    (OUT_DIR / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_plots:
        make_plots(all_res, panel, agg, hist_df)

    print(f"\n输出目录: {OUT_DIR}")
    print(agg.head(12).to_string(index=False))
    return 0


def run_grid(panel: Panel, samples: Samples, args, log: list[str]) -> int:
    """调参网格：hidden × patience，复现论文图 3（6 面板）与图 4（hidden_size 对比）。"""
    import time
    rows, t0 = [], time.time()
    total = len(args.hiddens) * len(args.patiences) * len(args.seeds)
    k = 0
    for hidden in args.hiddens:
        for patience in args.patiences:
            for s in args.seeds:
                k += 1
                r = train_one(panel, samples, seed=s, hidden=hidden, patience=patience,
                              epochs=args.epochs, lr=args.lr, batch=args.batch,
                              dropout=args.dropout, weight_decay=args.weight_decay,
                              variant="main", select=args.select)
                rows.append({
                    "hidden": hidden, "patience": patience, "seed": s,
                    "epochs_run": r["epochs_run"], "best_epoch": r["best_epoch"],
                    **{kk: vv for kk, vv in r.items()
                       if kk.startswith(("train_", "val_")) and not kk.startswith("val_WMAE_plus1")},
                })
                print(f"[grid {k}/{total}] {hidden}x{patience} seed={s} "
                      f"ep={r['epochs_run']:>3} val_WMAE_G={r['val_WMAE_true_gold']:.3f} "
                      f"val_WMAE_T={r['val_WMAE_true_total']:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"grid_results{args.tag}.csv", index=False, encoding="utf-8-sig")

    metrics = ["val_WMAE_true_gold", "val_WMAE_true_total", "val_MAE_gold", "val_MAE_total",
               "val_MSE_gold", "val_MSE_total", "val_RMSE_gold", "val_RMSE_total",
               "val_MAPE_gold", "val_MAPE_total", "epochs_run", "best_epoch",
               "train_MAE_gold", "train_MAE_total", "val_host_auc"]
    agg = {m: ["mean", "std"] for m in metrics}
    summ = df.groupby(["hidden", "patience"]).agg(agg)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(OUT_DIR / f"grid_summary{args.tag}.csv", index=False, encoding="utf-8-sig")

    log.append(f"网格运行完成: {len(df)} 次训练，用时 {time.time()-t0:.0f}s")
    (OUT_DIR / f"grid_log{args.tag}.txt").write_text("\n".join(log) + "\n", encoding="utf-8")

    make_grid_plots(summ, args.tag)

    show = summ[["hidden", "patience", "val_WMAE_true_gold_mean", "val_WMAE_true_total_mean"]]
    print("\n=== 网格结果（5 种子均值） ===")
    print(show.to_string(index=False))
    for h in args.hiddens:
        sub = summ[summ["hidden"] == h]
        best = sub.loc[(sub["val_WMAE_true_gold_mean"] + sub["val_WMAE_true_total_mean"]).idxmin()]
        print(f"hidden={h:>3} 最优 patience={int(best['patience'])}  "
              f"WMAE_Gold={best['val_WMAE_true_gold_mean']:.3f} WMAE_Total={best['val_WMAE_true_total_mean']:.3f}")
    print(f"\n图表: {FIG_DIR}/fig3_param_tuning{args.tag}.png , {FIG_DIR}/fig4_hidden_size{args.tag}.png")
    return 0


def make_grid_plots(summ: pd.DataFrame, tag: str = "") -> None:
    """图 3：3 个 hidden_size × {WMAE_Gold, WMAE_Total} 共 6 面板；图 4：hidden_size 对比。"""
    colors = {32: "#c8642a", 64: "#4a9d4a", 128: "#3b6fa0"}
    hiddens = sorted(summ["hidden"].unique().tolist())

    # ---------------- 图 3 ----------------
    fig, axes = plt.subplots(2, len(hiddens), figsize=(13.5, 7.2), sharex=False, squeeze=False)
    labels = "abcdef"
    for col, h in enumerate(hiddens):
        sub = summ[summ["hidden"] == h].sort_values("patience")
        x = np.arange(len(sub))
        xt = [f"{h}×{int(p)}" for p in sub["patience"]]
        for row, (metric, ylab) in enumerate([("val_WMAE_true_gold", "WMAE_Gold"),
                                              ("val_WMAE_true_total", "WMAE_Total")]):
            ax = axes[row, col]
            m = sub[f"{metric}_mean"].to_numpy()
            sd = sub[f"{metric}_std"].fillna(0).to_numpy()
            bars = ax.bar(x, m, yerr=sd, capsize=3, color=colors[h], alpha=.9,
                          error_kw=dict(ecolor="#555", lw=.8))
            i_min = int(np.argmin(m))
            ax.plot(x[i_min], m[i_min], "o", ms=9, mfc="none", mec="crimson", mew=1.6)  # 圆圈 = 误差最小值
            ax.annotate(f"{m[i_min]:.2f}", (x[i_min], m[i_min]), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=8, color="crimson")
            for xi, mi in zip(x, m):
                if xi != x[i_min]:
                    ax.annotate(f"{mi:.2f}", (xi, mi), textcoords="offset points",
                                xytext=(0, 4), ha="center", fontsize=7)
            ax.set_xticks(x); ax.set_xticklabels(xt, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(ylab, fontsize=9)
            ax.set_title(f"hidden_size = {h}", fontsize=10)
            ax.grid(alpha=.3, axis="y")
            ax.text(-0.16, 1.02, f"({labels[row*len(hiddens)+col]})", transform=ax.transAxes,
                    fontsize=11, fontweight="bold")
    fig.suptitle("Fig. 3（复现）: Variations in Parameter Tuning of WMAE_Gold and WMAE_Total", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / f"fig3_param_tuning{tag}.png", dpi=150)
    plt.close(fig)

    # ---------------- 图 4 ----------------
    # 只画本次实际完成的配置（论文正文点名的 32×6 / 64×8 / 64×10 / 128×10），不做派生的“自动挑最优”。
    picks = [(32, 6), (64, 8), (64, 10), (128, 10)]
    avail = {(int(r.hidden), int(r.patience)) for r in summ.itertuples()}
    picks = [p for p in picks if p in avail]
    if not picks:
        return

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(picks)); w = 0.38
    g, t = [], []
    for h, p in picks:
        row = summ[(summ["hidden"] == h) & (summ["patience"] == p)].iloc[0]
        g.append(row["val_WMAE_true_gold_mean"]); t.append(row["val_WMAE_true_total_mean"])
    b1 = ax.bar(x - w/2, g, w, label="WMAE_Gold", color="#c8642a")
    b2 = ax.bar(x + w/2, t, w, label="WMAE_Total", color="#7ab87a")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x()+b.get_width()/2, b.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}×{p}" for h, p in picks], fontsize=9.5)
    ax.set_xlabel("hidden_size × patience（本次实际完成的配置）")
    ax.set_ylabel("WMAE (validation 2020/2024)")
    ax.set_title("Fig. 4（复现）: Comparison on Hidden_size", fontsize=11)
    ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"fig4_hidden_size{tag}.png", dpi=150)
    plt.close(fig)


def make_plots(all_res, panel, agg28, hist_df) -> None:
    # 图 1：训练曲线（main，各 seed）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for r in all_res:
        if r["variant"] != "main":
            continue
        h = pd.DataFrame(r["_history"])
        axes[0].plot(h["epoch"], h["train_loss"], alpha=.7, label=f"seed{r['seed']}")
        axes[1].plot(h["epoch"], h["val_score"], alpha=.7, label=f"seed{r['seed']}")
    axes[0].set(title="Training loss (main)", xlabel="epoch", ylabel="0.7*MSE + 0.3*BCE (scaled)")
    axes[1].set(title="Validation WMAE score (main)", xlabel="epoch", ylabel="mean(WMAE_gold, WMAE_total)")
    for a in axes:
        a.grid(alpha=.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_training_curves.png", dpi=150); plt.close(fig)

    # 图 2：验证集（2020/2024）预测 vs 真实
    v = hist_df[(hist_df["variant"] == "main") & (hist_df["Split"] == "val")]
    v = v.groupby(["Country", "TargetEdition"], as_index=False).agg(
        Gold_true=("Gold_true", "first"), Gold_pred=("Gold_pred", "mean"),
        Total_true=("Total_true", "first"), Total_pred=("Total_pred", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (tc, pc, name) in zip(axes, [("Gold_true", "Gold_pred", "Gold"),
                                         ("Total_true", "Total_pred", "Total")]):
        ax.scatter(v[tc], v[pc], s=18, alpha=.75)
        lim = [0, max(v[tc].max(), v[pc].max()) * 1.05 + 1]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set(title=f"{name} medals, validation (2020/2024)", xlabel="actual", ylabel="predicted")
        ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_pred_vs_true.png", dpi=150); plt.close(fig)

    # 图 3：2028 趋势值 top 20（对照论文图 6）
    top = agg28.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["Country"], top["Total_trend_2028"], color="#8ab4d8", label="Total trend 2028")
    ax.barh(top["Country"], top["Gold_trend_2028"], color="#c8642a", label="Gold trend 2028")
    ax.legend(); ax.set(xlabel="predicted medals (trend value)")
    ax.set_title("LSTM trend values for 2028 (top 20 by total)")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_trend_2028.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
