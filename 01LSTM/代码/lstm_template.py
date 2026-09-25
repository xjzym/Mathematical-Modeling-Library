#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
==========================================================================================
【模型编号 01】LSTM 时序趋势建模 —— 通用可复用模板（配置驱动，可换数据集）
==========================================================================================

本模板把 2025 MCM Problem C（Team #2500759，论文 3.1.2 节）中验证过的 LSTM 趋势建模流程
抽象成“配置驱动”的形式：只要换一份 config JSON，就能在**不同背景、不同数据集**上工作。

核心思路（与论文一致，但不绑定具体数据）：
  1. 把长表（entity, year, 若干目标列）整理成「实体 × 期次」面板，缺失期次补 0；
  2. 可选地接入“主办/事件”表，构造 IsHost、Prep（事件前 W 期窗口）与 alpha（事件增益系数）；
  3. 用滑动窗口（连续 n_steps 期）预测下一期；
  4. 双通道 LSTM：目标通道（成绩序列）+ 事件通道（IsHost/Prep/alpha），末步隐状态拼接后回归；
  5. 分层损失 L = λ1·MSE + λ2·BCE（事件识别为辅助任务，防稀有正类塌缩）；
  6. 多种子训练 + 早停，输出“趋势值”（历史逐步 + 未来期外推）与评估指标。

用法：
  python lstm_template.py --config config_mcm2025C.json
  python lstm_template.py --config my_config.json --dry-run      # 只做数据检查，不训练
  python lstm_template.py --config my_config.json --override hidden_size=64 patience=8
==========================================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:                                     # 允许 --dry-run 在没有 torch 的机器上做数据检查
    torch = None
    nn = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ---------------------------------------------------------------------------------------
# 0. 工具：配置加载、随机种子、中文字体
# ---------------------------------------------------------------------------------------
DEFAULTS: dict = {
    "name": "unnamed-lstm",
    "data": {
        "panel_csv": "",                 # 长表：至少含 entity / year / 目标列
        "entity_col": "entity",
        "year_col": "year",
        "target_cols": [],               # 例如 ["Gold", "Total"]
        "entity_merge_map": {},          # 历史实体合并：{"旧名": "新名"}
        "drop_entities": [],             # 非实体（如“独立参赛者”）直接剔除
        "start_year": None,              # 只用 >= 该年份的期次
        "min_records": 1,                # 实体纳入门槛：至少有 N 期有记录
        "host_csv": None,                # 可选：事件/主办表
        "host_year_col": "year",
        "host_entity_col": "entity",
        "host_name_map": {},             # 事件表里的名字 → 面板里的名字
        "prep_window": 8,                # 事件前多少期计入“备战窗口”
    },
    "features": {
        "use_event_channel": True,       # 无 host 表时自动降级为单通道
        "use_alpha": True,               # 是否使用事件增益系数
        "alpha_basis_col": None,         # 默认取 target_cols[0]
        "n_steps": 3,                    # 输入窗口长度
    },
    "split": {
        "val_target_years": [],          # 验证目标的期次（按时间留出）
        "predict_years": [],             # 需要外推的期次（可不在数据里）
    },
    "model": {"hidden_size": 32, "dropout": 0.2, "lambda_medal": 0.7, "lambda_host": 0.3,
              "host_bias_init": None, "fc_hidden": 64},
    "train": {"epochs": 200, "patience": 6, "lr": 1e-3, "batch": 16, "weight_decay": 1e-5,
              "seeds": [0, 1, 2, 3, 4], "select": "best", "threads": 4},
    "metrics": {"wmae_weight_col": None},   # WMAE 的权重列（默认第一个目标列）
    "output_dir": "out",
}


def deep_update(base: dict, patch: dict) -> dict:
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = deep_update(json.loads(json.dumps(DEFAULTS)), json.loads(Path(path).read_text(encoding="utf-8")))
    for ov in overrides or []:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        try:
            node[keys[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[keys[-1]] = val
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def setup_cjk() -> bool:
    for name in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Source Han Sans SC"]:
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            matplotlib.rcParams["font.sans-serif"] = [name] + list(matplotlib.rcParams["font.sans-serif"])
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------------------
# 1. 数据层：长表 → 实体 × 期次面板（含可选事件特征）
# ---------------------------------------------------------------------------------------
@dataclass
class Panel:
    entities: list[str]
    periods: list[int]
    values: dict[str, np.ndarray]      # 每个目标列 → (E, P)
    is_event: np.ndarray | None        # (E, P) 0/1
    prep: np.ndarray | None            # (E, P) 0/1
    alpha: np.ndarray | None           # (E, P) 因果 alpha
    alpha_pred: dict[int, np.ndarray]  # 外推期次 → (E,) alpha
    event_years: dict[str, list[int]]  # 实体 → 事件期次列表


def build_panel(cfg: dict, log: list[str]) -> Panel:
    d = cfg["data"]
    df = pd.read_csv(d["panel_csv"])
    df[d["entity_col"]] = df[d["entity_col"]].astype(str).str.strip()
    df = df[~df[d["entity_col"]].isin(d.get("drop_entities") or [])].copy()
    if d.get("entity_merge_map"):
        df[d["entity_col"]] = df[d["entity_col"]].replace(d["entity_merge_map"])
    df[d["year_col"]] = df[d["year_col"]].astype(int)
    for c in d["target_cols"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # 同一 (实体, 期次) 多行 → 求和（合并历史实体时必需）
    agg = df.groupby([d["entity_col"], d["year_col"]], as_index=False)[d["target_cols"]].sum()
    if d.get("start_year") is not None:
        agg = agg[agg[d["year_col"]] >= int(d["start_year"])]

    cnt = agg.groupby(d["entity_col"])[d["year_col"]].nunique()
    entities = sorted(cnt[cnt >= int(d.get("min_records", 1))].index.tolist())
    periods = sorted(agg[d["year_col"]].unique().tolist())
    eidx = {e: i for i, e in enumerate(entities)}
    pidx = {p: i for i, p in enumerate(periods)}

    values = {c: np.zeros((len(entities), len(periods))) for c in d["target_cols"]}
    for _, r in agg.iterrows():
        e, p = r[d["entity_col"]], int(r[d["year_col"]])
        if e in eidx:
            for c in d["target_cols"]:
                values[c][eidx[e], pidx[p]] = float(r[c])
    log.append(f"面板：{len(entities)} 个实体 × {len(periods)} 个期次（{periods[0]}–{periods[-1]}），"
               f"目标列={d['target_cols']}，补 0 后的总单元格={len(entities)*len(periods)}")

    # ---- 事件特征（可选）----
    is_event = prep = alpha = None
    alpha_pred: dict[int, np.ndarray] = {}
    event_years: dict[str, list[int]] = {e: [] for e in entities}
    if d.get("host_csv") and cfg["features"].get("use_event_channel", True):
        h = pd.read_csv(d["host_csv"])
        h[d["host_entity_col"]] = h[d["host_entity_col"]].astype(str).str.strip()
        # 去掉 “Tokyo, Japan (postponed to 2021 …)” 这类括注（真实踩过的坑）
        h["_ent"] = (h[d["host_entity_col"]].str.split(",").str[-1]
                     .str.split("(").str[0].str.strip())
        h["_ent"] = h["_ent"].replace(d.get("host_name_map") or {})
        h["_ent"] = h["_ent"].replace(d.get("entity_merge_map") or {})
        h[d["host_year_col"]] = h[d["host_year_col"]].astype(int)
        h = h[~h[d["host_entity_col"]].str.contains("Cancelled", case=False, na=False)]
        for _, r in h.iterrows():
            e = r["_ent"]
            if e in event_years:
                event_years[e].append(int(r[d["host_year_col"]]))
        for e in event_years:
            event_years[e] = sorted(event_years[e])

        W = int(d.get("prep_window", 8))
        is_event = np.zeros((len(entities), len(periods)))
        prep = np.zeros((len(entities), len(periods)))
        for e, i in eidx.items():
            for p, j in pidx.items():
                if p in event_years[e]:
                    is_event[i, j] = 1.0
                if any(py - W <= p <= py for py in event_years[e]):
                    prep[i, j] = 1.0

        if cfg["features"].get("use_alpha", True):
            basis = cfg["features"].get("alpha_basis_col") or d["target_cols"][0]
            full = df.groupby([d["entity_col"], d["year_col"]])[basis].sum().reset_index()
            alpha = np.zeros((len(entities), len(periods)))
            for e, i in eidx.items():
                sub = full[full[d["entity_col"]] == e]
                for p, j in pidx.items():
                    alpha[i, j] = _causal_alpha(sub, event_years[e], p, d["year_col"], basis)
            for y in cfg["split"].get("predict_years", []):
                arr = np.zeros(len(entities))
                for e, i in eidx.items():
                    sub = full[full[d["entity_col"]] == e]
                    arr[i] = _causal_alpha(sub, event_years[e], int(y), d["year_col"], basis)
                alpha_pred[int(y)] = arr
        log.append(f"事件特征：{sum(1 for e in entities if event_years[e])} 个实体有过事件；"
                   f"事件正样本 = {int(is_event.sum())} / {is_event.size}")
    else:
        log.append("事件特征：未提供 host_csv 或已关闭 → 降级为单通道模型（仅目标通道）")

    return Panel(entities, periods, values, is_event, prep, alpha, alpha_pred, event_years)


def _causal_alpha(sub: pd.DataFrame, event_years: list[int], target_year: int,
                  year_col: str, basis: str) -> float:
    """事件增益系数（因果版）：只用 target_year 之前的历史；从未发生过事件则返回 0。"""
    past = sub[sub[year_col] < target_year][basis]
    ev = [y for y in event_years if y < target_year]
    if len(past) == 0 or len(ev) == 0:
        return 0.0
    avg = float(past.mean())
    ev_avg = float(np.mean([float(sub[sub[year_col] == y][basis].sum()) for y in ev]))
    return (ev_avg - avg) / avg if avg > 0 else 0.0


# ---------------------------------------------------------------------------------------
# 2. 样本层：滑动窗口（连续 n_steps 期 → 下一期）
# ---------------------------------------------------------------------------------------
@dataclass
class Samples:
    x_main: np.ndarray        # (N, n_steps, n_targets)
    x_event: np.ndarray|None  # (N, n_steps, n_event_feats)
    y: np.ndarray             # (N, n_targets)
    event_y: np.ndarray|None  # (N,)
    entity_idx: np.ndarray
    target_year: np.ndarray
    is_future: np.ndarray     # True 表示该样本是“外推期”的无标签样本


def make_samples(cfg: dict, panel: Panel, log: list[str]) -> Samples:
    d, f = cfg["data"], cfg["features"]
    tc = d["target_cols"]
    n_steps = int(f["n_steps"])
    E, P = panel.values[tc[0]].shape
    xm, xe, yy, ey, ei, ty, fut = [], [], [], [], [], [], []
    use_event = panel.is_event is not None and f.get("use_event_channel", True)

    def add(i, win, target_year, y_vals, ev_label, is_future, alpha_val=0.0):
        xm.append(np.stack([panel.values[c][i, win] for c in tc], axis=1))
        if use_event:
            feats = [panel.is_event[i, win], panel.prep[i, win]]
            if panel.alpha is not None:
                feats.append(np.full(len(panel.periods[win]), float(alpha_val)))
            xe.append(np.stack(feats, axis=1))
        yy.append(y_vals)
        ey.append(ev_label)
        ei.append(i)
        ty.append(target_year)
        fut.append(is_future)

    # 有标签样本：窗口 [j-n_steps+1, j] → 目标 j+1（alpha 取目标期次的值，因果计算）
    for j in range(n_steps - 1, P - 1):
        win = slice(j - n_steps + 1, j + 1)
        for i in range(E):
            add(i, win, panel.periods[j + 1],
                [panel.values[c][i, j + 1] for c in tc],
                (panel.is_event[i, j + 1] if use_event else 0.0), False,
                (panel.alpha[i, j + 1] if panel.alpha is not None else 0.0))

    # 外推样本：窗口 = 最后 n_steps 期 → 预测未来期次（数据中不存在）
    for y in cfg["split"].get("predict_years", []):
        y = int(y)
        if y in panel.periods:            # 若该期已在数据中，则按普通样本处理，避免重复
            continue
        win = slice(P - n_steps, P)
        for i in range(E):
            a = panel.alpha_pred.get(y)
            av = float(a[i]) if a is not None else 0.0
            add(i, win, y, [np.nan] * len(tc),
                1.0 if y in panel.event_years.get(panel.entities[i], []) else 0.0, True, av)

    S = Samples(
        x_main=np.asarray(xm, dtype=np.float32),
        x_event=(np.asarray(xe, dtype=np.float32) if use_event else None),
        y=np.asarray(yy, dtype=np.float32),
        event_y=(np.asarray(ey, dtype=np.float32) if use_event else None),
        entity_idx=np.asarray(ei, dtype=np.int64),
        target_year=np.asarray(ty, dtype=np.int64),
        is_future=np.asarray(fut, dtype=bool),
    )
    log.append(f"样本：总计 {len(S.target_year)}；有标签 {int((~S.is_future).sum())}；"
               f"外推 {int(S.is_future.sum())}（期次 {sorted(set(S.target_year[S.is_future].tolist()))}）")
    return S


# ---------------------------------------------------------------------------------------
# 3. 模型层：双通道 LSTM（事件通道可缺省）
# ---------------------------------------------------------------------------------------
def build_model(cfg: dict, n_targets: int, n_event_feats: int | None):
    if torch is None:
        raise RuntimeError("需要 PyTorch：pip install torch --index-url https://download.pytorch.org/whl/cpu")
    m = cfg["model"]
    hidden = int(m["hidden_size"])
    dropout = float(m["dropout"])
    fc_hidden = int(m.get("fc_hidden", 64))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.LSTM(input_size=n_targets, hidden_size=hidden, batch_first=True)
            self.event = (nn.LSTM(input_size=n_event_feats, hidden_size=hidden, batch_first=True)
                          if n_event_feats else None)
            in_dim = hidden * (2 if n_event_feats else 1)
            self.fc = nn.Sequential(nn.Linear(in_dim, fc_hidden), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(fc_hidden, n_targets))
            self.event_head = nn.Linear(hidden, 1) if n_event_feats else None
            if self.event_head is not None and m.get("host_bias_init") is not None:
                with torch.no_grad():
                    self.event_head.bias.fill_(float(m["host_bias_init"]))

        def forward(self, x_main, x_event=None):
            h_main, _ = self.main(x_main)
            h = h_main[:, -1, :]
            p = None
            if self.event is not None:
                h_ev, _ = self.event(x_event)
                h = torch.cat([h, h_ev[:, -1, :]], dim=1)
                p = torch.sigmoid(self.event_head(h_ev[:, -1, :])).squeeze(-1)
            return self.fc(h), p

    return Net()


# ---------------------------------------------------------------------------------------
# 4. 指标与训练
# ---------------------------------------------------------------------------------------
def wmae(y_true, y_pred, w) -> float:
    s = float(np.sum(w))
    return float(np.sum(w * np.abs(y_true - y_pred)) / s) if s > 0 else float("nan")


def metrics(y_true, y_pred, w) -> dict:
    err = np.abs(y_true - y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {"MSE": mse, "RMSE": math.sqrt(mse), "MAE": float(np.mean(err)),
            "WMAE": wmae(y_true, y_pred, w)}


def train_once(cfg: dict, panel: Panel, S: Samples, seed: int):
    tr_cfg = cfg["train"]
    d = cfg["data"]
    tc = d["target_cols"]
    wcol = cfg["metrics"].get("wmae_weight_col") or tc[0]
    wj = tc.index(wcol)
    set_seed(seed)
    ty = S.target_year
    lab = ~S.is_future
    val_years = set(int(y) for y in cfg["split"].get("val_target_years", []))
    tr = lab & ~np.isin(ty, list(val_years))
    va = lab & np.isin(ty, list(val_years))
    if va.sum() == 0:                                  # 未指定验证期 → 用最后 20% 有标签样本
        idx = np.where(lab)[0]
        cut = int(len(idx) * 0.8)
        tr = np.zeros_like(lab); va = np.zeros_like(lab)
        tr[idx[:cut]] = True; va[idx[cut:]] = True
        print("[warn] 未指定 val_target_years，自动按时间后 20% 作验证集")

    # 标准化（统计量只用训练样本）
    flat = S.x_main[tr].reshape(-1, len(tc))
    mu, sd = flat.mean(axis=0), np.where(flat.std(axis=0) < 1e-6, 1.0, flat.std(axis=0))
    X = ((S.x_main - mu) / sd).astype(np.float32)
    Y = ((S.y - mu) / sd).astype(np.float32)
    # 事件通道：alpha 列（最后一列）单独标准化
    XE = None
    if S.x_event is not None:
        XE = S.x_event.astype(np.float64).copy()
        for k in range(XE.shape[2]):
            col = XE[:, :, k]
            c_tr = col[tr]
            # 只标准化“量纲远大于 0/1”的特征列（如 alpha 系数）；0/1 标志位保持原样
            if float(col.max() - col.min()) <= 1.5:
                continue
            s = c_tr.std()
            if s > 1e-6:
                XE[:, :, k] = (col - c_tr.mean()) / s
        XE = XE.astype(np.float32)

    def loader(mask, shuffle):
        # 注意：这里必须喂 Y（标准化后的目标）。曾经误传 S.y（原始尺度），
        # 导致模型学出原始量级、又在 predict 里反标准化一次 → 结果放大约 10 倍。
        if XE is None:
            ds = TensorDataset(torch.from_numpy(X[mask]), torch.from_numpy(Y[mask]))
        else:
            ds = TensorDataset(torch.from_numpy(X[mask]), torch.from_numpy(XE[mask]),
                               torch.from_numpy(Y[mask]), torch.from_numpy(S.event_y[mask]))
        return DataLoader(ds, batch_size=int(tr_cfg["batch"]), shuffle=shuffle)

    model = build_model(cfg, len(tc), None if XE is None else XE.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=float(tr_cfg["lr"]),
                           weight_decay=float(tr_cfg["weight_decay"]))
    bce = nn.BCELoss()
    hist, best = [], {"score": float("inf"), "epoch": -1, "state": None}
    bad = 0
    for ep in range(1, int(tr_cfg["epochs"]) + 1):
        model.train()
        for batch in loader(tr, True):
            opt.zero_grad()
            if XE is None:
                xb, yb = batch
                y_hat, _ = model(xb); l_host = torch.tensor(0.0)
            else:
                xb, xeb, yb, hb = batch
                y_hat, p_hat = model(xb, xeb)
                l_host = bce(p_hat.clamp(1e-6, 1 - 1e-6), hb)
            l_main = torch.mean((y_hat - yb) ** 2)
            loss = float(cfg["model"]["lambda_medal"]) * l_main + float(cfg["model"]["lambda_host"]) * l_host
            loss.backward(); opt.step()
        # 验证
        model.eval()
        with torch.no_grad():
            yp, yt = [], []
            for batch in loader(va, False):
                if XE is None:
                    xb, yb = batch; y_hat, _ = model(xb)
                else:
                    xb, xeb, yb, _ = batch; y_hat, _ = model(xb, xeb)
                yp.append(y_hat.numpy()); yt.append(yb.numpy())
            yp = np.concatenate(yp) * sd + mu
            yt = np.concatenate(yt) * sd + mu
        w_true = S.y[va][:, wj].astype(float)
        sc = np.nanmean([metrics(yt[:, k], yp[:, k], w_true)["WMAE"] for k in range(len(tc))])
        if not np.isfinite(sc):
            sc = float(np.mean((yt - yp) ** 2))
        hist.append({"epoch": ep, "val_score": sc})
        if sc < best["score"] - 1e-9:
            best = {"score": sc, "epoch": ep,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= int(tr_cfg["patience"]):
                break
    if tr_cfg.get("select", "best") == "best":
        model.load_state_dict(best["state"])
    model.eval()

    def predict(mask):
        with torch.no_grad():
            out = []
            for batch in loader(mask, False):
                if XE is None:
                    xb = batch[0]; y_hat, p = model(xb); pr = np.zeros(len(xb))
                else:
                    xb, xeb = batch[0], batch[1]; y_hat, p = model(xb, xeb); pr = p.numpy()
                out.append((y_hat.numpy() * sd + mu, pr))
        y = np.concatenate([o[0] for o in out]); p = np.concatenate([o[1] for o in out])
        return y, p

    res = {"seed": seed, "epochs_run": len(hist), "best_epoch": best["epoch"]}
    res["_curve"] = hist
    for tag, mask in (("train", tr), ("val", va)):
        yp, _ = predict(mask)
        yt = S.y[mask].astype(float)
        w_true = S.y[mask][:, wj].astype(float)
        for k, c in enumerate(tc):
            for mk, mv in metrics(yt[:, k], yp[:, k], w_true).items():
                res[f"{tag}_{mk}_{c}"] = mv
    fut = S.is_future
    if fut.sum() > 0:
        yp, _ = predict(fut)
        res["_future"] = {"entity_idx": S.entity_idx[fut].tolist(),
                          "year": S.target_year[fut].tolist(),
                          "pred": yp.tolist()}
    yp_all, _ = predict(lab)
    res["_history"] = {"entity_idx": S.entity_idx[lab].tolist(),
                       "year": S.target_year[lab].tolist(),
                       "true": S.y[lab].tolist(), "pred": yp_all.tolist(),
                       "split": np.where(tr[lab], "train", "val").tolist()}
    return res


# ---------------------------------------------------------------------------------------
# 5. 主流程
# ---------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="【模型 01】LSTM 时序趋势建模 通用模板")
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="覆盖配置，如 hidden_size=64 train.patience=8")
    ap.add_argument("--dry-run", action="store_true", help="只做数据检查与样本构造，不训练")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.override)
    setup_cjk()
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    log = [f"配置：{args.config}", f"模型名：{cfg['name']}"]

    panel = build_panel(cfg, log)
    S = make_samples(cfg, panel, log)
    print("\n".join(log))
    if args.dry_run:
        print("\n[dry-run] 数据与样本检查通过，未训练。")
        return 0

    if torch is not None:
        torch.set_num_threads(int(cfg["train"].get("threads", 4)))

    runs = [train_once(cfg, panel, S, s) for s in cfg["train"]["seeds"]]
    tc = cfg["data"]["target_cols"]
    flat = [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]
    pd.DataFrame(flat).to_csv(out / "metrics_by_run.csv", index=False, encoding="utf-8-sig")
    summ = pd.DataFrame(flat).drop(columns=["seed"], errors="ignore").agg(["mean", "std"]).T
    summ.index.name = "metric"
    summ.to_csv(out / "metrics_summary.csv", encoding="utf-8-sig")
    print("\n=== 指标（多种子 mean/std）===")
    print(summ.head(20).to_string())

    # 未来期外推（跨种子平均）
    fut_rows = []
    for r, seed in zip(runs, cfg["train"]["seeds"]):
        f = r.get("_future")
        if not f:
            continue
        for i, y, p in zip(f["entity_idx"], f["year"], f["pred"]):
            fut_rows.append({"Entity": panel.entities[i], "Year": y, **{c: p[k] for k, c in enumerate(tc)},
                             "seed": seed})
    if fut_rows:
        fd = pd.DataFrame(fut_rows)
        agg = fd.groupby(["Entity", "Year"], as_index=False).agg(
            {**{c: ["mean", "std"] for c in tc}, "seed": "count"})
        agg.columns = ["Entity", "Year"] + [f"{c}_{s}" for c in tc for s in ("mean", "std")] + ["n_seeds"]
        agg.to_csv(out / "trend_future.csv", index=False, encoding="utf-8-sig")
        print(f"\n=== 外推趋势值（前 5 行，共 {len(agg)} 行）===")
        print(agg.head(5).to_string(index=False))

    # 历史趋势值
    hrows = []
    for r, seed in zip(runs, cfg["train"]["seeds"]):
        h = r["_history"]
        for i, y, t, p, sp in zip(h["entity_idx"], h["year"], h["true"], h["pred"], h["split"]):
            hrows.append({"Entity": panel.entities[i], "Year": y,
                          **{f"{c}_true": t[k] for k, c in enumerate(tc)},
                          **{f"{c}_trend": p[k] for k, c in enumerate(tc)},
                          "Split": sp, "seed": seed})
    hd = pd.DataFrame(hrows)
    agg_map = {}
    for c in tc:
        agg_map[f"{c}_true"] = (f"{c}_true", "first")
        agg_map[f"{c}_trend"] = (f"{c}_trend", "mean")
        agg_map[f"{c}_trend_std"] = (f"{c}_trend", "std")
    agg_map["Split"] = ("Split", "first")
    hagg = hd.groupby(["Entity", "Year"], as_index=False).agg(**agg_map)
    hagg.to_csv(out / "trend_history.csv", index=False, encoding="utf-8-sig")

    # 训练曲线（每个种子的验证 WMAE 轨迹）
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for r in runs:
        c = r.get("_curve", [])
        if c:
            ax.plot([h["epoch"] for h in c], [h["val_score"] for h in c],
                    alpha=.75, label=f"seed{r['seed']}")
    ax.set_title(f"{cfg['name']}: validation score (mean WMAE)")
    ax.set_xlabel("epoch"); ax.set_ylabel("val score")
    ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_training.png", dpi=150); plt.close(fig)
    print(f"\n输出目录：{out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
