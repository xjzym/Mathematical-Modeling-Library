#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
【模型 01 · LSTM】设计辅助工具：输入向量审计 / 损失权重对齐 / WMAE 与有效样本量

对应模型卡《01_LSTM_模型清单.md》第七节「如何设计输入向量、损失函数、WMAE」。
所有函数都可独立调用，无需 torch。

用法：
  python metrics_utils.py --demo          # 用交付数据演示全部工具
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =======================================================================================
# 一、WMAE 与加权指标
# =======================================================================================
def wmae(y_true, y_pred, w) -> float:
    """加权平均绝对误差（Weighted MAE）。Σw=0 时返回 nan（不要静默返回 0）。"""
    y_true, y_pred, w = map(lambda a: np.asarray(a, dtype=float), (y_true, y_pred, w))
    s = float(np.sum(w))
    return float(np.sum(w * np.abs(y_true - y_pred)) / s) if s > 0 else float("nan")


def wmse(y_true, y_pred, w) -> float:
    """加权均方误差（对大误差更敏感，用于与 WMAE 对照判断重尾）。"""
    y_true, y_pred, w = map(lambda a: np.asarray(a, dtype=float), (y_true, y_pred, w))
    s = float(np.sum(w))
    return float(np.sum(w * (y_true - y_pred) ** 2) / s) if s > 0 else float("nan")


def neff(w) -> float:
    """加权有效样本量 N_eff=(Σw)²/Σw²。用于判断“指标到底建立在几个样本上”。"""
    w = np.asarray(w, dtype=float)
    s2 = float(np.sum(w ** 2))
    return float(np.sum(w) ** 2 / s2) if s2 > 0 else float("nan")


def weighted_report(y_true, y_pred, w) -> dict:
    """一次给出：无权 MAE、加权 WMAE/WMSE、有效样本量、零权样本占比。"""
    y_true, y_pred, w = map(lambda a: np.asarray(a, dtype=float), (y_true, y_pred, w))
    err = np.abs(y_true - y_pred)
    return {
        "MAE": float(np.mean(err)),
        "RMSE": float(math.sqrt(np.mean((y_true - y_pred) ** 2))),
        "WMAE": wmae(y_true, y_pred, w),
        "WMSE": wmse(y_true, y_pred, w),
        "N_total": int(len(w)),
        "N_eff": neff(w),
        "N_zero_weight": int(np.sum(w == 0)),
        "zero_weight_ratio": float(np.mean(w == 0)),
    }


def weight_variants(y_true) -> dict:
    """给出三种常用权重口径的实现（规模 / 规模+1 平滑 / 无权），供敏感性对照。"""
    y = np.abs(np.asarray(y_true, dtype=float))
    return {"w_scale": y, "w_scale_plus1": y + 1.0, "w_uniform": np.ones_like(y)}


# =======================================================================================
# 二、输入向量审计
# =======================================================================================
def audit_features(df: pd.DataFrame, cols: list[str] | None = None,
                   standardize_ratio: float = 10.0) -> pd.DataFrame:
    """逐列审计输入特征：量纲跨度、缺失、是否 0/1、是否需要标准化、是否恒定（无信息）。

    判定规则：
      · span = max-min；若 span > standardize_ratio 倍（默认 10 倍）于该列最小值量级 → 建议标准化
      · 唯一值 ≤ 2 → 视为 0/1 标志位，不建议标准化
      · 标准差 = 0 → 恒定列，对模型无信息，应删除或核查
    """
    cols = cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        span = float(s.max() - s.min()) if s.notna().any() else 0.0
        nuniq = int(s.nunique(dropna=True))
        is_bin = nuniq <= 2
        const = bool(s.std() == 0 or math.isnan(float(s.std())))
        heavy = False
        if not is_bin and s.notna().any():
            med = float(s.abs().median())
            heavy = med > 0 and span > standardize_ratio * med      # 重尾：跨度远大于中位量级
        suggest = ("删除/核查（恒定列）" if const else
                   "保持原样（0/1 标志位）" if is_bin else
                   "建议标准化（重尾）" if heavy else "建议标准化")
        rows.append({
            "feature": c,
            "min": float(s.min()), "max": float(s.max()), "std": float(s.std()),
            "span": span, "n_unique": nuniq,
            "missing_ratio": float(s.isna().mean()),
            "is_binary": is_bin, "constant": const, "heavy_tail": heavy,
            "need_standardize": (not is_bin) and (not const),
            "suggest": suggest,
        })
    return pd.DataFrame(rows)


# =======================================================================================
# 三、损失函数设计辅助
# =======================================================================================
def recommend_lambdas(initial_losses: dict[str, float], target_ratio: dict[str, float] | None = None) -> dict:
    """按“初始量级对齐”给出损失权重建议：λ_i ∝ 1/L_i^(0)（可选再乘业务偏好比例）。

    示例：{'main': 250.0, 'aux': 0.7} → 让两项在训练起点贡献相同量级。
    """
    keys = list(initial_losses)
    inv = {k: 1.0 / max(float(initial_losses[k]), 1e-12) for k in keys}
    mean_inv = sum(inv.values()) / len(inv)
    lam = {k: inv[k] / mean_inv for k in keys}
    if target_ratio:                      # 业务偏好：把比例缩放到指定偏好上
        base = {k: max(float(target_ratio.get(k, 1.0)), 1e-12) for k in keys}
        s = sum(base.values())
        lam = {k: lam[k] * base[k] * len(keys) / s for k in keys}
    return lam


def event_bias_init(positive_rate: float) -> float:
    """稀有正类分类头的 bias 初始化：b = log(p/(1-p))，避免训练初期被多数类主导。

    例：事件基率 1.2% → b = log(0.012/0.988) ≈ -4.4；论文写法 0.1 先验 → b ≈ -2.197。
    """
    p = min(max(float(positive_rate), 1e-6), 1 - 1e-6)
    return float(math.log(p / (1 - p)))


def is_dead_loss(history, key: str, tol: float = 1e-6) -> bool:
    """死项检测：某损失项在训练中恒定不变 → 说明它与参数无关（例如被写成了常数）。

    history: list[dict]，每个元素形如 {'epoch':1,'loss_main':..,'loss_aux':..}
    """
    vals = [h[key] for h in history if key in h]
    if len(vals) < 3:
        return False
    return bool(np.ptp(np.asarray(vals, dtype=float)) <= tol)


def check_loss_scale(loss_terms: dict[str, float], warn_ratio: float = 100.0) -> list[str]:
    """量级失衡检查：任两项之比超过 warn_ratio → 提示先做尺度对齐，否则小项会被淹没。"""
    msgs = []
    items = {k: abs(float(v)) for k, v in loss_terms.items() if float(v) != 0}
    if len(items) >= 2:
        lo, hi = min(items.values()), max(items.values())
        if hi / max(lo, 1e-12) > warn_ratio:
            msgs.append(f"⚠ 损失项量级失衡 {hi/max(lo,1e-12):.1f}×（{items}）："
                        f"建议对目标做标准化，或用 recommend_lambdas 对齐 λ")
    return msgs


# =======================================================================================
# 四、一致性自检（输入 / 损失 / 指标 三者是否口径一致）
# =======================================================================================
def consistency_checklist(target_scale: str, loss_scale: str, weight_source: str) -> pd.DataFrame:
    """把最容易互相打架的三处口径列成检查表。"""
    rows = [
        {"环节": "输入向量", "检查项": "窗口只含 ≤t 的信息（无目标期泄漏）",
         "本项目": "t→t+1 错位；同届对齐实测使 WMAE 虚高到 0.4–2.7"},
        {"环节": "输入向量", "检查项": "同类变量同量纲、0/1 与比值不混放",
         "本项目": "只标准化 α_c，Host/Prep 保持 0/1"},
        {"环节": "损失函数", "检查项": "在标准化尺度算损失、在原始尺度报指标",
         "本项目": f"损失尺度={loss_scale}；指标尺度={target_scale}"},
        {"环节": "损失函数", "检查项": "无死项（每项都随训练变化）",
         "本项目": "原文 p̂=0.1 常数 → 该项梯度恒为 0，已改为 sigmoid 输出"},
        {"环节": "损失函数", "检查项": "各项量级不悬殊（<100×）",
         "本项目": "Gold/Total 先标准化，等价于量级对齐"},
        {"环节": "WMAE", "检查项": "权重口径与目标口径一致",
         "本项目": f"权重来源={weight_source}（当届真实值）"},
        {"环节": "WMAE", "检查项": "明确 w=0 处理并报告 N_eff",
         "本项目": "严格排除；N_eff=37.8/164（28% 样本零权）"},
        {"环节": "WMAE", "检查项": "同时报告无权 MAE/RMSE 与敏感性对照",
         "本项目": "已给出 w+1 平滑版；三变体排序不变"},
        {"环节": "共同", "检查项": "结论有统计支撑（多种子 + 配对检验）",
         "本项目": "5 种子；配对 t 检验 |t|<2.5 全部不显著"},
    ]
    return pd.DataFrame(rows)


# =======================================================================================
# 五、演示
# =======================================================================================
def _demo() -> None:
    here = Path(__file__).resolve().parent
    hist = pd.read_csv(here / ".." / "数据" / "lstm_trend_history.csv")
    val = hist[hist["Split"] == "val"]
    print("=" * 78)
    print("① WMAE 与加权有效样本量（验证集 2020/2024，main 变体）")
    rep = weighted_report(val["Gold_true"], val["Gold_trend"], val["Gold_true"])
    for k, v in rep.items():
        print(f"   {k:>18} = {v}" if not isinstance(v, float) else f"   {k:>18} = {v:.4f}")
    print("\n   三种权重口径对照：")
    for name, w in weight_variants(val["Gold_true"]).items():
        print(f"   {name:<14} WMAE={wmae(val['Gold_true'], val['Gold_trend'], w):.4f}  "
              f"N_eff={neff(w):.1f}")

    print("\n" + "=" * 78)
    print("② 输入向量审计（用交付面板的 4 类特征演示）")
    base = pd.read_csv(here / ".." / "数据" / "base_panel_country_year.csv")
    t28 = pd.read_csv(here / ".." / "数据" / "lstm_trend_2028.csv")
    audit = audit_features(pd.DataFrame({
        "Gold(自回归)": base["Gold"], "Total(自回归)": base["Total"],
        "IsHost(0/1)": base["IsHost"], "Prep(0/1)": base["Prep"],
        "Alpha_c(比值型)": base["Alpha_c"],
        "Gold_trend_2028": t28.set_index("Country").reindex(base["Country"]).reset_index(drop=True)["Gold_trend_2028"],
    }))
    print(audit.to_string(index=False))

    print("\n" + "=" * 78)
    print("③ 损失权重对齐与死项检测")
    init = {"main": 250.0, "aux": 0.7}
    print(f"   初始损失量级 {init} → 建议 λ = "
          f"{ {k: round(v, 4) for k, v in recommend_lambdas(init).items()} }")
    print(f"   业务偏好 main:aux = 7:3 → λ = "
          f"{ {k: round(v, 4) for k, v in recommend_lambdas(init, {'main': 7, 'aux': 3}).items()} }")
    print(f"   事件基率 1.2% 的 bias 初始化 = {event_bias_init(0.012):.3f}；"
          f"论文的 0.1 先验 → {event_bias_init(0.1):.3f}")
    print("   " + "\n   ".join(check_loss_scale(init)) if check_loss_scale(init) else "   量级检查通过")
    fake = [{"epoch": i, "loss_main": 1.0 / (i + 1), "loss_aux": 0.6931} for i in range(5)]
    print(f"   死项检测：loss_main 恒定? {is_dead_loss(fake, 'loss_main')} "
          f"／ loss_aux 恒定? {is_dead_loss(fake, 'loss_aux')}  ← 后者即“常数写进损失”的典型症状")

    print("\n" + "=" * 78)
    print("④ 三者一致性检查表")
    print(consistency_checklist("原始奖牌数", "标准化尺度", "真实当届金牌数").to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LSTM 设计辅助工具")
    ap.add_argument("--demo", action="store_true", help="用交付数据演示全部工具")
    a = ap.parse_args()
    if a.demo:
        _demo()
    else:
        ap.print_help()
