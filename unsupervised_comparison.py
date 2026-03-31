"""
=============================================================================
Unsupervised Method Comparison  —  Add-on for building_hybrid_pipeline.py
=============================================================================

PURPOSE
-------
Addresses Proposal Objective 7 / Risk mitigation:
  "Comparative analysis with alternative unsupervised methods"

Runs three alternative unsupervised anomaly detectors on the same test
transactions that the association rule pipeline scored, then produces a
side-by-side comparison report showing:

  • Per-method anomaly rates
  • Agreement matrix between all methods
  • Per-room breakdown of how methods disagree
  • Correlation of continuous scores where available
  • Combined consensus flagging (majority vote across all methods)

METHODS COMPARED
----------------
1. Association Rules (ARM)      — your pipeline (reads outputs/all_scores.csv)
2. Isolation Forest  (IF)       — tree-based anomaly score, no labels needed
3. K-Means Clustering (KM)      — distance-to-nearest-centroid as anomaly score
4. DBSCAN                       — points marked as noise (-1) are anomalies

All methods operate on the SAME numeric feature vector built from the same
1-minute window averages the pipeline uses:
    [mean_temp, mean_act, mean_sp_dev, mean_tdelta, mean_adelta, oo_mean_temp]


REQUIREMENTS
------------
    pip install pandas numpy scikit-learn matplotlib tqdm
=============================================================================
"""

import os
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# =============================================================================
#  CONFIGURATION
# =============================================================================

TEST_CSV_PATH  = "simulation_data_multi_prev_test.csv"
SCORES_CSV     = os.path.join("outputs", "all_scores.csv")   # ARM output
OUTPUT_DIR     = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_PATH    = os.path.join(OUTPUT_DIR, "comparison_report.csv")
SUMMARY_PATH   = os.path.join(OUTPUT_DIR, "comparison_summary.txt")
AGREEMENT_PATH = os.path.join(OUTPUT_DIR, "comparison_agreement.png")
ROC_PATH       = os.path.join(OUTPUT_DIR, "comparison_roc.png")


TRANSACTION_SIZE  = 6
OO_COLD_MAX       = 15.0
OO_NORMAL_MAX     = 20.0
OO_WARM_MAX       = 25.0
TEMP_COLD_MAX     = 19.0
TEMP_HOT_MIN      = 22.0
TREND_STABLE_BAND = 0.01
SP_DEV_BAND       = 1.5


ARM_ANOMALY_THRESHOLD = 0.01  


IF_CONTAMINATION  = 0.05   
IF_N_ESTIMATORS   = 200
IF_RANDOM_STATE   = 42

# K-Means 
KM_N_CLUSTERS     = 8      # one per room (A11–A24 = 8 indoor rooms)
KM_RANDOM_STATE   = 42

KM_CONTAMINATION  = 0.05

#DBSCAN 
DBSCAN_EPS        = 0.8    # neighbourhood radius in standardised feature space
DBSCAN_MIN_SAMPLES = 10    # minimum cluster size


# =============================================================================
#  STEP 1 — BUILD NUMERIC FEATURE MATRIX FROM TEST CSV
# =============================================================================

def build_feature_matrix(test_csv_path: str) -> pd.DataFrame:
    """
    Reads the test CSV, groups by room, applies tumbling 1-minute windows
    (TRANSACTION_SIZE records), computes window averages, and returns a
    DataFrame with one row per transaction containing:

        room, timestamp,
        mean_temp, mean_act, mean_sp_dev, mean_tdelta, mean_adelta,
        oo_mean_temp

    This is the continuous numeric equivalent of what the pipeline
    discretizes into symbolic items.
    """
    print(f"\n[Comparison] Loading test data: {test_csv_path}")
    t0 = time.time()

    df = pd.read_csv(test_csv_path, sep=";", low_memory=False)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
    )

    col_map = {"datetime": "timestamp", "timeline": "elapsed_s",
               "room_name": "room"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for i, variants in enumerate(
        [["prev_act_1", "previous_actuation_1"],
         ["prev_act_2", "previous_actuation_2"],
         ["prev_act_3", "previous_actuation_3"]], start=1
    ):
        canonical = f"prev_act_{i}"
        found = next((v for v in variants if v in df.columns), None)
        if found and found != canonical:
            df = df.rename(columns={found: canonical})
        if canonical not in df.columns:
            df[canonical] = 0.0

    for col in ["temperature", "actuation", "setpoint",
                "prev_act_1", "prev_act_2", "prev_act_3"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)

    # Kelvin → Celsius
    if df["temperature"].median() > 200:
        df["temperature"] -= 273.15
    if df["setpoint"].median() > 200:
        df["setpoint"] -= 273.15

    print(f"  Rows loaded     : {len(df):,}  ({time.time()-t0:.1f}s)")

    # ── Build OO timeline (outside temperature per timestamp) ─────────────────
    oo_df = df[df["room"] == "OO"][["timestamp", "temperature"]].copy()
    oo_df = oo_df.sort_values("timestamp").reset_index(drop=True)
    oo_df["oo_temp"] = oo_df["temperature"]

    # ── Build transactions per indoor room ────────────────────────────────────
    rows = []
    indoor_rooms = [r for r in df["room"].unique() if r != "OO"]

    for room in sorted(indoor_rooms):
        room_df = df[df["room"] == room].sort_values("timestamp").reset_index(drop=True)
        prev_temp = None

        for start in range(0, len(room_df) - TRANSACTION_SIZE + 1, TRANSACTION_SIZE):
            win = room_df.iloc[start: start + TRANSACTION_SIZE]

            temps  = win["temperature"].values
            acts   = win["actuation"].values
            sps    = win["setpoint"].values
            p_acts = win["prev_act_1"].values
            ts     = win["timestamp"].iloc[-1]

            mean_temp   = float(np.mean(temps))
            mean_act    = float(np.mean(acts))
            mean_sp_dev = mean_temp - float(np.mean(sps))
            mean_tdelta = float(np.mean(temps) - prev_temp) if prev_temp is not None else 0.0
            mean_adelta = float(np.mean(acts - p_acts))
            prev_temp   = mean_temp

            # Outside temperature: nearest OO reading
            if len(oo_df) > 0 and pd.notna(ts):
                idx = (oo_df["timestamp"] - ts).abs().idxmin()
                oo_mean = float(oo_df.loc[idx, "oo_temp"])
                if oo_mean > 200:
                    oo_mean -= 273.15
            else:
                oo_mean = float("nan")

            rows.append({
                "room":        room,
                "timestamp":   ts,
                "mean_temp":   round(mean_temp, 4),
                "mean_act":    round(mean_act, 4),
                "mean_sp_dev": round(mean_sp_dev, 4),
                "mean_tdelta": round(mean_tdelta, 4),
                "mean_adelta": round(mean_adelta, 4),
                "oo_mean_temp": round(oo_mean, 4) if not np.isnan(oo_mean) else 0.0,
            })

    feat_df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"  Transactions    : {len(feat_df):,}  across {len(indoor_rooms)} rooms")
    return feat_df


# =============================================================================
#  STEP 2 — LOAD ARM SCORES
# =============================================================================

def load_arm_scores(scores_csv: str, feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins ARM anomaly scores onto the feature DataFrame.
    Returns feat_df with new columns: arm_score, arm_severity, arm_anomaly (bool).
    """
    if not os.path.exists(scores_csv):
        print(f"  [Warning] ARM scores file not found: {scores_csv}")
        print("  Run building_hybrid_pipeline_v4.py first, then re-run this script.")
        feat_df["arm_score"]    = float("nan")
        feat_df["arm_severity"] = "UNKNOWN"
        feat_df["arm_anomaly"]  = False
        return feat_df

    arm = pd.read_csv(scores_csv)
    arm["timestamp"] = pd.to_datetime(arm["timestamp"], errors="coerce")
    arm = arm.rename(columns={"anomaly_score": "arm_score",
                               "severity":      "arm_severity"})
    arm = arm[["room", "timestamp", "arm_score", "arm_severity"]].drop_duplicates(
        subset=["room", "timestamp"]
    )

    feat_df = feat_df.merge(arm, on=["room", "timestamp"], how="left")
    feat_df["arm_score"]    = feat_df["arm_score"].fillna(0.0)
    feat_df["arm_severity"] = feat_df["arm_severity"].fillna("NONE")
    feat_df["arm_anomaly"]  = feat_df["arm_score"] >= ARM_ANOMALY_THRESHOLD

    matched = feat_df["arm_score"].notna().sum()
    print(f"  ARM scores matched: {matched:,} / {len(feat_df):,} transactions")
    return feat_df


# =============================================================================
#  STEP 3 — RUN ALTERNATIVE DETECTORS
# =============================================================================

FEATURE_COLS = ["mean_temp", "mean_act", "mean_sp_dev",
                "mean_tdelta", "mean_adelta", "oo_mean_temp"]


def _get_scaled(df: pd.DataFrame) -> np.ndarray:
    """Standardise the 6 numeric features. Fills any remaining NaN with 0."""
    X = df[FEATURE_COLS].fillna(0.0).values
    return StandardScaler().fit_transform(X), X


def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Isolation Forest: assigns a continuous anomaly score in [-1, 0] where
    lower = more anomalous.  We flip and normalise to [0, 1] so higher = worse,
    consistent with the ARM score direction.

    Returns df with new columns: if_score (0–1), if_anomaly (bool).
    """
    print("\n[Comparison] Running Isolation Forest...")
    t0 = time.time()

    X_scaled, _ = _get_scaled(df)

    clf = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_scaled)

    raw_scores = clf.decision_function(X_scaled)   # range roughly [-0.5, 0.5]
    flipped = -raw_scores
    min_s, max_s = flipped.min(), flipped.max()
    norm_scores = (flipped - min_s) / (max_s - min_s + 1e-9)

    df = df.copy()
    df["if_score"]   = np.round(norm_scores, 4)
    df["if_anomaly"] = clf.predict(X_scaled) == -1   # -1 = outlier in sklearn

    n_anom = df["if_anomaly"].sum()
    print(f"  Done ({time.time()-t0:.1f}s) | anomalies: {n_anom:,} "
          f"({100*n_anom/len(df):.1f}%)")
    return df


def run_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Comparison] Running K-Means clustering...")
    t0 = time.time()

    X_scaled, _ = _get_scaled(df)

    km = KMeans(
        n_clusters=KM_N_CLUSTERS,
        random_state=KM_RANDOM_STATE,
        n_init=10,
    )
    km.fit(X_scaled)

    # Distance of each point to its assigned centroid
    centres     = km.cluster_centers_
    labels      = km.labels_
    dists       = np.linalg.norm(X_scaled - centres[labels], axis=1)

    # Normalise to [0, 1]
    min_d, max_d = dists.min(), dists.max()
    norm_dists   = (dists - min_d) / (max_d - min_d + 1e-9)

    threshold    = np.quantile(norm_dists, 1.0 - KM_CONTAMINATION)

    df = df.copy()
    df["km_score"]   = np.round(norm_dists, 4)
    df["km_anomaly"] = norm_dists > threshold
    df["km_cluster"] = labels

    n_anom = df["km_anomaly"].sum()
    print(f"  Done ({time.time()-t0:.1f}s) | anomalies: {n_anom:,} "
          f"({100*n_anom/len(df):.1f}%)  threshold={threshold:.4f}")
    return df


def run_dbscan(df: pd.DataFrame) -> pd.DataFrame:
    """
    DBSCAN: points labelled -1 (noise) are anomalies.  No contamination
    parameter needed — the algorithm decides based on density.

    Returns df with new columns: dbscan_label (int), dbscan_anomaly (bool).
    """
    print("\n[Comparison] Running DBSCAN...")
    t0 = time.time()

    X_scaled, _ = _get_scaled(df)

    db = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        n_jobs=-1,
    )
    labels = db.fit_predict(X_scaled)

    df = df.copy()
    df["dbscan_label"]   = labels
    df["dbscan_anomaly"] = labels == -1

    n_anom   = (labels == -1).sum()
    n_clust  = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Done ({time.time()-t0:.1f}s) | clusters: {n_clust} | "
          f"noise points: {n_anom:,} ({100*n_anom/len(df):.1f}%)")
    return df


# =============================================================================
#  STEP 4 — CONSENSUS SCORE
# =============================================================================

def add_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a consensus_votes column (0–4) counting how many methods flag
    a transaction as anomalous, and a consensus_anomaly bool (votes >= 2).
    """
    df = df.copy()
    vote_cols = ["arm_anomaly", "if_anomaly", "km_anomaly", "dbscan_anomaly"]
    existing  = [c for c in vote_cols if c in df.columns]
    df["consensus_votes"]   = df[existing].astype(int).sum(axis=1)
    df["consensus_anomaly"] = df["consensus_votes"] >= 2
    return df


# =============================================================================
#  STEP 5 — AGREEMENT MATRIX + STATISTICS
# =============================================================================

def compute_agreement(df: pd.DataFrame) -> Dict:
    """
    Returns a dict with:
      - rates       : anomaly rate per method
      - agreement   : pairwise agreement fraction between all method pairs
      - by_room     : per-room anomaly rates per method
    """
    methods = {
        "ARM":    "arm_anomaly",
        "IsoFor": "if_anomaly",
        "KMeans": "km_anomaly",
        "DBSCAN": "dbscan_anomaly",
    }
    existing = {k: v for k, v in methods.items() if v in df.columns}

    rates = {}
    for name, col in existing.items():
        rates[name] = float(df[col].mean())

    pairs = {}
    names = list(existing.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = existing[names[i]], existing[names[j]]
            agree = float((df[a] == df[b]).mean())
            pairs[f"{names[i]} vs {names[j]}"] = agree

    by_room = {}
    for room, grp in df.groupby("room"):
        by_room[room] = {name: float(grp[col].mean())
                         for name, col in existing.items()}

    score_cols = {
        "ARM":    "arm_score",
        "IsoFor": "if_score",
        "KMeans": "km_score",
    }
    existing_scores = {k: v for k, v in score_cols.items() if v in df.columns}
    corr = df[[v for v in existing_scores.values()]].corr()
    corr.columns = list(existing_scores.keys())
    corr.index   = list(existing_scores.keys())

    return {
        "rates":    rates,
        "pairs":    pairs,
        "by_room":  by_room,
        "corr":     corr,
        "n_total":  len(df),
        "n_consensus": int(df["consensus_anomaly"].sum()),
    }


# =============================================================================
#  STEP 6 — VISUALISATIONS
# =============================================================================

def plot_agreement(df: pd.DataFrame, stats: Dict) -> None:
    """
    Single-panel academic-style clustered bar chart.
    White background, black axes and spines, hatched bars for B&W printing.
    """
    by_room  = stats["by_room"]
    rooms    = sorted(by_room.keys())
    methods  = ["ARM", "IsoFor", "KMeans", "DBSCAN"]
    # Accessible, print-safe colours + hatch patterns for B&W reproduction
    colors_m = ["#2166ac", "#d6604d", "#4dac26", "#878787"]
    hatches  = ["",        "///",     "xxx",     "..."]
    labels   = [
        "Assoc. Rule Mining (ARM)",
        "Isolation Forest",
        "K-Means Clustering",
        "DBSCAN",
    ]

    x     = np.arange(len(rooms))
    width = 0.19
    n     = len(methods)
    # Centre the group of bars around each room tick
    offsets = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for i, (method, color, hatch, label) in enumerate(
        zip(methods, colors_m, hatches, labels)
    ):
        vals = [by_room[r].get(method, 0.0) * 100 for r in rooms]
        bars = ax.bar(
            x + offsets[i], vals, width,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.88,
        )
        # Value labels on top of each bar
        for bar, v in zip(bars, vals):
            if v >= 0.2:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.08,
                    f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=7, color="black",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(rooms, fontsize=11)
    ax.set_xlabel("Room", fontsize=12, labelpad=6)
    ax.set_ylabel("Anomaly Rate (%)", fontsize=12, labelpad=6)
    ax.set_title(
        "Anomaly Detection Rate per Room — Comparison of Unsupervised Methods",
        fontsize=13, fontweight="bold", pad=10,
    )


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

 
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="both", length=3, labelsize=10)

 
    max_val = max(
        by_room[r].get(m, 0.0) * 100
        for r in rooms for m in methods
    )
    ax.set_ylim(0, max_val * 1.18)

    legend = ax.legend(
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        edgecolor="#aaaaaa",
        loc="upper right",
        ncol=2,
        columnspacing=1.0,
        handlelength=1.8,
    )
    legend.get_frame().set_linewidth(0.5)

    fig.tight_layout(pad=1.2)
    plt.savefig(AGREEMENT_PATH, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  Agreement chart → {AGREEMENT_PATH}")


def plot_roc_proxy(df: pd.DataFrame) -> None:
    """
    ROC-style curves treating ARM score (thresholded at each point) as
    the proxy ground truth, evaluating IF and KMeans against it.

    Since we have no true labels, this shows how well IF and KMeans
    agree with the ARM method across the full score range — effectively
    a method-agreement curve.  A curve close to the diagonal means the
    methods disagree; a curve in the upper-left means strong agreement.
    """
    if "if_score" not in df.columns or "km_score" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#0f3460"); sp.set_linewidth(1.2)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    arm_scores = df["arm_score"].values

    for method_col, label, color in [
        ("if_score", "Isolation Forest", "#e74c3c"),
        ("km_score", "K-Means",          "#f39c12"),
    ]:
        method_scores = df[method_col].values
        thresholds    = np.linspace(0, 1, 100)
        tprs, fprs    = [], []

        for t in thresholds:
            arm_pos    = arm_scores >= t
            method_pos = method_scores >= np.quantile(method_scores,
                             1.0 - arm_pos.mean() if arm_pos.mean() > 0 else 0.95)
            tp = np.sum(arm_pos & method_pos)
            fp = np.sum(~arm_pos & method_pos)
            fn = np.sum(arm_pos & ~method_pos)
            tn = np.sum(~arm_pos & ~method_pos)

            tpr = tp / (tp + fn + 1e-9)
            fpr = fp / (fp + tn + 1e-9)
            tprs.append(tpr)
            fprs.append(fpr)

        # AUC via trapezoidal rule
        sorted_pairs = sorted(zip(fprs, tprs))
        fprs_s, tprs_s = zip(*sorted_pairs)
        _trapz = getattr(np, "trapezoid", None) or np.trapz
        auc = float(_trapz(tprs_s, fprs_s))

        ax.plot(fprs_s, tprs_s, color=color, linewidth=2,
                label=f"{label}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#5F5E5A", linewidth=1,
            linestyle="--", label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate (vs ARM)", fontsize=10)
    ax.set_ylabel("True Positive Rate (vs ARM)", fontsize=10)
    ax.set_title(
        "Agreement ROC curve (ARM score as proxy truth)\n"
        "How well IF and KMeans agree with the association rule method",
        fontsize=11, pad=8,
    )
    ax.legend(framealpha=0.3, labelcolor="white",
              facecolor="#1a1a2e", edgecolor="#0f3460", fontsize=9)
    ax.grid(color="#0f3460", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(ROC_PATH, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ROC proxy chart → {ROC_PATH}")


# =============================================================================
#  STEP 7 — PRINT + SAVE SUMMARY
# =============================================================================

def print_and_save_summary(stats: Dict, df: pd.DataFrame) -> None:
    W    = 70
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * W)
    p("  UNSUPERVISED METHOD COMPARISON — SUMMARY")
    p("=" * W)
    p(f"  Total test transactions : {stats['n_total']:,}")
    p(f"  Consensus anomalies     : {stats['n_consensus']:,}  "
      f"(≥2 methods agree — {100*stats['n_consensus']/stats['n_total']:.1f}%)")
    p()

    p("  ANOMALY RATES PER METHOD:")
    p("  " + "-" * (W - 2))
    for method, rate in stats["rates"].items():
        bar = "█" * int(rate * 50)
        p(f"    {method:<10} {rate*100:>6.2f}%  {bar}")
    p()

    p("  PAIRWISE AGREEMENT (fraction of transactions where both methods agree):")
    p("  " + "-" * (W - 2))
    for pair, agree in stats["pairs"].items():
        strength = ("strong" if agree > 0.90 else
                    "moderate" if agree > 0.75 else "weak")
        p(f"    {pair:<25}  {agree*100:>6.2f}%  [{strength}]")
    p()

    if "corr" in stats and stats["corr"] is not None:
        p(" ")
        p("  " + "-" * (W - 2))
        corr = stats["corr"]
        cols = list(corr.columns)
        header = "    " + "".join(f"{c:>10}" for c in cols)
        
        for row_name in corr.index:
            vals = "".join(f"{corr.loc[row_name, c]:>10.3f}" for c in cols)
            p(f"    {row_name:<6}{vals}")
        p()

    p("  PER-ROOM ANOMALY RATES (%):")
    p("  " + "-" * (W - 2))
    methods = list(stats["rates"].keys())
    header = f"    {'Room':<8}" + "".join(f"{m:>10}" for m in methods)
    p(header)
    for room in sorted(stats["by_room"].keys()):
        vals = "".join(f"{stats['by_room'][room].get(m, 0)*100:>10.1f}"
                       for m in methods)
        p(f"    {room:<8}{vals}")
    p()

    p("  INTERPRETATION:")
    p("  " + "-" * (W - 2))

    arm_rate = stats["rates"].get("ARM", 0)
    if_rate  = stats["rates"].get("IsoFor", 0)
    km_rate  = stats["rates"].get("KMeans", 0)
    db_rate  = stats["rates"].get("DBSCAN", 0)
    consensus_rate = stats['n_consensus'] / stats['n_total']

    p(f"  • ARM flags {arm_rate*100:.1f}% of transactions as anomalous.")

    # Agreement analysis
    arm_if_agree = stats["pairs"].get("ARM vs IsoFor", 0)
    arm_km_agree = stats["pairs"].get("ARM vs KMeans", 0)
    if arm_if_agree > 0.85 and arm_km_agree > 0.85:
        p("  • Strong agreement between ARM, Isolation Forest, and K-Means.")
        p("    This validates that the association rules are capturing real")
        p("    anomalies, not statistical artefacts.")
    elif arm_if_agree > 0.75 or arm_km_agree > 0.75:
        p("  • Moderate agreement between methods. ARM anomalies are broadly")
        p("    consistent with density/distance-based methods, though some")
        p("    divergence exists — likely due to ARM's rule interpretability")
        p("    capturing behavioural violations that pure distance ignores.")
    else:
        p("  • Low agreement between methods. This is expected when ARM")
        p("    captures rule-based behavioural patterns that are not visible")
        p("    as outliers in raw feature space (e.g. wrong heater state at")
        p("    correct temperature).")

    p(f"  • Consensus rate ({consensus_rate*100:.1f}%) represents transactions")
    p("    flagged by ≥2 methods — highest-confidence anomaly candidates.")
    p("=" * W)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Summary saved → {SUMMARY_PATH}")


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

def run_comparison(
    test_csv_path: str = TEST_CSV_PATH,
    scores_csv:    str = SCORES_CSV,
) -> pd.DataFrame:
    """
    Full comparison pipeline.

    Parameters
    ----------
    test_csv_path : path to the test CSV (same used by the main pipeline)
    scores_csv    : path to outputs/all_scores.csv from the main pipeline

    Returns
    -------
    report_df : DataFrame with all method flags per transaction
                (also saved to outputs/comparison_report.csv)
    """
    print("\n" + "=" * 70)
    print("  UNSUPERVISED METHOD COMPARISON")
    print("  Methods: Association Rules | Isolation Forest | K-Means | DBSCAN")
    print("=" * 70)
    print(f"  Config:")
    print(f"    ARM threshold          : {ARM_ANOMALY_THRESHOLD}")
    print(f"    IF contamination       : {IF_CONTAMINATION}")
    print(f"    K-Means clusters       : {KM_N_CLUSTERS}")
    print(f"    DBSCAN eps / min_samp  : {DBSCAN_EPS} / {DBSCAN_MIN_SAMPLES}")

    # Step 1: build feature matrix
    df = build_feature_matrix(test_csv_path)

    # Step 2: load ARM scores
    df = load_arm_scores(scores_csv, df)

    # Step 3: run alternative methods
    df = run_isolation_forest(df)
    df = run_kmeans(df)
    df = run_dbscan(df)

    # Step 4: consensus
    df = add_consensus(df)

    # Step 5: statistics
    stats = compute_agreement(df)

    # Step 6: visualisations
    print("\n[Comparison] Generating visualisations...")
    plot_agreement(df, stats)
    plot_roc_proxy(df)

    # Step 7: summary
    print_and_save_summary(stats, df)

    # Save full report
    report_cols = [
        "room", "timestamp",
        "mean_temp", "mean_act", "mean_sp_dev",
        "arm_score", "arm_severity", "arm_anomaly",
        "if_score",  "if_anomaly",
        "km_score",  "km_anomaly",  "km_cluster",
        "dbscan_label", "dbscan_anomaly",
        "consensus_votes", "consensus_anomaly",
    ]
    existing_cols = [c for c in report_cols if c in df.columns]
    df[existing_cols].to_csv(REPORT_PATH, index=False)
    print(f"  Full report     → {REPORT_PATH}")

    return df


# =============================================================================
#  STANDALONE ENTRY POINT (OPTIONAL)
# =============================================================================

if __name__ == "__main__":
    test_csv = sys.argv[1] if len(sys.argv) > 1 else TEST_CSV_PATH
    scores   = sys.argv[2] if len(sys.argv) > 2 else SCORES_CSV
    run_comparison(test_csv_path=test_csv, scores_csv=scores)