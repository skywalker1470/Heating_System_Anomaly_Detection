"""
=============================================================================
Pattern Stability Analysis  —  Add-on for building_hybrid_pipeline_v3.py
=============================================================================

PURPOSE
-------
Validates that the association rules mined from training data are STABLE
across time — i.e. they are genuine recurring patterns, not artifacts of
one particular slice of data.

HOW IT WORKS

1. The training CSV is divided into N chronological folds.
2. Rules are mined on each fold using the same settings as the main pipeline (FP-Growth + RuleFilter).
3. For each rule found on ANY fold, the following is calculated:
    prevalence : proportion of folds where the rule is found (between 0.0 and 1.0)
    conf_mean : mean confidence over folds where it is found
    conf_std : std deviation of confidence (low if stable, high if unstable)
    conf_slope : direction of change in confidence (positive if strengthening, negative if degrading)
    sup_mean : mean support over folds where it is found
    sup_std : std deviation of support
    lift_mean : mean lift over folds where it is found
    stability_score : prevalence \* (1 - conf_std) : composite score
4. Rules are assigned STABLE, MODERATE, or UNSTABLE stability labels.
5. A "stable rule set" where prevalence == 1.0 and conf_std < threshold is written out separately.
6. Support drift is tracked over each fold.
7. The following outputs are produced:
    outputs/stability_report.csv : per-rule stability metrics
    outputs/stable_rules_only.csv : only the high-confidence stable rules
    outputs/stability_heatmap.png : heatmap of rules x folds, with cell values showing confidence
    outputs/support_drift.png : line chart showing support over folds for each rule



REQUIREMENTS
------------
    Same as building_hybrid_pipeline_v3.py:
    pip install pandas numpy mlxtend tqdm matplotlib
=============================================================================
"""

import os
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
#  CONFIGURATION  
# =============================================================================

# How many chronological folds to split the training data into 
# CHANGED: raised from 4 to 5.
# With 4 folds a single bad fold drops a rule from STABLE to MODERATE.
# 5 folds gives finer granularity 
N_FOLDS = 5

#Rule mining settings (copy from main pipeline) 
MIN_SUPPORT    = 0.10
MIN_CONFIDENCE = 0.92

#Rule filter settings (copy from main pipeline) 
MIN_LIFT               = 1.5
MIN_LEVERAGE           = 0.02
MIN_CONVICTION         = 1.2
MAX_CON_SUPPORT        = 0.70
MAX_CONSEQUENT_SIZE    = 2
MIN_ANTECEDENT_SUPPORT = 0.08
MAX_RULE_SET_SIZE      = 500

#Discretization thresholds (copy from main pipeline) 
OO_COLD_MAX       = 15.0
OO_NORMAL_MAX     = 20.0
OO_WARM_MAX       = 25.0
TEMP_COLD_MAX     = 19.0
TEMP_HOT_MIN      = 22.0
TREND_STABLE_BAND = 0.01
SP_DEV_BAND       = 1.5
TRANSACTION_SIZE  = 6
STRIDE            = 3   # sliding window stride (records advanced per step)

#Stability classification thresholds 
STABLE_PREVALENCE_MIN   = 1.00   # must appear in ALL folds
STABLE_CONF_STD_MAX     = 0.05   
MODERATE_PREVALENCE_MIN = 0.50   


# ── Output ──────────────
OUTPUT_DIR            = "outputs"
STABILITY_REPORT_PATH = os.path.join(OUTPUT_DIR, "stability_report.csv")
STABLE_RULES_PATH     = os.path.join(OUTPUT_DIR, "stable_rules_only.csv")
HEATMAP_PATH          = os.path.join(OUTPUT_DIR, "stability_heatmap.png")
DRIFT_PATH            = os.path.join(OUTPUT_DIR, "support_drift.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
#  STEP 1 — LOAD & SPLIT
# =============================================================================

def load_and_split(train_csv_path: str, n_folds: int = N_FOLDS) -> List[pd.DataFrame]:
    """
    Reads the training CSV in one shot and splits chronologically by row
    order — do NOT shuffle, since order matters for trend features.

    Returns a list of n_folds DataFrames.
    """
    print(f"\n[Stability] Loading training data from: {train_csv_path}")
    t0 = time.time()

    df = pd.read_csv(train_csv_path, sep=";", low_memory=False)
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
    )

    # Kelvin → Celsius
    for col in ["temperature", "setpoint"]:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").median() > 200:
            df[col] = pd.to_numeric(df[col], errors="coerce") - 273.15

    # Ensure numeric actuation columns
    for col in ["actuation", "previous_actuation_1", "previous_actuation_2",
                "previous_actuation_3"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    n = len(df)
    fold_size = n // n_folds
    folds = [df.iloc[i * fold_size: (i + 1) * fold_size].copy()
             for i in range(n_folds)]

    print(f"  Total rows      : {n:,}")
    print(f"  Folds           : {n_folds}  (~{fold_size:,} rows each)")
    print(f"  Load time       : {time.time() - t0:.1f}s")
    return folds


# =============================================================================
#  STEP 2 — BUILD TRANSACTIONS FOR ONE FOLDn.
#  FIX 1: OO temperature is now computed per-window using a positional
#          slice of the OO rows matching the window position, instead of
#          a single global mean for the entire fold. This matches the
#          rolling OOBuffer logic in the main pipeline.
#
#  FIX 2: Temperature delta is now computed as mean(np.diff(window_temps))
#          — the average of per-record differences within the window —
#          matching the main pipeline's StreamPreprocessor which computes
#          record-to-record temp_delta and then averages over the window.
# =============================================================================

def _oo_label(mean_temp: float) -> str:
    if mean_temp < OO_COLD_MAX:    return "OO_COLD"
    if mean_temp < OO_NORMAL_MAX:  return "OO_NORMAL"
    if mean_temp < OO_WARM_MAX:    return "OO_WARM"
    return "OO_HOT"


def _trend(prefix: str, delta: float) -> str:
    if delta > TREND_STABLE_BAND:   return f"{prefix}_INCREASING"
    if delta < -TREND_STABLE_BAND:  return f"{prefix}_DECREASING"
    return f"{prefix}_STABLE"


def build_transactions_from_fold(fold_df: pd.DataFrame) -> List[List[str]]:
    """
    Replicates the pipeline's TransactionBuilder logic on a DataFrame fold.
    Groups records by room, slides a window of TRANSACTION_SIZE rows
    with step STRIDE, computes mean features, then discretizes once.

    Returns a list of transactions (each transaction = list of string items).
    """
    # Normalise column names
    col_map = {"datetime": "timestamp", "timeline": "elapsed_s", "room_name": "room"}
    fold_df = fold_df.rename(columns={k: v for k, v in col_map.items()
                                       if k in fold_df.columns})

    if "room" not in fold_df.columns:
        candidate = next((c for c in fold_df.columns
                          if "room" in c or "name" in c), None)
        fold_df["room"] = fold_df[candidate] if candidate else "UNKNOWN"

    # Rename previous_actuation_* → prev_act_*
    for i, variants in enumerate(
        [["prev_act_1", "previous_actuation_1"],
         ["prev_act_2", "previous_actuation_2"],
         ["prev_act_3", "previous_actuation_3"]], start=1
    ):
        canonical = f"prev_act_{i}"
        found = next((v for v in variants if v in fold_df.columns), None)
        if found and found != canonical:
            fold_df = fold_df.rename(columns={found: canonical})
        if canonical not in fold_df.columns:
            fold_df[canonical] = 0.0

    #FIX 1: Build positional OO temperature array ──────────────────────────
    oo_df = fold_df[fold_df["room"] == "OO"].reset_index(drop=True)
    oo_temps_array = pd.to_numeric(
        oo_df["temperature"], errors="coerce"
    ).fillna(0.0).values if len(oo_df) > 0 else np.array([])

    # Fallback global mean — used only if positional slice is empty
    oo_mean_global = float(np.mean(oo_temps_array)) if len(oo_temps_array) > 0 else None

    transactions = []
    rooms = fold_df["room"].unique()

    for room in rooms:
        if room == "OO":
            continue

        room_df = fold_df[fold_df["room"] == room].reset_index(drop=True)

        for start in range(0, len(room_df) - TRANSACTION_SIZE + 1, STRIDE):
            window = room_df.iloc[start: start + TRANSACTION_SIZE]

            temps  = pd.to_numeric(window["temperature"], errors="coerce").fillna(0.0).values
            acts   = pd.to_numeric(window["actuation"],   errors="coerce").fillna(0.0).values
            sps    = pd.to_numeric(window["setpoint"],    errors="coerce").fillna(0.0).values
            p_acts = pd.to_numeric(window["prev_act_1"],  errors="coerce").fillna(0.0).values

            mean_temp   = float(np.mean(temps))
            mean_act    = float(np.mean(acts))
            mean_sp_dev = mean_temp - float(np.mean(sps))
            mean_adelta = float(np.mean(acts - p_acts))

            #FIX 2: Per-record temperature delta 
            if len(temps) > 1:
                mean_tdelta = float(np.mean(np.diff(temps)))
            else:
                mean_tdelta = 0.0

            items: List[str] = []

            # Temperature state
            if mean_temp < TEMP_COLD_MAX:
                items.append("TEMP_COLD")
            elif mean_temp > TEMP_HOT_MIN:
                items.append("TEMP_HOT")
            else:
                items.append("TEMP_NORMAL")

            # Temperature trend  (FIX 2 applied here)
            items.append(_trend("TEMP", mean_tdelta))

            # Heater state
            items.append("HEATER_ON" if mean_act > 0.01 else "HEATER_OFF")

            # Actuation trend
            items.append(_trend("ACT", mean_adelta))

            # Setpoint deviation
            if mean_sp_dev < -SP_DEV_BAND:
                items.append("BELOW_SETPOINT")
            elif mean_sp_dev > SP_DEV_BAND:
                items.append("ABOVE_SETPOINT")
            else:
                items.append("AT_SETPOINT")

            # FIX 1: Per-window OO temperature ──
           
            if len(oo_temps_array) > 0:
                oo_slice = oo_temps_array[start: start + TRANSACTION_SIZE]
                if len(oo_slice) > 0:
                    oo_mean = float(np.mean(oo_slice))
                else:
                    # Window extends beyond OO data
                    oo_mean = float(oo_temps_array[-1]) if len(oo_temps_array) > 0 \
                              else oo_mean_global
                items.append(_oo_label(oo_mean))
            elif oo_mean_global is not None:
                items.append(_oo_label(oo_mean_global))
            # If no OO data at all, omit OO item 

            transactions.append(items)

    return transactions


# =============================================================================
#  STEP 3 — MINE RULES FOR ONE FOLD
# =============================================================================

_DOMAINS = {
    "TEMPERATURE": {"TEMP_COLD", "TEMP_NORMAL", "TEMP_HOT"},
    "TEMP_TREND":  {"TEMP_INCREASING", "TEMP_STABLE", "TEMP_DECREASING"},
    "HEATER":      {"HEATER_ON", "HEATER_OFF"},
    "ACT_TREND":   {"ACT_INCREASING", "ACT_STABLE", "ACT_DECREASING"},
    "SETPOINT":    {"BELOW_SETPOINT", "AT_SETPOINT", "ABOVE_SETPOINT"},
    "OUTSIDE":     {"OO_COLD", "OO_NORMAL", "OO_WARM", "OO_HOT"},
}


def _get_domains(items: frozenset) -> set:
    domains = set()
    for item in items:
        for domain, members in _DOMAINS.items():
            if item in members:
                domains.add(domain)
    return domains


def _is_normal_heating_rule(ant: frozenset, con: frozenset) -> bool:
    """Mirror of RuleFilter._filter_normal_heating in the main pipeline."""
    if "HEATER_ON" in ant:
        normal_con = {"TEMP_HOT", "TEMP_INCREASING", "AT_SETPOINT"}
        if con.issubset(normal_con):
            return True
    ant_domains = _get_domains(ant)
    if (con == frozenset({"HEATER_OFF"})
            and ant_domains == {"TEMPERATURE"}
            and "OUTSIDE" not in ant_domains):
        return True
    return False


def mine_rules_for_fold(transactions: List[List[str]],
                         fold_idx: int) -> Optional[pd.DataFrame]:
    """
    Runs FP-Growth and applies all rule filters on a single fold's transactions.
    Returns a filtered DataFrame with an added 'rule_key' column for matching
    across folds, or None if no rules survive.
    """
    if not transactions:
        return None

    te = TransactionEncoder()
    te_arr = te.fit_transform(transactions)
    te_df  = pd.DataFrame(te_arr, columns=te.columns_)

    freq = fpgrowth(te_df, min_support=MIN_SUPPORT, use_colnames=True)
    if freq.empty:
        return None

    rules = association_rules(freq, metric="confidence",
                              min_threshold=MIN_CONFIDENCE)
    if rules.empty:
        return None

    # 1. Consequent size
    rules = rules[rules["consequents"].apply(len) <= MAX_CONSEQUENT_SIZE]
    # 2. Lift
    rules = rules[rules["lift"] > MIN_LIFT]
    # 3. Leverage
    rules = rules[rules["leverage"] > MIN_LEVERAGE]
    # 4. Conviction
    rules = rules[
        (rules["conviction"] > MIN_CONVICTION) |
        (rules["conviction"] == float("inf"))
    ]
    # 5. Trivial consequent
    rules = rules[rules["consequent support"] < MAX_CON_SUPPORT]
    # 6. Cross-domain
    mask = []
    for _, row in rules.iterrows():
        all_d = _get_domains(row["antecedents"]) | _get_domains(row["consequents"])
        mask.append(len(all_d) >= 2)
    rules = rules[mask]
    # 7. Antecedent support
    rules = rules[rules["antecedent support"] >= MIN_ANTECEDENT_SUPPORT]
    # 8. Normal-heating filter
    mask = [not _is_normal_heating_rule(r["antecedents"], r["consequents"])
            for _, r in rules.iterrows()]
    rules = rules[mask]
    # 9. Top-N cap
    if len(rules) > MAX_RULE_SET_SIZE:
        rules = (rules
                 .assign(_s=rules["confidence"] * rules["lift"])
                 .sort_values("_s", ascending=False)
                 .head(MAX_RULE_SET_SIZE)
                 .drop(columns=["_s"]))

    if rules.empty:
        return None

    # Canonical string key for cross-fold matching
    rules = rules.copy()
    rules["rule_key"] = rules.apply(
        lambda r: (
            " & ".join(sorted(r["antecedents"])) +
            " -> " +
            " & ".join(sorted(r["consequents"]))
        ),
        axis=1,
    )
    rules["fold"] = fold_idx
    return rules.reset_index(drop=True)


# =============================================================================
#  STEP 4 — COMPUTE STABILITY METRICS
# =============================================================================

def compute_stability_metrics(fold_rule_dfs: List[Optional[pd.DataFrame]],
                               n_folds: int) -> pd.DataFrame:
    """
    Given a list of fold-level rule DataFrames, compute stability metrics
    for every unique rule that appeared in at least one fold.

    Returns a DataFrame with columns:
        rule_key, antecedents, consequents,
        folds_present, prevalence,
        conf_mean, conf_std, conf_min, conf_max, conf_slope,
        sup_mean, sup_std,
        lift_mean,
        stability_score,
        label,
        conf_fold_0 … conf_fold_{N-1},
        sup_fold_0  … sup_fold_{N-1}
    """
    rule_records: Dict[str, dict] = {}

    for fold_df in fold_rule_dfs:
        if fold_df is None:
            continue
        for _, row in fold_df.iterrows():
            key = row["rule_key"]
            if key not in rule_records:
                rule_records[key] = {
                    "antecedents": " & ".join(sorted(row["antecedents"])),
                    "consequents": " & ".join(sorted(row["consequents"])),
                    "folds_conf": {},
                    "folds_sup":  {},
                    "folds_lift": {},
                }
            rule_records[key]["folds_conf"][int(row["fold"])] = float(row["confidence"])
            rule_records[key]["folds_sup"][int(row["fold"])]  = float(row["support"])
            rule_records[key]["folds_lift"][int(row["fold"])] = float(row["lift"])

    rows = []
    for key, data in rule_records.items():
        confs     = list(data["folds_conf"].values())
        sups      = list(data["folds_sup"].values())
        lifts     = list(data["folds_lift"].values())
        n_present = len(confs)
        prevalence = n_present / n_folds

        conf_mean = float(np.mean(confs))
        conf_std  = float(np.std(confs))  if n_present > 1 else 0.0
        sup_mean  = float(np.mean(sups))
        sup_std   = float(np.std(sups))   if n_present > 1 else 0.0
        lift_mean = float(np.mean(lifts))

     
        stability_score = prevalence * (1.0 - min(conf_std, 1.0))

        confs_ordered = [data["folds_conf"][i]
                         for i in range(n_folds)
                         if i in data["folds_conf"]]
        if len(confs_ordered) >= 2:
            conf_slope = (confs_ordered[-1] - confs_ordered[0]) / (len(confs_ordered) - 1)
        else:
            conf_slope = 0.0

        if (prevalence >= STABLE_PREVALENCE_MIN
                and conf_std <= STABLE_CONF_STD_MAX):
            label = "STABLE"
        elif prevalence >= MODERATE_PREVALENCE_MIN:
            label = "MODERATE"
        else:
            label = "UNSTABLE"

        row_out = {
            "rule_key":        key,
            "antecedents":     data["antecedents"],
            "consequents":     data["consequents"],
            "folds_present":   n_present,
            "prevalence":      round(prevalence, 4),
            "conf_mean":       round(conf_mean, 4),
            "conf_std":        round(conf_std, 4),
            "conf_min":        round(min(confs), 4),
            "conf_max":        round(max(confs), 4),
            "conf_slope":      round(conf_slope, 6),   # NEW
            "sup_mean":        round(sup_mean, 4),
            "sup_std":         round(sup_std, 4),
            "lift_mean":       round(lift_mean, 4),
            "stability_score": round(stability_score, 4),
            "label":           label,
        }

        for fi in range(n_folds):
            row_out[f"conf_fold_{fi}"] = round(data["folds_conf"].get(fi, float("nan")), 4)
            row_out[f"sup_fold_{fi}"]  = round(data["folds_sup"].get(fi, float("nan")), 4)

        rows.append(row_out)

    df = pd.DataFrame(rows)
    df = df.sort_values("stability_score", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
#  STEP 5 — VISUALISATIONS
# =============================================================================

def plot_stability_heatmap(stability_df: pd.DataFrame, n_folds: int) -> None:
    """
    Heatmap: rows = top-50 rules by stability_score,
             columns = folds,
             cell value = confidence in that fold (dark = rule absent).
    """
    top_n   = min(50, len(stability_df))
    plot_df = stability_df.head(top_n).copy()

    conf_cols = [f"conf_fold_{i}" for i in range(n_folds)]
    matrix    = plot_df[conf_cols].values.astype(float)
    labels    = [r[:80] for r in plot_df["rule_key"].tolist()]

    fig, ax = plt.subplots(
        figsize=(max(8, n_folds * 2.5), max(10, top_n * 0.35)),
        facecolor="#1a1a2e"
    )
    ax.set_facecolor("#16213e")

    masked = np.ma.masked_invalid(matrix)
    cmap   = plt.cm.RdYlGn
    cmap.set_bad(color="#2a2a4a")

    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0.88, vmax=1.0)

    ax.set_xticks(range(n_folds))
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)],
                       color="white", fontsize=10)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, color="white", fontsize=7)
    ax.set_title(
        f"Rule Confidence Heatmap — top {top_n} rules by stability score\n"
        "(green = high confidence, dark cell = rule absent in that fold)",
        color="white", fontsize=11, pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Confidence", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    for sp in ax.spines.values():
        sp.set_visible(False)

    plt.tight_layout()
    plt.savefig(HEATMAP_PATH, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Heatmap saved   → {HEATMAP_PATH}")


def plot_support_drift(stability_df: pd.DataFrame, n_folds: int,
                       top_n: int = 20) -> None:
    """
    Line chart showing each STABLE rule's support across chronological folds.
    A rule whose support drops over folds is 'fading' — relevant to the
    proposal risk about rules becoming outdated.
    """
    stable_df = stability_df[stability_df["label"] == "STABLE"].head(top_n)
    if stable_df.empty:
        print("  [Support drift] No STABLE rules to plot — skipping.")
        return

    sup_cols = [f"sup_fold_{i}" for i in range(n_folds)]
    x = list(range(1, n_folds + 1))

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#0f3460"); sp.set_linewidth(1.2)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    colors = plt.cm.tab20.colors
    for idx, (_, row) in enumerate(stable_df.iterrows()):
        sups  = [row[c] for c in sup_cols]
        label = row["rule_key"][:60] + ("…" if len(row["rule_key"]) > 60 else "")

        # Line style: dashed if conf_slope is negative (degrading rule)
        ls = "--" if row.get("conf_slope", 0) < -0.001 else "-"
        ax.plot(x, sups, marker="o", linewidth=1.5, markersize=5,
                color=colors[idx % len(colors)], alpha=0.85, linestyle=ls)
        ax.annotate(
            label, xy=(n_folds, sups[-1]),
            xytext=(n_folds + 0.05, sups[-1]),
            fontsize=6, color=colors[idx % len(colors)],
            va="center",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in x], color="white")
    ax.set_xlabel("Chronological fold", fontsize=10)
    ax.set_ylabel("Support (fraction of transactions)", fontsize=10)
    ax.set_title(
        f"Support drift — top {len(stable_df)} STABLE rules over {n_folds} folds\n"
        "(flat lines = stable; dashed = confidence degrading; downward = fading)",
        fontsize=11, pad=8,
    )
    ax.set_xlim(1, n_folds + 0.1)
    ax.grid(axis="y", color="#0f3460", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(DRIFT_PATH, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Support drift   → {DRIFT_PATH}")


# =============================================================================
#  STEP 6 — TERMINAL SUMMARY
# =============================================================================

def print_summary(stability_df: pd.DataFrame, n_folds: int) -> None:
    W = 72
    total      = len(stability_df)
    n_stable   = (stability_df["label"] == "STABLE").sum()
    n_moderate = (stability_df["label"] == "MODERATE").sum()
    n_unstable = (stability_df["label"] == "UNSTABLE").sum()

    print("\n" + "=" * W)
    print("  PATTERN STABILITY ANALYSIS — SUMMARY")
    print("=" * W)
    print(f"  Folds analysed     : {n_folds}")
    print(f"  Unique rules found : {total:,}  (across any fold)")
    print()
    print(f"  STABLE   (present in ALL {n_folds} folds, conf_std < {STABLE_CONF_STD_MAX}): "
          f"{n_stable:>5,}  ({100*n_stable/total:.1f}%)")
    print(f"  MODERATE (present in ≥{int(MODERATE_PREVALENCE_MIN*n_folds)}/{n_folds} folds)           : "
          f"{n_moderate:>5,}  ({100*n_moderate/total:.1f}%)")
    print(f"  UNSTABLE (present in fewer folds)              : "
          f"{n_unstable:>5,}  ({100*n_unstable/total:.1f}%)")

    print()
    print("  TOP 15 STABLE RULES (by stability score):")
    print("  " + "-" * (W - 2))
    print(f"  {'Rule':<48}  {'Stab':>5}  {'Conf̄':>5}  {'Conf σ':>6}  {'Slope':>7}  {'Sup̄':>5}")
    print("  " + "-" * (W - 2))

    top = stability_df[stability_df["label"] == "STABLE"].head(15)
    if top.empty:
        print("  (no STABLE rules found — try reducing N_FOLDS or lowering MIN_SUPPORT)")
    for _, row in top.iterrows():
        rule_str = row["rule_key"]
        if len(rule_str) > 48:
            rule_str = rule_str[:45] + "…"
        slope_flag = "↑" if row.get("conf_slope", 0) > 0.001 else \
                     "↓" if row.get("conf_slope", 0) < -0.001 else "→"
        print(f"  {rule_str:<48}  "
              f"{row['stability_score']:>5.3f}  "
              f"{row['conf_mean']:>5.3f}  "
              f"{row['conf_std']:>6.4f}  "
              f"{slope_flag} {row.get('conf_slope', 0.0):>+5.4f}  "
              f"{row['sup_mean']:>5.3f}")

    degrading = stability_df[
        (stability_df["label"] == "STABLE") &
        (stability_df.get("conf_slope", pd.Series(0, index=stability_df.index)) < -0.005)
    ]
    if len(degrading) > 0:
        print()
        print(f"  ⚠  WARNING: {len(degrading)} STABLE rule(s) show degrading confidence (↓):")
        for _, row in degrading.iterrows():
            print(f"     {row['rule_key'][:65]}  slope={row.get('conf_slope',0):+.4f}")

    print()
    print("  INTERPRETATION GUIDE:")
    print("  • STABLE rules  → reliable anomaly detection baseline.")
    print("    Present in ALL folds with low confidence variance.")
    print()
    print("  • MODERATE rules → worth monitoring; may be seasonal.")
    print("    Treat their anomaly signals with lower weight.")
    print()
    print("  • UNSTABLE rules → likely noise or time-specific artefacts.")
    print("    Exclude from production anomaly detection.")
    print()
    print("  • conf_slope ↑ → rule is STRENGTHENING over time (most trustworthy)")
    print("  • conf_slope → → rule is FLAT (stable)")
    print("  • conf_slope ↓ → rule is DEGRADING — monitor closely")
    print()
    print(f"  PROPOSAL EVALUATION LINK:")
    print(f"  Of {total} total rules, {n_stable} ({100*n_stable/total:.1f}%) are STABLE —")
    if n_stable / total > 0.5:
        print("  this is a healthy result. Your static baseline is representative")
        print("  of genuine recurring behaviour, not sampling artefacts.")
    else:
        print("  this is lower than ideal. Consider:")
        print("  • Increasing MIN_SUPPORT to mine only stronger patterns")
        print("  • Reducing N_FOLDS if the dataset period is short")
        print("  • Checking for concept drift in the simulation parameters")
    print("=" * W)

    sup_cols = [f"sup_fold_{i}" for i in range(n_folds)]
    drift_slopes = []
    for _, row in stability_df.iterrows():
        vals = [row[c] for c in sup_cols if not pd.isna(row[c])]
        if len(vals) >= 2:
            slope = (vals[-1] - vals[0]) / max(len(vals) - 1, 1)
            drift_slopes.append(slope)

    n_rising = sum(1 for s in drift_slopes if s >  0.005)
    n_fading = sum(1 for s in drift_slopes if s < -0.005)
    n_flat   = len(drift_slopes) - n_rising - n_fading

    print()
    print("  REASSESSMENT RECOMMENDATION  (Proposal risk: rules becoming outdated)")
    print("  " + "-" * (W - 2))
    print(f"  Support drift summary   : {n_rising} rules rising  |  "
          f"{n_flat} rules flat  |  {n_fading} rules fading")
    if n_fading == 0:
        print("  Drift verdict           : No fading rules detected. The static")
        print("                            baseline appears representative over")
        print("                            the training period.")
    else:
        print(f"  Drift verdict           : {n_fading} rule(s) show declining support.")
        print("                            Investigate these before deploying.")
    print()
    print("  Recommended re-mining cadence:")
    print("    * Re-run pattern_stability_analysis.py monthly, or after any")
    print("      major setpoint schedule change in the building.")
    print("    * Trigger immediate re-mining if >10% of STABLE rules drop")
    print("      below stability_score 0.90 in a fresh fold analysis.")
    print("    * Replace static_rules.csv / stable_rules_only.csv with the")
    print("      newly mined outputs and restart the detection pipeline.")
    print("  " + "-" * (W - 2))
    print("=" * W)


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

def run_stability_analysis(
    train_csv_path: str = "simulation_data_multi_prev_train.csv",
    n_folds: int = N_FOLDS,
) -> pd.DataFrame:
    """
    Full pattern stability analysis pipeline.

    Parameters
    ----------
    train_csv_path : path to the training CSV (same file used by main pipeline)
    n_folds        : number of chronological folds (default 5)

    Returns
    -------
    stability_df   : DataFrame with per-rule stability metrics
                     (also saved to outputs/stability_report.csv)
    """
    print("\n" + "=" * 70)
    print("  PATTERN STABILITY ANALYSIS  —  building_hybrid_pipeline add-on")
    print("=" * 70)
    print(f"  Config:")
    print(f"    N_FOLDS                : {n_folds}")
    print(f"    STABLE_PREVALENCE_MIN  : {STABLE_PREVALENCE_MIN}")
    print(f"    STABLE_CONF_STD_MAX    : {STABLE_CONF_STD_MAX}")
    print(f"    MODERATE_PREVALENCE_MIN: {MODERATE_PREVALENCE_MIN}")
    print(f"    MIN_SUPPORT / MIN_CONF : {MIN_SUPPORT} / {MIN_CONFIDENCE}")
    print(f"  Fixes applied:")
    print(f"    FIX 1 — OO temperature computed per-window (not fold-global mean)")
    print(f"    FIX 2 — Temp delta computed as mean(np.diff(window)) per record")
    print(f"    NEW   — conf_slope column added (strengthening/degrading indicator)")

    # ── Step 1: Load & split ────────────────────────
    folds = load_and_split(train_csv_path, n_folds)

    # ── Step 2 & 3: Build transactions + mine rules per fold ───────────────────
    fold_rule_dfs: List[Optional[pd.DataFrame]] = []

    for fold_idx, fold_df in enumerate(folds):
        print(f"\n[Stability] Fold {fold_idx + 1}/{n_folds} — "
              f"rows {fold_idx * len(fold_df):,} … {(fold_idx+1) * len(fold_df):,}")
        t0 = time.time()

        transactions = build_transactions_from_fold(fold_df)
        print(f"  Transactions built : {len(transactions):,}")

        if not transactions:
            print("  WARNING: No transactions built for this fold. Skipping.")
            fold_rule_dfs.append(None)
            continue

        fold_rules = mine_rules_for_fold(transactions, fold_idx)

        if fold_rules is None or fold_rules.empty:
            print(f"  WARNING: No rules survived filtering for fold {fold_idx + 1}.")
            fold_rule_dfs.append(None)
        else:
            print(f"  Rules after filter : {len(fold_rules):,}  "
                  f"({time.time() - t0:.1f}s)")
            fold_rule_dfs.append(fold_rules)

    #Step 4: Compute stability metrics 
    print("\n[Stability] Computing stability metrics across folds...")
    stability_df = compute_stability_metrics(fold_rule_dfs, n_folds)

    # Save reports
    stability_df.to_csv(STABILITY_REPORT_PATH, index=False)
    print(f"  Full report saved  → {STABILITY_REPORT_PATH}")

    stable_only = stability_df[stability_df["label"] == "STABLE"].copy()
    stable_only.to_csv(STABLE_RULES_PATH, index=False)
    print(f"  Stable rules saved → {STABLE_RULES_PATH}  ({len(stable_only):,} rules)")

    # Step 5: Visualisations 
    print("\n[Stability] Generating visualisations...")
    plot_stability_heatmap(stability_df, n_folds)
    plot_support_drift(stability_df, n_folds)

    #Step 6: Terminal summary 
    print_summary(stability_df, n_folds)

    return stability_df

def load_stable_rules_as_dataframe() -> Optional[pd.DataFrame]:
    """
    Loads the stable_rules_only.csv produced by run_stability_analysis()
    and returns it in the same format expected by StreamingDetector.

    NOTE: Per project design, this is for reference only. The main pipeline
    uses RuleFilter output (filtered_rules.csv) for detection. Stable rules
    serve as a validation cross-check, not a detection input.
    """
    if not os.path.exists(STABLE_RULES_PATH):
        print(f"[Warning] {STABLE_RULES_PATH} not found. "
              "Run run_stability_analysis() first.")
        return None

    df = pd.read_csv(STABLE_RULES_PATH)

    def parse_items(s: str) -> frozenset:
        return frozenset(item.strip() for item in s.split("&"))

    df["antecedents"] = df["antecedents"].apply(parse_items)
    df["consequents"] = df["consequents"].apply(parse_items)
    df = df.rename(columns={"conf_mean": "confidence", "sup_mean": "support"})

    if "lift_mean" in df.columns:
        df["lift"] = df["lift_mean"]

    print(f"[Stability] Loaded {len(df):,} stable rules from {STABLE_RULES_PATH}")
    return df


# =============================================================================
#  STANDALONE ENTRY POINT (Optional)
# =============================================================================

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "simulation_data_multi_prev_train.csv"
    n_folds  = int(sys.argv[2]) if len(sys.argv) > 2 else N_FOLDS

    stability_df = run_stability_analysis(
        train_csv_path=csv_path,
        n_folds=n_folds,
    )