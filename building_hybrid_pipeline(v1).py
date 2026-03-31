"""
=============================================================================
Detecting Inconsistent Behavior in Building Heating Systems  —  v2.1
=============================================================================

CHANGES FROM v1  (feedback after progress presentation)
────────────────────────────────────────────────────────
 1. OUTSIDE TEMPERATURE (OO room) is now included as a contextual feature.
    The OO sensor's temperature is joined into every indoor-room transaction
    as OO_COLD / OO_NORMAL / OO_HOT / OO_VERY_HOT so rules can reference
    external conditions (e.g. HEATER_ON is normal when OO_COLD).

 2. AVERAGE-BASED TRANSACTION GENERATION.
    Instead of discretizing every record and union-ing the labels
    (which produces contradictory items like TEMP_COLD + TEMP_HOT in the
    same transaction), we now:
      a) Collect raw numeric values for the 1-minute window (6 records).
      b) Compute the mean temperature, mean actuation, mean setpoint,
         mean temp_delta and mean act_delta over the window.
      c) Discretize the AVERAGES once → one clean label per dimension.
    This gives one unambiguous symbolic description per window.

 3. FREQUENCY-WEIGHTED PATTERN RECOGNITION.
    Each item is tracked by how often it appears across all training
    transactions.  During rule scoring, items whose support is < a
    threshold (RARE_ITEM_SUPPORT) are treated as "rare context" and
    violations involving rare antecedents contribute less to the score.

 4. CONFIDENCE + SUPPORT WEIGHTED ANOMALY SCORE.
    Old score  = violated / applicable   (unweighted ratio)
    New score  = Σ(conf_i × support_i  for violated rules i)
               / Σ(conf_i × support_i  for applicable rules i)
    This means a high-confidence, high-frequency rule violation hurts
    more than a low-confidence edge case.

 5. TIGHTER RULE FILTERING to reduce the active rule set.
    Added two new filters on top of the existing ones:
      • MIN_ANTECEDENT_SUPPORT  — antecedent itemset must itself appear
                                  frequently enough to be reliable.
      • MAX_RULE_SET_SIZE       — hard cap: keep only the top-N rules
                                  ranked by (confidence × lift) after
                                  all other filters pass.

PIPELINE:
  RAW DATA → PREPROCESSING → OO-JOIN → AVERAGE-DISCRETIZATION
           → TRANSACTIONS → RULE MINING → RULE FILTER
           → WEIGHTED STREAMING DETECTION → ALERTS

Requirements:
    pip install pandas numpy mlxtend tqdm
=============================================================================
"""

import os
import sys
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from tqdm import tqdm

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# =============================================================================
#  CONFIGURATION
# =============================================================================

TRAIN_PATH = "simulation_data_multi_prev_train.csv"
TEST_PATH  = "simulation_data_multi_prev_test.csv"

# --- Outside temperature discretization (OO room, in °C after K→C convert) ---
OO_COLD_MAX    = 15.0   # below → OO_COLD
OO_NORMAL_MAX  = 20.0   # 15–20 → OO_NORMAL
OO_WARM_MAX    = 25.0   # 20–25 → OO_WARM
                        # above → OO_HOT

# --- Indoor temperature discretization ---------------------------------------
TEMP_COLD_MAX     = 19.0    # °C
TEMP_HOT_MIN      = 22.0    # °C
TREND_STABLE_BAND = 0.01    # |mean_delta| below → STABLE
SP_DEV_BAND       = 1.5     # °C setpoint deviation band

# --- Transaction (1-minute window) -------------------------------------------
TRANSACTION_SIZE  = 6       # records per room per transaction (~10 s × 6 = 60 s)

# --- Rule mining (runs ONCE on training transactions) ------------------------
MIN_SUPPORT       = 0.10
MIN_CONFIDENCE    = 0.92   # raised from 0.88 — only very reliable rules

# --- Rule filter -------------------------------------------------------------
MIN_LIFT               = 1.5
MIN_LEVERAGE           = 0.02
MIN_CONVICTION         = 1.2
MAX_CON_SUPPORT        = 0.70   # lowered from 0.80 — TEMP_HOT (sup~0.54) and
                                 # AT_SETPOINT (sup~0.91) are now filtered out
                                 # as consequents, removing the dominant false
                                 # positive rules (HEATER_ON → TEMP_HOT etc.)
MAX_CONSEQUENT_SIZE    = 2
MIN_ANTECEDENT_SUPPORT = 0.08   # NEW: antecedent itemset must appear in ≥8% of txns
MAX_RULE_SET_SIZE      = 500    # NEW: hard cap — keep top-N by (confidence × lift)

# --- Frequency-weighted scoring ----------------------------------------------
# Items whose training support < this are "rare context" — violations of
# rules that rely on rare antecedents are down-weighted by RARE_WEIGHT.
RARE_ITEM_SUPPORT = 0.05
RARE_WEIGHT       = 0.4   # rare-antecedent violation counts this fraction

# --- Anomaly severity thresholds (weighted score) ----------------------------
# Score distribution is bimodal: normal txns cluster near 0.0, anomalies
# cluster near 0.35 / 0.65 / 1.0.  Thresholds are raised accordingly so
# the vast majority of normal behaviour is not flagged as LOW.
SEV_LOW_MIN    = 0.30   # was 0.20 — scores 0.0–0.30 are near-normal transitions
SEV_MED_MIN    = 0.55   # was 0.40 — genuine partial anomalies
SEV_HIGH_MIN   = 0.85   # was 0.65 — near-total rule violation (true anomaly)

# --- Output ------------------------------------------------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALERTS_PATH      = os.path.join(OUTPUT_DIR, "alerts_log.csv")
RULES_PATH       = os.path.join(OUTPUT_DIR, "static_rules.csv")
SCORES_PATH      = os.path.join(OUTPUT_DIR, "all_scores.csv")
CHARTS_PATH      = os.path.join(OUTPUT_DIR, "anomaly_dashboard.png")

NROWS_TRAIN = None
NROWS_TEST  = None


# =============================================================================
#  DATA CLASSES
# =============================================================================

@dataclass
class StreamRecord:
    """One raw sensor reading parsed from a CSV row."""
    timestamp:   pd.Timestamp
    room:        str
    temperature: float
    actuation:   float
    setpoint:    float
    prev_act_1:  float = 0.0
    prev_act_2:  float = 0.0
    prev_act_3:  float = 0.0
    is_outside:  bool  = False   # True when room == "OO"


@dataclass
class Transaction:
    """Symbolic itemset representing one room over TRANSACTION_SIZE records."""
    room:      str
    timestamp: pd.Timestamp
    items:     frozenset


@dataclass
class ScoredTransaction:
    """A transaction scored against the static rule set."""
    transaction_id:   int
    room:             str
    timestamp:        pd.Timestamp
    items:            frozenset
    anomaly_score:    float
    violated_count:   int
    applicable_count: int
    severity:         str
    violated_rules:   list   # [(antecedent, consequent, confidence, support), ...]


# =============================================================================
#  STAGE 1 — RAW DATA STREAM
# =============================================================================

class RawDataStream:
    """
    Yields one StreamRecord at a time (including OO records).
    OO records are flagged with is_outside=True so consumers can route them.
    """

    CHUNKSIZE = 50_000

    _COL_MAP = {
        "datetime":  "timestamp",
        "timeline":  "elapsed_s",
        "room_name": "room",
    }

    def __init__(self, path: str, nrows: Optional[int] = None):
        self.path  = path
        self.nrows = nrows

    def stream(self) -> Iterator[StreamRecord]:
        reader = pd.read_csv(
            self.path, sep=";", chunksize=self.CHUNKSIZE,
            nrows=self.nrows, low_memory=False,
        )
        for chunk in reader:
            chunk = self._clean_chunk(chunk)
            for _, row in chunk.iterrows():
                yield self._row_to_record(row)

    _columns_printed = False

    def _clean_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns.str.strip()
                      .str.lower()
                      .str.replace(" ", "_")
                      .str.replace("-", "_")
        )
        if not RawDataStream._columns_printed:
            print(f"\n[Stream] Raw columns: {list(df.columns)}")
            RawDataStream._columns_printed = True

        df = df.rename(columns=self._COL_MAP)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            df["timestamp"] = pd.NaT

        if "room" not in df.columns:
            candidate = next(
                (c for c in df.columns if "room" in c or "name" in c), None
            )
            df["room"] = df[candidate] if candidate else "UNKNOWN"

        for i, variants in enumerate(
            [["prev_act_1", "prev_actuation_1", "previous_actuation_1"],
             ["prev_act_2", "prev_actuation_2", "previous_actuation_2"],
             ["prev_act_3", "prev_actuation_3", "previous_actuation_3"]],
            start=1,
        ):
            canonical = f"prev_act_{i}"
            found = next((v for v in variants if v in df.columns), None)
            if found and found != canonical:
                df.rename(columns={found: canonical}, inplace=True)

        for col in ["temperature", "actuation", "setpoint",
                    "prev_act_1", "prev_act_2", "prev_act_3"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0

        # Kelvin → Celsius
        if df["temperature"].median() > 200:
            df["temperature"] = df["temperature"] - 273.15
        if "setpoint" in df.columns and df["setpoint"].median() > 200:
            df["setpoint"] = df["setpoint"] - 273.15

        df.dropna(subset=["temperature"], inplace=True)
        # NOTE: OO rows are now KEPT — no longer filtered here
        return df

    def _row_to_record(self, row) -> StreamRecord:
        room = str(row.get("room", "UNKNOWN"))
        return StreamRecord(
            timestamp   = row.get("timestamp", pd.NaT),
            room        = room,
            temperature = float(row["temperature"]),
            actuation   = float(row["actuation"]),
            setpoint    = float(row["setpoint"]),
            prev_act_1  = float(row["prev_act_1"]),
            prev_act_2  = float(row["prev_act_2"]),
            prev_act_3  = float(row["prev_act_3"]),
            is_outside  = (room == "OO"),
        )


# =============================================================================
#  STAGE 2 — PREPROCESSING
#  Derives trend features per record.  OO records contribute only temperature.
# =============================================================================

class StreamPreprocessor:
    """Per-room stateful preprocessor. Returns enriched dict."""

    def __init__(self):
        self._prev: Dict[str, StreamRecord] = {}

    def process(self, rec: StreamRecord) -> dict:
        prev = self._prev.get(rec.room)
        temp_delta = (rec.temperature - prev.temperature) if prev else 0.0
        act_delta  = (rec.actuation  - rec.prev_act_1)
        sp_deviation = rec.temperature - rec.setpoint

        self._prev[rec.room] = rec

        return {
            "room":         rec.room,
            "timestamp":    rec.timestamp,
            "temperature":  rec.temperature,
            "actuation":    rec.actuation,
            "setpoint":     rec.setpoint,
            "temp_delta":   temp_delta,
            "act_delta":    act_delta,
            "sp_deviation": sp_deviation,
            "is_outside":   rec.is_outside,
        }


# =============================================================================
#  STAGE 3 — AVERAGE-BASED DISCRETIZER
#
#  CHANGE FROM v1:
#    v1 discretized each record and unioned all labels.
#    v2 accumulates raw numerics for the window, then:
#       • computes the window mean for temperature, actuation, setpoint,
#         temp_delta, act_delta
#       • discretizes the MEANS → one label per dimension
#    Result: no contradictory items, one clean symbolic snapshot.
#
#  For OO records only the mean temperature is discretized (OO_* label).
# =============================================================================

def _discretize_oo(mean_temp: float) -> str:
    """Classify outside mean temperature into 4 symbolic bins."""
    if mean_temp < OO_COLD_MAX:
        return "OO_COLD"
    elif mean_temp < OO_NORMAL_MAX:
        return "OO_NORMAL"
    elif mean_temp < OO_WARM_MAX:
        return "OO_WARM"
    return "OO_HOT"


def _trend_label(prefix: str, mean_delta: float) -> str:
    if mean_delta > TREND_STABLE_BAND:
        return f"{prefix}_INCREASING"
    elif mean_delta < -TREND_STABLE_BAND:
        return f"{prefix}_DECREASING"
    return f"{prefix}_STABLE"


def discretize_window_averages(
    records: List[dict],
    oo_mean_temp: Optional[float],
) -> List[str]:
    """
    Given a list of preprocessed dicts for ONE indoor room over the window,
    compute per-dimension means and return a clean symbolic item list.
    Also appends the OO_* label if outside temperature is available.
    """
    if not records:
        return []

    mean_temp    = float(np.mean([r["temperature"]  for r in records]))
    mean_act     = float(np.mean([r["actuation"]    for r in records]))
    mean_sp_dev  = float(np.mean([r["sp_deviation"] for r in records]))
    mean_tdelta  = float(np.mean([r["temp_delta"]   for r in records]))
    mean_adelta  = float(np.mean([r["act_delta"]    for r in records]))

    items: List[str] = []

    # Temperature state
    if mean_temp < TEMP_COLD_MAX:
        items.append("TEMP_COLD")
    elif mean_temp > TEMP_HOT_MIN:
        items.append("TEMP_HOT")
    else:
        items.append("TEMP_NORMAL")

    # Temperature trend (based on average change per step)
    items.append(_trend_label("TEMP", mean_tdelta))

    # Heater state (mean actuation > 0.01 → ON)
    items.append("HEATER_ON" if mean_act > 0.01 else "HEATER_OFF")

    # Actuation trend
    items.append(_trend_label("ACT", mean_adelta))

    # Setpoint deviation
    if mean_sp_dev < -SP_DEV_BAND:
        items.append("BELOW_SETPOINT")
    elif mean_sp_dev > SP_DEV_BAND:
        items.append("ABOVE_SETPOINT")
    else:
        items.append("AT_SETPOINT")

    # Outside temperature context  ← NEW
    if oo_mean_temp is not None:
        items.append(_discretize_oo(oo_mean_temp))

    return items


# =============================================================================
#  STAGE 4 — TRANSACTION BUILDER  (average-based, tumbling 1-min windows)
#
#  Architecture:
#    • One RoomBuffer per indoor room accumulates raw preprocessed dicts.
#    • A shared OOBuffer holds the latest OO temperature readings so the
#      current 1-min window can look up the outside temperature at flush time.
#    • On flush, discretize_window_averages() is called once per window.
# =============================================================================

class OOBuffer:
    """
    Keeps a rolling deque of the most recent OO temperature readings.
    Returns the mean over the last TRANSACTION_SIZE readings.
    """

    def __init__(self, maxlen: int = TRANSACTION_SIZE * 2):
        self._buf: deque = deque(maxlen=maxlen)

    def push(self, temp: float):
        self._buf.append(temp)

    def current_mean(self) -> Optional[float]:
        if not self._buf:
            return None
        return float(np.mean(self._buf))


class RoomBuffer:
    """
    Per-indoor-room buffer.  Accumulates raw preprocessed dicts.
    On flush, averages are computed and discretized using the latest
    outside temperature from the shared OOBuffer.
    """

    def __init__(self, room: str, oo_buffer: OOBuffer):
        self.room       = room
        self._oo_buffer = oo_buffer
        self._buf: List[dict] = []

    def push(
        self, timestamp: pd.Timestamp, record: dict
    ) -> Optional[Transaction]:
        self._buf.append(record)
        if len(self._buf) >= TRANSACTION_SIZE:
            tx = self._flush(timestamp)
            self._buf = []
            return tx
        return None

    def _flush(self, timestamp: pd.Timestamp) -> Transaction:
        oo_temp = self._oo_buffer.current_mean()
        items   = discretize_window_averages(self._buf, oo_temp)
        return Transaction(
            room=self.room,
            timestamp=timestamp,
            items=frozenset(items),
        )


class TransactionBuilder:
    """Manages one RoomBuffer per indoor room + one shared OOBuffer."""

    def __init__(self):
        self._oo_buffer  = OOBuffer()
        self._buffers: Dict[str, RoomBuffer] = {}

    def push(
        self, rec_dict: dict, timestamp: pd.Timestamp
    ) -> Optional[Transaction]:
        room       = rec_dict["room"]
        is_outside = rec_dict["is_outside"]

        if is_outside:
            self._oo_buffer.push(rec_dict["temperature"])
            return None

        if room not in self._buffers:
            self._buffers[room] = RoomBuffer(room, self._oo_buffer)

        return self._buffers[room].push(timestamp, rec_dict)


# =============================================================================
#  STAGE 5a — STATIC RULE MINER
# =============================================================================

class StaticRuleMiner:
    """
    Collects ALL training transactions, runs FP-Growth ONCE, freezes rules.
    Also tracks per-item support for frequency-weighted scoring.
    """

    def __init__(self):
        self._transactions: List[List[str]] = []
        self.rules:         Optional[pd.DataFrame] = None
        self.item_support:  Dict[str, float] = {}   # item → support in [0,1]
        self._mined = False

    def collect(self, tx: Transaction):
        self._transactions.append(list(tx.items))

    def mine(self) -> pd.DataFrame:
        if self._mined:
            raise RuntimeError("mine() already called.")

        n = len(self._transactions)
        print(f"\n[Rule Mining] FP-Growth on {n:,} training transactions...")
        print(f"  min_support={MIN_SUPPORT}, min_confidence={MIN_CONFIDENCE}")

        te       = TransactionEncoder()
        te_array = te.fit_transform(self._transactions)
        te_df    = pd.DataFrame(te_array, columns=te.columns_)

        # ── item support (frequency) ──────────────────────────────────────────
        # Each column is a boolean; mean() gives the fraction of transactions
        # that contain that item.
        self.item_support = {col: float(te_df[col].mean()) for col in te_df.columns}
        print(f"  Unique items tracked       : {len(self.item_support):,}")

        freq  = fpgrowth(te_df, min_support=MIN_SUPPORT, use_colnames=True)
        print(f"  Frequent itemsets found    : {len(freq):,}")

        rules = association_rules(
            freq, metric="confidence", min_threshold=MIN_CONFIDENCE
        )
        rules = rules.sort_values("confidence", ascending=False).reset_index(drop=True)
        print(f"  Rules generated            : {len(rules):,}")

        self.rules  = rules
        self._mined = True

        # Save all raw rules
        rules_export = rules.copy()
        rules_export["antecedents"] = rules_export["antecedents"].apply(
            lambda x: " & ".join(sorted(x))
        )
        rules_export["consequents"] = rules_export["consequents"].apply(
            lambda x: " & ".join(sorted(x))
        )
        rules_export.to_csv(RULES_PATH, index=False)
        print(f"  Raw rules saved            : {RULES_PATH}")

        print("\n  Top 10 raw rules by confidence:")
        for _, row in rules.head(10).iterrows():
            ant = " & ".join(sorted(row["antecedents"]))
            con = " & ".join(sorted(row["consequents"]))
            print(f"    [{ant}] -> [{con}]  "
                  f"conf={row['confidence']:.3f}  "
                  f"sup={row['support']:.3f}  "
                  f"lift={row['lift']:.2f}")

        return rules

    @property
    def is_ready(self) -> bool:
        return self._mined and self.rules is not None and len(self.rules) > 0


# =============================================================================
#  STAGE 5b — RULE FILTER  (expanded with 2 new filters from feedback)
#
#  New filters added in v2:
#   7. MIN_ANTECEDENT_SUPPORT — antecedent itemset's OWN support must be ≥
#      MIN_ANTECEDENT_SUPPORT.  This removes rules built on rare premises.
#   8. MAX_RULE_SET_SIZE      — after all quality filters, keep only the top-N
#      rules ranked by confidence × lift (most informative & reliable first).
# =============================================================================

_DOMAINS = {
    "TEMPERATURE": {"TEMP_COLD", "TEMP_NORMAL", "TEMP_HOT"},
    "TEMP_TREND":  {"TEMP_INCREASING", "TEMP_STABLE", "TEMP_DECREASING"},
    "HEATER":      {"HEATER_ON", "HEATER_OFF"},
    "ACT_TREND":   {"ACT_INCREASING", "ACT_STABLE", "ACT_DECREASING"},
    "SETPOINT":    {"BELOW_SETPOINT", "AT_SETPOINT", "ABOVE_SETPOINT"},
    "OUTSIDE":     {"OO_COLD", "OO_NORMAL", "OO_WARM", "OO_HOT"},  # NEW domain
}


def _get_domains(items: frozenset) -> Set[str]:
    domains = set()
    for item in items:
        for domain, members in _DOMAINS.items():
            if item in members:
                domains.add(domain)
    return domains


class RuleFilter:
    """
    Multi-stage filter pipeline.  v2 adds antecedent-support and size-cap filters.
    """

    def apply(self, rules: pd.DataFrame) -> pd.DataFrame:
        original = len(rules)
        print(f"\n[RuleFilter] Starting with {original:,} rules...")

        rules, n_consize    = self._filter_consequent_size(rules)
        rules, n_lift       = self._filter_lift(rules)
        rules, n_leverage   = self._filter_leverage(rules)
        rules, n_conviction = self._filter_conviction(rules)
        rules, n_trivial    = self._filter_trivial_consequent(rules)
        rules, n_domain     = self._filter_cross_domain(rules)
        rules, n_ant        = self._filter_antecedent_support(rules)
        rules, n_normal     = self._filter_normal_heating(rules)
        rules, n_cap        = self._filter_top_n(rules)

        kept = len(rules)
        print(f"\n  Filter summary:")
        print(f"    Removed by consequent size      : {n_consize:>6,}  (<=  {MAX_CONSEQUENT_SIZE} items)")
        print(f"    Removed by lift filter          : {n_lift:>6,}  (> {MIN_LIFT})")
        print(f"    Removed by leverage filter      : {n_leverage:>6,}  (> {MIN_LEVERAGE})")
        print(f"    Removed by conviction filter    : {n_conviction:>6,}  (> {MIN_CONVICTION})")
        print(f"    Removed by trivial consequent   : {n_trivial:>6,}  (con. sup < {MAX_CON_SUPPORT})")
        print(f"    Removed by cross-domain filter  : {n_domain:>6,}  (2+ domains required)")
        print(f"    Removed by antecedent support   : {n_ant:>6,}  (ant. sup >= {MIN_ANTECEDENT_SUPPORT})")
        print(f"    Removed by normal-heating filter: {n_normal:>6,}  (HEATER_ON->TEMP_HOT/INCREASING are normal)")
        print(f"    Removed by top-N cap            : {n_cap:>6,}  (top {MAX_RULE_SET_SIZE} by conf*lift)")
        print(f"    {'─'*46}")
        print(f"    Total removed                   : {original - kept:>6,}")
        print(f"    Rules remaining                 : {kept:>6,}")

        if kept == 0:
            print("\n  WARNING: All rules were filtered out! Relax CONFIG thresholds.")
        else:
            print(f"\n  Top 10 rules after filtering (ranked by conf×lift):")
            rules_sorted = rules.sort_values(
                by=["confidence", "lift"], ascending=False
            )
            for _, row in rules_sorted.head(10).iterrows():
                ant = " & ".join(sorted(row["antecedents"]))
                con = " & ".join(sorted(row["consequents"]))
                print(f"    [{ant}]")
                print(f"      -> [{con}]")
                print(f"      conf={row['confidence']:.3f}  sup={row['support']:.3f}  "
                      f"lift={row['lift']:.3f}  leverage={row['leverage']:.4f}")

        filtered_path = os.path.join(OUTPUT_DIR, "filtered_rules.csv")
        export = rules.copy()
        export["antecedents"] = export["antecedents"].apply(
            lambda x: " & ".join(sorted(x))
        )
        export["consequents"] = export["consequents"].apply(
            lambda x: " & ".join(sorted(x))
        )
        export.to_csv(filtered_path, index=False)
        print(f"\n  Filtered rules saved: {filtered_path}")

        return rules.reset_index(drop=True)

    # ── individual filters ────────────────────────────────────────────────────

    def _filter_consequent_size(self, rules):
        before = len(rules)
        rules  = rules[rules["consequents"].apply(len) <= MAX_CONSEQUENT_SIZE]
        return rules, before - len(rules)

    def _filter_lift(self, rules):
        before = len(rules)
        rules  = rules[rules["lift"] > MIN_LIFT]
        return rules, before - len(rules)

    def _filter_leverage(self, rules):
        before = len(rules)
        rules  = rules[rules["leverage"] > MIN_LEVERAGE]
        return rules, before - len(rules)

    def _filter_conviction(self, rules):
        before = len(rules)
        rules  = rules[
            (rules["conviction"] > MIN_CONVICTION) |
            (rules["conviction"] == float("inf"))
        ]
        return rules, before - len(rules)

    def _filter_trivial_consequent(self, rules):
        before = len(rules)
        rules  = rules[rules["consequent support"] < MAX_CON_SUPPORT]
        return rules, before - len(rules)

    def _filter_cross_domain(self, rules):
        before = len(rules)
        mask = []
        for _, row in rules.iterrows():
            all_domains = _get_domains(row["antecedents"]) | _get_domains(row["consequents"])
            mask.append(len(all_domains) >= 2)
        rules = rules[mask]
        return rules, before - len(rules)

    def _filter_antecedent_support(self, rules):
        """Drop rules where the antecedent itemset is too rare."""
        before = len(rules)
        rules  = rules[rules["antecedent support"] >= MIN_ANTECEDENT_SUPPORT]
        return rules, before - len(rules)

    def _filter_normal_heating(self, rules):
        """
        Drop rules that encode NORMAL heating behaviour and will therefore
        fire constantly during regular operation, producing false positives.

        Patterns removed:
          A) HEATER_ON (any antecedent) -> TEMP_HOT or TEMP_INCREASING
             A heater that is ON driving temperature up is the expected
             physical response — never an anomaly by itself.

          B) HEATER_ON (any antecedent) -> AT_SETPOINT
             Being at setpoint while the heater is on is a normal steady-
             state (the heater is maintaining temperature).

          C) TEMP_NORMAL/TEMP_COLD -> HEATER_OFF  when nothing else is violated
             These are too general: a room at normal/cold temperature with
             the heater off is only anomalous in specific sub-conditions
             already captured by more specific rules (e.g. with OO context).
             Only remove pure HEATER_OFF consequents with no OO context in
             the antecedent AND the antecedent is just TEMP_* alone (too weak).

        A rule is kept if its antecedent contains HEATER_ON AND its consequent
        is anything OTHER than the normal-response items listed above.
        """
        before = len(rules)

        def _is_normal(row) -> bool:
            ant = set(row["antecedents"])
            con = set(row["consequents"])

            # Pattern A & B: HEATER_ON present in antecedent
            if "HEATER_ON" in ant:
                # Consequent is purely a "heater working as expected" label
                normal_con = {"TEMP_HOT", "TEMP_INCREASING", "AT_SETPOINT"}
                if con.issubset(normal_con):
                    return True

            # Pattern C: very weak antecedent (only TEMP_* items, no OO / setpoint)
            # with HEATER_OFF as the sole consequent — too imprecise
            ant_domains = _get_domains(frozenset(ant))
            con_domains = _get_domains(frozenset(con))
            if (con == {"HEATER_OFF"}
                    and ant_domains == {"TEMPERATURE"}     # only temp items in ant
                    and "OUTSIDE" not in ant_domains):     # no OO context
                return True

            return False

        mask = [not _is_normal(row) for _, row in rules.iterrows()]
        rules = rules[mask]
        return rules, before - len(rules)

    def _filter_top_n(self, rules):
        """NEW: keep only the top MAX_RULE_SET_SIZE rules by confidence × lift."""
        before = len(rules)
        if len(rules) <= MAX_RULE_SET_SIZE:
            return rules, 0
        rules = (
            rules
            .assign(_score=rules["confidence"] * rules["lift"])
            .sort_values("_score", ascending=False)
            .head(MAX_RULE_SET_SIZE)
            .drop(columns=["_score"])
        )
        return rules, before - len(rules)


# =============================================================================
#  STAGE 6 — STREAMING DETECTOR  (confidence × support weighted scoring)
#
#  CHANGE FROM v1:
#    v1 score = violated / applicable   (pure rule-count ratio)
#
#    v2 score = Σ(weight_i  for violated rule i)
#              / Σ(weight_i  for applicable rule i)
#
#    where weight_i = confidence_i × support_i
#
#    Rules with high confidence AND high support (frequent, reliable patterns)
#    carry more weight — a violation of such a rule is a stronger anomaly
#    signal than violating a low-confidence, rare rule.
#
#    Additionally, if ALL items in a rule's antecedent have support < RARE_ITEM_SUPPORT
#    (i.e. the triggering condition is very rare), the violation weight is
#    multiplied by RARE_WEIGHT to avoid over-sensitive scoring on edge cases.
# =============================================================================

def _severity(score: float) -> str:
    if score >= SEV_HIGH_MIN:
        return "HIGH"
    elif score >= SEV_MED_MIN:
        return "MEDIUM"
    elif score >= SEV_LOW_MIN:
        return "LOW"
    return "NONE"


class StreamingDetector:
    """
    Scores each transaction against the frozen rule set using
    confidence × support weighted anomaly scoring.
    """

    def __init__(self, rules: pd.DataFrame, item_support: Dict[str, float]):
        self._tx_counter = 0
        self._item_support = item_support
        print(f"  [Detector] Pre-compiling {len(rules):,} rules... ", end="", flush=True)

        self._compiled: List[Tuple[frozenset, frozenset, float, float]] = [
            (
                frozenset(row["antecedents"]),
                frozenset(row["consequents"]),
                float(row["confidence"]),
                float(row["support"]),
            )
            for _, row in rules.iterrows()
        ]
        print("done.")

    def _rule_weight(self, ant: frozenset, conf: float, sup: float) -> float:
        """
        Base weight = confidence × support.
        Down-weighted if ALL antecedent items are individually rare.
        """
        base = conf * sup
        ant_supports = [self._item_support.get(item, 0.0) for item in ant]
        if ant_supports and max(ant_supports) < RARE_ITEM_SUPPORT:
            base *= RARE_WEIGHT
        return base

    def score(self, tx: Transaction) -> ScoredTransaction:
        self._tx_counter += 1
        tx_items = tx.items
        total_weight    = 0.0
        violated_weight = 0.0
        violated_rules  = []
        applicable_count = 0
        violated_count   = 0

        for ant, con, conf, sup in self._compiled:
            if ant.issubset(tx_items):
                w = self._rule_weight(ant, conf, sup)
                applicable_count += 1
                total_weight     += w
                if not con.issubset(tx_items):
                    violated_count   += 1
                    violated_weight  += w
                    violated_rules.append((ant, con, conf, sup))

        anomaly_score = violated_weight / total_weight if total_weight > 0 else 0.0

        return ScoredTransaction(
            transaction_id   = self._tx_counter,
            room             = tx.room,
            timestamp        = tx.timestamp,
            items            = tx.items,
            anomaly_score    = round(anomaly_score, 4),
            violated_count   = violated_count,
            applicable_count = applicable_count,
            severity         = _severity(anomaly_score),
            violated_rules   = violated_rules,
        )


# =============================================================================
#  STAGE 7 — ALERT WRITER
# =============================================================================

class AlertWriter:

    _COLUMNS = [
        "transaction_id", "timestamp", "room", "severity", "anomaly_score",
        "violated_count", "applicable_count", "top_violated_rule",
    ]

    def __init__(self, path: str = ALERTS_PATH):
        self.path   = path
        self._file  = open(path, "w", buffering=1, encoding="utf-8")
        self._file.write(",".join(self._COLUMNS) + "\n")
        self._count = 0

    def write(self, st: ScoredTransaction):
        if st.severity == "NONE":
            return
        top_rule = ""
        if st.violated_rules:
            ant, con, conf, sup = st.violated_rules[0]
            top_rule = (
                f"[{' & '.join(sorted(ant))}] -> "
                f"[{' & '.join(sorted(con))}] "
                f"(conf={conf:.3f}, sup={sup:.3f})"
            )
        row = [
            st.transaction_id, str(st.timestamp), st.room, st.severity,
            st.anomaly_score, st.violated_count, st.applicable_count,
            f'"{top_rule}"',
        ]
        self._file.write(",".join(str(v) for v in row) + "\n")
        self._count += 1

    def close(self):
        self._file.close()
        print(f"\n[Alerts] {self._count:,} alerts written → {self.path}")

    @property
    def alert_count(self) -> int:
        return self._count


# =============================================================================
#  PIPELINE ORCHESTRATOR
# =============================================================================

class Pipeline:
    """
    Phase 1 — Training:
      Stream → Preprocessor → OO-Buffer (outside temp) + RoomBuffer
      → Average-discretize on flush → StaticRuleMiner

    Phase 2 — Detection:
      Stream → same preprocessing → TransactionBuilder
      → Weighted StreamingDetector → AlertWriter
    """

    def __init__(self):
        self.preprocessor = StreamPreprocessor()

    # ── Phase 1: Training ──────────────────────────────────────────────────────

    def run_training(self) -> StaticRuleMiner:
        print("\n" + "=" * 60)
        print("  PHASE 1 — TRAINING STREAM  (v2: average-based transactions)")
        print("=" * 60)

        source  = RawDataStream(TRAIN_PATH, nrows=NROWS_TRAIN)
        builder = TransactionBuilder()
        miner   = StaticRuleMiner()

        rec_count, tx_count = 0, 0
        oo_count = 0
        t0 = time.time()
        _sample_shown = False

        for rec in source.stream():
            rec_count += 1
            processed = self.preprocessor.process(rec)

            if rec.is_outside:
                oo_count += 1
                # Feed OO temp into OO buffer only (no transaction built)
                builder.push(processed, rec.timestamp)
                continue

            tx = builder.push(processed, rec.timestamp)
            if tx is not None:
                miner.collect(tx)
                tx_count += 1

                if not _sample_shown:
                    _sample_shown = True
                    print(f"\n  [Sample transaction — room {tx.room}]")
                    print(f"    Items (avg-discretized): {sorted(tx.items)}")
                    print(f"    OO context included    : {'OO_' in str(tx.items)}")
                    print()

            if rec_count % 200_000 == 0:
                print(f"  → {rec_count:>8,} records | "
                      f"{oo_count:>6,} OO records | "
                      f"{tx_count:>6,} transactions | "
                      f"{time.time()-t0:.1f}s")

        print(f"\n  Training complete.")
        print(f"  Records processed    : {rec_count:,}")
        print(f"  OO records used      : {oo_count:,}")
        print(f"  Transactions built   : {tx_count:,}")

        miner.mine()
        return miner

    # ── Phase 2: Detection ─────────────────────────────────────────────────────

    def run_detection(self, miner: StaticRuleMiner) -> List[ScoredTransaction]:
        print("\n" + "=" * 60)
        print("  PHASE 2 — STREAMING DETECTION  (v2: weighted scoring)")
        print("=" * 60)

        source  = RawDataStream(TEST_PATH, nrows=NROWS_TEST)
        builder = TransactionBuilder()

        filtered_rules = RuleFilter().apply(miner.rules)
        if len(filtered_rules) == 0:
            print("\n  WARNING: No rules after filtering — using unfiltered rules.")
            filtered_rules = miner.rules

        detector = StreamingDetector(filtered_rules, miner.item_support)
        alerter  = AlertWriter()

        # Reset preprocessor state for fresh test stream
        self.preprocessor = StreamPreprocessor()

        all_scores: List[ScoredTransaction] = []
        rec_count, tx_count, alert_count = 0, 0, 0
        t0 = time.time()

        SEV_COLOR = {"HIGH": "\033[91m", "MEDIUM": "\033[93m", "LOW": "\033[96m"}
        RESET = "\033[0m"
        BOLD  = "\033[1m"

        room_counts: Dict[str, Dict[str, int]] = {}
        _reported: set = set()

        print(f"\n  {'TIMESTAMP':<22} {'ROOM':<8} {'SEV':<8} "
              f"{'SCORE':>6}  VIOLATED RULE")
        print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*6}  {'-'*45}")

        for rec in source.stream():
            rec_count += 1
            processed = self.preprocessor.process(rec)
            tx = builder.push(processed, rec.timestamp)

            if tx is None:
                continue

            tx_count += 1
            scored = detector.score(tx)
            all_scores.append(scored)

            if scored.severity != "NONE":
                alert_count += 1

                if scored.room not in room_counts:
                    room_counts[scored.room] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
                room_counts[scored.room][scored.severity] += 1

                if pd.notna(scored.timestamp):
                    bucket = (scored.timestamp.hour // 3) * 3
                    ts_key = f"{str(scored.timestamp)[:10]}_{bucket:02d}h"
                else:
                    ts_key = "unknown"

                dedup_key = (scored.room, scored.severity, ts_key)
                if dedup_key not in _reported:
                    _reported.add(dedup_key)
                    alerter.write(scored)

                    rule_str = ""
                    if scored.violated_rules:
                        ant, con, conf, sup = scored.violated_rules[0]
                        rule_str = (
                            f"[{' & '.join(sorted(ant))}] -> "
                            f"[{' & '.join(sorted(con))}]"
                        )
                        if len(rule_str) > 50:
                            rule_str = rule_str[:47] + "..."

                    color = SEV_COLOR.get(scored.severity, "")
                    ts    = str(scored.timestamp)[:19] if pd.notna(scored.timestamp) else "N/A"
                    print(f"  {ts:<22} "
                          f"{scored.room:<8} "
                          f"{color}{BOLD}{scored.severity:<8}{RESET} "
                          f"{scored.anomaly_score:>6.3f}  "
                          f"{rule_str}")

            if rec_count % 100_000 == 0:
                elapsed = time.time() - t0
                pct = 100 * alert_count / tx_count if tx_count else 0
                print(f"\n  {'='*70}")
                print(f"  PROGRESS | {rec_count:>8,} records | "
                      f"{tx_count:>6,} tx | "
                      f"{alert_count:>4,} alerts ({pct:.1f}%) | "
                      f"{elapsed:.1f}s")
                if room_counts:
                    print("  Anomalies by room:")
                    for room in sorted(room_counts):
                        c = room_counts[room]
                        bar = (SEV_COLOR["HIGH"]   + "H" * c["HIGH"]   + RESET +
                               SEV_COLOR["MEDIUM"] + "M" * c["MEDIUM"] + RESET +
                               SEV_COLOR["LOW"]    + "L" * c["LOW"]    + RESET)
                        print(f"    {room:<8}  H={c['HIGH']:>4}  M={c['MEDIUM']:>4}  "
                              f"L={c['LOW']:>4}  {bar}")
                print(f"  {'='*70}")
                print(f"  {'TIMESTAMP':<22} {'ROOM':<8} {'SEV':<8} "
                      f"{'SCORE':>6}  VIOLATED RULE")
                print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*6}  {'-'*45}")

        alerter.close()
        self._save_all_scores(all_scores)
        self._print_summary(all_scores, alerter)
        return all_scores

    # ── helpers ────────────────────────────────────────────────────────────────

    def _save_all_scores(self, scores: List[ScoredTransaction]):
        rows = [{
            "transaction_id":   s.transaction_id,
            "timestamp":        s.timestamp,
            "room":             s.room,
            "anomaly_score":    s.anomaly_score,
            "severity":         s.severity,
            "violated_count":   s.violated_count,
            "applicable_count": s.applicable_count,
        } for s in scores]
        pd.DataFrame(rows).to_csv(SCORES_PATH, index=False)
        print(f"[Scores]  All scores saved → {SCORES_PATH}")

    def _print_summary(self, scores: List[ScoredTransaction], alerter: AlertWriter):
        df = pd.DataFrame([{
            "room": s.room, "score": s.anomaly_score,
            "severity": s.severity,
            "violated_count": s.violated_count,
            "applicable_count": s.applicable_count,
            "violated_rules": s.violated_rules,
        } for s in scores])

        sev_counts = df["severity"].value_counts()
        total      = len(df)
        anomalies  = df[df["severity"] != "NONE"]

        W = 68
        print("\n" + "=" * W)
        print("  DETECTION SUMMARY  (v2 — weighted scoring)")
        print("=" * W)
        print(f"  Total transactions scored : {total:,}")
        print(f"  Total anomalies flagged   : {len(anomalies):,}  "
              f"({100*len(anomalies)/total:.1f}%)")
        print(f"  Mean anomaly score        : {df['score'].mean():.4f}")
        print(f"  Max anomaly score         : {df['score'].max():.4f}")
        print()
        print("  Severity breakdown:")
        for sev in ["HIGH", "MEDIUM", "LOW", "NONE"]:
            n   = sev_counts.get(sev, 0)
            pct = 100 * n / total if total else 0
            bar = "█" * int(pct / 2)
            print(f"    {sev:<8}: {n:>6,}  ({pct:5.1f}%)  {bar}")

        print()
        print("  Anomaly rate per room:")
        room_anom  = anomalies.groupby("room").size()
        room_total = df.groupby("room").size()
        room_rate  = (room_anom / room_total).fillna(0).sort_values(ascending=False)
        for room, rate in room_rate.items():
            n = int(room_anom.get(room, 0))
            bar = "█" * int(rate * 40)
            print(f"    {room:>6}: {rate:.3f}  ({n:>5,} anomalies)  {bar}")

        # Violation analysis
        violation_counts: Dict[tuple, dict] = {}
        for s in scores:
            for ant, con, conf, sup in s.violated_rules:
                key = (frozenset(ant), frozenset(con))
                if key not in violation_counts:
                    violation_counts[key] = {
                        "ant": ant, "con": con, "conf": conf, "sup": sup,
                        "count": 0, "rooms": set(),
                    }
                violation_counts[key]["count"] += 1
                violation_counts[key]["rooms"].add(s.room)

        if violation_counts:
            sorted_v = sorted(violation_counts.values(),
                              key=lambda x: x["count"], reverse=True)
            total_v  = sum(v["count"] for v in sorted_v)

            print()
            print("  " + "=" * (W - 2))
            print("  TOP VIOLATED RULES")
            print("  " + "=" * (W - 2))
            print(f"  Total violations : {total_v:,}  |  "
                  f"Unique violated rules : {len(sorted_v)}")
            print(f"\n  {'#':<4} {'COUNT':>6} {'%TOT':>6} {'CONF':>6} {'SUP':>6}  RULE")
            print(f"  {'-'*66}")

            for i, v in enumerate(sorted_v[:15], 1):
                ant_s  = " & ".join(sorted(v["ant"]))
                con_s  = " & ".join(sorted(v["con"]))
                rule_s = f"[{ant_s}] -> [{con_s}]"
                pct    = 100 * v["count"] / total_v
                if len(rule_s) > 36:
                    rule_s = rule_s[:33] + "..."
                print(f"  {i:<4} {v['count']:>6,} {pct:>5.1f}% "
                      f"{v['conf']:>6.3f} {v['sup']:>6.3f}  {rule_s}")

        # Score distribution
        print()
        print("  " + "-" * (W - 2))
        print("  SCORE DISTRIBUTION:")
        bins   = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        labels = ["0.0-0.1","0.1-0.2","0.2-0.3","0.3-0.4","0.4-0.5",
                  "0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9","0.9-1.0"]
        df["score_bin"] = pd.cut(df["score"], bins=bins, labels=labels, right=False)
        bin_counts = df["score_bin"].value_counts().sort_index()
        print()
        for label, count in bin_counts.items():
            pct = 100 * count / total
            bar = "█" * int(pct / 1.5)
            marker = (" <- LOW"  if label == "0.2-0.3" else
                      " <- MED"  if label == "0.4-0.5" else
                      " <- HIGH" if label == "0.6-0.7" else "")
            print(f"    {label} : {count:>6,} ({pct:5.1f}%)  {bar}{marker}")

        print("=" * W)


# =============================================================================
#  VISUALIZATION
# =============================================================================

SEV_COLORS = {
    "HIGH":   "#e74c3c",
    "MEDIUM": "#f39c12",
    "LOW":    "#3498db",
    "NONE":   "#ecf0f1",
}


def generate_dashboard(scores: List[ScoredTransaction]) -> None:
    print("\n[Charts] Building dashboard...", flush=True)

    df = pd.DataFrame([{
        "timestamp": s.timestamp,
        "room":      s.room,
        "score":     s.anomaly_score,
        "severity":  s.severity,
    } for s in scores])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    alerts = df[df["severity"] != "NONE"].copy()
    rooms  = sorted(df["room"].unique())

    fig = plt.figure(figsize=(22, 16), facecolor="#1a1a2e")
    fig.suptitle(
        "Building Heating Anomaly Detection v2 — Dashboard\n"
        "(Average-based transactions | Weighted scoring | OO context)",
        fontsize=14, fontweight="bold", color="white", y=0.98,
    )
    gs = GridSpec(2, 2, figure=fig,
                  hspace=0.38, wspace=0.30,
                  left=0.07, right=0.97,
                  top=0.93, bottom=0.06,
                  height_ratios=[1.6, 1.0])

    ax_timeline = fig.add_subplot(gs[0, :])
    ax_hist     = fig.add_subplot(gs[1, 0])
    ax_donut    = fig.add_subplot(gs[1, 1])

    for ax in [ax_timeline, ax_hist, ax_donut]:
        ax.set_facecolor("#16213e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#0f3460"); sp.set_linewidth(1.5)
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # Panel 1: Anomaly Timeline
    ax = ax_timeline
    ax.set_title("Anomaly Timeline — All Rooms (v2)", fontsize=13, pad=8)
    room_y    = {r: i for i, r in enumerate(rooms)}
    sev_order = ["LOW", "MEDIUM", "HIGH"]
    for sev in sev_order:
        sub = alerts[alerts["severity"] == sev]
        if sub.empty:
            continue
        ax.scatter(
            sub["timestamp"], sub["room"].map(room_y),
            c=SEV_COLORS[sev],
            s=10 if sev == "LOW" else (18 if sev == "MEDIUM" else 28),
            alpha=0.65 if sev == "LOW" else (0.80 if sev == "MEDIUM" else 1.0),
            label=sev, zorder=3,
        )
    ax.set_yticks(range(len(rooms)))
    ax.set_yticklabels(rooms, fontsize=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax.grid(axis="x", color="#0f3460", linestyle="--", alpha=0.5)
    ax.set_ylim(-0.6, len(rooms) - 0.4)
    ax.legend(
        handles=[Patch(facecolor=SEV_COLORS[s], label=s) for s in sev_order],
        loc="upper right", framealpha=0.3, labelcolor="white",
        facecolor="#1a1a2e", edgecolor="#0f3460", fontsize=9,
    )

    # Panel 2: Score Distribution
    ax = ax_hist
    ax.set_title("Anomaly Score Distribution (weighted)", fontsize=11, pad=6)
    ax.hist(df["score"], bins=40, color="#3498db", alpha=0.85, edgecolor="#1a1a2e")
    ymax = ax.get_ylim()[1]
    for thresh, label, color in [
        (SEV_LOW_MIN,  "LOW",    SEV_COLORS["LOW"]),
        (SEV_MED_MIN,  "MEDIUM", SEV_COLORS["MEDIUM"]),
        (SEV_HIGH_MIN, "HIGH",   SEV_COLORS["HIGH"]),
    ]:
        ax.axvline(thresh, color=color, linewidth=2, linestyle="--", alpha=0.9)
        ax.text(thresh + 0.01, ymax * 0.92, label, color=color, fontsize=8, va="top")
    ax.set_xlabel("Weighted Anomaly Score", fontsize=9)
    ax.set_ylabel("Transactions", fontsize=9)

    # Panel 3: Severity Donut
    ax = ax_donut
    ax.set_title("Severity Breakdown", fontsize=11, pad=6)
    sev_vc = df["severity"].value_counts()
    labels_d, sizes_d, colors_d = [], [], []
    for sev in ["HIGH", "MEDIUM", "LOW", "NONE"]:
        n = sev_vc.get(sev, 0)
        if n:
            labels_d.append(f"{sev}  {n:,}")
            sizes_d.append(n)
            colors_d.append(SEV_COLORS[sev])
    wedges, _ = ax.pie(
        sizes_d, colors=colors_d, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.5, edgecolor="#1a1a2e", linewidth=1.5),
    )
    ax.legend(wedges, labels_d,
              loc="center left", bbox_to_anchor=(0.88, 0.5),
              framealpha=0.2, labelcolor="white",
              facecolor="#1a1a2e", edgecolor="#0f3460", fontsize=9)
    anom_pct = 100 * len(alerts) / len(df) if len(df) else 0
    ax.text(0, 0, f"{anom_pct:.1f}%\nanomaly",
            ha="center", va="center", fontsize=12,
            color="white", fontweight="bold")

    plt.savefig(CHARTS_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Charts] Dashboard saved → {CHARTS_PATH}")


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    print("=" * 70)
    print("  BUILDING HEATING ANOMALY DETECTION  v2.1")
    print("  Fixes from post-presentation review:")
    print("    FIX 1: Severity thresholds raised to match bimodal score distribution")
    print("           LOW>=0.30 | MED>=0.55 | HIGH>=0.85  (was 0.20/0.40/0.65)")
    print("    FIX 2: Normal-heating filter added to RuleFilter")
    print("           HEATER_ON->TEMP_HOT, HEATER_ON->AT_SETPOINT,")
    print("           and weak TEMP_COLD->HEATER_OFF rules are removed.")
    print("           These encode expected physical behaviour, not anomalies.")
    print("    FIX 3: MIN_CONFIDENCE raised 0.88->0.92; MAX_CON_SUPPORT 0.80->0.70")
    print("           Fewer but more trustworthy rules; common consequents purged.")
    print("=" * 70)
    print(f"\n  Config:")
    print(f"    Transaction size     : {TRANSACTION_SIZE} records (~1 min)")
    print(f"    Min support          : {MIN_SUPPORT}")
    print(f"    Min confidence       : {MIN_CONFIDENCE}")
    print(f"    Max rules (cap)      : {MAX_RULE_SET_SIZE}")
    print(f"    Min antecedent sup.  : {MIN_ANTECEDENT_SUPPORT}")
    print(f"    Rare item threshold  : {RARE_ITEM_SUPPORT} (weight {RARE_WEIGHT})")
    print(f"    Severity LOW / MED / HIGH : "
          f"{SEV_LOW_MIN} / {SEV_MED_MIN} / {SEV_HIGH_MIN}")

    pipeline = Pipeline()
    miner    = pipeline.run_training()

    if not miner.is_ready:
        print("\n  No rules mined. Lower MIN_SUPPORT or MIN_CONFIDENCE.")
        return

    scores = pipeline.run_detection(miner)
    generate_dashboard(scores)

    print(f"\n  Pipeline complete. Outputs in: {OUTPUT_DIR}/")
    print(f"    {RULES_PATH}")
    print(f"    {ALERTS_PATH}")
    print(f"    {SCORES_PATH}")
    print(f"    {CHARTS_PATH}")


if __name__ == "__main__":
    main()