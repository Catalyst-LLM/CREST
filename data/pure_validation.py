import os
import json
import math
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# ---------- Utility Functions ----------
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ---------- Edit Distance ----------
try:
    import Levenshtein
except ImportError:
    Levenshtein = None

def levenshtein_similarity(s1: str, s2: str) -> float:
    if Levenshtein:
        return Levenshtein.ratio(s1, s2)
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return 1.0 - dp[len1][len2] / max(len1, len2)

def compare_with_seg(str1, str2, seg=" "):
    str1 = str1.strip().lower().split(seg)
    str2 = str2.strip().lower().split(seg)
    for s in str1:
        if s in str2:
            return True
    return False

def compare_with_digit_char(str1, str2):
    str1 = str1.replace('₂', '2').replace('₁', '1').replace('₃', '3').replace('₄', '4').replace('₅', '5')\
        .replace('₆', '6').replace('₇', '7').replace('₈', '8').replace('₉', '9').replace('₀', '0')
    str2 = str2.replace('₂', '2').replace('₁', '1').replace('₃', '3').replace('₄', '4').replace('₅', '5')\
        .replace('₆', '6').replace('₇', '7').replace('₈', '8').replace('₉', '9').replace('₀', '0')
    str1 = str1.replace(')', '').lower()
    str2 = str2.replace('(', '').lower()
    return str1 == str2

def normal_compare(str1, str2):
    str1 = str1.strip().lower()
    str2 = str2.strip().lower()
    return str1 == str2

def preprocess_unit_str(s):
    if isinstance(s, str):
        if s in ['NA', 'N/A', 'na', 'n/a', 'unknown','unknown', 'Unknown']:
            return None
        else:
            return s.strip().lower()
    return s

# ---------- Match Strategy Definitions ----------
class MatchStrategy(Enum):
    EXACT = "exact"
    EDIT_DISTANCE = "edit_distance"
    UNIT_TOLERANT = "unit_tolerant"
    CUSTOM = "custom"
    ATM = "atm"
    CATALYST = "catalyst_name"
    MENTAL = "catalyst_metal"
    COCATALYST = "cocatalyst"
    COMONOMER_1_NAME = "comonomer_1_name"

@dataclass
class FieldConfig:
    match_type: MatchStrategy
    params: Dict[str, Any] = field(default_factory=dict)
    custom_fn: Optional[callable] = None

# ---------- Core Evaluator ----------
class FieldEvaluator:
    def __init__(self, config: Dict[str, FieldConfig],
                 default_strategy=MatchStrategy.EXACT,
                 include_nulls=False,
                 ignore_fields: Optional[List[str]] = None):   # New: ignore fields list
        self.config = config
        self.default_strategy = default_strategy
        self.include_nulls = include_nulls
        self.ignore_fields = ignore_fields if ignore_fields is not None else []

    def compare(self, gt_val, pred_val, field):
        strategy = self.config.get(field, FieldConfig(self.default_strategy))
        if strategy.match_type == MatchStrategy.EXACT:
            return self._exact(gt_val, pred_val)
        elif strategy.match_type == MatchStrategy.EDIT_DISTANCE:
            thr = strategy.params.get("threshold", 0.9)
            return self._edit_dist(gt_val, pred_val, thr)
        elif strategy.match_type == MatchStrategy.UNIT_TOLERANT:
            ratios = strategy.params.get("allowed_ratios", [1])
            rtol = strategy.params.get("relative_tolerance", 0.0)
            return self._unit_tol(gt_val, pred_val, ratios, rtol)
        elif strategy.match_type == MatchStrategy.CUSTOM:
            if strategy.custom_fn:
                return strategy.custom_fn(gt_val, pred_val)
            raise ValueError(f"Missing custom_fn for '{field}'")
        elif strategy.match_type == MatchStrategy.ATM:
            return self._unit_atm(gt_val, pred_val)
        elif strategy.match_type == MatchStrategy.CATALYST:
            return self._unit_catalyst_name(gt_val, pred_val)
        elif strategy.match_type == MatchStrategy.MENTAL:
            return self._unit_catalyst_metal(gt_val, pred_val)
        elif strategy.match_type == MatchStrategy.COCATALYST:
            return self._unit_cocatalyst(gt_val, pred_val)
        elif strategy.match_type == MatchStrategy.COMONOMER_1_NAME:
            return self._unit_comonomer_1_name(gt_val, pred_val)
        else:
            raise ValueError(f"Unknown strategy: {strategy.match_type}")

    @staticmethod
    def _unit_comonomer_1_name(a, b):
        a = a.lower().strip()
        b = b.lower().strip()
        name_mapping = {
            "norbornene": "nbe",
            "nbe": "nbe",
            "5-norbornene-2-methanol": "nbmo",
            "nbmo": "nbmo",
        }
        if name_mapping.get(a) is not None and name_mapping.get(b) is not None:
            return name_mapping.get(a) == name_mapping.get(b)
        return normal_compare(a, b)

    @staticmethod
    def _unit_cocatalyst(a, b):
        # Fix: Unify as list comparison to avoid character-level traversal
        items_a = a if isinstance(a, list) else [a]
        items_b = b if isinstance(b, list) else [b]
        for i in items_a:
            for j in items_b:
                if compare_with_digit_char(str(i), str(j)):
                    return True
                if normal_compare(str(i), str(j)):
                    return True
        return False

    @staticmethod
    def _unit_catalyst_metal(a, b):
        if compare_with_digit_char(a, b):
            return True
        a = a.lower().strip()
        b = b.lower().strip()
        name_mapping = {
            "v": "vanadium",
            "cr": "chromium",
            "mn": "manganese",
            "fe": "iron",
            "co": "cobalt",
            "ni": "nickel",
            "vanadium": "vanadium",
            "chromium": "chromium",
            "manganese": "manganese",
            "iron": "iron",
            "cobalt": "cobalt",
            "nickel": "nickel"
        }
        if name_mapping.get(a) is not None and name_mapping.get(b) is not None:
            return name_mapping.get(a) == name_mapping.get(b)
        return str(a).strip().lower() == str(b).strip().lower()

    @staticmethod
    def _unit_catalyst_name(a, b):
        if isinstance(a, str) and isinstance(b, str):
            if compare_with_seg(a, b, " "):
                return True
            if compare_with_seg(a, b, "/"):
                return True
            if compare_with_digit_char(a, b):
                return True
        return str(a).strip().lower() == str(b).strip().lower()

    @staticmethod
    def _unit_atm(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b
        return str(a).strip() == str(b).strip()

    @staticmethod
    def _exact(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a == b
        return str(a).strip() == str(b).strip()

    @staticmethod
    def _edit_dist(a, b, thr):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return levenshtein_similarity(str(a).strip(), str(b).strip()) >= thr

    @staticmethod
    def _unit_tol(a, b, ratios, rtol):
        """Unit tolerance: Check scaling factors bidirectionally"""
        if a == b:
            return True
        try:
            na, nb = float(a), float(b)
        except (ValueError, TypeError):
            return False
        if na == 0 and nb == 0:
            return True
        if na == 0 or nb == 0:
            return False
        if rtol > 0 and math.isclose(na, nb, rel_tol=rtol):
            return True
        ratio = nb / na
        for r in ratios:
            if math.isclose(ratio, r, rel_tol=1e-9) or math.isclose(1.0 / ratio, r, rel_tol=1e-9):
                return True
        return False

    def normal_k(self, kv):
        rkv = {}
        for k,v in kv.items():
            k = k.lower().strip()
            rkv[k] = v
        return rkv

    def evaluate_record(self, gt, pred):
        tp = fp = fn = 0
        details = []
        gt = self.normal_k(gt)
        pred = self.normal_k(pred)

        for field in set(gt.keys()) | set(pred.keys()):
            if field in self.ignore_fields:   # Skip ignored fields
                continue
            print(field, gt.get(field), pred.get(field))
            gv = gt.get(field)
            pv = pred.get(field)

            gv = preprocess_unit_str(gv)
            pv = preprocess_unit_str(pv)
            gnull = gv is None
            pnull = pv is None

            match = None
            if not gnull and not pnull:
                match = self.compare(gv, pv, field)
                if match:
                    tp += 1
                    status = "TP"
                else:
                    fp += 1
                    fn += 1
                    status = "FP+FN"
            elif not gnull and pnull:
                fn += 1
                status = "FN"
            elif gnull and not pnull:
                fp += 1
                status = "FP"
            else:
                if self.include_nulls:
                    tp += 1
                    match = True
                    status = "TP"
                else:
                    status = "TN"
            details.append({"field": field, "gt": gv, "pred": pv, "match": match, "status": status})

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1, "details": details}

# ---------- Batch Evaluation Function ----------
def evaluate_batch(records, field_config, include_nulls=False, ignore_fields=None):
    evaluator = FieldEvaluator(field_config, include_nulls=include_nulls, ignore_fields=ignore_fields)
    total_tp = total_fp = total_fn = 0
    per_rec = []
    field_tp = {}
    field_fp = {}
    field_fn = {}
    all_fields_global = set()
    error_list = []

    for i, rec in enumerate(records):
        res = evaluator.evaluate_record(rec["gt"], rec["pred"])
        res["record_index"] = i
        per_rec.append(res)
        total_tp += res["tp"]
        total_fp += res["fp"]
        total_fn += res["fn"]

        for d in res["details"]:
            field = d["field"]
            all_fields_global.add(field)
            status = d["status"]
            if status == "TP":
                field_tp[field] = field_tp.get(field, 0) + 1
            elif status == "FP":
                field_fp[field] = field_fp.get(field, 0) + 1
            elif status == "FN":
                error_list.append({
                    "field": field,
                    "gt": d["gt"],
                    "pred": d["pred"],
                    "status": status,
                    "meta": rec.get("meta")  # Extract meta info from record
                })
                field_fn[field] = field_fn.get(field, 0) + 1
            elif status == "FP+FN":
                field_fp[field] = field_fp.get(field, 0) + 1
                field_fn[field] = field_fn.get(field, 0) + 1
                # Record error details with meta info of the record

    field_metrics = []
    for field in sorted(all_fields_global):
        tp = field_tp.get(field, 0)
        fp = field_fp.get(field, 0)
        fn = field_fn.get(field, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        field_metrics.append({
            "field": field,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })
    field_metrics.sort(key=lambda x: x["f1"], reverse=True)

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                if (micro_prec + micro_rec) > 0 else 0.0)

    n = len(per_rec)
    macro_prec = sum(r["precision"] for r in per_rec) / n if n > 0 else 0.0
    macro_rec = sum(r["recall"] for r in per_rec) / n if n > 0 else 0.0
    macro_f1 = sum(r["f1"] for r in per_rec) / n if n > 0 else 0.0

    return {
        "macro_avg": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
        "micro_avg": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1},
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "per_record": per_rec,
        "per_field_metrics": field_metrics,
        "error_details": error_list
    }

# ---------- Read Raw Data ----------
def get_test_data(json_path):
    data = load_json(json_path)

    return data.get('data'), data.get('data_id'), data.get('index'), data.get('ignore')

# ---------- Main Run Function ----------
def run_evaluation(gt_path, pred_path, field_config, include_nulls=False, ignore_fields=None):
    all_records = []
    record_meta = []
    cnt = 0
    for fname in os.listdir(gt_path):
        if 'check' not in fname:
            continue

        val_file = os.path.join(gt_path, fname)
        gts, table_names, table_ranges, ignores = get_test_data(val_file)
        folder = fname.replace('.json', '').replace('-check', '')

        for idx, (gt, table_id, table_range) in enumerate(zip(gts, table_names, table_ranges)):

            pred_filename = f'{table_id}_{table_range[0]}_{table_range[1]}_parsed.json'
            pred_file = os.path.join(pred_path, folder, pred_filename)
            if ignores is not None:
                if ignores[idx]:
                    print(f"Info: Record {fname} {table_id} is marked as ignore, skipping.")
                    continue
            try:
                pred_data = load_json(pred_file)
            except Exception as e:
                print(f"Warning: cannot load {pred_file}, skipping. Error: {e}")
                continue

            # Ensure gt and pred_data are lists
            if not isinstance(gt, list):
                gt = [gt]
            if not isinstance(pred_data, list):
                pred_data = [pred_data]

            ng = []
            np = []
            idxs = []
            if len(gt) == len(pred_data):
                for idx, (g, p) in enumerate(zip(gt, pred_data)):
                    flage = False
                    for k, v in g.items():
                        if v == "traces" or v == "inactive" or v == "trace":
                            flage = True
                            break
                    if not flage:
                        ng.append(g)
                        np.append(p)
                        idxs.append(idx)
            for idx, (g, p) in enumerate(zip(ng, np)):

                if not isinstance(g, dict) or not isinstance(p, dict):
                    print(f"Skipping non-dict entry in {fname} {table_id}")
                    continue

                all_records.append({
                    "gt": g,
                    "pred": p,
                    "meta": {   # Attach file information directly to each record
                        "source_file": val_file,
                        "pred_file": pred_file,
                        "table_id": table_id,
                        "table_range": table_range,
                        "record_index": idx,  # Index of record in current file
                    }
                })

    print(f"Evaluating {len(all_records)} records...")
    results = evaluate_batch(all_records, field_config, include_nulls, ignore_fields)
    return results, all_records  # Return records directly, or only results (with meta)

# ---------- Usage Example ----------
if __name__ == "__main__":
    GT_PATH = "./data/ground_truth"
    PRED_PATH = "./data/predictions"

    FIELD_CONFIG = {
        "catalyst_ligand_type": FieldConfig(
            MatchStrategy.EDIT_DISTANCE, params={"threshold": 0.9}
        ),
        "catalyst_reference": FieldConfig(
            MatchStrategy.EDIT_DISTANCE, params={"threshold": 0.9}
        ),
        "activity": FieldConfig(
            MatchStrategy.UNIT_TOLERANT, params={"allowed_ratios": [1, 1e10]}
        ),
        "Mw_kgmol": FieldConfig(
            MatchStrategy.UNIT_TOLERANT, params={"allowed_ratios": [1, 1e10]}
        ),
        "catalyst_name": FieldConfig(
            MatchStrategy.CATALYST
        ),
        "catalyst_ligand_type": FieldConfig(
            MatchStrategy.CATALYST
        ),
        "cocatalyst": FieldConfig(
            MatchStrategy.COCATALYST
        ),
        "catalyst_metal": FieldConfig(
            MatchStrategy.MENTAL
        ),
        "comonomer_1_name": FieldConfig(
            MatchStrategy.COMONOMER_1_NAME
        )
    }

    # Fields to ignore
    IGNORE_FIELDS = ["additional_items", "2d_structure", "3d_structure", "catalyst_reference"]

    results, records_with_meta = run_evaluation(
        GT_PATH, PRED_PATH, FIELD_CONFIG,
        include_nulls=True,
        ignore_fields=IGNORE_FIELDS
    )

    # Output field metrics
    print("\n========== Per-Field Metrics (sorted by F1) ==========")
    for fm in results["per_field_metrics"]:
        print(f"{fm['field']:30s}  TP:{fm['tp']:3d}  FP:{fm['fp']:3d}  FN:{fm['fn']:3d}  "
              f"P:{fm['precision']:.3f}  R:{fm['recall']:.3f}  F1:{fm['f1']:.3f}")

    # Output error details (first 10)
    print('result:', results['macro_avg'])
    print("\n========== Error Details (first 10) ==========")

    with open("evaluation_errors1.json", "w") as f:
        json.dump(results["error_details"], f, indent=2, ensure_ascii=False)

    print(f"\nTotal errors written to evaluation_errors.json: {len(results['error_details'])}")
