"""Central experiment results logger.

Every number that ends up in the manuscript must be written here — never
just printed to the console — so it is traceable to a timestamp, git commit,
seed, and config, and mapped to a specific paper table/figure via
outputs/tables/MANIFEST.md. This module never invents or hardcodes a result;
it only records numbers handed to it by real experiment code.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DATA_DIR = PROJECT_ROOT / "outputs" / "figures" / "data"

# Config fields worth snapshotting on every record for reproducibility.
_TRACKED_CONFIG_KEYS = (
    "window_hours",
    "jaccard_threshold_tau",
    "retrieval_M",
    "gamma_threshold",
    "epoch",
    "seed",
)


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


class ResultsLogger:
    """Append-only JSON logger for paper table/figure data.

    Each `outputs/tables/{table_id}.json` (or `outputs/figures/data/{curve_id}.json`)
    holds:
        {"table_id": ..., "records": [...], "aggregates": [...]}
    `records` is the raw, append-only log (one entry per call). `aggregates`
    is recomputed on every write: mean/std per metric, grouped by
    (key_field, dataset), for any group with >= 2 distinct seeds.
    """

    def __init__(
        self,
        experiment_version: str = None,
        tables_dir: Path = None,
        figures_data_dir: Path = None,
    ):
        """
        Args:
            experiment_version: "v1" (S-VLG) or "v2" (SU-MedVQA). When given,
                results are written under a `{version}/` subfolder of
                `tables_dir`/`figures_data_dir` (e.g.
                outputs/tables/v1/table6_overall.json) so V1 and V2 numbers
                never overwrite each other. None keeps the old unversioned
                layout (used e.g. for dataset-level, not model-run, artifacts).
            tables_dir / figures_data_dir: root directories (default: the
                real project outputs/tables, outputs/figures/data). Override
                with a temp directory in tests to avoid touching real results.
        """
        self.experiment_version = experiment_version
        tables_root = Path(tables_dir) if tables_dir is not None else TABLES_DIR
        figures_root = Path(figures_data_dir) if figures_data_dir is not None else FIGURES_DATA_DIR
        self.tables_dir = tables_root / experiment_version if experiment_version else tables_root
        self.figures_data_dir = figures_root / experiment_version if experiment_version else figures_root
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ helpers --

    def _make_metadata(self, dataset, config, seed) -> dict:
        cfg = dict(config or {})
        if seed is not None:
            cfg["seed"] = seed
        tracked_config = {k: cfg[k] for k in _TRACKED_CONFIG_KEYS if k in cfg}
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit_hash(),
            "experiment_version": self.experiment_version,
            "dataset": dataset,
            "config": tracked_config,
        }

    @staticmethod
    def _load(path: Path) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"records": []}

    @staticmethod
    def _save(path: Path, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _aggregate(records: list, key_field: str) -> list:
        """Mean +/- std of each numeric metric, grouped by (key_field, dataset),
        for groups with >= 2 distinct seeds. Groups with a single seed (or no
        seed) are left for the raw `records` to represent — no aggregate row.
        """
        groups: dict = {}
        for r in records:
            key = (r.get(key_field), r.get("dataset"))
            groups.setdefault(key, []).append(r)

        aggregates = []
        for (key_value, dataset), group_records in groups.items():
            seeds = {r.get("seed") for r in group_records if r.get("seed") is not None}
            if len(seeds) < 2:
                continue

            metric_names = set()
            for r in group_records:
                metric_names.update(r.get("metrics", {}).keys())

            mean_vals, std_vals = {}, {}
            for name in metric_names:
                values = [r["metrics"][name] for r in group_records if name in r.get("metrics", {})]
                if not values:
                    continue
                mean_vals[name] = mean(values)
                std_vals[name] = stdev(values) if len(values) >= 2 else 0.0

            aggregates.append({
                key_field: key_value,
                "dataset": dataset,
                "n_seeds": len(seeds),
                "mean": mean_vals,
                "std": std_vals,
            })
        return aggregates

    def _append_and_save(self, path: Path, table_id: str, record: dict, key_field: str) -> Path:
        data = self._load(path)
        data.setdefault("table_id", table_id)
        data.setdefault("records", []).append(record)
        data["aggregates"] = self._aggregate(data["records"], key_field=key_field)
        self._save(path, data)
        return path

    # -------------------------------------------------------------- API ---

    def log_metrics(
        self,
        table_id: str,
        model_name: str,
        metrics_dict: dict,
        seed: int = None,
        dataset: str = None,
        config: dict = None,
    ) -> Path:
        """Append one metrics record for a model-comparison table (e.g. Table
        6/7/8/11 — see outputs/tables/MANIFEST.md for the exact mapping).

        `table_id` is the output file stem, e.g. "table6_overall".
        """
        path = self.tables_dir / f"{table_id}.json"
        record = {
            "model_name": model_name,
            "metrics": dict(metrics_dict),
            "seed": seed,
            **self._make_metadata(dataset, config, seed),
        }
        return self._append_and_save(path, table_id, record, key_field="model_name")

    def log_ablation(
        self,
        variant_name: str,
        metrics_dict: dict,
        seed: int = None,
        dataset: str = None,
        config: dict = None,
    ) -> Path:
        """Append one ablation-variant record — Table 9 (8 variants)."""
        table_id = "table9_ablation"
        path = self.tables_dir / f"{table_id}.json"
        record = {
            "variant_name": variant_name,
            "metrics": dict(metrics_dict),
            "seed": seed,
            **self._make_metadata(dataset, config, seed),
        }
        return self._append_and_save(path, table_id, record, key_field="variant_name")

    def log_risk_coverage(
        self,
        config_name: str,
        coverage_points,
        risk_values,
        auc: float,
        seed: int = None,
        dataset: str = None,
        config: dict = None,
    ) -> Path:
        """Append one risk-coverage curve — Table 10 (AUC) + Figure 8 (curve).

        `coverage_points`/`risk_values` are the raw per-run curve (not
        averaged across seeds — only the scalar `auc` participates in the
        usual mean/std seed aggregation). This call also mirrors the curve
        into `outputs/figures/data/fig8_risk_coverage.json` via
        `log_curve_data`, so Table 10 and Figure 8 can never drift apart.
        """
        table_id = "table10_risk_coverage"
        path = self.tables_dir / f"{table_id}.json"
        record = {
            "config_name": config_name,
            "coverage_points": list(coverage_points),
            "risk_values": list(risk_values),
            "metrics": {"auc": auc},
            "seed": seed,
            **self._make_metadata(dataset, config, seed),
        }
        result_path = self._append_and_save(path, table_id, record, key_field="config_name")

        self.log_curve_data(
            curve_id="fig8_risk_coverage",
            x=coverage_points,
            y=risk_values,
            label=config_name,
            seed=seed,
            dataset=dataset,
            config=config,
        )
        return result_path

    def log_curve_data(
        self,
        curve_id: str,
        x,
        y,
        label: str,
        seed: int = None,
        dataset: str = None,
        config: dict = None,
    ) -> Path:
        """Append one raw (x, y) curve series for a figure (PR/ROC, risk-
        coverage, ablation bar, ...) under outputs/figures/data/{curve_id}.json.
        """
        path = self.figures_data_dir / f"{curve_id}.json"
        data = self._load(path)
        data.setdefault("curve_id", curve_id)
        record = {
            "label": label,
            "x": list(x),
            "y": list(y),
            "seed": seed,
            **self._make_metadata(dataset, config, seed),
        }
        data.setdefault("records", []).append(record)
        self._save(path, data)
        return path


def _self_test() -> bool:
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        logger = ResultsLogger(
            tables_dir=tmp_path / "tables",
            figures_data_dir=tmp_path / "tables" / "figures" / "data",
        )

        # --- log_metrics with 3 seeds -> mean/std aggregation expected ---
        vqa_accs = [0.80, 0.82, 0.78]
        for seed, acc in enumerate(vqa_accs):
            logger.log_metrics(
                table_id="table6_overall_demo",
                model_name="S-VLG",
                metrics_dict={"vqa_acc": acc, "bleu4": 0.30 + seed * 0.01},
                seed=seed,
                dataset="mimic-test",
                config={
                    "window_hours": 48,
                    "jaccard_threshold_tau": 0.3,
                    "retrieval_M": 5,
                    "gamma_threshold": 1.0,
                    "epoch": 10,
                },
            )

        data6 = json.loads((logger.tables_dir / "table6_overall_demo.json").read_text(encoding="utf-8"))
        if len(data6["records"]) != 3:
            print(f"FAIL: expected 3 records, got {len(data6['records'])}")
            ok = False
        if not data6["aggregates"]:
            print("FAIL: expected an aggregate row with >= 2 seeds")
            ok = False
        else:
            agg = data6["aggregates"][0]
            expected_mean = sum(vqa_accs) / len(vqa_accs)
            if abs(agg["mean"]["vqa_acc"] - expected_mean) > 1e-9:
                print(f"FAIL: mean vqa_acc {agg['mean']['vqa_acc']} != {expected_mean}")
                ok = False
            if agg["n_seeds"] != 3:
                print(f"FAIL: n_seeds {agg['n_seeds']} != 3")
                ok = False

        rec0 = data6["records"][0]
        for key in ("timestamp", "git_commit", "dataset", "config"):
            if key not in rec0:
                print(f"FAIL: record missing metadata field {key!r}")
                ok = False
        if rec0["config"].get("seed") != 0:
            print("FAIL: config snapshot missing/incorrect seed")
            ok = False

        # --- log_ablation, single seed -> no aggregation expected ---
        logger.log_ablation("full_model", {"vqa_acc": 0.81}, seed=0, dataset="mimic-test")
        data9 = json.loads((logger.tables_dir / "table9_ablation.json").read_text(encoding="utf-8"))
        if data9["aggregates"]:
            print("FAIL: expected no aggregates with a single seed")
            ok = False

        # --- log_risk_coverage + auto-mirrored figure data ---
        risk_values = [0.05, 0.08, 0.12, 0.20]
        logger.log_risk_coverage(
            config_name="S-VLG",
            coverage_points=[0.5, 0.7, 0.9, 1.0],
            risk_values=risk_values,
            auc=0.09,
            seed=0,
            dataset="mimic-test",
        )
        data10 = json.loads((logger.tables_dir / "table10_risk_coverage.json").read_text(encoding="utf-8"))
        if len(data10["records"]) != 1 or data10["records"][0]["metrics"]["auc"] != 0.09:
            print("FAIL: risk-coverage record not saved correctly")
            ok = False

        fig8_path = logger.figures_data_dir / "fig8_risk_coverage.json"
        if not fig8_path.exists():
            print("FAIL: expected fig8_risk_coverage.json to be auto-mirrored")
            ok = False
        else:
            fig8_data = json.loads(fig8_path.read_text(encoding="utf-8"))
            if fig8_data["records"][0]["y"] != risk_values:
                print("FAIL: mirrored figure data does not match risk_values")
                ok = False

        # --- log_curve_data directly ---
        logger.log_curve_data("fig7_pr_roc_demo", x=[0.1, 0.5, 0.9], y=[0.9, 0.7, 0.3], label="S-VLG")
        if not (logger.figures_data_dir / "fig7_pr_roc_demo.json").exists():
            print("FAIL: fig7 curve data not written")
            ok = False

        # --- experiment_version separates V1/V2 output directories ---
        shared_root = tmp_path / "versioned_tables"
        logger_v1 = ResultsLogger(experiment_version="v1", tables_dir=shared_root)
        logger_v2 = ResultsLogger(experiment_version="v2", tables_dir=shared_root)
        logger_v1.log_metrics(table_id="table6_overall", model_name="S-VLG", metrics_dict={"vqa_acc": 0.80})
        logger_v2.log_metrics(table_id="table6_overall", model_name="SU-MedVQA", metrics_dict={"vqa_acc": 0.75})

        if logger_v1.tables_dir == logger_v2.tables_dir:
            print("FAIL: v1/v2 loggers resolved to the same tables_dir")
            ok = False

        v1_data = json.loads((logger_v1.tables_dir / "table6_overall.json").read_text(encoding="utf-8"))
        v2_data = json.loads((logger_v2.tables_dir / "table6_overall.json").read_text(encoding="utf-8"))
        if len(v1_data["records"]) != 1 or v1_data["records"][0]["model_name"] != "S-VLG":
            print("FAIL: v1 table6_overall.json does not contain only the V1 record")
            ok = False
        if len(v2_data["records"]) != 1 or v2_data["records"][0]["model_name"] != "SU-MedVQA":
            print("FAIL: v2 table6_overall.json does not contain only the V2 record")
            ok = False
        if v1_data["records"][0].get("experiment_version") != "v1":
            print("FAIL: v1 record missing experiment_version metadata")
            ok = False

    print("PASS: results_logger" if ok else "FAIL: results_logger")
    return ok


if __name__ == "__main__":
    _self_test()
