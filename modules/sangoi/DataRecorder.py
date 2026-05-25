import gzip
import json
import math
import os
import traceback

from modules.sangoi.logFun import logFun


class DataRecorder:
    def __init__(self, debug: bool = True, verbose: bool = True):
        self.records: list[dict[str, int | str | float | None]] = []
        self.debug = debug
        self.verbose = verbose

    @staticmethod
    def __optional_finite_float(name: str, value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"DataRecorder {name} must be numeric or None")
        float_value = float(value)
        if not math.isfinite(float_value):
            raise ValueError(f"DataRecorder {name} must be finite")
        return float_value

    def log_metrics_step(
        self,
        step: int,
        group: str,
        d_num_pdgy: float | None = None,
        d_den_pdgy: float | None = None,
        dlr_pdgy: float | None = None,
        grad_norm: float | None = None,
    ):
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("DataRecorder step must be a non-negative integer")
        if not isinstance(group, str) or not group:
            raise ValueError("DataRecorder group must be a non-empty string")

        self.records.append(
            {
                "step": step,
                "group": group,
                "d_num_pdgy": self.__optional_finite_float("d_num_pdgy", d_num_pdgy),
                "d_den_pdgy": self.__optional_finite_float("d_den_pdgy", d_den_pdgy),
                "dlr_pdgy": self.__optional_finite_float("dlr_pdgy", dlr_pdgy),
                "grad_norm": self.__optional_finite_float("grad_norm", grad_norm),
            }
        )

    def dump(self, path: str):
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        payload = {
            "schema_version": 1,
            "kind": "sangoi_data_recorder",
            "records": self.records,
        }

        try:
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            if self.verbose:
                logFun(f"[DataRecorder] Saved metrics to {path}", lvl="success")
        except Exception as error:
            logFun(f"[DataRecorder] Failed to save metrics to {path}: {error}", lvl="error")
            traceback.print_exc()
            raise
