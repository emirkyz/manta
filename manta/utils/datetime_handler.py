"""
Centralized datetime detection, format inference, and conversion for MANTA.

Replaces scattered datetime logic across data_pipeline, visualizer, and cache_manager
with a single source of truth.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_COMMON_DATETIME_NAMES = {
    "date", "datetime", "timestamp", "time",
    "created_at", "updated_at",
    "rev_submit_millis_since_epoch",
}


class DatetimeFormat(Enum):
    FULL_DATETIME = "full_datetime"
    DATE_ONLY = "date_only"
    YEAR_MONTH_COMBINED = "year_month_combined"
    YEAR_ONLY = "year_only"
    EPOCH_MILLIS = "epoch_millis"
    EPOCH_SECONDS = "epoch_seconds"


@dataclass
class DatetimeInfo:
    column_name: str
    format: DatetimeFormat
    source: str  # "user", "auto_combined", "auto_detected"


class DatetimeDetector:

    @staticmethod
    def detect(df: pd.DataFrame, user_column: Optional[str] = None) -> Optional[DatetimeInfo]:
        """Detect datetime column with clear priority order.

        Priority 1: User-provided column (if valid)
        Priority 2: Year + month column combination
        Priority 3: Auto-detect by scoring all columns
        """
        # Priority 1: User-provided column
        if user_column and user_column in df.columns:
            fmt = DatetimeDetector.infer_format(df[user_column], user_column)
            return DatetimeInfo(column_name=user_column, format=fmt, source="user")

        if user_column and user_column not in df.columns:
            logger.warning(
                f"User-specified datetime_column '{user_column}' not found in data. "
                f"Falling back to auto-detection."
            )

        # Priority 2: Year + month combination
        if "year" in df.columns and "month" in df.columns:
            return DatetimeInfo(
                column_name="datetime_combined",
                format=DatetimeFormat.YEAR_MONTH_COMBINED,
                source="auto_combined",
            )

        # Priority 3: Auto-detect best candidate
        best_col = None
        best_score = 0.0

        for col in df.columns:
            score = DatetimeDetector._score_column(df[col], col)
            if score > best_score:
                best_score = score
                best_col = col

        if best_col and best_score > 0.3:
            fmt = DatetimeDetector.infer_format(df[best_col], best_col)
            return DatetimeInfo(column_name=best_col, format=fmt, source="auto_detected")

        return None

    @staticmethod
    def infer_format(series: pd.Series, column_name: str = "") -> DatetimeFormat:
        """Infer datetime format from actual data values and column name hints."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return DatetimeFormat.FULL_DATETIME

        sample = series.dropna()
        if len(sample) == 0:
            return DatetimeFormat.FULL_DATETIME

        numeric = pd.to_numeric(sample, errors="coerce").dropna()
        col_lower = column_name.lower()

        if len(numeric) > 0 and len(numeric) >= len(sample) * 0.8:
            min_val, max_val = numeric.min(), numeric.max()

            if 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100:
                return DatetimeFormat.YEAR_ONLY
            if min_val > 1e12 or "millis" in col_lower or "epoch" in col_lower:
                return DatetimeFormat.EPOCH_MILLIS
            if min_val > 1e9:
                return DatetimeFormat.EPOCH_SECONDS

        # Try parsing a small sample as datetime
        test_sample = sample.head(20)
        try:
            parsed = pd.to_datetime(test_sample, errors="coerce")
            success_rate = parsed.notna().sum() / len(test_sample)
            if success_rate > 0.8:
                has_time = any(
                    t.hour != 0 or t.minute != 0 or t.second != 0
                    for t in parsed.dropna()
                )
                return DatetimeFormat.FULL_DATETIME if has_time else DatetimeFormat.DATE_ONLY
        except Exception:
            pass

        return DatetimeFormat.FULL_DATETIME  # fallback

    @staticmethod
    def _score_column(series: pd.Series, column_name: str) -> float:
        """Score how likely a column is to be a datetime column (0.0 to 1.0)."""
        if series.dtype == bool or series.dtype == np.bool_:
            return 0.0

        score = 0.0
        col_lower = column_name.lower()

        # Already datetime type
        if pd.api.types.is_datetime64_any_dtype(series):
            return 1.0

        # Name-based scoring
        if col_lower in _COMMON_DATETIME_NAMES:
            score += 0.5
        elif any(hint in col_lower for hint in ("date", "time", "epoch", "millis", "created", "updated")):
            score += 0.3

        # Data-based scoring: try parsing a sample
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return 0.0

        # Check numeric ranges (year or epoch)
        numeric = pd.to_numeric(sample, errors="coerce").dropna()
        if len(numeric) >= len(sample) * 0.8:
            min_val, max_val = numeric.min(), numeric.max()
            if 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100:
                score += 0.2  # could be year, but also could be generic int
            elif min_val > 1e9:
                score += 0.4  # likely epoch
        else:
            # Non-numeric: try datetime parsing
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                success_rate = parsed.notna().sum() / len(sample)
                score += success_rate * 0.5
            except Exception:
                pass

        return min(score, 1.0)

    @staticmethod
    def convert_to_datetime(series: pd.Series, fmt: DatetimeFormat) -> pd.Series:
        """Convert series to datetime64 based on detected format."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return series

        if fmt == DatetimeFormat.YEAR_ONLY:
            return pd.to_datetime(series.astype(int).astype(str), format="%Y", errors="coerce")
        elif fmt == DatetimeFormat.EPOCH_MILLIS:
            return pd.to_datetime(series, unit="ms", errors="coerce")
        elif fmt == DatetimeFormat.EPOCH_SECONDS:
            return pd.to_datetime(series, unit="s", errors="coerce")
        else:
            return pd.to_datetime(series, errors="coerce")

    @staticmethod
    def combine_year_month(df: pd.DataFrame) -> pd.Series:
        """Combine separate year and month columns into a datetime series."""
        month_numeric = df["month"].apply(_convert_month_to_numeric)
        return pd.to_datetime(
            df["year"].astype(int).astype(str)
            + "-"
            + month_numeric.astype(int).astype(str).str.zfill(2)
            + "-01",
            format="%Y-%m-%d",
            errors="coerce",
        )

    @staticmethod
    def suggest_time_grouping(datetime_info: Optional[DatetimeInfo], series: Optional[pd.Series] = None) -> str:
        """Suggest appropriate time grouping based on format and data range."""
        if datetime_info is None:
            return "year"

        if datetime_info.format == DatetimeFormat.YEAR_ONLY:
            return "year"
        if datetime_info.format == DatetimeFormat.YEAR_MONTH_COMBINED:
            return "month"

        # For full datetime, decide based on data range
        if series is not None and len(series) > 0:
            try:
                dt = pd.to_datetime(series, errors="coerce").dropna()
                if len(dt) > 0:
                    span_days = (dt.max() - dt.min()).days
                    if span_days < 365 * 2:
                        return "month"
                    if span_days < 365 * 5:
                        return "quarter"
            except Exception:
                pass

        return "year"


def _convert_month_to_numeric(month_val) -> int:
    """Convert month name or number to numeric (1-12)."""
    if pd.isna(month_val):
        return 1

    if isinstance(month_val, (int, float)):
        month_num = int(month_val)
        return month_num if 1 <= month_num <= 12 else 1

    month_str = str(month_val).strip()
    if month_str == "None":
        return 1

    month_lower = month_str.lower()
    if month_lower in _MONTH_MAP:
        return _MONTH_MAP[month_lower]

    try:
        month_num = int(month_str)
        return month_num if 1 <= month_num <= 12 else 1
    except ValueError:
        return 1
