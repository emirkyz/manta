#!/usr/bin/env python3
"""
TEMPORAL TOPIC FORECASTING SCRIPT
==================================

Description:
    Performs Facebook Prophet-based time series forecasting on temporal topic
    distribution data. Designed for quarterly data with multiple topics.

Input Format:
    CSV file with structure:
        period,Topic 1,Topic 2,...,Topic N
        2000Q1,80.20,139.43,...,6.40
        2000Q2,86.49,127.41,...,11.48
        ...

Output:
    - CSV: Combined historical + forecast data
    - PNG: Static matplotlib visualization
    - HTML: Interactive ECharts visualization

Usage:
    1. Edit the CONFIGURATION section below with your file path and preferences
    2. Run: python temporal_forecast.py

Installation:
    pip install pandas numpy prophet matplotlib seaborn tqdm

    Note: Prophet requires PyStan. On some systems you may need:
        conda install -c conda-forge prophet

Dependencies:
    - pandas >= 1.5.0
    - numpy >= 1.23.0
    - prophet >= 1.1.5 (Facebook Prophet)
    - matplotlib >= 3.5.0
    - seaborn >= 0.12.0
    - tqdm >= 4.65.0 (optional, for progress bars)

Author: Claude Code
License: MIT
Version: 1.0.0
"""

# ==================== CONFIGURATION ====================
# Edit these settings before running the script

# Path to your input CSV file (REQUIRED)
INPUT_FILE = "/Users/emirkarayagiz/Work/radiology-topic-analysis/to_generate_from/heart_failure_with_pagerank_nmtf_bpe_34/heart_failure_with_pagerank_nmtf_bpe_34_temporal_topic_dist_quarter.csv"

# Output directory (set to None to use same directory as input file)
OUTPUT_DIR = "forecast_deneme"  # or specify a path like "./output/"

# Forecasting parameters
FORECAST_PERIODS = 20  # Number of quarters to forecast (20 = 5 years)
CONFIDENCE_INTERVAL = 0.95  # Confidence level (0.95 = 95%)

# Validation/Backtesting mode
VALIDATION_MODE = False  # Set to True to test model on last 5 years of known data
VALIDATION_PERIODS = 20  # Number of quarters to hold out for validation (20 = 5 years)

# Topic selection (set to None to forecast all topics, or list specific topics)
# Examples:
#   TOPIC_SUBSET = None  # Forecast all topics
#   TOPIC_SUBSET = [1, 5, 10, 15]  # Forecast only these topic numbers
TOPIC_SUBSET = ""

# Prophet settings
SEASONALITY_MODE = 'multiplicative'  # 'multiplicative' or 'additive'

# Output control
GENERATE_CSV = True   # Generate CSV output files
GENERATE_PLOT = True  # Generate static PNG plot (combined)
GENERATE_INDIVIDUAL_PLOTS = True  # Generate individual PNG plots for each topic
GENERATE_HTML = True  # Generate interactive HTML visualization

# Logging
VERBOSE = False  # Set to True for detailed debug output

# ==================== END CONFIGURATION ====================

import logging
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
import colorsys

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import seaborn as sns

# Optional: tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ==================== UTILITY FUNCTIONS ====================

def setup_logging(verbose: bool) -> logging.Logger:
    """
    Configure logging with appropriate level.

    Args:
        verbose: If True, set DEBUG level, otherwise INFO

    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Suppress Prophet's verbose output unless in debug mode
    if not verbose:
        logging.getLogger('prophet').setLevel(logging.WARNING)
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

    return logger


def _generate_distinct_colors(n_topics: int) -> List[tuple]:
    """
    Generate maximally distinct colors for topics.

    Uses predefined high-contrast palettes for small numbers, and greedy
    algorithm for larger numbers of topics.

    Args:
        n_topics: Number of distinct colors needed

    Returns:
        List of RGB tuples with maximally distinct colors
    """
    # Predefined high-contrast color palettes
    distinct_palettes = {
        2: ['#E31A1C', '#1F78B4'],
        3: ['#E31A1C', '#33A02C', '#1F78B4'],
        4: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4'],
        5: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A'],
        6: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99'],
        7: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99', '#B15928'],
        8: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99', '#B15928', '#FDBF6F'],
        9: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99', '#B15928', '#FDBF6F', '#CAB2D6'],
        10: ['#E31A1C', '#FF7F00', '#33A02C', '#1F78B4', '#6A3D9A', '#FB9A99', '#B15928', '#FDBF6F', '#CAB2D6', '#FFFF99'],
    }

    if n_topics <= 10 and n_topics in distinct_palettes:
        colors = distinct_palettes[n_topics]
        return [mcolors.hex2color(color) for color in colors]

    # For larger numbers, use tab20 with optimized ordering
    if n_topics <= 20:
        base_colors = plt.cm.tab20(np.arange(20))
        optimized_order = [0, 10, 2, 12, 4, 14, 6, 16, 8, 18, 1, 11, 3, 13, 5, 15, 7, 17, 9, 19]
        reordered_colors = [base_colors[i] for i in optimized_order[:n_topics]]
        return [(r, g, b, a) for r, g, b, a in reordered_colors]

    # For very large numbers, use greedy color selection
    return _generate_greedy_distinct_colors(n_topics)


def _generate_greedy_distinct_colors(n_topics: int) -> List[tuple]:
    """
    Generate colors using greedy algorithm for maximum perceptual distance.

    Args:
        n_topics: Number of colors needed

    Returns:
        List of RGB tuples
    """
    if n_topics <= 1:
        return [(0.8, 0.2, 0.2, 1.0)]

    colors = [(0.8, 0.2, 0.2, 1.0)]  # Start with red

    for i in range(1, n_topics):
        best_color = None
        best_min_distance = 0

        for _ in range(100):
            h = np.random.uniform(0, 1)
            s = np.random.uniform(0.6, 1.0)
            v = np.random.uniform(0.4, 0.9)

            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            candidate = (r, g, b, 1.0)

            min_distance = min(
                np.sqrt((r - c[0])**2 + (g - c[1])**2 + (b - c[2])**2)
                for c in colors
            )

            if min_distance > best_min_distance:
                best_min_distance = min_distance
                best_color = candidate

        if best_color:
            colors.append(best_color)
        else:
            h = i / n_topics
            r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.7)
            colors.append((r, g, b, 1.0))

    return colors


# ==================== INPUT HANDLING ====================

def load_configuration():
    """Load configuration from global variables."""
    class Config:
        def __init__(self):
            self.input = INPUT_FILE
            self.output_dir = OUTPUT_DIR
            self.forecast_periods = FORECAST_PERIODS
            self.confidence_interval = CONFIDENCE_INTERVAL
            self.validation_mode = VALIDATION_MODE
            self.validation_periods = VALIDATION_PERIODS
            self.topic_subset = TOPIC_SUBSET
            self.seasonality_mode = SEASONALITY_MODE
            self.no_csv = not GENERATE_CSV
            self.no_plot = not GENERATE_PLOT
            self.no_individual_plots = not GENERATE_INDIVIDUAL_PLOTS
            self.no_html = not GENERATE_HTML
            self.verbose = VERBOSE

    return Config()


def parse_quarter_to_datetime(quarter_str: str) -> pd.Timestamp:
    """
    Convert "YYYYQN" format to datetime.

    Args:
        quarter_str: Quarter string like "2000Q1", "2025Q3"

    Returns:
        pandas Timestamp for the first day of that quarter

    Raises:
        ValueError: If quarter format is invalid
    """
    try:
        if not isinstance(quarter_str, str) or 'Q' not in quarter_str:
            raise ValueError(f"Invalid quarter format: {quarter_str}")

        year_str, quarter_str = quarter_str.split('Q')
        year = int(year_str)
        quarter = int(quarter_str)

        if quarter < 1 or quarter > 4:
            raise ValueError(f"Quarter must be 1-4, got {quarter}")

        # Map quarter to month (Q1→1, Q2→4, Q3→7, Q4→10)
        month = (quarter - 1) * 3 + 1

        return pd.Timestamp(year=year, month=month, day=1)

    except Exception as e:
        raise ValueError(f"Failed to parse quarter '{quarter_str}': {e}")


def validate_and_load_csv(file_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Load and validate input CSV structure.

    Args:
        file_path: Path to CSV file
        logger: Logger instance

    Returns:
        Validated DataFrame with period as index

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV format is invalid
    """
    # Check file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # Load CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    # Validate structure
    if df.empty:
        raise ValueError("CSV file is empty")

    if 'period' not in df.columns:
        raise ValueError(
            f"CSV must have 'period' as first column. "
            f"Found columns: {list(df.columns)}"
        )

    # Check for topic columns
    topic_cols = [col for col in df.columns if col.startswith('Topic ')]
    if len(topic_cols) < 2:
        raise ValueError(
            f"CSV must have at least 2 topic columns (e.g., 'Topic 1', 'Topic 2'). "
            f"Found {len(topic_cols)} topic columns"
        )

    # Validate period format
    try:
        df['datetime'] = df['period'].apply(parse_quarter_to_datetime)
    except Exception as e:
        raise ValueError(f"Invalid period format in CSV: {e}")

    # Check minimum data points
    if len(df) < 8:
        logger.warning(
            f"Only {len(df)} data points found. "
            f"Prophet recommends at least 8 quarters for quarterly seasonality."
        )

    # Check for numeric topic values
    for col in topic_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Topic column '{col}' must contain numeric values")

    # Check for duplicates
    if df['period'].duplicated().any():
        duplicates = df[df['period'].duplicated()]['period'].tolist()
        raise ValueError(f"Duplicate periods found: {duplicates}")

    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)

    # Check for gaps
    expected_quarters = pd.date_range(
        start=df['datetime'].min(),
        end=df['datetime'].max(),
        freq='QS'
    )
    if len(df) < len(expected_quarters):
        logger.warning(
            f"Detected {len(expected_quarters) - len(df)} missing quarters in timeline. "
            f"Prophet will interpolate missing values."
        )

    logger.info(f"Loaded {len(df)} quarters from {df['period'].iloc[0]} to {df['period'].iloc[-1]}")
    logger.info(f"Found {len(topic_cols)} topics")

    return df


# ==================== FORECASTING ENGINE ====================

def prepare_prophet_data(df: pd.DataFrame, topic_col: str) -> pd.DataFrame:
    """
    Convert topic time series to Prophet's required format.

    Args:
        df: DataFrame with 'datetime' and topic columns
        topic_col: Name of topic column to prepare

    Returns:
        DataFrame with 'ds' (datetime) and 'y' (value) columns
    """
    prophet_df = pd.DataFrame({
        'ds': df['datetime'],
        'y': df[topic_col]
    })

    # Remove NaN values
    prophet_df = prophet_df.dropna()

    return prophet_df


def forecast_single_topic(
    data: pd.DataFrame,
    topic_name: str,
    n_periods: int,
    seasonality_mode: str = 'multiplicative',
    interval_width: float = 0.95,
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, Prophet]:
    """
    Forecast a single topic using Facebook Prophet.

    Args:
        data: DataFrame with 'ds' and 'y' columns
        topic_name: Name of topic (for logging)
        n_periods: Number of quarters to forecast
        seasonality_mode: 'additive' or 'multiplicative'
        interval_width: Confidence interval width (e.g., 0.95 for 95%)
        logger: Logger instance

    Returns:
        Tuple of (forecast DataFrame, trained model)
    """
    if logger:
        logger.debug(f"Forecasting {topic_name}...")

    # Check for constant or zero values
    if data['y'].std() == 0:
        if logger:
            logger.warning(f"{topic_name}: All values are constant. Adding small noise.")
        data = data.copy()
        data['y'] = data['y'] + np.random.normal(0, 1e-6, len(data))

    # Configure Prophet
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = Prophet(
            seasonality_mode=seasonality_mode,
            interval_width=interval_width,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )

        # Add explicit quarterly seasonality
        model.add_seasonality(
            name='quarterly',
            period=365.25 / 4,
            fourier_order=5
        )

        # Fit model
        model.fit(data)

    # Create future dataframe
    future = model.make_future_dataframe(
        periods=n_periods,
        freq='QS'  # Quarter start
    )

    # Make forecast
    forecast = model.predict(future)

    # Check for negative forecasts
    if (forecast['yhat'] < 0).any():
        if logger:
            logger.warning(
                f"{topic_name}: Forecast contains negative values. "
                f"This is common with Prophet. Consider using additive seasonality."
            )

    return forecast, model


def forecast_all_topics(
    df: pd.DataFrame,
    topic_cols: List[str],
    n_periods: int,
    seasonality_mode: str = 'multiplicative',
    interval_width: float = 0.95,
    verbose: bool = False,
    logger: logging.Logger = None
) -> Dict[str, Tuple[pd.DataFrame, Prophet]]:
    """
    Forecast all topics with progress tracking.

    Args:
        df: Input DataFrame with datetime and topic columns
        topic_cols: List of topic column names to forecast
        n_periods: Number of quarters to forecast
        seasonality_mode: Prophet seasonality mode
        interval_width: Confidence interval width
        verbose: Enable verbose output
        logger: Logger instance

    Returns:
        Dictionary mapping topic names to (forecast_df, model) tuples
    """
    forecasts = {}
    failed_topics = []

    # Progress tracking
    if TQDM_AVAILABLE and not verbose:
        iterator = tqdm(topic_cols, desc="Forecasting topics")
    else:
        iterator = topic_cols
        if logger:
            logger.info(f"Forecasting {len(topic_cols)} topics...")

    for i, topic in enumerate(iterator):
        if not TQDM_AVAILABLE and not verbose and logger:
            print(f"  [{i+1}/{len(topic_cols)}] {topic}...", end='\r')

        try:
            # Prepare data
            prophet_data = prepare_prophet_data(df, topic)

            # Forecast
            forecast_df, model = forecast_single_topic(
                data=prophet_data,
                topic_name=topic,
                n_periods=n_periods,
                seasonality_mode=seasonality_mode,
                interval_width=interval_width,
                logger=logger if verbose else None
            )

            forecasts[topic] = (forecast_df, model)

        except Exception as e:
            failed_topics.append(topic)
            if logger:
                logger.error(f"Failed to forecast {topic}: {e}")
            continue

    if not TQDM_AVAILABLE and not verbose:
        print()  # Clear progress line

    if logger:
        logger.info(f"Successfully forecasted {len(forecasts)}/{len(topic_cols)} topics")
        if failed_topics:
            logger.warning(f"Failed topics: {', '.join(failed_topics)}")

    return forecasts


def calculate_validation_metrics(
    actual_values: np.ndarray,
    forecast_values: np.ndarray,
    topic_name: str
) -> dict:
    """
    Calculate accuracy metrics for validation.

    Args:
        actual_values: Actual values from validation period
        forecast_values: Forecasted values
        topic_name: Name of topic

    Returns:
        Dictionary with accuracy metrics
    """
    # Remove any NaN values
    mask = ~(np.isnan(actual_values) | np.isnan(forecast_values))
    actual = actual_values[mask]
    forecast = forecast_values[mask]

    if len(actual) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'mape': np.nan,
            'r2': np.nan
        }

    # Mean Absolute Error
    mae = np.mean(np.abs(actual - forecast))

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))

    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - forecast) / (actual + 1e-10))) * 100

    # R-squared
    ss_res = np.sum((actual - forecast) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


# ==================== OUTPUT GENERATION ====================

def create_combined_dataframe(
    historical_df: pd.DataFrame,
    forecasts: Dict[str, Tuple[pd.DataFrame, Prophet]],
    n_historical: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Combine historical data with forecasts.

    Args:
        historical_df: Original historical data
        forecasts: Dictionary of forecast results
        n_historical: Number of historical periods

    Returns:
        Tuple of (main_df, lower_df, upper_df) DataFrames
    """
    # Get forecast periods from first topic
    first_topic = list(forecasts.keys())[0]
    all_dates = forecasts[first_topic][0]['ds'].values
    forecast_dates = all_dates[n_historical:]

    # Create period strings for forecast
    forecast_periods = []
    for date in forecast_dates:
        dt = pd.Timestamp(date)
        quarter = (dt.month - 1) // 3 + 1
        period_str = f"{dt.year}Q{quarter}"
        forecast_periods.append(period_str)

    # Initialize forecast DataFrames
    forecast_main = pd.DataFrame({'period': forecast_periods})
    forecast_lower = pd.DataFrame({'period': forecast_periods})
    forecast_upper = pd.DataFrame({'period': forecast_periods})

    # Add forecast values for each topic
    for topic, (forecast_df, _) in forecasts.items():
        forecast_values = forecast_df['yhat'].values[n_historical:]
        lower_values = forecast_df['yhat_lower'].values[n_historical:]
        upper_values = forecast_df['yhat_upper'].values[n_historical:]

        forecast_main[topic] = forecast_values
        forecast_lower[topic] = lower_values
        forecast_upper[topic] = upper_values

    # Combine with historical
    topic_cols = [col for col in historical_df.columns if col.startswith('Topic ')]
    historical_main = historical_df[['period'] + topic_cols].copy()

    # Concatenate
    combined_main = pd.concat([historical_main, forecast_main], ignore_index=True)

    return combined_main, forecast_lower, forecast_upper


def save_forecast_csv(
    combined_df: pd.DataFrame,
    lower_df: pd.DataFrame,
    upper_df: pd.DataFrame,
    output_path: Path,
    logger: logging.Logger
) -> List[Path]:
    """
    Save forecast results to CSV files.

    Args:
        combined_df: Main forecast DataFrame
        lower_df: Lower confidence bounds
        upper_df: Upper confidence bounds
        output_path: Base output path
        logger: Logger instance

    Returns:
        List of created file paths
    """
    created_files = []

    # Save main forecast
    combined_df.to_csv(output_path, index=False)
    created_files.append(output_path)
    logger.info(f"Saved forecast CSV: {output_path}")

    # Save confidence bounds
    lower_path = output_path.parent / f"{output_path.stem}_lower.csv"
    lower_df.to_csv(lower_path, index=False)
    created_files.append(lower_path)
    logger.info(f"Saved lower bounds CSV: {lower_path}")

    upper_path = output_path.parent / f"{output_path.stem}_upper.csv"
    upper_df.to_csv(upper_path, index=False)
    created_files.append(upper_path)
    logger.info(f"Saved upper bounds CSV: {upper_path}")

    return created_files


def create_static_plot(
    historical_df: pd.DataFrame,
    forecasts: Dict[str, Tuple[pd.DataFrame, Prophet]],
    output_path: Path,
    logger: logging.Logger
) -> None:
    """
    Generate static matplotlib visualization.

    Args:
        historical_df: Historical data
        forecasts: Forecast results
        output_path: Output file path
        logger: Logger instance
    """
    n_topics = len(forecasts)
    colors = _generate_distinct_colors(n_topics)

    # Get historical periods
    n_historical = len(historical_df)

    # Decide between single plot or faceted
    if n_topics <= 10:
        # Single plot with overlapping lines
        fig, ax = plt.subplots(figsize=(16, 10))

        for idx, (topic, (forecast_df, _)) in enumerate(forecasts.items()):
            color = colors[idx]

            # Get data
            all_dates = forecast_df['ds'].values
            all_values = forecast_df['yhat'].values
            lower_bounds = forecast_df['yhat_lower'].values
            upper_bounds = forecast_df['yhat_upper'].values

            hist_dates = all_dates[:n_historical]
            hist_values = all_values[:n_historical]

            forecast_dates = all_dates[n_historical:]
            forecast_values = all_values[n_historical:]
            forecast_lower = lower_bounds[n_historical:]
            forecast_upper = upper_bounds[n_historical:]

            # Plot historical (solid line)
            ax.plot(hist_dates, hist_values,
                   color=color, linewidth=2, label=topic, alpha=0.8)

            # Plot forecast (dashed line)
            ax.plot(forecast_dates, forecast_values,
                   color=color, linewidth=2, linestyle='--', alpha=0.8)

            # Confidence interval (shaded region)
            ax.fill_between(forecast_dates, forecast_lower, forecast_upper,
                           color=color, alpha=0.2)

        # Vertical separator
        separator_date = all_dates[n_historical - 1]
        ax.axvline(x=separator_date, color='red', linestyle=':', linewidth=2,
                  label='Forecast Start', alpha=0.7)

        # Formatting
        ax.set_xlabel('Quarter', fontsize=12)
        ax.set_ylabel('Topic Weight', fontsize=12)
        ax.set_title('Temporal Topic Forecast (5 Years)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(bbox_to_anchor=(0.5, -0.08), loc='upper center',
                 ncol=min(6, n_topics), fontsize=10)

        plt.tight_layout()

    else:
        # Faceted subplots
        n_cols = 5
        n_rows = (n_topics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, (topic, (forecast_df, _)) in enumerate(forecasts.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            color = colors[idx]

            # Get data
            all_dates = forecast_df['ds'].values
            all_values = forecast_df['yhat'].values
            lower_bounds = forecast_df['yhat_lower'].values
            upper_bounds = forecast_df['yhat_upper'].values

            hist_dates = all_dates[:n_historical]
            hist_values = all_values[:n_historical]

            forecast_dates = all_dates[n_historical:]
            forecast_values = all_values[n_historical:]
            forecast_lower = lower_bounds[n_historical:]
            forecast_upper = upper_bounds[n_historical:]

            # Plot
            ax.plot(hist_dates, hist_values, color=color, linewidth=1.5, alpha=0.8)
            ax.plot(forecast_dates, forecast_values, color=color, linewidth=1.5,
                   linestyle='--', alpha=0.8)
            ax.fill_between(forecast_dates, forecast_lower, forecast_upper,
                           color=color, alpha=0.2)

            # Separator
            separator_date = all_dates[n_historical - 1]
            ax.axvline(x=separator_date, color='red', linestyle=':', linewidth=1, alpha=0.5)

            # Formatting
            ax.set_title(topic, fontsize=10, fontweight='bold')
            ax.grid(alpha=0.2, linestyle='--')
            ax.tick_params(labelsize=8)

            # Rotate x-axis labels
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

        # Hide empty subplots
        for idx in range(n_topics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        fig.suptitle('Temporal Topic Forecasts (5 Years)', fontsize=16, fontweight='bold')
        plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved forecast plot: {output_path}")


def create_individual_topic_plots(
    historical_df: pd.DataFrame,
    forecasts: Dict[str, Tuple[pd.DataFrame, Prophet]],
    output_dir: Path,
    base_name: str,
    logger: logging.Logger,
    validation_mode: bool = False
) -> List[Path]:
    """
    Generate individual PNG plots for each topic.

    Args:
        historical_df: Historical data
        forecasts: Forecast results
        output_dir: Output directory path
        base_name: Base name for output files
        logger: Logger instance
        validation_mode: Whether in validation mode

    Returns:
        List of created file paths
    """
    created_files = []
    n_historical = len(historical_df)
    n_topics = len(forecasts)
    colors = _generate_distinct_colors(n_topics)

    # Create subdirectory for individual plots
    suffix = "_validation" if validation_mode else "_forecast"
    individual_dir = output_dir / f"{base_name}{suffix}_individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating individual topic plots in: {individual_dir}")

    for idx, (topic, (forecast_df, _)) in enumerate(forecasts.items()):
        color = colors[idx]

        # Get data
        all_dates = forecast_df['ds'].values
        all_values = forecast_df['yhat'].values
        lower_bounds = forecast_df['yhat_lower'].values
        upper_bounds = forecast_df['yhat_upper'].values

        hist_dates = all_dates[:n_historical]
        hist_values = all_values[:n_historical]

        forecast_dates = all_dates[n_historical:]
        forecast_values = all_values[n_historical:]
        forecast_lower = lower_bounds[n_historical:]
        forecast_upper = upper_bounds[n_historical:]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot historical (solid line)
        ax.plot(hist_dates, hist_values,
               color=color, linewidth=2.5, label='Historical', alpha=0.9)

        # Plot forecast (dashed line)
        ax.plot(forecast_dates, forecast_values,
               color=color, linewidth=2.5, linestyle='--', label='Forecast', alpha=0.9)

        # Confidence interval (shaded region)
        ax.fill_between(forecast_dates, forecast_lower, forecast_upper,
                       color=color, alpha=0.25, label='95% CI')

        # Vertical separator
        separator_date = all_dates[n_historical - 1]
        ax.axvline(x=separator_date, color='red', linestyle=':', linewidth=2,
                  label='Forecast Start', alpha=0.7)

        # Formatting
        ax.set_xlabel('Quarter', fontsize=12)
        ax.set_ylabel('Topic Weight', fontsize=12)
        ax.set_title(f'{topic} - Temporal Forecast', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Create safe filename from topic name
        safe_topic_name = topic.replace(' ', '_').replace('/', '_')
        plot_path = individual_dir / f"{safe_topic_name}{suffix}.png"

        # Save
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        created_files.append(plot_path)

    logger.info(f"Saved {len(created_files)} individual topic plots")

    return created_files


def create_interactive_html(
    historical_df: pd.DataFrame,
    forecasts: Dict[str, Tuple[pd.DataFrame, Prophet]],
    output_path: Path,
    logger: logging.Logger,
    validation_data: pd.DataFrame = None
) -> None:
    """
    Generate interactive HTML visualization.

    Args:
        historical_df: Historical data (training data in validation mode)
        forecasts: Forecast results
        output_path: Output file path
        logger: Logger instance
        validation_data: Optional validation data for comparison
    """
    n_historical = len(historical_df)
    is_validation_mode = validation_data is not None

    # Prepare data
    topic_names = list(forecasts.keys())

    # Get all periods
    first_topic = topic_names[0]
    all_dates = forecasts[first_topic][0]['ds'].values

    # Create period strings
    all_periods = []
    for date in all_dates:
        dt = pd.Timestamp(date)
        quarter = (dt.month - 1) // 3 + 1
        period_str = f"{dt.year}Q{quarter}"
        all_periods.append(period_str)

    # Build data structure
    chart_data = []
    for i, period in enumerate(all_periods):
        row = {'period': period, 'is_forecast': i >= n_historical}
        for topic in topic_names:
            forecast_df, _ = forecasts[topic]
            row[topic] = float(forecast_df['yhat'].values[i])
            row[f"{topic}_lower"] = float(forecast_df['yhat_lower'].values[i])
            row[f"{topic}_upper"] = float(forecast_df['yhat_upper'].values[i])

            # Add actual values if in validation mode
            if is_validation_mode and i >= n_historical:
                validation_idx = i - n_historical
                if validation_idx < len(validation_data):
                    row[f"{topic}_actual"] = float(validation_data[topic].values[validation_idx])
                else:
                    row[f"{topic}_actual"] = None
            else:
                row[f"{topic}_actual"] = None

        chart_data.append(row)

    # Generate colors
    colors = _generate_distinct_colors(len(topic_names))
    color_map = {
        topic: f'rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},0.8)'
        for topic, c in zip(topic_names, colors)
    }

    # Create HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topic Forecast Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}

        .header p {{
            opacity: 0.9;
            font-size: 1.1em;
        }}

        .controls {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}

        .topic-selector {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}

        .topic-checkbox {{
            display: flex;
            align-items: center;
            background: white;
            padding: 8px 12px;
            border-radius: 5px;
            border: 2px solid #dee2e6;
            cursor: pointer;
            transition: all 0.3s;
        }}

        .topic-checkbox:hover {{
            border-color: #667eea;
            transform: translateY(-2px);
        }}

        .topic-checkbox input {{
            margin-right: 8px;
        }}

        #chart {{
            width: 100%;
            height: 600px;
            padding: 20px;
        }}

        .info {{
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
            text-align: center;
        }}

        .info p {{
            margin: 5px 0;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Temporal Topic Forecast</h1>
            <p>Interactive 5-Year Forecast with 95% Confidence Intervals</p>
        </div>

        <div class="controls">
            <h3>Select Topics to Display:</h3>
            <div class="topic-selector" id="topicSelector">
                <!-- Topic checkboxes will be inserted here -->
            </div>
        </div>

        <div id="chart"></div>

        <div class="info">
            <p><strong>Historical Data:</strong> {historical_df['period'].iloc[0]} - {historical_df['period'].iloc[-1]} ({n_historical} quarters)</p>
            <p><strong>Forecast:</strong> {all_periods[n_historical]} - {all_periods[-1]} ({len(all_periods) - n_historical} quarters)</p>
            <p><em>Dashed lines indicate forecasted values. Shaded areas show 95% confidence intervals.</em></p>
        </div>
    </div>

    <script>
        // Embedded data
        const CHART_DATA = {json.dumps(chart_data, indent=2)};
        const TOPIC_NAMES = {json.dumps(topic_names)};
        const COLOR_MAP = {json.dumps(color_map)};
        const N_HISTORICAL = {n_historical};

        // Initialize
        let selectedTopics = TOPIC_NAMES.slice(0, Math.min(5, TOPIC_NAMES.length));

        // Create topic selector
        const selectorDiv = document.getElementById('topicSelector');
        TOPIC_NAMES.forEach(topic => {{
            const label = document.createElement('label');
            label.className = 'topic-checkbox';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = topic;
            checkbox.checked = selectedTopics.includes(topic);
            checkbox.addEventListener('change', updateChart);

            const span = document.createElement('span');
            span.textContent = topic;

            label.appendChild(checkbox);
            label.appendChild(span);
            selectorDiv.appendChild(label);
        }});

        // Initialize chart
        const chart = echarts.init(document.getElementById('chart'));

        function updateChart() {{
            // Get selected topics
            selectedTopics = Array.from(
                document.querySelectorAll('.topic-checkbox input:checked')
            ).map(cb => cb.value);

            if (selectedTopics.length === 0) {{
                chart.setOption({{
                    title: {{
                        text: 'Please select at least one topic',
                        left: 'center',
                        top: 'center'
                    }}
                }});
                return;
            }}

            // Prepare series
            const series = [];
            selectedTopics.forEach(topic => {{
                const color = COLOR_MAP[topic];

                // Main line
                series.push({{
                    name: topic,
                    type: 'line',
                    data: CHART_DATA.map((d, i) => [
                        d.period,
                        d[topic]
                    ]),
                    lineStyle: {{
                        color: color,
                        width: 2,
                        type: function(params) {{
                            return params.dataIndex >= N_HISTORICAL ? 'dashed' : 'solid';
                        }}
                    }},
                    itemStyle: {{
                        color: color
                    }},
                    symbol: 'circle',
                    symbolSize: 4,
                    emphasis: {{
                        focus: 'series'
                    }}
                }});

                // Actual values line (validation mode only)
                const hasActualData = CHART_DATA.some(d => d[topic + '_actual'] !== null);
                if (hasActualData) {{
                    series.push({{
                        name: topic + ' (Actual)',
                        type: 'line',
                        data: CHART_DATA.map((d, i) => [
                            d.period,
                            d[topic + '_actual']
                        ]),
                        lineStyle: {{
                            color: color,
                            width: 3,
                            type: 'solid'
                        }},
                        itemStyle: {{
                            color: color,
                            borderWidth: 2,
                            borderColor: '#fff'
                        }},
                        symbol: 'circle',
                        symbolSize: 6,
                        emphasis: {{
                            focus: 'series'
                        }},
                        z: 10  // Render on top
                    }});
                }}

                // Confidence interval as filled area - simpler approach
                // Build polygon coordinates: lower bounds forward + upper bounds backward
                const forecastIndices = [];
                for (let i = N_HISTORICAL; i < CHART_DATA.length; i++) {{
                    forecastIndices.push(i);
                }}

                if (forecastIndices.length > 0) {{
                    // Create coordinates for polygon (lower path + upper path reversed)
                    const lowerPath = forecastIndices.map(i => [
                        CHART_DATA[i].period,
                        CHART_DATA[i][topic + '_lower']
                    ]);
                    const upperPath = forecastIndices.map(i => [
                        CHART_DATA[i].period,
                        CHART_DATA[i][topic + '_upper']
                    ]).reverse();

                    const polygonData = [...lowerPath, ...upperPath];

                    series.push({{
                        name: topic + ' CI',
                        type: 'custom',
                        renderItem: function(params, api) {{
                            if (params.dataIndex !== 0) return null;

                            const points = polygonData.map(point =>
                                api.coord(point)
                            );

                            return {{
                                type: 'polygon',
                                shape: {{
                                    points: points
                                }},
                                style: {{
                                    fill: color.replace('0.8', '0.15'),
                                    stroke: 'none'
                                }},
                                silent: true
                            }};
                        }},
                        data: [1],  // Single item to trigger render
                        tooltip: {{ show: false }},
                        z: -1
                    }});
                }}
            }});

            // Set options
            const option = {{
                tooltip: {{
                    trigger: 'axis',
                    axisPointer: {{
                        type: 'cross'
                    }},
                    formatter: function(params) {{
                        const period = params[0].axisValue;
                        const isForecast = CHART_DATA.findIndex(d => d.period === period) >= N_HISTORICAL;

                        let html = `<strong>${{period}}</strong>`;
                        if (isForecast) {{
                            html += ' <span style="color: #f00;">(Forecast)</span>';
                        }}
                        html += '<br/>';

                        params.forEach(param => {{
                            if (!param.seriesName.includes('CI')) {{
                                const seriesName = param.seriesName;

                                // Skip if data is null (e.g., actual values in historical period)
                                if (param.data[1] === null || param.data[1] === undefined) {{
                                    return;
                                }}

                                const value = param.data[1].toFixed(2);

                                // Extract base topic name (remove " (Actual)" suffix if present)
                                const topic = seriesName.replace(' (Actual)', '');

                                const dataPoint = CHART_DATA.find(d => d.period === period);
                                if (isForecast && dataPoint && dataPoint[topic + '_lower'] !== undefined) {{
                                    const lower = dataPoint[topic + '_lower'].toFixed(2);
                                    const upper = dataPoint[topic + '_upper'].toFixed(2);
                                    html += `${{param.marker}} ${{seriesName}}: <strong>${{value}}</strong> [${{lower}}, ${{upper}}]<br/>`;
                                }} else {{
                                    html += `${{param.marker}} ${{seriesName}}: <strong>${{value}}</strong><br/>`;
                                }}
                            }}
                        }});

                        return html;
                    }}
                }},
                legend: {{
                    data: selectedTopics,
                    bottom: 10,
                    type: 'scroll'
                }},
                grid: {{
                    left: '3%',
                    right: '4%',
                    bottom: '15%',
                    top: '5%',
                    containLabel: true
                }},
                xAxis: {{
                    type: 'category',
                    boundaryGap: false,
                    data: CHART_DATA.map(d => d.period),
                    axisLabel: {{
                        rotate: 45
                    }}
                }},
                yAxis: {{
                    type: 'value',
                    name: 'Topic Weight',
                    nameLocation: 'middle',
                    nameGap: 50
                }},
                dataZoom: [
                    {{
                        type: 'inside',
                        start: 0,
                        end: 100
                    }},
                    {{
                        start: 0,
                        end: 100
                    }}
                ],
                series: series
            }};

            chart.setOption(option, true);
        }}

        // Initial render
        updateChart();

        // Handle window resize
        window.addEventListener('resize', () => {{
            chart.resize();
        }});
    </script>
</body>
</html>"""

    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"Saved interactive HTML: {output_path}")


# ==================== SUMMARY & REPORTING ====================

def calculate_trend_summary(
    forecast_df: pd.DataFrame,
    n_historical: int
) -> dict:
    """
    Calculate trend statistics for a topic.

    Args:
        forecast_df: Prophet forecast DataFrame
        n_historical: Number of historical periods

    Returns:
        Dictionary with trend statistics
    """
    last_historical = forecast_df['yhat'].values[n_historical - 1]
    final_forecast = forecast_df['yhat'].values[-1]

    pct_change = ((final_forecast - last_historical) / last_historical * 100
                 if last_historical != 0 else 0)

    avg_forecast = np.mean(forecast_df['yhat'].values[n_historical:])
    ci_width = np.mean(
        forecast_df['yhat_upper'].values[n_historical:] -
        forecast_df['yhat_lower'].values[n_historical:]
    )

    if pct_change > 10:
        trend = 'growing'
    elif pct_change < -10:
        trend = 'declining'
    else:
        trend = 'stable'

    return {
        'pct_change': pct_change,
        'avg_forecast': avg_forecast,
        'ci_width': ci_width,
        'trend': trend
    }


def print_summary_report(
    forecasts: Dict[str, Tuple[pd.DataFrame, Prophet]],
    n_historical: int,
    n_forecast: int,
    output_files: List[Path],
    logger: logging.Logger
) -> None:
    """
    Print summary report to console.

    Args:
        forecasts: Forecast results
        n_historical: Number of historical periods
        n_forecast: Number of forecast periods
        output_files: List of generated output files
        logger: Logger instance
    """
    print("\n" + "=" * 60)
    print("FORECASTING SUMMARY")
    print("=" * 60)
    print(f"Topics Forecasted: {len(forecasts)}")
    print(f"Forecast Horizon: {n_forecast} quarters (~{n_forecast/4:.1f} years)")
    print(f"Confidence Level: 95%")

    # Trend analysis
    growing = []
    declining = []
    stable = []
    warnings_list = []

    for topic, (forecast_df, _) in forecasts.items():
        stats = calculate_trend_summary(forecast_df, n_historical)

        if stats['trend'] == 'growing':
            growing.append((topic, stats['pct_change']))
        elif stats['trend'] == 'declining':
            declining.append((topic, stats['pct_change']))
        else:
            stable.append(topic)

        # Check for warnings
        if (forecast_df['yhat'] < 0).any():
            warnings_list.append(f"{topic}: Contains negative forecast values")

        if stats['ci_width'] > stats['avg_forecast']:
            warnings_list.append(f"{topic}: Very wide confidence intervals (high uncertainty)")

    print("\nTREND ANALYSIS:")

    if growing:
        print(f"  Growing Topics (>{10}% increase):")
        for topic, pct in sorted(growing, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {topic}: +{pct:.1f}%")

    if declining:
        print(f"  Declining Topics (>{10}% decrease):")
        for topic, pct in sorted(declining, key=lambda x: x[1])[:5]:
            print(f"    - {topic}: {pct:.1f}%")

    if stable:
        print(f"  Stable Topics (±10%): {len(stable)} topics")

    print("\nOUTPUT FILES:")
    for path in output_files:
        print(f"  ✓ {path}")

    if warnings_list:
        print("\nWARNINGS:")
        for warning in warnings_list[:10]:  # Limit to first 10
            print(f"  ⚠ {warning}")
        if len(warnings_list) > 10:
            print(f"  ... and {len(warnings_list) - 10} more warnings")

    print("=" * 60 + "\n")


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution flow."""
    # Load configuration
    config = load_configuration()

    # Setup logging
    logger = setup_logging(config.verbose)
    logger.info("Starting temporal topic forecasting...")

    # Validate and load input
    try:
        input_path = Path(config.input)
        df = validate_and_load_csv(input_path, logger)
    except Exception as e:
        logger.error(f"Failed to load input: {e}")
        return 1

    # Determine topics to forecast
    topic_cols = [col for col in df.columns if col.startswith('Topic ')]

    if config.topic_subset:
        # topic_subset is now a list of integers
        topic_cols = [f'Topic {i}' for i in config.topic_subset if f'Topic {i}' in topic_cols]
        if not topic_cols:
            logger.error("No valid topics found in TOPIC_SUBSET")
            return 1

    logger.info(f"Forecasting {len(topic_cols)} topics: {', '.join(topic_cols)}")

    # Handle validation mode
    validation_data = None
    validation_metrics = {}

    if config.validation_mode:
        logger.info(f"\n{'='*60}")
        logger.info("VALIDATION MODE ENABLED")
        logger.info(f"{'='*60}")
        logger.info(f"Using last {config.validation_periods} quarters for validation")

        # Split data
        n_training = len(df) - config.validation_periods
        if n_training < 8:
            logger.error(f"Not enough training data ({n_training} quarters). Need at least 8.")
            return 1

        training_df = df.iloc[:n_training].copy()
        validation_df = df.iloc[n_training:].copy()

        logger.info(f"Training data: {training_df['period'].iloc[0]} to {training_df['period'].iloc[-1]} ({n_training} quarters)")
        logger.info(f"Validation data: {validation_df['period'].iloc[0]} to {validation_df['period'].iloc[-1]} ({len(validation_df)} quarters)")

        # Store validation data for comparison
        validation_data = validation_df

        # Use training data for forecasting
        df_for_forecast = training_df
        n_periods_to_forecast = config.validation_periods
    else:
        # Normal forecasting mode
        df_for_forecast = df
        n_periods_to_forecast = config.forecast_periods

    # Perform forecasting
    try:
        forecasts = forecast_all_topics(
            df=df_for_forecast,
            topic_cols=topic_cols,
            n_periods=n_periods_to_forecast,
            seasonality_mode=config.seasonality_mode,
            interval_width=config.confidence_interval,
            verbose=config.verbose,
            logger=logger
        )

        if not forecasts:
            logger.error("No topics were successfully forecasted")
            return 1

    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
        return 1

    # Calculate validation metrics if in validation mode
    if config.validation_mode and validation_data is not None:
        logger.info("\nCalculating validation metrics...")
        for topic in topic_cols:
            if topic not in forecasts:
                continue

            forecast_df, _ = forecasts[topic]
            n_training = len(df_for_forecast)

            # Get forecasted values
            forecast_values = forecast_df['yhat'].values[n_training:]

            # Get actual values
            actual_values = validation_data[topic].values

            # Calculate metrics
            metrics = calculate_validation_metrics(actual_values, forecast_values, topic)
            validation_metrics[topic] = metrics

            if not np.isnan(metrics['mae']):
                logger.info(f"  {topic}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, "
                          f"MAPE={metrics['mape']:.1f}%, R²={metrics['r2']:.3f}")

    # Prepare output directory
    output_dir = Path(config.output_dir) if config.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_path.stem

    output_files = []

    # Generate outputs
    try:
        # CSV output
        if not config.no_csv:
            suffix = "_validation" if config.validation_mode else "_forecast"
            csv_path = output_dir / f"{base_name}{suffix}.csv"
            combined_df, lower_df, upper_df = create_combined_dataframe(
                df_for_forecast, forecasts, len(df_for_forecast)
            )
            csv_files = save_forecast_csv(combined_df, lower_df, upper_df, csv_path, logger)
            output_files.extend(csv_files)

        # Static plot (combined)
        if not config.no_plot:
            suffix = "_validation" if config.validation_mode else "_forecast"
            plot_path = output_dir / f"{base_name}{suffix}.png"
            create_static_plot(df_for_forecast, forecasts, plot_path, logger)
            output_files.append(plot_path)

        # Individual topic plots
        if not config.no_individual_plots:
            individual_files = create_individual_topic_plots(
                df_for_forecast, forecasts, output_dir, base_name, logger,
                validation_mode=config.validation_mode
            )
            output_files.extend(individual_files)

        # Interactive HTML
        if not config.no_html:
            suffix = "_validation" if config.validation_mode else "_forecast"
            html_path = output_dir / f"{base_name}{suffix}.html"
            create_interactive_html(df_for_forecast, forecasts, html_path, logger,
                                   validation_data if config.validation_mode else None)
            output_files.append(html_path)

    except Exception as e:
        logger.error(f"Failed to generate outputs: {e}")
        return 1

    # Print summary
    if config.validation_mode:
        # Print validation metrics summary
        print(f"\n{'='*60}")
        print("VALIDATION METRICS SUMMARY")
        print(f"{'='*60}")
        print(f"Training Period: {df_for_forecast['period'].iloc[0]} - {df_for_forecast['period'].iloc[-1]}")
        print(f"Validation Period: {validation_data['period'].iloc[0]} - {validation_data['period'].iloc[-1]}")
        print(f"\nAccuracy Metrics (averaged across {len(validation_metrics)} topics):")

        avg_mae = np.mean([m['mae'] for m in validation_metrics.values() if not np.isnan(m['mae'])])
        avg_rmse = np.mean([m['rmse'] for m in validation_metrics.values() if not np.isnan(m['rmse'])])
        avg_mape = np.mean([m['mape'] for m in validation_metrics.values() if not np.isnan(m['mape'])])
        avg_r2 = np.mean([m['r2'] for m in validation_metrics.values() if not np.isnan(m['r2'])])

        print(f"  Average MAE:  {avg_mae:.2f}")
        print(f"  Average RMSE: {avg_rmse:.2f}")
        print(f"  Average MAPE: {avg_mape:.1f}%")
        print(f"  Average R²:   {avg_r2:.3f}")

        print("\nBest Performing Topics (by R²):")
        sorted_topics = sorted(validation_metrics.items(), key=lambda x: x[1]['r2'], reverse=True)
        for topic, metrics in sorted_topics[:5]:
            if not np.isnan(metrics['r2']):
                print(f"  {topic}: R²={metrics['r2']:.3f}, MAPE={metrics['mape']:.1f}%")

        print("\nWorst Performing Topics (by R²):")
        for topic, metrics in sorted_topics[-5:]:
            if not np.isnan(metrics['r2']):
                print(f"  {topic}: R²={metrics['r2']:.3f}, MAPE={metrics['mape']:.1f}%")

        print(f"{'='*60}\n")

    print_summary_report(forecasts, len(df_for_forecast), n_periods_to_forecast, output_files, logger)

    logger.info("Forecasting completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
