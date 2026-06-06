"""
Implied Volatility (IV) Surface Imputation Pipeline.

This script implements a multi-scale Principal Component Analysis (PCA) ensemble
to reconstruct missing implied volatility data across a financial options surface.
It combines spatial structural reconstruction (via PCA) with temporal smoothing
(via Savitzky-Golay filtering) to achieve robust imputation, particularly in scenarios
with contiguous block missingness (e.g., expiry-day liquidity blackouts).

Pipeline Flow:
    1. Data Loading & Preprocessing
    2. Baseline 1D Interpolation (Akima/Linear)
    3. Multi-Scale PCA Engine Inference
    4. Temporal Smoothing & Blending
    5. Final Output Generation
"""

import re
import warnings
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator, interp1d
from scipy.signal import savgol_filter
from scipy.signal.windows import gaussian
from sklearn.decomposition import PCA

# Suppress noisy warnings from scipy/sklearn interpolations during missing data boundaries
warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================

# Load the core dataset
df_wide = pd.read_csv('dataset.csv')
df_wide['row_id'] = np.arange(len(df_wide))
option_cols = [c for c in df_wide.columns if c not in ('datetime', 'underlying_price', 'row_id')]

def parse_ticker(col: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Parses an option ticker string to extract the strike price and option type.

    Args:
        col (str): The column name representing the option ticker (e.g., 'NIFTY27JAN2624100PE').

    Returns:
        Tuple[Optional[int], Optional[str]]: A tuple containing the strike price as an integer 
                                             and the option type ('CE' or 'PE'). Returns 
                                             (None, None) if the format is invalid.
    """
    m = re.match(r'NIFTY\d{2}[A-Z]{3}\d{2}(\d+)(CE|PE)$', col)
    if m:
        return int(m.group(1)), m.group(2)
    return None, None

# Transform the wide-format matrix into a long-format dataframe to facilitate groupby operations
df_long = df_wide.melt(id_vars=['row_id', 'datetime', 'underlying_price'], 
                       value_vars=option_cols, 
                       var_name='ticker', 
                       value_name='iv')

# Extract structured features (strike and option_type) from the ticker strings
parsed = df_long['ticker'].apply(lambda x: pd.Series(parse_ticker(x), index=['strike', 'option_type']))
df_long = pd.concat([df_long.drop(columns='ticker'), parsed], axis=1)

# Standardize data types and time indexing
df_long['strike'] = df_long['strike'].astype(float)
df_long['datetime_parsed'] = pd.to_datetime(df_long['datetime'], dayfirst=True)
df_long['is_missing'] = df_long['iv'].isna()

# Sort to ensure chronological and spatial continuity for the mathematical interpolators
df_long = df_long.sort_values(['datetime_parsed', 'option_type', 'strike']).reset_index(drop=True)

# Reconstruct a standardized ticker column ensuring uniformity
df_long['ticker'] = df_long.apply(lambda r: f"NIFTY27JAN26{int(r['strike'])}{r['option_type']}", axis=1)


# =============================================================================
# 2. BASELINE 1D INTERPOLATION (AKIMA)
# =============================================================================

def get_valid_ivs(sub: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a dataframe to return only strictly positive, non-null Implied Volatility values.

    Args:
        sub (pd.DataFrame): Subset of options data.

    Returns:
        pd.DataFrame: Filtered data containing valid IV points.
    """
    return sub[sub['iv'].notna() & (sub['iv'] >= 0.001)]

def fill_row_akima(group: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Akima 1D interpolation across the strike dimension for a specific time snapshot.
    
    Why Akima? 
    Akima interpolation preserves local convexity (the "volatility smile") without introducing 
    extreme oscillatory artifacts (Runge's phenomenon) that standard cubic splines produce 
    when dealing with sparse financial data.
    
    Args:
        group (pd.DataFrame): Data belonging to a single timestamp, containing various strikes.

    Returns:
        pd.DataFrame: The group with baseline predictions (`iv_pred`) and bounds tracking (`is_internal`).
    """
    group = group.copy()
    group['iv_pred'] = np.nan
    group['is_internal'] = False
    
    for opt_type in ['PE', 'CE']:
        mask = group['option_type'] == opt_type
        sub = group[mask].sort_values('strike')
        if len(sub) == 0: 
            continue
            
        clean = get_valid_ivs(sub)
        n_known = len(clean)
        if n_known == 0: 
            continue
            
        k = clean['strike'].values
        iv = clean['iv'].values
        xs = sub['strike'].values
        pred = np.full(len(xs), np.nan)
        
        # If only one point is known, propagate it as a flat assumption
        if n_known == 1:
            pred[:] = iv[0]
            group.loc[sub.index, 'is_internal'] = True
        else:
            # Track internal points vs extrapolated tails
            in_bounds_mask = (xs >= k[0]) & (xs <= k[-1])
            group.loc[sub.index, 'is_internal'] = in_bounds_mask
            
            # Akima requires at least 4 points to compute continuous derivatives
            if n_known >= 4:
                try:
                    akima_f = Akima1DInterpolator(k, iv)
                    if in_bounds_mask.any(): 
                        pred[in_bounds_mask] = akima_f(xs[in_bounds_mask])
                except Exception: 
                    pass
            
            # Fallback to linear extrapolation for tail strikes (outside known bounds)
            missing = np.isnan(pred)
            if missing.any():
                lin_f = interp1d(k, iv, kind="linear", fill_value="extrapolate", bounds_error=False)
                pred[missing] = lin_f(xs[missing])
                
        # IV cannot be zero or negative; clip to a reasonable mathematical floor
        group.loc[sub.index, 'iv_pred'] = np.clip(pred, 0.001, None)
        
    return group

# Apply baseline spatial interpolation independently at every timestep
df_long_filled = df_long.groupby('datetime', group_keys=False).apply(fill_row_akima)
preds_1d_real = df_long_filled['iv_pred'].values
is_internal_real = df_long_filled['is_internal'].values

# Create an initialized full matrix where missing values are seeded with baseline Akima predictions
df_long_completed = df_long.copy()
df_long_completed['iv'] = np.where(df_long_completed['iv'].isna(), preds_1d_real, df_long_completed['iv'])

# Pivot back to wide format to feed the linear algebra matrices
df_real_raw_pivot = df_long.pivot(index='row_id', columns='ticker', values='iv')
X_real_raw = df_real_raw_pivot[option_cols].values

df_real_completed_pivot = df_long_completed.pivot(index='row_id', columns='ticker', values='iv')
X_real_completed_init = df_real_completed_pivot[option_cols].values

# Track exactly where the dataset expects predictions
mask_missing_real = np.isnan(X_real_raw)


# =============================================================================
# 3. MULTI-SCALE PCA ENGINE INFERENCE
# =============================================================================

def rolling_pca_impute_gaussian(
    X_init: np.ndarray, 
    mask: np.ndarray, 
    window_size: int, 
    step_size: int, 
    k: int, 
    max_iters: int, 
    alpha: float
) -> np.ndarray:
    """
    Executes a rolling-window Principal Component Analysis to iteratively reconstruct missing data.
    
    Why this approach?
    Options surfaces are highly correlated. Standard interpolation fails during massive 
    liquidity blackouts (e.g., expiry blocks). PCA extracts the underlying structural factors 
    (level, slope, curvature) of the surface and reconstructs missing points based on the 
    global market structure rather than local neighboring noise.
    
    Args:
        X_init (np.ndarray): The initial data matrix (seeded with baseline predictions).
        mask (np.ndarray): Boolean matrix marking missing indices.
        window_size (int): Size of the temporal window for PCA calculation.
        step_size (int): Stride for the rolling window.
        k (int): Number of principal components to retain (4 optimal for level, slope, curvature, +1).
        max_iters (int): Iterations per window for the reconstruction to converge.
        alpha (float): Momentum term governing the decay back to the seed (regularization).

    Returns:
        np.ndarray: The finalized, fully reconstructed data matrix.
    """
    N, D = X_init.shape
    X_accum = np.zeros((N, D))
    C_accum = np.zeros((N, D))
    
    # Gaussian weights ensure that predictions near the center of a window are trusted 
    # more than predictions at the edges, preventing jarring artifacts at window boundaries.
    g_weights = gaussian(window_size, std=window_size / 4.0).reshape(-1, 1)
    
    starts = list(range(0, N - window_size + 1, step_size))
    if not starts or starts[-1] + window_size < N:
        starts.append(max(0, N - window_size))
        
    for start in starts:
        end = min(start + window_size, N)
        X_window = X_init[start:end, :].copy()
        mask_window = mask[start:end, :]
        
        # Iterative reconstruction: Fit PCA, inverse transform, blend with momentum
        for _ in range(max_iters):
            col_means = X_window.mean(axis=0)
            col_stds = X_window.std(axis=0)
            col_stds = np.where(col_stds == 0, 1.0, col_stds)
            
            X_scaled = (X_window - col_means) / col_stds
            pca = PCA(n_components=k, random_state=42)
            X_recon = pca.inverse_transform(pca.fit_transform(X_scaled)) * col_stds + col_means
            
            # Regularize by blending the newly reconstructed points with the previous iteration
            X_window[mask_window] = alpha * X_window[mask_window] + (1.0 - alpha) * X_recon[mask_window]
            
        al = end - start
        w_slice = g_weights[:al]
        
        # Accumulate weighted predictions to be averaged later
        X_accum[start:end, :] += X_window * w_slice
        C_accum[start:end, :] += w_slice
        
    return X_accum / np.maximum(C_accum, 1e-8)

# Execute an ensemble of multiple PCA window scales.
# This prevents the model from overfitting to the boundaries of any single window size.
windows = [90, 105, 120, 135, 150]
ensemble_matrices = []

for w in windows:
    X_imputed = rolling_pca_impute_gaussian(
        X_init=X_real_completed_init,
        mask=mask_missing_real,
        window_size=w,
        step_size=10,
        k=4,
        max_iters=5,
        alpha=0.8,
    )
    ensemble_matrices.append(X_imputed)

# The final spatial reconstruction is the mean of all multi-scale matrices
X_multi_scale = np.mean(ensemble_matrices, axis=0)


# =============================================================================
# 4. TEMPORAL SMOOTHING & BLENDING
# =============================================================================

# Map the matrix outputs back into the long-format dataframe structure
df_recon_pivoted = pd.DataFrame(X_multi_scale, columns=option_cols, index=np.arange(len(df_wide)))
df_recon_pivoted['row_id'] = df_recon_pivoted.index
df_recon_long = df_recon_pivoted.melt(id_vars=['row_id'], value_vars=option_cols, var_name='ticker', value_name='iv_pca')
df_long = df_long.merge(df_recon_long, on=['row_id', 'ticker'], how='left')

# Sort chronologically per ticker to prepare for temporal processing
df_sorted = df_long.sort_values(['ticker', 'datetime_parsed'])

def apply_savgol(x: pd.Series) -> pd.Series:
    """
    Applies Savitzky-Golay filtering over the temporal dimension.
    
    Why Savgol?
    Unlike an Exponential Moving Average (EMA) which flattens legitimate spikes, 
    Savgol fits a local polynomial, preserving the dynamic properties of the 
    volatility surface while filtering out high-frequency PCA noise.
    
    Note: Window length is explicitly tuned to 9 to better span isolated 5-min drops.
    """
    if len(x) >= 9:
        return savgol_filter(x, window_length=9, polyorder=2, mode='interp')
    return x

df_long['iv_pca_smoothed'] = df_sorted.groupby('ticker')['iv_pca'].transform(apply_savgol).loc[df_long.index]

# Blend the raw spatial PCA prediction with the temporally smoothed prediction.
# Note: We only apply temporal smoothing to strictly internal interpolations. Extrapolated 
# tail strikes are too volatile and should rely purely on structural PCA to avoid erratic swings.
temporal_blend = 0.95
df_long['iv_final'] = np.where(
    is_internal_real,
    temporal_blend * df_long['iv_pca'] + (1.0 - temporal_blend) * df_long['iv_pca_smoothed'],
    df_long['iv_pca'],
)

# Merge predictions securely back into the original dataset structure
df_long['value'] = np.where(df_long['is_missing'], df_long['iv_final'], df_long['iv'])


# =============================================================================
# 5. OUTPUT & EXPORT
# =============================================================================

# Secondary fallback: Forward-fill / Backward-fill isolated nulls ensuring structural completeness
df_long_sorted_time = df_long.sort_values(['ticker', 'datetime_parsed'])
df_long_sorted_time['value'] = df_long_sorted_time.groupby('ticker')['value'].transform(lambda s: s.ffill().bfill())
df_long['value'] = df_long_sorted_time['value'].loc[df_long.index]

# Final safety net for edge-cases where the ticker is entirely null across time
remaining = df_long['value'].isna().sum()
if remaining > 0: 
    df_long['value'] = df_long['value'].fillna(0.05)

# Hard limit to mathematically logical boundaries
df_long['value'] = np.clip(df_long['value'], 0.001, None)

# Extract only the strictly requested missing targets for the Kaggle submission file
df_sub = df_long[df_long['is_missing']].copy()
df_sub['id'] = df_sub.apply(lambda r: f"{r['datetime']}||{r['ticker']}", axis=1)
df_sub = df_sub[['id', 'value']].sort_values('id').reset_index(drop=True)

df_sub.to_csv('submission.csv', index=False)

