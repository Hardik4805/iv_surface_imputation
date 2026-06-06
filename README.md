# Implied Volatility (IV) Surface Imputation

Reconstructing missing Implied Volatility (IV) surfaces using a Multi-Scale PCA ensemble and Savitzky-Golay temporal smoothing. Built for high-frequency NIFTY options data.

## Overview
This repository contains a robust data-imputation pipeline designed specifically for financial time-series, specifically Implied Volatility (IV) surfaces. Standard interpolation techniques often fail on financial data due to extreme liquidity blackouts and structural market volatility. 

To solve this, this pipeline implements a hybrid approach:
1. **Spatial Reconstruction:** Uses a Multi-Scale Principal Component Analysis (PCA) ensemble with momentum decay to rebuild the underlying structural factors (level, slope, curvature) of the volatility surface.
2. **Temporal Smoothing:** Applies Savitzky-Golay filtering to preserve the dynamic properties of the volatility surface while filtering out high-frequency noise. 

## File Structure
- `run_pipeline38.py`: The core imputation engine combining Akima 1D interpolation, PCA ensembles, and Savitzky-Golay smoothing.

## Usage
Simply run the python script. Ensure `dataset.csv` is present in the working directory.
```bash
python run_pipeline38.py
```
The script will output `submission.csv` containing the final imputed values.
