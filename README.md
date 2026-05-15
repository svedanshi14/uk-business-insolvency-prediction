# UK Business Insolvency Prediction

## Project Overview
Predicting quarterly UK business insolvency rates using machine learning and distributed computing (PySpark).
Study period: Q3 2016 — Q4 2025 | 1,468 observations

## Research Question
Can macroeconomic indicators (Bank of England base rate, energy prices, regional GDP) predict UK business insolvency rates across sectors and locations?

## Datasets Used
| Dataset | Source | Role |
|---------|--------|------|
| Company Insolvency Statistics | Gov.uk | Target variable |
| Regional GDP | ONS | Economic predictor |
| Industrial Energy Price Indices | DESNZ Gov.uk | Cost predictor |
| Bank of England Base Rate | Bank of England | Borrowing cost |

## Methods
### Machine Learning
- Linear Regression (baseline)
- Random Forest
- Gradient Boosting
- TimeSeriesSplit Cross Validation (5-fold)
- Residual Analysis

### HPC
- Apache PySpark Distributed Random Forest

### EDA
- 11 visualisations
- K-Means Clustering (K=3, elbow method)
- PCA (variance explained + scatter)

## Results
| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Linear Regression | 34.66 | 23.18 | 0.9830 |
| Random Forest | 34.69 | 17.71 | 0.9830 |
| Gradient Boosting | 38.46 | 18.79 | 0.9791 |
| PySpark RF (HPC) | 35.32 | 17.31 | 0.9821 |

## Key Findings
- All models achieved R²>0.977 — exceptional predictive accuracy
- Random Forest most reliable (CV Mean R²=0.948, Std=0.043)
- K-Means independently discovered 3 economic regimes
- PySpark 30x slower on small dataset but provides scalability

## Technologies
Python, Pandas, NumPy, Scikit-learn, PySpark, 
Matplotlib, Seaborn, Google Colab

## Repository Structure
```
├── Code/
│   └── uk_business_insolvency_prediction.ipynb
├── Datasets/
│   ├── vedanshi_interest_rate.csv
│   ├── merged_complete_dataset.csv
│   ├── cleaned_interest_rate.csv
│   └── bank_rate_data.csv
├── Results/
│   ├── ml_results.csv
│   ├── hpc_results.csv
│   ├── cv_results.csv
│   └── all_results.csv
└── Charts/
    ├── eda_chart1_trend.png
    ├── eda_chart2_correlation.png
    ├── eda_chart3_sectors.png
    ├── eda_chart4_sectors_time.png
    ├── eda_chart5_location.png
    ├── eda_chart6_scatter.png
    ├── eda_chart7_elbow.png
    ├── eda_chart8_kmeans.png
    ├── eda_chart9_sector_clusters.png
    ├── eda_chart10_pca_variance.png
    ├── eda_chart11_pca_scatter.png
    ├── feature_importance.png
    ├── performance_comparison.png
    ├── predicted_vs_actual.png
    └── residual_analysis
