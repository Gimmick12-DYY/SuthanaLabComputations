# SuthanaLabComputations
This repository contains the projects developed within the Suthana Lab Group at Duke University:

## Structure

```
.
├── TR-PTSD/           # Notebooks and scripts for the Cosinor Regression Model used for Analysis
├── /     # 
├── LICENSE
└── README.md
```

1. **TR-PTSD/**: Construction of a Cosinor Regression Model that analyzes the daily recordings of TR-PTSD patients. The model utilizes a multiple-component Cosine Linear Regression algorithm to fit the daily rhythmic data. Special adjustments to the original CosinorPy package were implemented to adapt to the specialized datasets.

Key files:
   - `cosinor_24h.ipynb` – Notebook for fitting daily (24-hour) rhythmic models.
   - `cosinor_7ds.ipynb` – Notebook for fitting weekly (7-day) rhythmic models.
   - `cosinormodel.ipynb` – Core notebook for model parsing and general usage.
   - `cosinormodel_backbone.ipynb` – Backbone logic used for modular development and heatmap extension.
   - `compare_csv_rows.py` – Script for comparing and validating CSV data rows across datasets.
   - `matrix_analysis.py` – Additional analysis tools based on matrix logic.
   - `cosinordemo/` – Demo folder for CosinorPy-based baseline testing.
   
2. **/**: 
