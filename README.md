# NBA MVP Prediction – End-to-End Machine Learning

## Project Description
This end-to-end machine-learning project builds a classifier to predict which NBA player will win the Most Valuable Player (MVP) award in a given season. We combine historical data from 2004/05 through 2023/24 to train and evaluate our models.

---

## Project Highlights & Results

- **Chronological Test Split:** Last 4 seasons (2020/21–2023/24) used as test set to simulate future predictions.  
- **Baseline CV F1:** 0.693  
- **Best CV F1 After Tuning:** 0.725  
- **Final Test F1-Score (MVP @ threshold 0.27):** 0.727  
  - **Recall (MVP):** 1.00 (all 4 actual MVPs identified)  
  - **Precision (MVP):** 0.57 (4 out of 7 predictions correct)  
  - **ROC-AUC:** 0.998  

---

## Links

| Name               | URL                                                                                      |
|--------------------|------------------------------------------------------------------------------------------|
| Hugging Face Space | [Hugging Face](https://huggingface.co/spaces/ovogoeky/termproject)                                        |
| GitHub Repository  | [GitHub](https://github.com/ovogoeky/nba-mvp-prediction)                                            |

---

## Data Sources & Features

| Data Source                                      | Features                                                                                                                |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| [Basketball-Reference.com](https://www.basketball-reference.com/) | MVP Awards (Player, Team, Season)                                                                                       |
| [Stathead.com](https://www.stathead.com/) – Basic      | PPG, RPG, APG, Win Shares (WS)                                                                                           |
| [Stathead.com](https://www.stathead.com/) – Advanced   | ORtg, DRtg, PER, BPM, USG%, VORP                                                                                        |
| [Stathead.com](https://www.stathead.com/) – Team        | Team Win%                                                                                                               |

---

## Dataset Versions

- **final_dataset_complete.csv**  
  Merged raw data (player stats + team Win% + MVP labels).

- **final_dataset_with_outlier_flag.csv**  
  Adds an outlier flag from EDA to remove extreme non-MVP entries.

- **final_dataset_with_features.csv**  
  Final features after engineering, used for training and testing.

---

## Feature Engineering

| Feature           | Description                                                      | Used in Final Model |
|-------------------|------------------------------------------------------------------|:-------------------:|
| **Standings**     | Team rank by Win% (1 = best)                                     | Yes                 |
| Top3_Team         | 1 if team in top 3 by Win%, else 0                               | No                  |
| **Efficiency_score** | (PPG + RPG + APG) × Win%                                      | Yes                 |
| PPG_weighted      | PPG × Win%                                                       | No                  |
| PPG_x_Win%        | Interaction term: PPG × Win%                                     | No                  |
| BPM_x_USG%        | Interaction term: BPM × USG%                                     | No                  |
| **Net_Rating**    | ORtg − DRtg                                                      | Yes                 |
| PPG_rel           | PPG − Avg PPG_Season                                             | No                  |
| PER_rel           | PER − Avg PER_Season                                             | No                  |
| WS_rel            | WS / Avg WS_Season                                               | No                  |
| WinPct_rel        | Win% − Avg Win%_Season                                           | No                  |
| is_outlier        | Binary flag for non-MVP outliers (IQR & Z-score)                 | No                  |

> **Note:** All “_rel” features are excluded in production, since the app ingests single-season CSVs without cross-season context.

---

## Model Training & Optimization

- **Data Size:**  
  - 3,684 player-seasons  
  - 20 MVP cases (2004/05–2023/24)

- **Splitting Strategy:**  
  1. **Train:** 2004/05–2019/20  
  2. **Test:** 2020/21–2023/24  
  3. **CV:** 5-fold stratified on training set for hyperparameter tuning

| Metric                      | Value | Description                                       |
|-----------------------------|-------|---------------------------------------------------|
| **Test F1-Score (MVP)**     | 0.727 | Balance of Precision & Recall                     |
| **Recall (MVP)**            | 1.00  | All 4 MVPs correctly identified                   |
| **Precision (MVP)**         | 0.57  | 57% of MVP predictions were correct               |
| **Test ROC-AUC**            | 0.998 | Excellent discriminative ability                  |
| **Baseline CV F1**          | 0.693 | Initial model performance                         |
| **Best CV F1 (after tuning)** | 0.725 | After hyperparameter optimization                 |

---

## Deployment

- **App Script:** `app.py` (Gradio Blocks)  
- **Dependencies:** listed in `requirements.txt`  
- **Usage:**  
  1. Upload a CSV with columns:  
     ```
     Player, Team, PPG, RPG, APG, Win%, WS,
     ORtg, DRtg, PER, BPM, VORP
     ```  
  2. Or click “Fill with 24/25 season data”  
  3. View top-3 MVP candidates & their probabilities  

---

## Visualizations & Analyses

1. **Feature Distributions** – Histograms split by MVP status  
2. **Correlation Matrix** – Correlations between features & MVP label  
3. **Scatter Matrix** – Pairwise plots of top features (PPG, WS, PER, BPM, VORP)  
4. **Permutation Importance** – Top feature importances from Random Forest  

> See the full Jupyter Notebook for interactive plots, PCA, time-series analyses, and detailed code.

---

## Conclusion & Outlook

### Conclusion
- End-to-end NBA MVP prediction pipeline implemented.  
- Data merged from Basketball-Reference and Stathead.  
- Key features engineered: Efficiency_score, Standings, Net_Rating, etc.  
- Random Forest evaluated on realistic test set:  
  - **F1-Score:** 0.727  
  - **Recall:** 1.00  
  - **Precision:** 0.57  
  - **ROC-AUC:** 0.998  

### Outlook
1. **Expand the Data**  
   - Include more seasons for robustness  
2. **Deepen Feature Engineering**  
   - Explore complex interactions & time-series features  
3. **Explore Imbalance Techniques**  
   - Apply SMOTE or specialized sampling  
4. **Build Ensembles**  
   - Combine multiple models for stability  

Overall, this project lays a solid foundation for MVP prediction and demonstrates ML’s power in sports analytics.  
