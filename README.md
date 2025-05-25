# NBA MVP Prediction – End-to-End Machine Learning


## Project Description
This end-to-end machine-learning project builds a classifier to predict which NBA player will win the Most Valuable Player (MVP) award in a given season. We combine historical data from 2004/05 through 2023/24 to train and evaluate our models.

---

## Project Highlights & Results

Our final optimized Random Forest model, trained on historical data and evaluated on the latest four seasons (2020/21-2023/24), demonstrates **robust and practical performance** in NBA MVP prediction.

* **Chronological Test Split:** The last 4 seasons (2020/21, 2021/22, 2022/23, 2023/24) were used as the test set to simulate a realistic prediction of future MVPs.
* **Baseline CV F1:** 0.693 (Indicator of initial performance on the training set).
* **Best CV F1 After Tuning:** 0.725 (Improved performance through hyperparameter optimization on the training set).
* **Final Test F1-Score (MVP Class) @ Optimal Threshold (0.27): 0.727**
    * **Recall (MVP Class): 1.00** – The model **correctly identified all 4 actual MVPs** in the test set.
    * **Precision (MVP Class): 0.57** – Of all players predicted as MVP, 57% were actual MVPs (4 out of 7 predicted).
    * **Test ROC-AUC: 0.998** – Confirms the model's excellent discriminative ability.

**Confusion Matrix on the Test Set:**  
|722|3|
|0|4|

The confusion matrix shows that the model correctly identified all 4 MVPs as True Positives (TP) and had no False Negatives (FN). The 3 False Positives (FP) are non-MVPs incorrectly classified as MVPs. This balance, of finding all MVPs, is crucial for the project's goal and demonstrates the model's high relevance.

---

| Name           | URL                                               |
|----------------|---------------------------------------------------|
| Huggingface Space | [Hugging Face](https://huggingface.co/spaces/ovogoeky/termproject) |
| GitHub         | [NBA-MVP-Prediction](https://github.com/ovogoeky/nba-mvp-prediction) |

## Data Sources and Features Used Per Source

For this project, relevant player, team, and MVP data were extracted and combined from various online sources.

| Data Source                                       | Features Used                                                                                                               |
|---------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| [Basketball-Reference.com](https://www.basketball-reference.com/) | MVP Awards (Player, Team, Season)                                                                                           |
| [Stathead.com](https://www.stathead.com/)         | **Basic Player Statistics:** Points Per Game (PPG), Rebounds Per Game (RPG), Assists Per Game (APG), Win Shares (WS)        |
| [Stathead.com](https://www.stathead.com/)         | **Advanced Player Statistics:** Offensive Rating (ORtg), Defensive Rating (DRtg), Player Efficiency Rating (PER), Box Plus/Minus (BPM), Usage Percentage (USG%), Value Over Replacement Player (VORP) |
| [Stathead.com](https://www.stathead.com/)         | **Team Performance:** Winning Percentage (Win%)   

---
## Dataset Versions Used

Throughout the project, I worked with multiple versions of the dataset that reflect different stages of the ML pipeline:

- **`final_dataset_complete.csv`**  
  → This dataset was created after merging the three original data sources (player stats, team win percentages, and MVP labels).

- **`final_dataset_with_outlier_flag.csv`**  
  → This version includes the merged data **plus an outlier flag**, generated during Exploratory Data Analysis (EDA) to identify and remove extreme non-MVP entries.

- **`final_dataset_with_features.csv`**  
  → This is the **final dataset used for training and testing**, created after performing feature engineering. It contains all selected and engineered features and excludes flagged outliers.

---

## Feature Engineering
## Features Created (with indication of those used in the final model)

| Feature | Description | Used in Final Model |
|---|---|:---:|
| `Standings` | Team rank in the season by Win% (1 = best team) | Yes |
| `Top3_Team` | Binary indicator if the team is among the top 3 in season Win% (1 = Yes, 0 = No) | No |
| `Efficiency_score` | Weighted combination of production and team success: (PPG + RPG + APG) × Win% | Yes |
| `PPG_weighted` | Points per game multiplied by team Win% (PPG × Win%) | No |
| `PPG_x_Win%` | Interaction term: PPG × Win% | No |
| `BPM_x_USG%` | Interaction term: Box Plus/Minus × Usage Rate (BPM × USG%) | No |
| `Net_Rating` | Difference between Offensive and Defensive Rating (ORtg − DRtg) | Yes |
| `PPG_rel` | Deviation of PPG from league average per season (PPG − Avg PPG_Season) | No |
| `PER_rel` | Deviation of PER from league average per season (PER − Avg PER_Season) | No |
| `WS_rel` | Ratio of Win Shares to league average (WS / Avg WS_Season) | No |
| `WinPct_rel` | Deviation of Team Win% from league average per season (Win% − Avg Win%_Season) | No |
| `is_outlier` | Binary flag for outliers among non-MVP players (IQR- & Z-score–based) | No |

> **Note:** We exclude all “relative” features in the final model since our Gradio app ingests only a single-season CSV (24/25) and does not require cross-season normalization.

---

## Model Training & Optimization

### Amount of Data
* **Player-Seasons**: 3684 data points 
* **MVP Cases**: 20 (1 per season from 2004/05 to 2023/24)

### Data Splitting
1.  **Train/Test Split**: Chronological split to simulate predicting future seasons.
    * **Training Set:** Seasons 2004/05 – 2019/20
    * **Test Set:** Last 4 seasons (2020/21 – 2023/24) – this final test set is used only once for model evaluation.
2.  **Hyperparameter Tuning & Model Selection**: 5-fold, stratified cross-validation on the **training set** for robust hyperparameter optimization of the Random Forest model.
3.  **Final Evaluation**: Model evaluation on the separate, unseen test set to assess generalization ability.

| Metric | Value | Description |
|---|---|---|
| **Test F1-Score (MVP)** | **0.727** | High balance between Precision and Recall for the MVP class. |
| **Recall (MVP)** | **1.00** | The model **correctly identified all 4 actual MVPs** in the test set. |
| **Precision (MVP)** | **0.57** | 57% of players predicted as MVP were correct. |
| **Test ROC-AUC** | **0.998** | Excellent ability of the model to distinguish MVPs from non-MVPs. |
| **Baseline CV F1** | 0.693 | Initial model performance on the training set. |
| **Best CV F1 (after Tuning)** | 0.725 | Improved performance through hyperparameter optimization. |
---

## Deployment

- **App**: `app.py` (Gradio Blocks)  
- **Usage**:  
  1. Upload a CSV with columns:  
     ```
     Player, Team, PPG, RPG, APG, Win%, WS,
     ORtg, DRtg, PER, BPM, VORP
     ```  
  2. Or click **“Mit den Daten der 24/25 Saison füllen”**  
  3. View Top-3 MVP candidates & probabilities  


## Visualizations and Analyses

To better understand the data, support feature selection, and interpret model performance, various visualizations were created. 

### 1. Feature Distributions and MVP Separation
![image](https://github.com/user-attachments/assets/56833893-6de4-4dc3-8103-34922625ebeb)
These histograms with density estimates show the distribution of each numerical feature, split by MVP status (blue for Non-MVP, orange for MVP). It is clear that MVPs consistently have significantly higher values in almost all key statistics, which forms a good basis for classification.

### 2. Correlation Matrix
![image](https://github.com/user-attachments/assets/3f87cb47-2fee-49c2-b261-f4dcf339fe8e)
The correlation matrix visualizes the correlations between all numerical features and the MVP target. It shows that advanced stats such as PER, BPM, VORP, and WS, in particular, exhibit a strong positive correlation with MVP status, underscoring their importance for prediction. Win% and PPG also show relevant correlations.

### 3. Scatter Matrix of Selected Top Features
![image](https://github.com/user-attachments/assets/e02676c2-c4c4-474e-a628-eeda4906e8b9)
This scatter matrix displays the pairwise relationships between the top features (PPG, WS, PER, BPM, VORP). The red points (MVPs) are concentrated in the upper right corners of the plots, visually confirming that MVPs perform exceptionally well in these statistics and stand out from the majority of Non-MVPs.

### 4. Feature Importance of the Final Model (without relative features)
![image](https://github.com/user-attachments/assets/9269f1cb-51cb-4770-9305-793b179e0a50)
This bar chart illustrates the importance of the top 20 features for the Random Forest model based on Permutation Importance (ΔF1), after relative features were removed from the dataset. It highlights the dominance of features such as `WS`, `VORP`, `Efficiency_score`, and `BPM`, which are the most influential factors for MVP prediction.

### Further Analyses in the Notebook
For deeper insights into the data and modeling, including the following visualizations and analyses, please refer to the main Jupyter Notebook in the GitHub Repository:
* **PCA (Principal Component Analysis) of the numerical features**
* **Average PPG across seasons (MVP vs. Non-MVP)**
* **Detailed Feature Interactions (examples)**
* **Code for data acquisition, preprocessing, and model training**

## Conclusion and Outlook

This end-to-end Machine Learning project has successfully developed and deployed a classification model for NBA MVP prediction. The project covered all essential steps of an ML workflow, from data acquisition from various sources like Basketball-Reference.com and Stathead.com to the deployment of the application on Huggingface.

The challenge of extreme class imbalance (only one MVP per season) was addressed through the targeted use of metrics like the F1-Score and ROC-AUC. The final Random Forest model, evaluated on a realistic test set spanning the last four seasons, achieved an **F1-Score of 0.727** for the MVP class, with a **Recall of 1.00** and a Precision of 0.57. This means the model **correctly identified all four actual MVPs** in the test set – a crucial success for the objective. The high ROC-AUC value of 0.998 further underscores the model's strong intrinsic discriminative ability.

Through comprehensive feature engineering, relevant features such as `Efficiency_score`, `Standings`, and `Net_Rating` were created and utilized. The Permutation Importance analysis was crucial for selecting the most important features, with `Efficiency_score`, `VORP`, `WS`, and `BPM` identified as the most influential factors.

**Potential next steps and improvements could include:**

* **Expanding the Data Basis:** A larger number of seasons could further strengthen the model's robustness and improve its generalization to future seasons.
* **Further Feature Engineering:** Exploring more complex interaction terms or time-based features could uncover new patterns.
* **Applying Advanced Techniques for Class Imbalance:** Techniques such as SMOTE or specialized sampling methods could further optimize handling of the extreme imbalance.
* **Ensemble Methods:** Combining multiple models (ensembles) could further stabilize and improve predictive performance.

Overall, this project provides a solid foundation for MVP prediction and demonstrates the practical application of Machine Learning to solve complex sports analytics problems.

