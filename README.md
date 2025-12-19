
Customer Churn Prediction — End-to-End Machine Learning Pipeline

📌 About This Project

This project is an end-to-end Telecom Customer Churn Prediction system built to reflect how real businesses approach churn analytics. 
It goes beyond model training by including data cleaning, leakage-aware feature engineering, class imbalance handling, baseline vs. advanced model comparison, decision threshold tuning, deployment-style inference, and explainability through feature importance and SHAP. The goal was not only to predict churn, but to understand why customers churn and convert insights into actionable business strategy. This project reflects my focus on building practical, production-aligned AI solutions that solve meaningful problems.


## Dataset

Source: IBM Telco Customer Churn Dataset

Target variable: Churn Value

1 → Customer churned
0 → Customer retained

The dataset contains customer demographics, service subscriptions, billing information, and contract details.

Data Description
7043 observations with 33 variables
The dataset contains customer demographics, service usage, and billing information.

7043 observations with 33 variables

CustomerID: A unique ID that identifies each customer.

Count: A value used in reporting/dashboarding to sum up the number of customers in a filtered set.

Country: The country of the customer’s primary residence.

State: The state of the customer’s primary residence.

City: The city of the customer’s primary residence.

Zip Code: The zip code of the customer’s primary residence.

Lat Long: The combined latitude and longitude of the customer’s primary residence.

Latitude: The latitude of the customer’s primary residence.

Longitude: The longitude of the customer’s primary residence.

Gender: The customer’s gender: Male, Female

Senior Citizen: Indicates if the customer is 65 or older: Yes, No

Partner: Indicate if the customer has a partner: Yes, No

Dependents: Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.

Tenure Months: Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.

Phone Service: Indicates if the customer subscribes to home phone service with the company: Yes, No

Multiple Lines: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No

Internet Service: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.

Online Security: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No

Online Backup: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No

Device Protection: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No

Tech Support: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No

Streaming TV: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.

Streaming Movies: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.

Contract: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.

Paperless Billing: Indicates if the customer has chosen paperless billing: Yes, No

Payment Method: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check

Monthly Charge: Indicates the customer’s current total monthly charge for all their services from the company.

Total Charges: Indicates the customer’s total charges, calculated to the end of the quarter specified above.

Churn Label: Yes = the customer left the company this quarter. No = the customer remained with the company. Directly related to Churn Value.

Churn Value: 1 = the customer left the company this quarter. 0 = the customer remained with the company. Directly related to Churn Label.

Churn Score: A value from 0-100 that is calculated using the predictive tool IBM SPSS Modeler. The model incorporates multiple factors known to cause churn. The higher the score, the more likely the customer will churn.

CLTV: Customer Lifetime Value. A predicted CLTV is calculated using corporate formulas and existing data. The higher the value, the more valuable the customer. High value customers should be monitored for churn.

Churn Reason: A customer’s specific reason for leaving the company. Directly related to Churn Category.


Methodology & Workflow

This project follows a modular, pipeline-based machine learning workflow, separating exploration, preprocessing, modeling, and evaluation.

Step 1 — Exploratory Data Analysis (EDA)

Tool: Jupyter Notebook (notebooks/01_eda.ipynb)

EDA was performed on mostly raw data to:

Understand the distribution of the target variable

Identify class imbalance (~26.5% churn)

Explore relationships between churn and numeric features

Analyze categorical features and detect primary churn drivers

Identify data quality issues and potential leakage

Key EDA Findings

Churn is imbalanced, requiring stratified splits and recall-focused evaluation

Churned customers have:

Significantly lower tenure

Higher monthly charges

Contract type shows the strongest association with churn

Several columns contain post-outcome or derived information and must be excluded

EDA outputs were used to inform feature selection and preprocessing decisions, not to train models.

Step 2 — Feature Selection & Exclusion

Feature decisions were finalized based on EDA insights.

Target

Churn Value (binary numeric target)

Dropped Features (Intentional Exclusion)

These features were removed to prevent leakage, overfitting, or noise:

Identifiers: CustomerID, Count

Geographic fields: Country, State, City, Zip Code, Latitude, Longitude, Lat Long

Redundant or leakage-prone fields:

Churn Label

Churn Score

Churn Reason

CLTV

Step 3 — Feature Grouping (Preprocessing Design)

Features were grouped before preprocessing, enabling clean and reproducible pipelines.

Numeric Features

Tenure Months

Monthly Charges

Total Charges

Categorical Features

Contract & billing:
Contract, Payment Method, Paperless Billing

Internet & phone services:
Internet Service, Phone Service, Multiple Lines

Service add-ons:
Tech Support, Online Security, Online Backup,
Device Protection, Streaming TV, Streaming Movies

Demographics:
Senior Citizen, Gender, Partner, Dependents

These decisions are centralized in config.py, acting as a single source of truth.

Step 4 — Preprocessing Pipeline (Execution)

Tools: scikit-learn Pipelines & ColumnTransformer

Preprocessing is executed inside the modeling pipeline, not manually, to avoid leakage.

Numeric preprocessing

Median imputation for missing values

Standard scaling

Categorical preprocessing

Most-frequent imputation

One-hot encoding with handle_unknown="ignore"

This ensures:

Consistent preprocessing across train and test sets

Reproducibility

Safe deployment-ready design

Step 5 — Model Training

Baseline Model: Logistic Regression
File: src/train.py

Stratified train/test split on Churn Value to preserve class balance

class_weight="balanced" to address class imbalance

Full preprocessing + model combined in a single scikit-learn Pipeline

Trained pipeline saved using joblib as models/logistic_regression_pipeline.joblib

This baseline was chosen for:

Interpretability (coefficients and direction of effects)

Probabilistic outputs (predict_proba)

Strong performance on mostly monotonic relationships (e.g., tenure vs churn)

Nonlinear Model: Random Forest
File: src/train_rf.py

Same stratified train/test split and preprocessing pipeline

RandomForestClassifier with class_weight="balanced" to further handle class imbalance

Captures nonlinear relationships and feature interactions (e.g., Contract × Monthly Charges)

Trained pipeline saved as models/random_forest_pipeline.joblib

This model was added to:

Explore performance gains from nonlinear modeling

Better capture interactions between contract type, pricing, and service features

Compare interpretability vs performance trade-offs against Logistic Regression

Step 6 — Model Evaluation

Files:

Logistic Regression: src/evaluate.py

Random Forest: src/evaluate_rf.py

Evaluation focuses on business-relevant metrics, not just accuracy, and uses the same stratified test set for fair comparison.

Metrics Reported

Accuracy

Precision

Recall (primary metric)

F1-score

ROC-AUC

Baseline Results — Logistic Regression

Accuracy: 0.7431

Precision: 0.5105

Recall: 0.7807

F1-score: 0.6173

ROC-AUC: 0.8488

These results indicate strong separation of churners while prioritizing recall, minimizing the number of missed churn cases.

Random Forest Results

Accuracy: 0.7658

Precision: 0.5415

Recall: 0.7674

F1-score: 0.6350

ROC-AUC: 0.8538

Random Forest improves overall performance (accuracy, precision, F1, ROC-AUC) while maintaining high recall, making it a strong candidate model for churn prediction.

Model Comparison Summary
Metric	Logistic Regression	Random Forest
Accuracy	0.7431	0.7658
Precision	0.5105	0.5415
Recall	0.7807	0.7674
F1-score	0.6173	0.6350
ROC-AUC	0.8488	0.8538

Logistic Regression serves as an interpretable baseline.

Random Forest offers better overall performance and similar recall, making it preferable when performance is prioritized over interpretability.

Step 7 — Decision Threshold Tuning

Default binary classifiers use a 0.50 probability threshold, but in churn prediction this is not always optimal. The business objective is to:

Minimize false negatives (missed churners) while still maintaining reasonable precision.

A dedicated threshold tuning script (src/tune_threshold.py) evaluated multiple thresholds using the same test set:

Threshold	Accuracy	Precision	Recall	F1
0.30	0.6870	0.4553	0.9118	0.6073
0.35	0.7069	0.4719	0.8770	0.6137
0.40	0.7346	0.5000	0.8583	0.6319
0.45	0.7530	0.5223	0.8128	0.6360
0.50	0.7658	0.5415	0.7674	0.6350
0.55	0.7857	0.5766	0.7246	0.6422

📌 Selected Threshold: 0.45

Maintains strong recall (> 0.80)

Improves F1 relative to higher thresholds

Balances cost of outreach vs. missed churn risk

This threshold is applied in inference for the final churn decision.

Step 8 — Inference Pipeline (Real-World Usability)

A deployment-ready inference script (src/infer.py) was implemented to simulate production usage.

It supports:

Single customer prediction

Probability output

Human-readable risk classification

Example output:

{
  "prediction": 1,
  "churn_probability": 0.78,
  "risk_level": "HIGH RISK"
}

Model Used for Deployment

Random Forest was selected for deployment because it:

Outperformed Logistic Regression on accuracy, precision, F1, and ROC-AUC

Maintained strong recall

Captures nonlinear churn behavior

The system is modular and supports swapping models if priorities change.

Step 9 — Feature Importance (Model Explainability)

To move beyond prediction and understand why customers churn, Random Forest feature importance was evaluated using the full preprocessing pipeline.

🔝 Top Drivers of Churn
Rank	 Feature	                             Interpretation
1	    Contract: Month-to-Month	             Short-term customers are most likely to churn
2	    Tenure Months	                         Newer customers churn significantly more
3	    Total Charges	                         Financial accumulation influences churn
4	    Online Security = No	                 Lack of protection services increases churn
5	    Contract: Two-Year	                     Long contracts reduce churn risk
6   	Tech Support = No	                     Lower perceived support → higher churn
7	    Monthly Charges	                         Higher bills drive dissatisfaction
8–10	Dependents / Internet Service Type	     Secondary demographic and service effects

📌 Business Insight
Churn is primarily influenced by customer lifecycle, pricing pressure, and service experience, which aligns with real-world telecom churn research.

Plots saved to:

reports/rf_feature_importance.csv
reports/figures/rf_feature_importance.png

Step 10 — SHAP Explainability (Advanced Model Transparency)

To provide deeper interpretability, SHAP (SHapley Additive exPlanations) was applied.

SHAP assigns each feature a “contribution value,” explaining how much it increased or decreased churn probability for a prediction.

This provides both global and local explainability.

Global Insights (Summary + Bar Plots)

SHAP confirms:

Tenure Months has the largest average impact; low tenure increases churn.

Monthly Charges strongly affect churn probability; higher charges contribute to churn risk.

Contract type and lack of support/security services significantly shift churn probabilities.

This validates model behavior and supports actionable strategy.

Plots saved to:

reports/figures/shap_summary_plot.png
reports/figures/shap_bar_plot.png


Project Structure

churn-ml-end-to-end/
│
├── data/
│   ├── raw/                 # Raw dataset (not committed)
│   └── processed/           # Any intermediate processed data
│
├── notebooks/
│   └── 01_eda.ipynb         # Exploratory Data Analysis
│
├── src/
│   ├── config.py            # Global settings: paths, target, feature groups
│   │
│   ├── data_prep.py         # Data loading & minimal cleaning (no leakage)
│   ├── features.py          # Preprocessing pipelines (scaling, OHE, imputation)
│   │
│   ├── train.py             # Logistic Regression training pipeline
│   ├── train_rf.py          # Random Forest training pipeline
│   │
│   ├── evaluate.py          # Logistic Regression evaluation
│   ├── evaluate_rf.py       # Random Forest evaluation
│   │
│   ├── tune_threshold.py    # Decision threshold tuning for imbalanced churn
│   │
│   ├── infer.py             # Deployment-style inference (single-customer prediction)
│   │
│   ├── feature_importance_rf.py   # Random Forest feature importance analysis
│   └── shap_analysis.py     # SHAP model explainability (global interpretability)
│
├── models/                  # Saved trained model pipelines (.joblib)
│
├── reports/
│   ├── figures/             # Visual outputs
│   │   ├── confusion_matrix_logreg.png
│   │   ├── roc_curve_logreg.png
│   │   ├── confusion_matrix_rf.png
│   │   ├── roc_curve_rf.png
│   │   ├── rf_feature_importance.png
│   │   ├── shap_summary_plot.png
│   │   └── shap_bar_plot.png
│   │
│   └── rf_feature_importance.csv   # Ranked feature importance table
│
├── requirements.txt
└── README.md

Final Model Summary & Business Impact:

This project delivers a production-aligned churn prediction system with:

✔ Clean EDA & leakage-aware feature engineering
✔ Robust ML pipeline design
✔ Baseline + advanced model comparison
✔ Imbalance handling & stratified evaluation
✔ Decision threshold optimization
✔ Deployment-oriented inference pipeline
✔ Explainability via feature importance + SHAP
✔ Actionable business insights

What This Means for a Telecom Business?

Identify high-risk customers early
Prioritize month-to-month & new users
Focus on pricing communication & support quality
Offer retention incentives before churn happens
Final Methodology Terminology 

This project demonstrates:

End-to-End Machine Learning Pipeline
Pipeline-based preprocessing
Leakage-aware feature selection
Imbalanced classification strategy
Baseline vs nonlinear model comparison
Threshold tuning for decision optimization
Model interpretability (Feature Importance + SHAP)
Deployment-ready inference architecture

How to Run This Project

1️⃣ Clone & enter project

git clone <your-repo-url>
cd churn-ml-end-to-end


2️⃣ Create virtual environment & install dependencies

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Mac/Linux
pip install -r requirements.txt


3️⃣ Add dataset
Place:

Telco_customer_churn.csv


into:

data/raw/


4️⃣ Train models

python src/train.py        # Logistic Regression
python src/train_rf.py     # Random Forest


5️⃣ Evaluate

python src/evaluate.py
python src/evaluate_rf.py


6️⃣ Threshold tuning:

python src/tune_threshold.py


Explainability:

python src/feature_importance_rf.py
python src/shap_analysis.py


7️⃣ Run inference

python src/infer.py


Models, figures, and reports are automatically saved to:

models/
reports/figures/
reports/



👤 Author

Enas Elhaj
Graduate Student — Applied Artificial Intelligence & Data Science
University of Denver | Ritchie School of Engineering & Computer Science

Former telecom & computer engineering lecturer with strong background in:

Python

Machine Learning & AI

Data Science

Databases

Software engineering

Currently building advanced AI and data-driven projects, exploring:

Predictive modeling

AI explainability / responsible AI

End-to-end ML systems

Business decision intelligence

Data-driven product strategy

📌 Passionate about
Turning data into actionable insights that solve real human + business problems.

📫 Contact / Profiles

LinkedIn: https://www.linkedin.com/in/enas-elhaj/

GitHub: https://github.com/enase-elhaj

Email: enas.elhaj@du.edu
