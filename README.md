# Incremental Uplift Modeling: Optimizing Marketing ROI
### Causal Inference & Machine Learning on the Hillstrom (MineThatData) Dataset

## Business Problem
Standard propensity models predict *who is likely to buy*, but they fail to distinguish between:
1. **Sure Things:** Customers who buy even without an email.
2. **Lost Causes:** Customers who won't buy regardless.
3. **Persuadables:** Customers who buy **only if** they receive an email.

Marketing to "Sure Things" is a waste of budget ($0.05/email), and marketing to "Sleeping Dogs" can actually decrease conversion. This project uses **Causal ML** to target only the **Persuadables**.

## Tech Stack
* **Language:** Python 3.x (Google Colab)
* **Causal Framework:** `scikit-uplift` (sklift)
* **ML Engine:** `XGBoost` (Tuned via `GridSearchCV`)
* **Analysis:** `Pandas`, `NumPy`, `SciPy` (Power Analysis)
* **Visualization:** `Matplotlib`, `Seaborn`

## Project Workflow

### 1. Statistical Rigor (A/B Testing)
Before modeling, I conducted a **Power Analysis** to ensure the sample size (64k) was sufficient to detect a 1% lift. I then performed a T-Test to confirm the statistical significance ($p < 0.05$) of the email campaign across the entire population.

### 2. Modeling Strategy: The S-Learner
I implemented a **Solo Model (S-Learner)** approach using an XGBoost Classifier. Unlike standard classification, the model treats the "Treatment" (Email vs. No Email) as a feature to predict the **Conditional Average Treatment Effect (CATE)**.

### 3. Hyperparameter Tuning
Used `GridSearchCV` to optimize XGBoost depth and learning rates, ensuring the model captured non-linear interactions between customer history and the treatment.

## Business Impact Simulation
I developed an ROI calculator to compare a "Mass Marketing" strategy vs. a "Causal Targeting" strategy.

| Strategy | Targeting | Revenue | Marketing Cost | Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Everyone (100%) | $High | $High | Low ROI |
| **Uplift Model** | Top 20% | ~85% of Total | -80% Cost | **High ROI** |

## Key Results
* **Qini Coefficient:** Successfully outperformed a random targeting strategy.
* **Segmentation Insight:** Customers with high "Past Spend" (History) showed a significantly higher incremental response to Men's E-mail campaigns compared to new users.
* **Recommendation:** **SHIP.** Deploy the model to target the top 20% of Persuadables to maximize profit while minimizing "Inbox fatigue."

## How to Run
1. Open the `Uplift_Modeling_Hillstrom.ipynb` in Google Colab.
2. Run the first cell to install `scikit-uplift`.
3. Execute all cells to view the EDA, Model Training, and ROI Simulation.

---

