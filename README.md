Objective
Implement and understand Simple & Multiple Linear Regression using Scikit-learn, Pandas, and Matplotlib on the House Prices dataset.

Project Structure
ml-task3/
├── train.csv                      # Dataset (House Prices)
├── linear_regression.py           # Main Python script
├── linear_regression_results.png  # Output plots 
└── README.md                      # This file

Requirements
bashpip install pandas numpy scikit-learn matplotlib
Python 3.8 or higher.

How to Run
bashpython linear_regression.py

Make sure train.csv is in the same folder as the script.


Dataset
Source: House Prices — Kaggle
PropertyValueRows1,460Columns81TargetSalePrice (in USD)
Features Used
FeatureDescriptionGrLivAreaAbove-grade living area (sq ft)OverallQualOverall material and finish quality (1–10)TotalBsmtSFTotal basement area (sq ft)GarageAreaGarage size (sq ft)YearBuiltYear the house was builtFullBathNumber of full bathrooms above gradeTotRmsAbvGrdTotal rooms above grade

Steps Followed
1. Import & Preprocess

Loaded train.csv using Pandas
Selected 7 numeric features relevant to house pricing
Filled missing values with column median

2. Train-Test Split
pythonX_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

80% training, 20% testing
random_state=42 ensures reproducibility

3. Model Training
Simple Linear Regression (1 feature)
pythonfrom sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X_train[["GrLivArea"]], y_train)
Multiple Linear Regression (7 features)
pythonfrom sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

mlr = LinearRegression()
mlr.fit(X_train_scaled, y_train)
Features are standardised so coefficients are directly comparable.
4. Evaluation Metrics
MetricFormulaWhat it tells youMAEmean |y − ŷ|Average error in dollarsMSEmean (y − ŷ)²Penalises large errors more heavilyRMSE√MSEError in same unit as target ($)R²1 − SS_res/SS_tot% of variance explained (higher = better)
5. Plots Generated
PlotDescriptionRegression LineScatter + fitted line for Simple LRActual vs PredictedHow close MLR predictions are to real valuesFeature CoefficientsWhich features impact price most (MLR)Residuals — Simple LRChecks for prediction error patternsResiduals — Multiple LRSame check for the richer modelMetrics ComparisonSide-by-side bar chart: Simple vs Multiple LR

Results
Simple Linear Regression (GrLivArea → SalePrice)
Intercept  : $24,899.75
Coefficient:    $102.49  per sq ft

MAE  : $38,341
MSE  : $3,418,946,311
RMSE : $58,472
R²   :  0.5543

Every additional sq ft of living area adds ~$102 to the predicted price.
The model explains 55.4% of price variance.


Multiple Linear Regression (7 features → SalePrice)
Intercept : $181,441.54

Feature          Coefficient (standardised)
─────────────────────────────────────────
OverallQual            +$28,938
GrLivArea              +$26,049
YearBuilt              +$11,648
TotalBsmtSF            +$10,029
GarageArea             +$10,001
TotRmsAbvGrd            +$1,141
FullBath                −$2,255

MAE  : $24,846
MSE  : $1,575,206,652
RMSE : $39,689
R²   :  0.7946

Adding 6 more features improved R² by +43.4% and reduced RMSE by $18,783.


Coefficient Interpretation
Feature Meaning
OverallQual (+$28,938)Highest impact — buyers pay most for quality
GrLivArea (+$26,049)More living space → higher priceYearBuilt (+$11,648)Newer homes are valued moreTotalBsmtSF (+$10,029)Larger basement adds significant valueGarageArea (+$10,001)Garage space is nearly as important as basementFullBath (−$2,255)Slight negative due to multicollinearity with room countTotRmsAbvGrd (+$1,141)Smallest independent effect
