# House Price Prediction

## Overview
This project predicts house prices using the **Ames Housing Dataset** from Kaggle. It involves **exploratory data analysis (EDA), feature selection, preprocessing, and model training** using different machine learning algorithms. The best-performing model is saved and deployed using **Streamlit**.

## Dataset
- **Source:** [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Contains **79 features** describing various aspects of residential homes in Ames, Iowa.

## Workflow

### 1. Load Dataset
- Load the Ames Housing dataset for analysis.

### 2. Exploratory Data Analysis (EDA)
- Perform data visualization and statistical analysis.

### 3. Feature Selection
- Use the following **10 key features** for prediction:

  ```python
  selected_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'YearBuilt', 
                       'Neighborhood', 'GarageCars', 'BsmtQual', 'KitchenQual', 
                       'FullBath', 'LotArea']
### 4. Data Preprocessing
- Handle missing values and apply ordinal encoding.
- Encode categorical variables appropriately.
- Apply log transformation to skewed features.
### 5. Model Training & Evaluation
- Train and evaluate three models:
    - Linear Regression → MSE: `0.0234`
    - Random Forest → MSE: `0.0251`
    - XGBoost → MSE: `0.0269`

- The best model is saved as:
    ```python
    house_price_model.pkl

- A separate file for encoding neighborhoods is also saved:
    ```python
    neighborhood_encoder.pkl

### 6. Training the Model
- Run all cells in `code.ipynb` to train the model and save the weights.

### 7. Deploying with Streamlit
- A Streamlit app (`app.py`) is created for real-time predictions.

- To run the app, execute:
    ```python
    streamlit run app.py
---