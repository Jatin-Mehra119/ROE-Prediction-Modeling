# Return On Equity Prediction Project

This project aims to predict the Return on Equity (ROE) of companies based on various financial metrics using a machine learning model. The project includes data analysis, model training, and a web application for making predictions.

#### Data source - [Link](https://dataverse.harvard.edu/dataset.xhtml;jsessionid=41510459a96a8d0e58860d844857?persistentId=doi:10.7910/DVN/XMVD5L)

 - Notebook.ipynb - It includes all the EDA, Preprocessing, Model Training and Model Evaluation. 
 - ROE_model.pkl - Pipelined Model Saved as a pickle file.  
 - app.py - Streamlit app used for making prediction using pickled model.

### Key Insights gained more training the model-

1.  **Top Contributors**:
    
    -   **Equity** (`log__Equity`): The most important feature, suggesting that the equity of a company is a significant predictor of its ROE.
    -   **Current Ratio** (`log__Current_Ratio`): Indicates the liquidity of a company and its ability to pay short-term obligations, which is a strong predictor.
    -   **Market Cap** (`log__Market_Cap`): Reflects the company's size and market perception, contributing substantially to predicting ROE.
    -   **Profit/Loss** (`log__Profit_Loss`): Indicates the profitability of the company, also a significant predictor.
2.  **Moderate Contributors**:
    
    -   **Assets** (`log__Assets`): Although lower than the top contributors, still has a meaningful impact.
    -   **Dividends** (`log__Dividends`): Reflects the company's dividend policy, showing some importance.
    -   **Liabilities** (`log__Liabilities`): Indicates the company's obligations, but with lesser importance compared to assets and equity.
    -   **Expenses** (`norm__Expenses`): Operating expenses also contribute, but to a lesser degree.
    -   **Growth Rate** (`norm__Growth_Rate`): The company's growth rate is considered, but with minimal importance.
3.  **Low Contributors**:
    
    -   **Revenue by Region** (`log__Revenue_Central China`, `log__Revenue_West China`, `log__Revenue_South China`, `log__Revenue_North China`, `log__Revenue_East China`): These features have very low importance, indicating regional revenues might not significantly influence ROE.
    -   **PE Ratio** (`log__PE_Ratio`): Also shows low importance in predicting ROE.
    -   **Total Revenue** (`norm__Revenue`): Surprisingly low impact on predicting ROE.
4.  **Minimal or Negligible Contributors**:
    
    -   **City and Sector**: Categorical variables like `City` and `Sector` have very low importance.
    -   **Company Size**: The size of the company (`onehot__Company_Size_Medium`, `onehot__Company_Size_Small`, `onehot__Company_Size_Large`) has almost no importance in predicting ROE.