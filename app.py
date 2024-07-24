import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('ROE_model.pkl')

# Function to display the app
def main():
    st.set_page_config(
        page_title="Return On Equity Prediction",
        page_icon="📈",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title("Return On Equity Prediction")
    st.markdown("""
    Welcome to the Return On Equity (ROE) prediction app. This tool leverages machine learning to estimate the ROE for companies based on various financial metrics. Please fill out the following information to get a prediction.
    """)

    # Collect input data from the user
    with st.form(key='input_form'):
        col1, col2 = st.columns(2)

        with col1:
            industry = st.selectbox('Select the Industry', ['Finance', 'Services', 'Healthcare', 'Manufacturing', 'Technology'])
            company_size = st.selectbox('Select Company Size', ['Large', 'Medium', 'Small'])
            city = st.selectbox('Select City', ['Chengdu', 'Guangzhou', 'Beijing', 'Shenzhen', 'Shanghai'])
            year = st.number_input('Year', min_value=2000, max_value=2024, value=2012, format="%d")
            revenue_north_china = st.number_input('Revenue North China', value=0.0, format="%f")
            revenue_east_china = st.number_input('Revenue East China', value=0.0, format="%f")
            revenue_south_china = st.number_input('Revenue South China', value=0.0, format="%f")
            revenue_west_china = st.number_input('Revenue West China', value=0.0, format="%f")
            revenue_central_china = st.number_input('Revenue Central China', value=0.0, format="%f")
            revenue = st.number_input('Revenue', value=0.0, format="%f")

        with col2:
            assets = st.number_input('Assets', value=0.0, format="%f")
            liabilities = st.number_input('Liabilities', value=0.0, format="%f")
            equity = st.number_input('Equity', value=0.0, format="%f")
            profit_loss = st.number_input('Profit Loss', value=0.0, format="%f")
            dividends = st.number_input('Dividends', value=0.0, format="%f")
            market_cap = st.number_input('Market Cap', value=0.0, format="%f")
            pe_ratio = st.number_input('PE Ratio', value=0.0, format="%f")
            current_ratio = st.number_input('Current Ratio', value=0.0, format="%f")
            expenses = st.number_input('Expenses', value=0.0, format="%f")
            growth_rate = st.number_input('Growth Rate', value=0.0, format="%f")

        # Submit button
        submit_button = st.form_submit_button(label='Predict Return On Equity')

    # Prediction
    if submit_button:
        with st.spinner('Predicting...'):
            # Combine all inputs into a single DataFrame
            input_data = {
                'Assets': assets,
                'Liabilities': liabilities,
                'Equity': equity,
                'Profit_Loss': profit_loss,
                'Revenue_North China': revenue_north_china,
                'Revenue_East China': revenue_east_china,
                'Revenue_South China': revenue_south_china,
                'Revenue_West China': revenue_west_china,
                'Revenue_Central China': revenue_central_china,
                'Dividends': dividends,
                'Market_Cap': market_cap,
                'PE_Ratio': pe_ratio,
                'Current_Ratio': current_ratio,
                'Revenue': revenue,
                'Expenses': expenses,
                'Growth_Rate': growth_rate,
                'Sector': industry,
                'Company_Size': company_size,
                'City': city
            }

            input_df = pd.DataFrame([input_data])

            # Predict
            try:
                prediction = model.predict(input_df)[0]
                st.success(f'The predicted Return On Equity (ROE) is: {prediction:.5f}')
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

            st.markdown("""
                ### Disclaimer
                This is a sample prediction only. The actual ROE may vary. Please consult with a financial advisor before making any financial decisions.

                **Model Information:**
                - Preprocessing: Cleaning, transforming, feature engineering, encoding, scaling
                - Model: Random Forest Regressor (default parameters)
                - Accuracy: ~100% (R2 Score on test data)
                - Mean Squared Error (MSE): 0 (on test data)
            """)

            # Footer
            st.markdown(
                """
                <style>
                .footer {
                    position: fixed;
                    left: 0;
                    bottom: 0;
                    width: 100%;
                    background-color: #f1f1f1;
                    text-align: center;
                    padding: 20px;
                    font-size: 12px;
                    color: #333;
                }
                .footer a {
                    color: #333;
                    text-decoration: none;
                    margin: 0 10px;
                }
                </style>
                <div class="footer">
                    Developed by Jatin Mehra. Powered by Streamlit.
                    <a href="https://github.com/Jatin-Mehra119" target="_blank">GitHub</a>
                    <a href="https://dataverse.harvard.edu/dataset.xhtml;jsessionid=41510459a96a8d0e58860d844857?persistentId=doi:10.7910/DVN/XMVD5L" target="_blank">Data Source</a>
                </div>
                """, unsafe_allow_html=True
            )

if __name__ == '__main__':
    main()
