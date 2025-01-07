import streamlit as st
import numpy as np
from datetime import datetime
from utils import (
    process_file_data,
    calculate_greeks,
    plot_spot_gamma,
    plot_greeks_profiles,
    plot_put_call_ratios
)

st.set_page_config(page_title="Options Analysis Dashboard", layout="wide")

def main():
    st.title("Options Market Analysis Dashboard")

    st.markdown("""
    ## How to Use This Tool

    This dashboard provides comprehensive options market analysis including:
    - Gamma Exposure Analysis
    - Greeks Profiles
    - Put/Call Ratios
    - Volume Analysis
    
    ### Data Requirements
    1. Download options chain data from CBOE
    2. Ensure the following filters are set to ALL:
        - Volume
        - Expiration Type
        - Options Range
        
    ### Source
    [CBOE Options Chain](https://www.cboe.com/delayed_quotes/spx/quote_table)
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Process the uploaded file
            df, spotPrice, fromStrike, toStrike, todayDate = process_file_data(uploaded_file)
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Spot Price", f"${spotPrice:,.2f}")
            with col2:
                st.metric("Strike Range", f"${fromStrike:,.0f} - ${toStrike:,.0f}")
            with col3:
                st.metric("Date", todayDate.strftime('%B %d, %Y'))

            # Calculate all Greeks and profiles
            results = calculate_greeks(df, spotPrice, fromStrike, toStrike, todayDate)
            
            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["Gamma Analysis", "Greeks Profiles", "Market Ratios"])

            with tab1:
                st.subheader("Gamma Exposure Analysis")
                plot_spot_gamma(results['dfAgg'], results['strikes'], 
                              fromStrike, toStrike, spotPrice)

            with tab2:
                st.subheader("Greeks Profiles")
                plot_greeks_profiles(
                    results['levels'], spotPrice, 
                    results['totalGamma'], results['totalGammaExNext'],
                    results['totalGammaExFri'], results['totalVanna'],
                    results['totalCharm'], todayDate, fromStrike, toStrike
                )

            with tab3:
                st.subheader("Put/Call Analysis")
                plot_put_call_ratios(df, spotPrice, fromStrike, toStrike)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()