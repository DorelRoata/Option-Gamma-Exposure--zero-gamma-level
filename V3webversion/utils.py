import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
from config import PLOT_STYLE, DATA_CONFIG, GREEK_SCALING

def calcGammaEx(S, K, vol, T, r, q, optType, OI):
    """Calculate gamma exposure"""
    if T == 0 or vol == 0:
        return 0
    
    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T)
    
    gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
    return OI * 100 * S * S * 0.01 * gamma

def calcVannaEx(S, K, vol, T, r, q, optType, OI):
    """Calculate vanna exposure"""
    if T == 0 or vol == 0:
        return 0
    
    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T)
    
    vanna = -np.exp(-q*T) * norm.pdf(dp) * dm / (S * vol)
    if optType == 'put':
        vanna = -vanna
    return OI * 100 * S * 0.01 * vanna

def calcCharmEx(S, K, vol, T, r, q, optType, OI):
    """Calculate charm exposure"""
    if T == 0 or vol == 0:
        return 0
    
    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T)
    
    charm = q*np.exp(-q*T)*norm.cdf(dp) - np.exp(-q*T)*norm.pdf(dp)*(2*(r-q)*T - dp*vol*np.sqrt(T))/(2*T*vol*np.sqrt(T))
    if optType == 'put':
        charm = charm - q*np.exp(-q*T)
    return OI * 100 * charm

def isThirdFriday(d):
    """Check if date is third Friday of month"""
    return d.weekday() == 4 and 15 <= d.day <= 21

def process_file_data(uploaded_file):
    """Process the uploaded CSV file and extract key data"""
    # Read file content
    uploaded_file.seek(0)
    file_data = uploaded_file.read().decode("utf-8").splitlines()
    
    # Extract spot price
    spot_line = file_data[1]
    spotPrice = float(spot_line.split('Last:')[1].split(',')[0])
    fromStrike = DATA_CONFIG['strike_range']['lower'] * spotPrice
    toStrike = DATA_CONFIG['strike_range']['upper'] * spotPrice
    
    # Extract date
    date_line = file_data[2]
    today_date = date_line.split('Date: ')[1].strip()
    if " at " in today_date:
        today_date = today_date.split(" at ")[0].strip()
    elif "," in today_date:
        today_date = today_date.split(",")[0].strip()
    todayDate = datetime.strptime(today_date, '%B %d, %Y')
    
    # Read CSV data
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=",", header=None, skiprows=4, dtype={
        'StrikePrice': float,
        'CallIV': float,
        'PutIV': float,
        'CallGamma': float,
        'PutGamma': float,
        'CallOpenInt': float,
        'PutOpenInt': float
    })
    
    # Set column names
    df.columns = ['ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
                 'CallIV','CallDelta','CallGamma','CallOpenInt','StrikePrice','Puts','PutLastSale',
                 'PutNet','PutBid','PutAsk','PutVol','PutIV','PutDelta','PutGamma','PutOpenInt']
    
    # Process date columns
    df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y') + timedelta(hours=16)
    
    return df, spotPrice, fromStrike, toStrike, todayDate

@st.cache_data
def calculate_greeks(_df, spotPrice, fromStrike, toStrike, todayDate):
    """Calculate all Greeks and return results dictionary"""
    df = _df.copy()  # Make a copy to avoid modifying original
    
    # Calculate days till expiration
    df['daysTillExp'] = [1/DATA_CONFIG['trading_days_per_year'] if (np.busday_count(todayDate.date(), x.date())) == 0 
                         else np.busday_count(todayDate.date(), x.date())/DATA_CONFIG['trading_days_per_year'] 
                         for x in df.ExpirationDate]

    # Get next expiry dates
    nextExpiry = df['ExpirationDate'].min()
    df['IsThirdFriday'] = [isThirdFriday(x) for x in df.ExpirationDate]
    thirdFridays = df.loc[df['IsThirdFriday'] == True]
    nextMonthlyExp = thirdFridays['ExpirationDate'].min()

    # Calculate spot gamma
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1
    df['TotalGamma'] = (df.CallGEX + df.PutGEX) / GREEK_SCALING['gamma']

    # Calculate aggregate values by strike (dfAgg)
    numeric_columns = ['CallGEX', 'PutGEX', 'TotalGamma', 'CallOpenInt', 'PutOpenInt']
    dfAgg = df.groupby('StrikePrice')[numeric_columns].sum()
    strikes = dfAgg.index.values

    # Calculate Greeks profiles
    levels = np.linspace(fromStrike, toStrike, DATA_CONFIG['plot_points'])
    
    totalGamma = []
    totalGammaExNext = []
    totalGammaExFri = []
    totalVanna = []
    totalCharm = []

    for level in levels:
        # Calculate Gamma
        df['callGammaEx'] = df.apply(lambda row: calcGammaEx(level, row['StrikePrice'], 
            row['CallIV'], row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), axis=1)
        df['putGammaEx'] = df.apply(lambda row: calcGammaEx(level, row['StrikePrice'], 
            row['PutIV'], row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis=1)
        
        # Calculate Vanna
        df['callVannaEx'] = df.apply(lambda row: calcVannaEx(level, row['StrikePrice'], 
            row['CallIV'], row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), axis=1)
        df['putVannaEx'] = df.apply(lambda row: calcVannaEx(level, row['StrikePrice'], 
            row['PutIV'], row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis=1)
        
        # Calculate Charm
        df['callCharmEx'] = df.apply(lambda row: calcCharmEx(level, row['StrikePrice'], 
            row['CallIV'], row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), axis=1)
        df['putCharmEx'] = df.apply(lambda row: calcCharmEx(level, row['StrikePrice'], 
            row['PutIV'], row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis=1)

        gammaEx = df['callGammaEx'].sum() - df['putGammaEx'].sum()
        totalGamma.append(gammaEx)
        totalVanna.append(df['callVannaEx'].sum() - df['putVannaEx'].sum())
        totalCharm.append(df['callCharmEx'].sum() - df['putCharmEx'].sum())

        # Calculate ex-expiry exposures
        exNxt = df[df['ExpirationDate'] != nextExpiry]
        totalGammaExNext.append(exNxt['callGammaEx'].sum() - exNxt['putGammaEx'].sum())

        exFri = df[df['ExpirationDate'] != nextMonthlyExp]
        totalGammaExFri.append(exFri['callGammaEx'].sum() - exFri['putGammaEx'].sum())

    # Scale all Greeks
    totalGamma = np.array(totalGamma) / GREEK_SCALING['gamma']
    totalGammaExNext = np.array(totalGammaExNext) / GREEK_SCALING['gamma']
    totalGammaExFri = np.array(totalGammaExFri) / GREEK_SCALING['gamma']
    totalVanna = np.array(totalVanna) / GREEK_SCALING['vanna']
    totalCharm = np.array(totalCharm) / GREEK_SCALING['charm']

    return {
        'dfAgg': dfAgg,
        'strikes': strikes,
        'levels': levels,
        'totalGamma': totalGamma,
        'totalGammaExNext': totalGammaExNext,
        'totalGammaExFri': totalGammaExFri,
        'totalVanna': totalVanna,
        'totalCharm': totalCharm
    }

def find_gamma_flip(levels, totalGamma):
    """Find the gamma flip point"""
    zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]
    if len(zeroCrossIdx) > 0:
        negGamma = totalGamma[zeroCrossIdx[0]]
        posGamma = totalGamma[zeroCrossIdx[0]+1]
        negStrike = levels[zeroCrossIdx[0]]
        posStrike = levels[zeroCrossIdx[0]+1]
        return posStrike - ((posStrike - negStrike) * posGamma/(posGamma-negGamma))
    return None

@st.cache_data
def plot_spot_gamma(dfAgg, strikes, fromStrike, toStrike, spotPrice):
    """Plot spot gamma exposure"""
    # Total Gamma
    fig1, ax1 = plt.subplots(figsize=PLOT_STYLE['figsize'])
    plt.grid(True)
    plt.bar(strikes, dfAgg['TotalGamma'].to_numpy(), 
            width=6, linewidth=0.1, edgecolor='k', 
            alpha=PLOT_STYLE['alpha']['bar'],
            label="Gamma Exposure")
    plt.xlim([fromStrike, toStrike])
    
    chartTitle = f"Total Gamma: ${dfAgg['TotalGamma'].sum():.2f} Bn per 1% Move"
    plt.title(chartTitle, fontsize=PLOT_STYLE['fontsize']['title'], fontweight="bold")
    plt.xlabel('Strike', fontweight="bold")
    plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
    plt.axvline(x=spotPrice, color='r', lw=1, label=f"Spot: {spotPrice:,.0f}")
    plt.legend()
    st.pyplot(fig1)
    plt.close()

    # Calls vs Puts
    fig2, ax2 = plt.subplots(figsize=PLOT_STYLE['figsize'])
    plt.grid(True)
    plt.bar(strikes, dfAgg['CallGEX'].to_numpy() / GREEK_SCALING['gamma'], 
            width=6, linewidth=0.1, edgecolor='k', 
            alpha=PLOT_STYLE['alpha']['bar'],
            color=PLOT_STYLE['colors']['call'], 
            label="Call Gamma")
    plt.bar(strikes, dfAgg['PutGEX'].to_numpy() / GREEK_SCALING['gamma'], 
            width=6, linewidth=0.1, edgecolor='k', 
            alpha=PLOT_STYLE['alpha']['bar'],
            color=PLOT_STYLE['colors']['put'], 
            label="Put Gamma")
    plt.xlim([fromStrike, toStrike])
    plt.title(chartTitle, fontsize=PLOT_STYLE['fontsize']['title'], fontweight="bold")
    plt.xlabel('Strike', fontweight="bold")
    plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
    plt.axvline(x=spotPrice, color='r', lw=1, label=f"Spot: {spotPrice:,.0f}")
    plt.legend()
    st.pyplot(fig2)
    plt.close()

@st.cache_data
def plot_greeks_profiles(levels, spotPrice, totalGamma, totalGammaExNext, totalGammaExFri, 
                        totalVanna, totalCharm, todayDate, fromStrike, toStrike):
    """Plot Greeks profiles"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Find gamma flip point
    zeroGamma = find_gamma_flip(levels, totalGamma)
    
    # Gamma Profile
    ax1.grid(True)
    ax1.plot(levels, totalGamma, label="All Expiries")
    ax1.plot(levels, totalGammaExNext, label="Ex-Next Expiry")
    ax1.plot(levels, totalGammaExFri, label="Ex-Next Monthly Expiry")
    ax1.set_title("Gamma Exposure Profile", fontweight="bold", fontsize=14)
    ax1.set_xlabel('Index Price', fontweight="bold")
    ax1.set_ylabel('Gamma ($ billions/1% move)', fontweight="bold")
    ax1.axvline(x=spotPrice, color='r', lw=1, label=f"Spot: {spotPrice:,.0f}")
    if zeroGamma is not None:
        ax1.axvline(x=zeroGamma, color='g', lw=1, label=f"Flip: {zeroGamma:,.0f}")
    ax1.axhline(y=0, color='grey', lw=1)
    ax1.set_xlim([fromStrike, toStrike])
    ax1.legend()
    

    # Vanna Profile
    ax2.grid(True)
    ax2.plot(levels, totalVanna)
    ax2.set_title("Vanna Exposure Profile", fontweight="bold", fontsize=14)
    ax2.set_xlabel('Index Price', fontweight="bold")
    ax2.set_ylabel('Vanna ($ billions/1% vol)', fontweight="bold")
    ax2.axvline(x=spotPrice, color='r', lw=1, label=f"Spot: {spotPrice:,.0f}")
    ax2.axhline(y=0, color='grey', lw=1)
    ax2.set_xlim([fromStrike, toStrike])
    ax2.legend()
    
    # Charm Profile
    ax3.grid(True)
    ax3.plot(levels, totalCharm)
    ax3.set_title("Charm Exposure Profile", fontweight="bold", fontsize=14)
    ax3.set_xlabel('Index Price', fontweight="bold")
    ax3.set_ylabel('Charm ($ billions/day)', fontweight="bold")
    ax3.axvline(x=spotPrice, color='r', lw=1, label=f"Spot: {spotPrice:,.0f}")
    ax3.axhline(y=0, color='grey', lw=1)
    ax3.set_xlim([fromStrike, toStrike])
    ax3.legend()
    
    plt.suptitle(f"SPX Options Greeks Profiles\n{todayDate.strftime('%d %b %Y')}", 
                 fontweight="bold", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

@st.cache_data
def plot_put_call_ratios(df, spotPrice, fromStrike, toStrike):
    """Plot put/call ratios"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ratio_data = df.groupby('StrikePrice').agg({
        'CallVol': 'sum',
        'PutVol': 'sum',
        'CallOpenInt': 'sum',
        'PutOpenInt': 'sum'
    }).reset_index()
    
    ratio_data['VolumeRatio'] = ratio_data['PutVol'] / ratio_data['CallVol']
    ratio_data['OIRatio'] = ratio_data['PutOpenInt'] / ratio_data['CallOpenInt']
    
    # Volume ratio
    ax1.plot(ratio_data['StrikePrice'], ratio_data['VolumeRatio'], 
             color='blue', linewidth=2)
    ax1.set_title('Put/Call Volume Ratio', fontweight="bold")
    ax1.set_ylabel('P/C Volume Ratio')
    ax1.grid(True)
    ax1.axvline(x=spotPrice, color='r', linestyle='--', label='Spot')
    ax1.axhline(y=1, color='gray', linestyle='--', label='1:1')
    ax1.set_xlim([fromStrike, toStrike])
    ax1.legend()

    # OI ratio
    ax2.plot(ratio_data['StrikePrice'], ratio_data['OIRatio'], 
             color='green', linewidth=2)
    ax2.set_title('Put/Call Open Interest Ratio', fontweight="bold")
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('P/C OI Ratio')
    ax2.grid(True)
    ax2.axvline(x=spotPrice, color='r', linestyle='--', label='Spot')
    ax2.axhline(y=1, color='gray', linestyle='--', label='1:1')
    ax2.set_xlim([fromStrike, toStrike])
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()