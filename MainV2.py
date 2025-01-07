import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

pd.options.display.float_format = '{:,.4f}'.format

def calcGammaEx(S, K, vol, T, r, q, optType, OI):
    if T == 0 or vol == 0:
        return 0
    
    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T)
    
    gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
    return OI * 100 * S * S * 0.01 * gamma

def calcVannaEx(S, K, vol, T, r, q, optType, OI):
    if T == 0 or vol == 0:
        return 0
    
    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T)
    
    vanna = -np.exp(-q*T) * norm.pdf(dp) * dm / (S * vol)
    if optType == 'put':
        vanna = -vanna
    return OI * 100 * S * 0.01 * vanna

def calcCharmEx(S, K, vol, T, r, q, optType, OI):
    if T == 0 or vol == 0:
        return 0
    
    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T)
    
    charm = q*np.exp(-q*T)*norm.cdf(dp) - np.exp(-q*T)*norm.pdf(dp)*(2*(r-q)*T - dp*vol*np.sqrt(T))/(2*T*vol*np.sqrt(T))
    if optType == 'put':
        charm = charm - q*np.exp(-q*T)
    return OI * 100 * charm

def isThirdFriday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21

# File reading and initial data processing
def process_spx_data(filename):
    # Read the file and get spot price
    optionsFile = open(filename)
    optionsFileData = optionsFile.readlines()
    optionsFile.close()
    
    # Get SPX Spot
    spotLine = optionsFileData[1]
    spotPrice = float(spotLine.split('Last:')[1].split(',')[0])
    fromStrike = 0.8 * spotPrice
    toStrike = 1.2 * spotPrice
    
    # Get Today's Date
    dateLine = optionsFileData[2]
    todayDate = dateLine.split('Date: ')[1].split(' at ')[0].strip()
    todayDate = datetime.strptime(todayDate, '%B %d, %Y')
    
    # Read and process the options data
    df = pd.read_csv(filename, sep=",", header=None, skiprows=4)
    df.columns = ['ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
                'CallIV','CallDelta','CallGamma','CallOpenInt','StrikePrice','Puts','PutLastSale',
                'PutNet','PutBid','PutAsk','PutVol','PutIV','PutDelta','PutGamma','PutOpenInt']
    
    df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y')
    df['ExpirationDate'] = df['ExpirationDate'] + timedelta(hours=16)
    df['StrikePrice'] = df['StrikePrice'].astype(float)
    df['CallIV'] = df['CallIV'].astype(float)
    df['PutIV'] = df['PutIV'].astype(float)
    df['CallGamma'] = df['CallGamma'].astype(float)
    df['PutGamma'] = df['PutGamma'].astype(float)
    df['CallOpenInt'] = df['CallOpenInt'].astype(float)
    df['PutOpenInt'] = df['PutOpenInt'].astype(float)
    
    return df, spotPrice, fromStrike, toStrike, todayDate

def calculate_spot_gamma(df, spotPrice):
    # Calculate spot gamma exposure
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1
    df['TotalGamma'] = (df.CallGEX + df.PutGEX) / 10**9
    
    return df.groupby(['StrikePrice']).sum(numeric_only=True)

def plot_spot_gamma(dfAgg, strikes, fromStrike, toStrike, spotPrice):
    # Plot absolute gamma exposure
    plt.figure(figsize=(12, 5))
    plt.grid()
    plt.bar(strikes, dfAgg['TotalGamma'].to_numpy(), width=6, linewidth=0.1, edgecolor='k', label="Gamma Exposure")
    plt.xlim([fromStrike, toStrike])
    chartTitle = "Total Gamma: $" + str("{:.2f}".format(dfAgg['TotalGamma'].sum())) + " Bn per 1% SPX Move"
    plt.title(chartTitle, fontweight="bold", fontsize=20)
    plt.xlabel('Strike', fontweight="bold")
    plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
    plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot: " + str("{:,.0f}".format(spotPrice)))
    plt.legend()
    plt.show()
    
    # Plot calls vs puts
    plt.figure(figsize=(12, 5))
    plt.grid()
    plt.bar(strikes, dfAgg['CallGEX'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Call Gamma")
    plt.bar(strikes, dfAgg['PutGEX'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Put Gamma")
    plt.xlim([fromStrike, toStrike])
    chartTitle = "Total Gamma: $" + str("{:.2f}".format(dfAgg['TotalGamma'].sum())) + " Bn per 1% SPX Move"
    plt.title(chartTitle, fontweight="bold", fontsize=20)
    plt.xlabel('Strike', fontweight="bold")
    plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold")
    plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot: " + str("{:,.0f}".format(spotPrice)))
    plt.legend()
    plt.show()

def calculate_greeks_profile(df, levels, spotPrice, todayDate, nextExpiry, nextMonthlyExp):
    totalGamma = []
    totalVanna = []
    totalCharm = []
    totalGammaExNext = []
    totalGammaExFri = []
    
    for level in levels:
        # Calculate Gamma Exposure
        df['callGammaEx'] = df.apply(lambda row: calcGammaEx(level, row['StrikePrice'], 
            row['CallIV'], row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), axis=1)
        df['putGammaEx'] = df.apply(lambda row: calcGammaEx(level, row['StrikePrice'], 
            row['PutIV'], row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis=1)
        
        # Calculate Vanna Exposure
        df['callVannaEx'] = df.apply(lambda row: calcVannaEx(level, row['StrikePrice'], 
            row['CallIV'], row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), axis=1)
        df['putVannaEx'] = df.apply(lambda row: calcVannaEx(level, row['StrikePrice'], 
            row['PutIV'], row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis=1)
        
        # Calculate Charm Exposure
        df['callCharmEx'] = df.apply(lambda row: calcCharmEx(level, row['StrikePrice'], 
            row['CallIV'], row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), axis=1)
        df['putCharmEx'] = df.apply(lambda row: calcCharmEx(level, row['StrikePrice'], 
            row['PutIV'], row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis=1)
        
        # Calculate total exposures
        gammaEx = df['callGammaEx'].sum() - df['putGammaEx'].sum()
        totalGamma.append(gammaEx)
        totalVanna.append(df['callVannaEx'].sum() - df['putVannaEx'].sum())
        totalCharm.append(df['callCharmEx'].sum() - df['putCharmEx'].sum())
        
        # Calculate ex-expiry exposures
        exNxt = df.loc[df['ExpirationDate'] != nextExpiry]
        totalGammaExNext.append(exNxt['callGammaEx'].sum() - exNxt['putGammaEx'].sum())
        
        exFri = df.loc[df['ExpirationDate'] != nextMonthlyExp]
        totalGammaExFri.append(exFri['callGammaEx'].sum() - exFri['putGammaEx'].sum())
    
    return (np.array(totalGamma) / 10**9, 
            np.array(totalGammaExNext) / 10**9,
            np.array(totalGammaExFri) / 10**9,
            np.array(totalVanna) / 10**9, 
            np.array(totalCharm) / 10**9)
def find_gamma_flip(levels, totalGamma):
    zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]
    if len(zeroCrossIdx) > 0:
        negGamma = totalGamma[zeroCrossIdx[0]]
        posGamma = totalGamma[zeroCrossIdx[0]+1]
        negStrike = levels[zeroCrossIdx[0]]
        posStrike = levels[zeroCrossIdx[0]+1]
        return posStrike - ((posStrike - negStrike) * posGamma/(posGamma-negGamma))
    return None

def plot_greeks_profiles(levels, spotPrice, totalGamma, totalGammaExNext, totalGammaExFri, 
                        totalVanna, totalCharm, todayDate, fromStrike, toStrike):
    # Create figure with subplots
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
    plt.show()

def main():
    # Input Parameters
    filename = 'spx_quotedata.csv'
    
    # Process data
    df, spotPrice, fromStrike, toStrike, todayDate = process_spx_data(filename)
    
    # Calculate days till expiration
    df['daysTillExp'] = [1/262 if (np.busday_count(todayDate.date(), x.date())) == 0 
                         else np.busday_count(todayDate.date(), x.date())/262 for x in df.ExpirationDate]
    
    # Get next expiry dates
    nextExpiry = df['ExpirationDate'].min()
    df['IsThirdFriday'] = [isThirdFriday(x) for x in df.ExpirationDate]
    thirdFridays = df.loc[df['IsThirdFriday'] == True]
    nextMonthlyExp = thirdFridays['ExpirationDate'].min()

    # Calculate spot gamma
    dfAgg = calculate_spot_gamma(df, spotPrice)
    strikes = dfAgg.index.values

    # Plot spot gamma
    plot_spot_gamma(dfAgg, strikes, fromStrike, toStrike, spotPrice)

    # Calculate greeks profiles
    levels = np.linspace(fromStrike, toStrike, 60)
    totalGamma, totalGammaExNext, totalGammaExFri, totalVanna, totalCharm = calculate_greeks_profile(
        df, levels, spotPrice, todayDate, nextExpiry, nextMonthlyExp
    )

    # Plot greeks profiles
    plot_greeks_profiles(
        levels, spotPrice, totalGamma, totalGammaExNext, totalGammaExFri,
        totalVanna, totalCharm, todayDate, fromStrike, toStrike
    )

if __name__ == "__main__":
    main()