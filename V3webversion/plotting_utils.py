@st.cache_data
def plot_spot_gamma(dfAgg, strikes, fromStrike, toStrike, spotPrice):
    """Plot absolute gamma exposure"""
    # Plot 1: Total Gamma
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

    # Plot 2: Call vs Put Gamma
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
def plot_greeks_profiles(levels, spotPrice, totalGamma, totalGammaExNext, totalGammaExFri, 
                        totalVanna, totalCharm, todayDate, fromStrike, toStrike):
    """Plot Greeks profiles with Streamlit"""
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
    ax1.