PLOT_STYLE = {
    'figsize': (12, 6),
    'fontsize': {
        'title': 20,
        'label': 12,
        'tick': 10
    },
    'colors': {
        'call': 'green',
        'put': 'red',
        'total': 'blue',
        'spot': 'black',
        'flip': 'purple'
    },
    'alpha': {
        'bar': 0.6,
        'line': 0.8,
        'fill': 0.2
    }
}

DATA_CONFIG = {
    'trading_days_per_year': 262,
    'strike_range': {
        'lower': 0.8,
        'upper': 1.2
    },
    'plot_points': 60
}

GREEK_SCALING = {
    'gamma': 1e9,  # billions
    'vanna': 1e9,  # billions
    'charm': 1e9,  # billions
    'delta': 1e6   # millions
}