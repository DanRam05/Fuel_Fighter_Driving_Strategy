import numpy as np
import pandas as pd

def load_track(path):
    df = pd.read_csv(path)

    # Support multiple possible column names for distance
    if 'Distance' in df.columns:
        s_col = 'Distance'
    elif 'Distance from Lap Line (m)' in df.columns:
        s_col = 'Distance from Lap Line (m)'
    else:
        s_col = df.columns[0]

    s = df[s_col].values
    x = df['UTMX'].values
    y = df['UTMY'].values

    return s, x, y