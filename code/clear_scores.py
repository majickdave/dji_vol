import pandas as pd
for col in ['handle_time', 'volume', 'aht']:
    df = pd.read_csv('./scores/'+col+'_score.csv', index_col=0)
    df.drop(df.index).to_csv('./scores/'+col+'_score.csv')