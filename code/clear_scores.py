import pandas as pd

def main():
    names = ['bu','old','prophet','kpi','start_train', 'end_train']
    dfs = {}
    for kpi in ['aht','handle_time','volume']:
        path = './scores/'+kpi+'_score.csv'
        dfs[kpi] = pd.read_csv(path, names=names, header=0)
        dfs[kpi][['old','prophet']] = dfs[kpi][['old','prophet']].applymap(lambda x: float('inf'))
        dfs[kpi].to_csv(path, index=False)

if __name__ == "__main__":
    main()