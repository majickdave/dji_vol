import os
from run_model import *
from split_dataset import *

def main():
    # load a dataframe
    df = load_rename('./data/full_dataset.csv')

    # create csv for individual business unit
    write_data(df)

    def run_prediction(num_bus=10, 
    kpis=['handle_time', 'volume', 'aht']):
        """
        obtain predictions and score logging for 10 business units
        """
        for k in kpis:
            # set end date for training data for
            end = ['2020-11-01']*num_bus
            # set start date for training data
            start = ['2019-11-01'] *num_bus
            kpi = [k]*num_bus
            files = os.listdir('./data/')
            files = list(filter(lambda x: (x[-7:] == 'vol.csv'), files))

            for line in list(zip(files, kpi, start, end)):
                get_pred_score(line[0],line[1],line[2],line[3], periods=365)

    run_prediction()

if __name__ == "__main__":
    main()