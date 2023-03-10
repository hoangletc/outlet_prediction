import argparse

import numpy as np
import yaml
from sklearn.preprocessing import MinMaxScaler

from sale_prediction.preprocess import Preprocess

scaling_cols = ['store_nbr', 'family', 'sales', 'onpromotion',
                'dcoilwtico', 'city', 'state', 'type', 'cluster', 'transactions',
                'date_type_local', 'date_name_local', 'loc_local', 'date_type_region',
                'date_name_region', 'loc_region', 'date_type_nation_1',
                'date_name_nation_1', 'date_type_nation_2', 'date_name_nation_2',
                'sales_last1', 'sales_last2', 'sales_last3', 'sales_last4']


def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Data processing trigger')

    parser.add_argument('--version', '-v', type=str, default='v0', help='Data version')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = init_args()

    path = "/media/DataLinux/outlet_prediction/conf.yaml"
    with open(path) as fp:
        paths = yaml.safe_load(fp)['PATH']

    preprocessor = Preprocess(paths)

    df_fact = preprocessor.create_fact_table()

    # df_fact['sales'] = np.log(df_fact['sales'] + 1e-4)
    # df_fact['sales_last1'] = np.log(df_fact['sales_last1'] + 1e-4)
    # df_fact['sales_last2'] = np.log(df_fact['sales_last2'] + 1e-4)
    # df_fact['sales_last3'] = np.log(df_fact['sales_last3'] + 1e-4)
    # df_fact['sales_last4'] = np.log(df_fact['sales_last4'] + 1e-4)

    df_train, df_test = preprocessor.split_traintest(df_fact)

    # Encode and save training split
    dat_train = preprocessor.encode(df_train)
    dat_test = preprocessor.encode(df_test)

    # Applly scaler
    dat_tmp = np.vstack((dat_train, dat_test))

    scaler = MinMaxScaler()
    scaler.fit(dat_tmp)

    dat_train = scaler.transform(dat_train)
    dat_test = scaler.transform(dat_test)

    # Save
    preprocessor.save_split("train", args.version, dat_train)
    preprocessor.save_split("test", args.version, dat_test)
