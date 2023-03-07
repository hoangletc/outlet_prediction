import yaml

from sale_prediction.preprocess import Preprocess

if __name__ == '__main__':
    path = "/media/DataLinux/tcdata/outlet_prediction/conf.yaml"
    with open(path) as fp:
        paths = yaml.safe_load(fp)['PATH']

    preprocessor = Preprocess(paths)

    df_fact = preprocessor.create_fact_table()
    df_train, df_test = preprocessor.split_traintest(df_fact)

    # # Encode and save training split
    dat_train = preprocessor.encode(df_train)
    preprocessor.save_split("train", dat_train)

    # Encode and save testing split
    # dat_test = preprocessor.encode(df_test)
    # preprocessor.save_split("test", dat_test)
