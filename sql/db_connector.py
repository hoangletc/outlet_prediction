import argparse
import csv
import traceback

import pandas as pd
import psycopg2
import yaml
from loguru import logger
from psycopg2.extras import RealDictCursor
from tqdm import trange

from sale_prediction import preprocess

PATH_CONFIG = "/media/DataLinux/tcdata/outlet_prediction/conf.yaml"
with open(PATH_CONFIG) as fp:
    conf = yaml.safe_load(fp)


def _insert_db(conn):
    cur = conn.cursor()

    def _core(cur, table: str, df: pd.DataFrame, step: int = 10000):
        cols = ','.join(df.columns)
        presenter = ','.join(['%s'] * len(df.columns))

        for i in trange(0, len(df), step):
            rows = tuple(df[i:i+step].itertuples(index=False, name=None))

            args_str = ','.join(
                cur.mogrify(f"({presenter})", x).decode("utf-8")
                for x in rows
            )
            sql = f"INSERT INTO \"{table}\" ({cols}) VALUES {args_str}"

            cur.execute(sql)
            conn.commit()

    try:
        logger.info("Load table: oilprice")
        _core(cur, "oilprice", preprocessor.oilprice.oil)

        logger.info("Load table: total_trans")
        _core(cur, "total_trans", preprocessor.tot_trans.tot_trans)

        logger.info("Load table: stores")
        _core(cur, "stores", preprocessor.store.stores)

        logger.info("Load table: holidays")
        _core(cur, "holidays", preprocessor.holiday.holiday)

        logger.info("Load table: transactions")
        _core(cur, "transactions", preprocessor.trans.transaction)

    except Exception:
        traceback.print_exc()
    finally:
        cur.close()


def _select_db(table: str = 'final_v1'):
    logger.info(f"Select all from table: {table}")

    cur = conn.cursor(cursor_factory=RealDictCursor)
    sql = f"SELECT * FROM {table}"

    try:
        cur.execute(sql)
        results = cur.fetchall()

        keys = results[0].keys()

        with open(conf['PATH']['raw_fact'], 'w+', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
    except Exception:
        traceback.print_exc()
    finally:
        cur.close()


def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DB connector')

    parser.add_argument('--insert', action='store_true',
                        help='sum the integers (default: find the max)')
    parser.add_argument('--select-db', type=str, default="final_v1",
                        help="DB name to be selected")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = init_args()

    conn = psycopg2.connect(
        host=conf['DB']['HOST'],
        database=conf['DB']['DB'],
        user=conf['DB']['USER'],
        password=conf['DB']['PWD']
    )

    if args.insert is True:
        logger.info("Start inserting")

        preprocessor = preprocess.Preprocess(conf['PATH'], read_idx=False)

        _insert_db(conn)
    else:
        logger.info("Start selecting")

        _select_db(args.select_db)

    conn.close()
