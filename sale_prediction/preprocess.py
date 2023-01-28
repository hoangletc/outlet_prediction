import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from sklego.preprocessing import RepeatingBasisFunction
from tqdm import tqdm

RE_DESCRIPTION = re.compile(r"((\w|\s|\:)+)(\+|\-)\d+")
NATION = 'ecuador'
NUM_STORES = 54


def _scale_minmax(df: pd.DataFrame, col: str):
    scaler = MinMaxScaler()
    df.loc[:, col] = scaler.fit_transform(
        np.expand_dims(df[col].to_numpy(), 1)
    ).squeeze()


def _proc_des(des: str):

    tmp = re.findall(RE_DESCRIPTION, des)
    if len(tmp) > 0:
        des = tmp[0][0]

    des = des.removeprefix("traslado ")
    des = des.removeprefix("puente ")

    if des.startswith("cantonizacion de"):
        des = "cantonizacion de"
    elif des.startswith("dia de"):
        des = "dia de"
    elif des.startswith("fundacion"):
        des = "fundacion"
    elif des.count("independencia") != 0:
        des = "independencia"
    elif des.startswith("recupero"):
        des = "recupero"
    elif des.startswith("provincializacion"):
        des = "provincializacion"
    elif "de futbol" in des:
        des = "futbol"
    return des


def _process_entry(entry, df: pd.DataFrame) -> Tuple[Union[str, None], Union[str, None],
                                                     Union[str, None], Union[str, None], bool]:
    locale_name, date = entry.locale_name, entry.date
    raw_date_type = entry.type
    des = entry.description
    date_type = "work day"
    ignored = False

    # Process case type = 'Event'
    if raw_date_type == "event":
        date_type = "event"

    # Process case type = 'transfer'
    if raw_date_type == "holiday":
        if entry.transferred is False:
            date_type = "holiday"
        else:
            date_type = "work day"

    if raw_date_type == "bridge":
        date_type = "holiday"

    if raw_date_type == "transfer":
        date_type = "holiday"

    # Process case type = 'additional'
    if raw_date_type == "additional":
        # Check whether there is "bridge" occuring in the same date
        c1 = df['date'] == date
        c2 = df['type'].isin(['bridge', 'event', 'transfer', 'holiday'])
        c3 = df['description'] == des
        c4 = df['locale_name'] == locale_name
        df_tmp = df[c1 & c2 & c3 & c4]
        assert len(df_tmp) <= 1
        if len(df_tmp) == 1:
            ignored = True
            return None, None, None, None, ignored
        else:
            date_type = "additional"

    # Some default set
    if date_type == "work day":
        des = "work day"
    if date_type in ["work day", "weekend"]:
        locale_name = "ecuador"

    return date, date_type, locale_name, des, ignored


class Mapping():
    def __init__(self, path_map) -> None:
        logger.info("Init Mapping")

        path_map = Path(path_map)

        assert path_map.exists(), f"Path to map not existed: {path_map}"

        self.mapping = self._establish_db(path_map)

    def _establish_db(self, path_map: Path):
        out = {}

        with open(path_map) as fp:
            d = json.load(fp)
        for k, v in d.items():
            out[k] = {x: i for i, x in enumerate(v)}

        return out

    def __getitem__(self, k):
        return self.mapping[k]


class HolidayInfo:
    def __init__(self, mapping: Mapping,  path_raw_holiday: str, path_res_holiday: str) -> None:
        logger.info("Init HolidayInfo")

        self.mapping = mapping

        if Path(path_res_holiday).exists():
            self.holiday = pd.read_csv(path_res_holiday)
        else:
            assert path_raw_holiday is not None and \
                Path(path_raw_holiday).exists(), \
                "As 'path_res_holiday' not existed, 'path_raw_holiday' must be specified"

            self.holiday = self._establish_db(path_raw_holiday)

            # Save to CSV
            self.holiday.to_csv(path_res_holiday, index=False)

    def get_holiday_info(self, date: str, city: str, state: str) -> tuple[int, int, int, int, int, int, int, int]:
        """Get info of given date

        Args:
            date (str): format "yyyy-mm-dd"
            city (str): city name (used to look for 'local')
            state (str): state name (used to look for 'regional')

        Returns:
            tuple: tuple of encoded holiday info
        """

        def _set_default(default: Literal['work day', 'weekend'] = 'work day'):
            date_type = self.mapping['date_type'][default]
            date_name = self.mapping['date_name'][default]

            return date_type, date_name

        date_type_local, date_name_local, loc_name_local = -1, -1, -1
        date_type_region, date_name_region, loc_name_region = -1, -1, -1
        date_type_nation_1, date_name_nation_1 = -1, -1
        date_type_nation_2, date_name_nation_2 = -1, -1

        df = self.holiday[self.holiday['date'] == date]
        date_obj = datetime.strptime(date, "%Y-%m-%d")

        if len(df) == 0:
            if date_obj.weekday() >= 5:
                date_type_nation_1, date_name_nation_1 = _set_default('weekend')
            else:
                date_type_nation_1, date_name_nation_1 = _set_default()
        else:
            # Get local holiday info
            d_local = df[df['locale_name'] == city]
            assert len(d_local) <= 1

            if len(d_local) == 1:
                row = d_local.iloc[0]

                date_type_local = self.mapping['date_type'][row['date_type']]
                date_name_local = self.mapping['date_name'][row['date_name']]
                loc_name_local = self.mapping['city'][city]

            # Get regional holiday info
            d_reg = df[df['locale_name'] == state]
            assert len(d_reg) <= 1

            if len(d_reg) == 1:
                row = d_reg.iloc[0]

                date_type_region = self.mapping['date_type'][row['date_type']]
                date_name_region = self.mapping['date_name'][row['date_name']]
                loc_name_region = self.mapping['state'][state]

            # Get national holiday info
            d_nat = df[df['locale_name'] == NATION]
            assert len(d_nat) <= 2

            if len(d_nat) >= 1:
                row = d_nat.iloc[0]

                date_type_nation_1 = self.mapping['date_type'][row['date_type']]
                date_name_nation_1 = self.mapping['date_name'][row['date_name']]

                if len(d_nat) == 2:
                    row = d_nat.iloc[1]

                    date_type_nation_2 = self.mapping['date_type'][row['date_type']]
                    date_name_nation_2 = self.mapping['date_name'][row['date_name']]
            else:
                if date_obj.weekday() >= 5:
                    date_type_nation_1, date_name_nation_1 = _set_default('weekend')
                else:
                    date_type_nation_1, date_name_nation_1 = _set_default()

        out = (date_type_local, date_name_local, loc_name_local,
               date_type_region, date_name_region, loc_name_region,
               date_type_nation_1, date_name_nation_1,
               date_type_nation_2, date_name_nation_2)

        return out

    def _establish_db(self, path_raw: str):
        holidayinfo = []
        df = pd.read_csv(path_raw)

        # Preprocess some columns
        df.loc[:, 'type'] = df['type'].str.lower()
        df.loc[:, 'locale_name'] = df['locale_name'].str.lower()
        df.loc[:, 'locale'] = df['locale'].str.lower()
        df.loc[:, 'description'] = df['description'].str.lower()
        df.loc[:, 'description'] = df['description'].apply(_proc_des)

        # Start looping
        for entry in df.itertuples():
            date, date_type, locale_name, date_name, ignored = _process_entry(entry, df)

            if ignored is False:
                holidayinfo.append({
                    'date_type': date_type,
                    'date': date,
                    'date_name': date_name,
                    'locale_name': locale_name,
                })

        out = pd.DataFrame.from_records(holidayinfo)

        return out


class OilPrice():
    def __init__(self, path_raw_oil: str, path_res_oil: str) -> None:
        logger.info("Init OilPrice")

        path_raw, path_res = Path(path_raw_oil), Path(path_res_oil)

        if not path_res.exists():
            # Create db
            assert path_raw.exists(), f"Path to raw oil not existed: {path_raw}"

            self.oil = self._establish_db(path_raw)
            self.oil.to_csv(path_res_oil, index=False)
        else:
            # Load processed
            self.oil = pd.read_csv(path_res, index_col='date')

    def _establish_db(self, path_raw):
        d = pd.read_csv(path_raw)

        # Fill NaN
        for i in range(len(d)):
            if not math.isnan(d.at[i, 'dcoilwtico']):
                continue

            if i == 0:
                j = i + 1
                while math.isnan(d.at[j, 'dcoilwtico']) and j < len(d) - 1:
                    j += 1

                d.at[i, 'dcoilwtico'] = d.at[j, 'dcoilwtico']
            elif i == len(d) - 1:
                j = i - 1
                while math.isnan(d.at[j, 'dcoilwtico']) and j > 0:
                    j -= 1

                d.at[i, 'dcoilwtico'] = d.at[j, 'dcoilwtico']
            else:
                j = i + 1
                while math.isnan(d.at[j, 'dcoilwtico']) and j < len(d) - 1:
                    j += 1
                k = i - 1
                while math.isnan(d.at[k, 'dcoilwtico']) and k > 0:
                    k -= 1
                d.at[i, 'dcoilwtico'] = (d.at[j, 'dcoilwtico'] + d.at[j, 'dcoilwtico']) / 2

        # Scale: MinMaxScaler
        _scale_minmax(d, 'dcoilwtico')

        return d

    def get_oilprice(self, date: str) -> float:
        oilprice = -1
        if date in self.oil.index:
            oilprice = self.oil.loc[date].dcoilwtico
        else:
            date_obj = datetime.strptime(date, "%Y-%m-%d")

            while date_obj > datetime(2011, 12, 31):
                date_obj = date_obj - timedelta(days=1)

                if date_obj.strftime("%Y-%m-%d") in self.oil.index:
                    oilprice = self.oil.loc[date_obj.strftime("%Y-%m-%d")].dcoilwtico
                    break

        return oilprice


class DateTime():
    start, end = '2012-01-01', '2017-12-31'

    def __init__(self) -> None:
        logger.info("Init DateTime")

        self.dt = self._establish_db()

    def _establish_db(self):

        range_of_dates = pd.date_range(start=DateTime.start, end=DateTime.end)

        X = pd.DataFrame(index=range_of_dates)
        X["day_nr"] = range(len(X))
        X["day_of_year"] = X.index.day_of_year

        rbf = RepeatingBasisFunction(n_periods=12, column="day_of_year",
                                     input_range=(1, 365), remainder="drop")
        rbf.fit(X)
        X = pd.DataFrame(index=X.index, data=rbf.transform(X))

        return X

    def get_dt_enc(self, datetime: str):
        assert DateTime.start <= datetime <= DateTime.end, f'Given datetime invalid: {datetime}'

        return list(self.dt.loc[datetime])


class Store():
    def __init__(self, mapping: Mapping,  path_raw_store: str, path_res_store: str) -> None:
        logger.info("Init Store")

        self.mapping = mapping

        path_raw, path_res = Path(path_raw_store), Path(path_res_store)

        if not path_res.exists():
            # Create db
            assert path_raw.exists(), f"Path to store's data not existed: {path_raw}"

            self.stores: pd.DataFrame = self._establish_db(path_raw)
            self.stores.to_csv(path_res)
        else:
            # Load processed
            self.stores = pd.read_csv(path_res, index_col='store_nbr')

    def _establish_db(self, path_raw):
        stores = pd.read_csv(path_raw, index_col='store_nbr')

        stores['city'] = stores['city'].str.lower()
        stores['state'] = stores['state'].str.lower()

        return stores

    def get_store_info(self, storeid: int):
        try:
            row = self.stores.loc[storeid]

            info = {
                'city': row.city,
                'state': row.state
            }
            enc = (self.mapping['store_type'][row.type], row.cluster)
        except KeyError:
            raise AssertionError(f"Invalid storeid: {storeid}")
        return info, enc


class Preprocess():
    def __init__(self, paths: dict[str, str]) -> None:
        self.paths = paths

        self.mapping = Mapping(paths['mapping'])

        self.dt = DateTime()
        self.oilprice = OilPrice(paths['raw_oil'], paths['res_oil'])
        self.holiday = HolidayInfo(self.mapping, paths['raw_holiday'], paths['res_holiday'])
        self.store = Store(self.mapping, paths['raw_store'], paths['res_store'])

    def _preprocess(self, idx: int, data: pd.DataFrame) -> np.ndarray:

        # Some basic transforms: Apply scaling on 'onpromotion', lowercase
        _scale_minmax(data, 'onpromotion')
        data.loc[:, 'family'] = data['family'].str.lower()

        # Start processing
        processed = []
        for r in tqdm(data.itertuples(), total=data.shape[0], desc=f"Start preprocess {idx}"):
            entry = []

            entry.append(r.sales)
            entry.append(r.store_nbr)

            # Get and process store info
            info, enc = self.store.get_store_info(r.store_nbr)
            entry.extend(enc)

            # Process oil price
            oilprice = self.oilprice.get_oilprice(r.date)
            entry.append(oilprice)

            # Process product
            enc_product = self.mapping['product'][r.family]
            entry.append(enc_product)
            entry.append(r.onpromotion)

            # Process datetime
            enc_datetime = self.dt.get_dt_enc(r.date)
            entry.extend(enc_datetime)

            # Process holiday
            enc_holiday = self.holiday.get_holiday_info(r.date, info['city'],
                                                        info['state'])
            entry.extend(enc_holiday)

            processed.append(entry)

        # Convert to np array and save
        processed = np.array(processed, dtype=np.float32)

        return processed

    def preprocess(self, split: Literal["train", "test"] = "train"):
        if split == "train":
            path_raw, path_res = self.paths['raw_train'], self.paths['res_train']
        elif split == "test":
            path_raw, path_res = self.paths['raw_test'], self.paths['res_test']
        else:
            raise NotImplementedError()

        df = pd.read_csv(path_raw)
        n_files_created = len(list(Path(path_res).parent.glob("store_*.npz")))
        if n_files_created == 0:
            logger.info("No file created. Starting...")

            start = 1
        else:
            logger.info(f"Found {n_files_created} files created. Resuming...")

            start = n_files_created + 1

        for i in range(start, NUM_STORES + 1):
            df_store = df[df['store_nbr'] == i].copy()

            out = self._preprocess(i, df_store)

            path_save = path_res.replace("<id>", f"{i:02d}")
            Path(path_save).parent.mkdir(exist_ok=True)
            np.savez_compressed(path_save, out)


if __name__ == '__main__':
    path = "/media/DataLinux/tcdata/outlet_prediction/conf.yaml"
    with open(path) as fp:
        paths = yaml.safe_load(fp)['PATH']

    Preprocess(paths).preprocess()

    # mapping = Mapping(paths['mapping'])
    # holiday = HolidayInfo(mapping, paths['raw_holiday'], paths['res_holiday'])
    # print(holiday.get_holiday_info("2012-12-24", 'mila', 'mali'))
