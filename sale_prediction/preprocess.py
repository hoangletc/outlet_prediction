import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import pandas as pd

RE_DESCRIPTION = re.compile(r"((\w|\s|\:)+)(\+|\-)\d+")


def _proc_des(des: str):
    des = des.lower()

    tmp = re.findall(RE_DESCRIPTION, des)
    if len(tmp) > 0:
        des = tmp[0][0]

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

    return des


def _process_entry(entry) -> Tuple[str, str, Union[str, None]]:
    date_type = "work day"
    locale_name = entry.locale_name.lower()
    raw_date_type = entry.type.lower()

    # Process case type = 'weekend'
    date_obj = datetime.strptime(entry.date, "%Y-%m-%d")
    if date_obj.weekday() < 5 and date_type != "work day":
        date_type = "weekend"

    # Process case type = 'Event'
    if raw_date_type == "event":
        date_type = "event"

    # Process case type = 'transfer'
    if raw_date_type == "holiday":
        if entry.transferred is False:
            date_type = "holiday"
        else:
            date_type = "work day"

    if raw_date_type == "transfer":
        date_type = "holiday"

    # Process case type = 'additional'
    if raw_date_type == "additional":
        date_type = "holiday"

    # Set description
    if date_type == "work day":
        des = ""
    else:
        des = _proc_des(entry.description)

    # Set locale_name
    if date_type in ["work day", "weekend"]:
        locale_name = "ecuador"

    return date_type, locale_name, des


class Day:
    def __init__(self, path_holidayinfo: str, path_holiday: str = None) -> None:
        if Path(path_holidayinfo).exists():
            with open(path_holidayinfo) as fp:
                self.holidayinfo = json.load(fp)
        else:
            assert path_holiday is not None and \
                Path(path_holiday).exists(), \
                "As 'path_holidayinfo' not existed, 'path_holiday' must be specified"

            self.holidayinfo = self.establish_db(path_holiday)

            # Save to JSON
            with open(path_holidayinfo, "w+") as fp:
                json.dump(self.holidayinfo, fp, indent=2)

    def get_date_info(self, date: str, locale_name) -> dict:
        if date in self.holidayinfo and locale_name in self.holidayinfo[date]:
            dateinfo = self.holidayinfo[date][locale_name]

        else:
            dateinfo = {
                'date_type': "work day",
                'locale_name': "ecuardo",
                'description': ""
            }

            date_obj = datetime.strptime(date, "%Y-%m-%d")
            if date_obj.weekday() < 5:
                dateinfo['date_type'] = "weekend"

        return dateinfo

    def establish_db(self, path_holiday: str):
        """Get info of given date

        Args:
            date (str): date (str): format "yyyy-mm-dd"

        Returns:
            Union[str, str, str]: Date info
        """

        holidayinfo = defaultdict(dict)

        df = pd.read_csv(path_holiday)
        for entry in df.itertuples():
            date_type, locale_name, des = _process_entry(entry)

            holidayinfo[entry.date][locale_name] = {
                'date_type': date_type,
                'description': des
            }

        return holidayinfo
