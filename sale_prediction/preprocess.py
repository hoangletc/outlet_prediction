import re
from datetime import datetime
from typing import Tuple, Union

from pandas.core.series import Series

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
    elif des.startswith("traslado"):
        des = "traslado"

    return des

class Day:
    def __init__(self) -> None:
        self.

    
        

    def get_date_info(self, entry: Series) -> Tuple[str, str, Union[str, None], Union[str, None], str]:
        pass

    def establish_db(self):
        """Get info of given date

        Args:
            date (str): date (str): format "yyyy-mm-dd"

        Returns:
            Union[str, str, str, str]: Date info
        """



        date_type, locale, locale_name = "Work day", "National", "Ecuador"
        event_type = None

        # Clean description
        

        # Process case type = 'Weekend'
        date_obj = datetime.strptime(entry['date'], "%Y-%m-%d")
        if date_obj.weekday() < 5:
            date_type = "Weekend"

        # Process case type = 'Event'
        if entry['type'] == "Event":
            date_type = "Event"

        return date_type, locale, locale_name, event_type



