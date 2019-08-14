import unittest
import pandas as pd

from data_utils import get_start_end


class TestPynoplyStats(unittest.TestCase):
    def test1(self):
        get_date = lambda x: pd.date_range('2018-01-01 00:00:00', '2018-01-01 0{}:00:00'.format(x), freq='1h')
        df1_list = [pd.DataFrame({'Date': get_date(i)}) for i in range(5)]
        df2_list = [pd.DataFrame({'Date': get_date(i)}) for i in range(5, 10)]
        start, end = get_start_end(df1_list + df2_list)
        start_true, end_true = pd.to_datetime('2018-01-01 00:00:00'), pd.to_datetime('2018-01-01 09:00:00')
        self.assertEqual((start, end), (start_true, end_true))
