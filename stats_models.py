from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_models import PointsDataSource
from data_utils import get_start_end, get_corr
import numpy as np
import pandas as pd


class Stats:
    def __init__(self, name1_df1_list: tuple, name2_df2_list: tuple):
        """
        suggest Date as index column
        :param name1_df1_list: name1 and df1_list
        :param name2_df2_list: name2 and df2_list
        """
        self.name1, self.df1_list = name1_df1_list
        self.name2, self.df2_list = name2_df2_list

    def calc_stats(self, start=None, end=None, freq='1h', pq_list=('T', 'Po', 'Ff')):
        """
        MAE, RMSE, Pearson correlation for each station
        :param start: Date start
        :param end: Date start
        :param freq:
        :param pq_list:
        :return:
        """
        if not start and not end:
            start, end = get_start_end(self.df1_list + self.df2_list)

        if end and start:
            # proper_dates = pd.date_range(start, end, freq=freq)
            # df_proper = pd.DataFrame({'Date': proper_dates})
            # print('proper dates', start, end)
            #
            # self.df1_list = [pd.merge(df_proper, df, on='Date', how='left') for df in self.df1_list]
            # self.df1_list = [pd.merge(df_proper, df, on='Date', how='left') for df in self.df2_list]

            # df_common_list = [pd.merge(df1, df2, on='Date', how='inner',
            #                            suffixes=['_' + self.name1, '_' + self.name2])
            #                   for df1, df2 in zip(dfs1, dfs2)]
            data_source1 = PointsDataSource(name=self.name1, df_list=self.df1_list)
            data_source2 = PointsDataSource(name=self.name2, df_list=self.df2_list)
            merged_data_source = data_source1 + data_source2  # merge dfs from both sources
            df_common_list = merged_data_source.df_list
            df_stats = get_df_stats(df_common_list, pq_list, self.name1, self.name2)
            return df_stats
        else:
            raise Exception('nan value found in dates')

    def __repr__(self):
        return 'Stats: {} {}'.format(self.name1, self.name2)


def get_df_stats(df_common_list, pq_list, name1, name2):
    scores = []
    for pq in pq_list:
        for i, df in enumerate(df_common_list):  # for each point
            init_len = len(df)
            if init_len == 0:
                continue

            _id1 = '_id_{}'.format(name1)
            _id2 = '_id_{}'.format(name1)

            if _id1 in df.columns:
                id_uniq = df[_id1].unique()
            elif _id2 in df.columns:
                id_uniq = df[_id2].unique()
            else:
                raise Exception('_id not found {} {} {}'.format(_id1, _id2, df.columns.tolist()))

            print('id_uniq', id_uniq)
            _id = id_uniq[0] if str(id_uniq[0]) != 'nan' else id_uniq[1]
            col1 = '{}_{}'.format(pq, name1)
            col2 = '{}_{}'.format(pq, name2)
            df_pq = df[[col1, col2]].dropna()
            score = get_df_pq_stats(_id, pq, df_pq, col1, col2, init_len)
            scores.append(score)
    return pd.DataFrame(scores)


def get_df_pq_stats(_id, pq, df_pq, col1, col2, init_len):
    if len(df_pq) > 0:
        a = df_pq[col1].values
        b = df_pq[col2].values
        score = dict(
            _id=_id,
            pq=pq,
            total_len=len(a),
            corr=get_corr(a, b),
            mae=mean_absolute_error(a, b),
            rmse=mean_squared_error(a, b) ** 0.5,
            r2=r2_score(a, b),
            missed_rows=init_len - len(df_pq),
            missed_perc=round((init_len - len(df_pq)) / len(df_pq), 2) * 100
        )
        score['var_' + col1] = np.var(a)
        score['var_' + col2] = np.var(b)
        score['mean_' + col1] = np.mean(a)
        score['mean_' + col2] = np.mean(b)
    else:
        score = dict(
            _id=_id,
            pq=pq,
            total_len=0,
            corr=np.NaN,
            mae=np.NaN,
            rmse=np.NaN,
            r2=np.NaN,
            missed_rows=init_len,
            missed_perc=100,
        )
        score['var_' + col1] = np.NaN
        score['var_' + col2] = np.NaN
        score['mean_' + col1] = np.NaN
        score['mean_' + col2] = np.NaN

    return score
