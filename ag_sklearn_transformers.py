import numpy as np
import pandas as pd
import datetime
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from bisect import bisect_right

class AlphaNumericSplitTransformer(TransformerMixin, BaseEstimator):
    '''
    Splits given text columns into 2 columns (_alpha and _numeric).
    Drops the original text columns.
    '''

    def __init__(self, columns=None):
        self.columns = columns

    def __get_numeric(self, s):
        r = ''.join(c for c in s if c.isnumeric())
        return '0' if not r else r

    def __parse_number(self, s):
        split = s.split(".", 1)
        if len(split) > 1:
            return float("{0}.{1}".format(self.__get_numeric(split[0]), self.__get_numeric(split[1])))
        else:
            return float(self.__get_numeric(split[0]))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            X['{0}_alpha'.format(c)] = X[c].apply(lambda x: ''.join(c for c in x if c.isalpha()).lower())
            X['{0}_numeric'.format(c)] = X[c].apply(lambda x: self.__parse_number(x)).fillna(0)
        X = X.drop(columns=self.columns)
        return X

class GroupScalerTransformer(TransformerMixin, BaseEstimator):
    '''
    Scales values (column_to_scale) using a separate scaler (scaler_factory) per category value (group_column).
    '''

    def __init__(self, group_column, column_to_scale, scaler_factory):
        self.group_column = group_column
        self.column_to_scale = column_to_scale
        self.scaler_factory = scaler_factory
        self.scalers_per_name = {}

    def fit(self, X, y=None):
        Y = X[[self.group_column, self.column_to_scale]]
        for n, g in Y.groupby([self.group_column]):
            scaler = self.scaler_factory()
            scaler.fit(g[[self.column_to_scale]])
            self.scalers_per_name[n] = scaler
        return self

    def transform(self, X):
        X = X.copy()
        Y = X[[self.group_column, self.column_to_scale]]
        for n, g in Y.groupby([self.group_column]):
            if(n in self.scalers_per_name):
                scaler = self.scalers_per_name[n]
            else:
                scaler = self.scaler_factory()
                scaler.fit(g[[self.column_to_scale]])
            u = scaler.transform(g[[self.column_to_scale]])
            X.loc[X[self.group_column] == n, self.column_to_scale] = u
        return X

class IsLatestTransformer(TransformerMixin, BaseEstimator):
    '''
    For the given time column (represent a timestamp),
    feature name (name_column) and its version (version_column),
    marks the occurences of the latest version (is_latest_column).
    '''

    def __init__(self, time_column, name_column, version_column, is_latest_column):
        self.time_column = time_column
        self.name_column = name_column
        self.version_column = version_column
        self.is_latest_column = is_latest_column
        self.min_times_per_name = {}

    def __fit(self, X, D):
        for i, r in X.iterrows():
            time = r[self.time_column]
            name = r[self.name_column]
            version = r[self.version_column]
            if name in D:
                min_times = D[name]
                if version in min_times:
                    if time < min_times[version]:
                        min_times[version] = time
                else:
                    min_times[version] = time
            else:
                D[name] = {}
                D[name][version] = time

    def fit(self, X, y=None):
        self.__fit(X, self.min_times_per_name)
        return self

    def __find_le(self, a, x):
        i = bisect_right(a, x)
        if i:
            return i-1
        return 0

    def transform(self, X):
        X = X.copy()
        temp = self.min_times_per_name.copy()
        self.__fit(X, temp)

        fast_dict = {}
        for name, min_times in temp.items():
            version_and_time = list(min_times.items())
            version_and_time.sort(key = lambda p: p[1])
            version_and_time_adjusted = []
            version_and_time_adjusted.append(version_and_time[0])
            previous_version = version_and_time[0][0]
            for i in range(1, len(version_and_time)):
                v = version_and_time[i]
                if v[0] > previous_version:
                    version_and_time_adjusted.append(v)
                    previous_version = v[0]

            times = list(map(lambda p: p[1], version_and_time_adjusted))
            fast_dict[name] = (version_and_time_adjusted, times)

        results = []
        for i, r in X.iterrows():
            time = r[self.time_column]
            name = r[self.name_column]
            version = r[self.version_column]
            
            (version_and_time, times) = fast_dict[name]
            last_i = self.__find_le(times, time)
            latest_version_at_time = version_and_time[last_i][0]
            results.append(version == latest_version_at_time)

        X[self.is_latest_column] = results

        return X

class OneHotDummyTransformer(TransformerMixin, BaseEstimator):
    '''
    A transformer for one hot encoding.
    Supports strings, handles unknown values (keeps the fit columns).
    '''

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        X_oneHot = pd.get_dummies(X, columns=self.columns)
        self.fit_columns = X_oneHot.columns
        return self

    def transform(self, X):
        X_oneHot = pd.get_dummies(X, columns=self.columns)
        X_oneHot_columns = set(X_oneHot.columns)
        for c in self.fit_columns:
            if c not in X_oneHot_columns:
                X_oneHot[c] = 0
        return X_oneHot[self.fit_columns]

class ScalerTransformer(TransformerMixin, BaseEstimator):
    '''
    A transformer that wrapps Scaler.
    '''

    def __init__(self, columns, scaler):
        self.columns = columns
        self.scaler = scaler

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()
        scaled_values = self.scaler.transform(X[self.columns])
        scaled_df = pd.DataFrame(scaled_values, index=X.index, columns=self.columns)
        X[self.columns] = scaled_df[self.columns]
        return X

class TrueFalseTransformer(TransformerMixin, BaseEstimator):
    '''
    A transformer for true/false categorical columns.
    '''

    def __init__(self, columns=None, true_value='T'):
        self.columns = columns
        self.true_value = true_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            X[c] = X[c].apply(lambda x: x == self.true_value)
        return X