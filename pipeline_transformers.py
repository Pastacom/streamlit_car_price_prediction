from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re


def clean_torque(torque) -> pd.Series:
    if not isinstance(torque, str):
      return pd.Series([None, None])

    torque = torque.lower()
    torque = torque.replace(',', '')
    torque = torque.replace('at', '@')

    numeric_regex = r'[\d\.]+'
    nums_result = re.findall(numeric_regex, torque)

    if len(nums_result) < 1:
      return pd.Series([None, None])

    torque_num = None
    rpm_num = None

    if len(nums_result) == 1:
      if re.search(r'rpm', torque):
        rpm_num = float(nums_result[0])
      else:
        torque_num = float(nums_result[0])
    elif len(nums_result) == 2:
      torque_num = float(nums_result[0])
      rpm_num = float(nums_result[1])
    else:
      torque_num = float(nums_result[0])
      deviation_regex = r'\d\+\/-\d'

      if re.search(deviation_regex, torque):
        rpm_num = float(nums_result[1]) + float(nums_result[2])
      else:
        rpm_num = float(nums_result[2])

    if re.search(r'kgm', torque) and torque_num is not None:
      torque_num *= 9.8

    return pd.Series([torque_num, rpm_num])


def clean_features_content(df: pd.DataFrame) -> pd.DataFrame:
    FIELDS_TO_CLEAN = {'mileage': [' kmpl', ' km/kg'],
                       'engine': [' cc'],
                       'max_power': [' bhp']}

    df = df.copy()

    for col in FIELDS_TO_CLEAN.keys():
        df[col] = df[col].astype(str).str.lower()

        for field in FIELDS_TO_CLEAN[col]:
            df[col] = df[col].str.replace(field, '', regex=False)

        df[col] = df[col].replace(['nan', ''], None).astype(float)

    df[['torque', 'max_torque_rpm']] = df['torque'].apply(clean_torque).replace(['nan', ''], None)

    return df


def extract_brand(name: str) -> pd.Series:
    MULTIPLE_WORDS_BRANDS = ['Land Rover', 'MG Hector', 'Mercedes-Benz']

    if pd.isna(name):
        return pd.Series({'brand': None})

    name = name.strip()

    brand = None
    for b in MULTIPLE_WORDS_BRANDS:
        if name.startswith(b):
            brand = b
            break

    if brand is None:
        brand = name.split(' ')[0]

    return pd.Series({'brand': brand})


class CleanFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, clean_func, median_cols):
        self.clean_func = clean_func
        self.median_cols = median_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = self.clean_func(X)
        medians = X[self.median_cols].median()

        for i in medians.index:
            X[i] = X[i].fillna(medians[i])

        X["engine"] = X["engine"].astype(int)
        X["seats"] = X["seats"].astype(int)

        return X


class ExtractBrand(BaseEstimator, TransformerMixin):
    def __init__(self, brand_func):
        self.brand_func = brand_func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["brand"] = X["name"].apply(self.brand_func)
        X.drop(columns=["name"], inplace=True)
        return X


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["hp_per_liter"] = X["max_power"] / X["engine"]
        X["year_sq"] = X["year"] ** 2
        return X