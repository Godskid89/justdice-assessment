import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):

    # Fill missing values in adspend with 0
    df['total_adspend'].fillna(0, inplace=True)

    # convert the month_year column to only months
    df['month'] = pd.to_datetime(df['month_year'].astype(str) + '-01').dt.month
    df['month'] = df['month'].astype('object')
    df['network_id'] = df['network_id'].astype('object')
    df['country_id'] = df['country_id'].astype('object')
    df.drop('month_year', axis=1, inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    df['network_id'] = le.fit_transform(df['network_id'])
    df['country_id'] = le.fit_transform(df['country_id'])
    df['month'] = le.fit_transform(df['month'])

    return df


