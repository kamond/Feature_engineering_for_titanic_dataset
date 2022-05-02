import pandas as pd
from helpers.eda import *
from helpers.data_prep import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()

def titanic_data_prep(df):

    df.columns = [col.upper() for col in df.columns]
    # Cabin bool
    df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
    # Name count
    df["NEW_NAME_COUNT"] = df["NAME"].str.len()
    # name word count
    df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
    # name dr
    df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
    # name title
    df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    # family size
    df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
    # age_pclass
    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
    # is alone
    df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
    # age level
    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    # sex x age
    df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    #############################################
    # 2. Outliers (Aykırı Değerler)
    #############################################

    for col in num_cols:
        replace_with_thresholds(df, col)
    #############################################
    # 3. Missing Values (Eksik Değerler)
    #############################################

    df.drop("CABIN", inplace=True, axis=1)
    remove_cols = ["TICKET", "NAME"]
    df.drop(remove_cols, inplace=True, axis=1)

    df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

    df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

    df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    #############################################
    # 4. Label Encoding
    #############################################

    binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
                   and df[col].nunique() == 2]

    for col in binary_cols:
        df = label_encoder(df, col)


    #############################################
    # 5. Rare Encoding
    #############################################

    df = rare_encoder(df, 0.01, cat_cols)

    #############################################
    # 6. One-Hot Encoding
    #############################################

    ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
    df = one_hot_encoder(df, ohe_cols)
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    #############################################
    # 7. Standart Scaler
    #############################################

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


df_prepared = titanic_data_prep(df)

check_df(df_prepared)


# df_prepared.to_pickle("hafta06/df_prepared.pkl")
# titanic_df_prepared = pd.read_pickle("hafta06/df_prepared.pkl")


''''

#############################################
# 8. Model
#############################################

y = df_prepared["SURVIVED"]
X = df_prepared.drop(["PASSENGERID", "SURVIVED"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

'''