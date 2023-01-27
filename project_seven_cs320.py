# project: p7
# submitter: mroytman
# partner: none
# hours: 11

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import sqlite3

pd.options.mode.chained_assignment = None

def combine(users, logs):
    conn=sqlite3.connect('test_info')
    cursor = conn.cursor()
    users.to_sql('users', conn, if_exists='replace', index = False)
    logs.to_sql('logs', conn, if_exists='replace', index = False)
    
    sql_df = pd.read_sql("""
    SELECT *
    FROM logs
    Inner JOIN users
    ON users.user_id = logs.user_id
""", conn)

    column_names = sql_df.columns.values
    column_names
    column_names[4] = 'drop_user'
    sql_df.columns = column_names
    sql_df.drop(sql_df.columns[[4]], axis=1, inplace=True)
    return sql_df
    
    
def onehot(df):
    x=df["badge"].value_counts()
    df.badge[df.badge==x.index[0]]=1
    df.badge[df.badge==x.index[1]]=2
    df.badge[df.badge==x.index[2]]=3
    return df

class UserPredictor:
    def __init__(self):
        self.model = Pipeline([
            ("pf", PolynomialFeatures()),
            ("lr", LogisticRegression()),
            ])
        self.xcols = ["past_purchase_amt", "age", "badge", "seconds"]
        
    def fit(self, users, logs, y):
        combined_df = combine(users, logs)
        grouped_df=combined_df.groupby(["user_id"])
        users["seconds"] = grouped_df["seconds"].sum()
        users=users.fillna(0)
        users=onehot(users)
        self.model.fit(users[self.xcols], y["y"])

    def predict(self, users, logs):
        combined_df = combine(users, logs)
        grouped_df=combined_df.groupby(["user_id"])
        seconds = grouped_df["seconds"].sum()
        seconds=seconds.reset_index()
        predict_data=pd.merge(users, seconds, on="user_id", how="left")
        predict_data=predict_data.fillna(0)
        predict_data=onehot(predict_data)
        return self.model.predict(predict_data[self.xcols])