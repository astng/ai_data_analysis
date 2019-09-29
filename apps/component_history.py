import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

pd.set_option('display.max_columns', 50)

def main(database: str, table: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()

    all_data = pd.DataFrame(query_fetch)
    component = 1784
    client = set(all_data[all_data['component'] == component]['client'])
    cols = all_data.columns.difference(['client', 'component'])
    df = all_data[all_data['component'] == component][cols]
    ensayo = 0 # luego separare por ensayos AFQ y ensayos de metal
    df = df[cols[ensayo]].dropna()
    plt.plot(df)
    plt.title(client)
    plt.legend([cols[ensayo]])
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table)