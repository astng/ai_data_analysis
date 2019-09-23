import argparse

import pandas as pd
from pymongo import MongoClient


def main(database: str, table: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()

    # to pandas
    all_data = pd.DataFrame(query_fetch)

    # printing first rows
    print(all_data.head())
    # printing last rows
    print(all_data.tail())
    # printing some data stats
    print(all_data.describe())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table)
