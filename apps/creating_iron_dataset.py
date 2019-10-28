import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

pd.set_option('display.max_columns', 50)


def main(database: str, table: str, outfile: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()

    all_data = pd.DataFrame(query_fetch)
    grouped_by_component = all_data.groupby(by='component')

    all_iron_component_series = list()
    for id_component, group in grouped_by_component:
        #all_iron_component_series.append(group[["iron", "ironLSC", "ironLSM", "component_type", "component"]].values)
        normalized = (group["iron"] - group["iron"].mean())/group["iron"].std()
        all_iron_component_series.append(normalized.values)
    all_iron_component_series = pd.Series(all_iron_component_series)
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    all_iron_component_series.to_hdf(outfile, key='df')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfile', type=str, default="../datasets/normalized-iron_dataset.h5")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfile)
