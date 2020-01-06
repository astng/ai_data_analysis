import numpy as np
import argparse
import os
import pandas as pd
from pymongo import MongoClient

pd.set_option('display.max_columns', 50)
pd.options.mode.chained_assignment = None


def main(database: str, table: str, outfile: str, essay: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()
    print(essay)
    all_data = pd.DataFrame(query_fetch)
    grouped_by_component = all_data.groupby(by='component')
    for id_component, group in grouped_by_component:
        predicted_changes = all_data[all_data['component'] == id_component][essay] < \
                            0.6 * all_data[all_data['component'] == id_component][essay].shift()
        all_data["change"][all_data["component"] == id_component] = predicted_changes.values
    grouped_by_correlative = all_data.groupby(by='correlativo_muestra')
    all_iron_component_series = list()
    for correlativo, group in grouped_by_correlative:
        aux = group[["component", "component_type", "machine_type", "change", essay + "LSC", essay + "LSM", "h_k_lubricante",
                     essay]].values
        if pd.isna(aux).any():
            continue
        else:
            #all_iron_component_series.append(aux[~np.isnan(aux[:, 1:])])
            all_iron_component_series.append(aux[0])
    all_iron_component_series = pd.Series(all_iron_component_series)
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    if essay != "iron":
        outfile = outfile.replace("iron", essay)
    all_iron_component_series.to_hdf(outfile, key='df')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--essay', type=str, default="iron")
    parser.add_argument('--outfile', type=str, default="../datasets/iron_dataset.h5")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfile, cmd_args.essay)
