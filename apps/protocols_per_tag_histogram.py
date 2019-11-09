import argparse
import matplotlib.pyplot as plt

import pandas as pd
from pymongo import MongoClient


def main(database: str, table: str, outfolder: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()

    all_data = pd.DataFrame(query_fetch)

    grouped_by_tag = all_data.groupby(by='tag')
    amount_protocols_per_tag = list()

    for group in grouped_by_tag:
        protocols = all_data[all_data['tag'] == group[0]]['id_protocol'].dropna().reset_index(drop=True)
        amount_protocols_per_tag.append(len(set(protocols)))

    pd.Series(amount_protocols_per_tag).hist(bins=30)
    plt.title('Histograma de protocolos por tag')
    plt.xlabel("Protocolos por tag")
    plt.ylabel("Tags")
    plt.savefig(outfolder + "protocols_per_tag_histogram.pdf")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfolder', type=str, default="../figures/iron-plots/")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfolder)