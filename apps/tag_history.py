import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

time_horizon = 250
n_tags = 4  # first n_tags with more data are plotted

def main(database: str, table: str, outfolder: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()
    data_numbers = []
    all_data = pd.DataFrame(query_fetch)
    grouped_by_tag = all_data.groupby(by='tag')
    for group_id in grouped_by_tag:
        tag = group_id[0]
        group = grouped_by_tag.groups[tag]
        results = all_data.iloc[group]["iron"].dropna().reset_index(drop=True)
        data_numbers.append((len(results), tag))
    data_numbers.sort()
    data_numbers.reverse()

    for cnt, data in enumerate(data_numbers):
        tag = data[1]
        if cnt + 1 > n_tags:
            break
        legend = []
        tag_results = all_data[all_data['tag'] == tag]['iron'].dropna().reset_index(drop=True)
        id_component = set(all_data[all_data['tag'] == tag]['component'].dropna().reset_index(drop=True))
        legend.append("id_component:" + str(list(id_component)[0]))
        limits = all_data[all_data['tag'] == tag]['ironLSC'].dropna().reset_index(drop=True)
        protocols = all_data[all_data['tag'] == tag]['id_protocol'].dropna().reset_index(drop=True)
        legend.append("LSC")
        legend.append("id_protocol")
        plt.plot(tag_results[:time_horizon])
        plt.plot(limits[:time_horizon], linestyle='dashed')
        plt.plot(protocols[:time_horizon], linestyle='dotted')
        plt.legend(legend)
        plt.xlabel("muestras")
        plt.title('Analisis del nivel de fierro para tag ' + str(tag))
        plt.grid(True)
        plt.savefig(outfolder + 'iron-tag' + str(tag) + '.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfolder', type=str, default="../figures/iron-plots/")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfolder)