import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

time_horizon = 250
n_components = 600

def main(database: str, table: str, outfolder: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()

    all_data = pd.DataFrame(query_fetch)
    grouped_by_component = all_data.groupby(by='component')

    for cnt, id_component in enumerate(set(all_data['component'])):
        if cnt > n_components:
            break
        component_group = grouped_by_component.groups[id_component]
        component_results = all_data.iloc[component_group]["iron"].dropna().reset_index(drop=True)
        plt.plot(component_results[:time_horizon])
    plt.xlabel("muestras")
    plt.title('Analisis del nivel de fierro')
    plt.grid(True)
    plt.savefig(outfolder + 'iron-' + str(n_components) + '-components.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfolder', type=str, default="../Informe2/figs/")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfolder)