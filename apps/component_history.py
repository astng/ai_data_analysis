import argparse
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

    components = {50: 'U.Hidraulica', 54: 'Reductor'}

    for id_component in components.keys():
        component_group = grouped_by_component.groups[id_component]
        component_results = all_data.iloc[component_group]["iron"].dropna().reset_index(drop=True)
        plt.plot(component_results)
    plt.legend(list(components.values()))
    plt.xlabel("muestras")
    plt.title('Analisis del nivel de fierro')
    plt.grid(True)
    plt.savefig(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfile', type=str, default="../Informe2/figs/iron.pdf")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfile)
