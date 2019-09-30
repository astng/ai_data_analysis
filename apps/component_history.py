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

    components = {50: 'U.Hidraulica', 54: 'Reductor'}
    for idc in components.keys():
        cols = all_data.columns.difference(['component'])
        comp_results = all_data[all_data['component'] == idc][cols]
        comp_results = comp_results['iron'].dropna()
        plt.plot(comp_results)
        plt.legend(list(components.values()))
        plt.xlabel("muestras")
    plt.title('Analisis del nivel de fierro')
    plt.grid(True)
    plt.savefig("../Informe2/figs/iron.pdf")
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table)