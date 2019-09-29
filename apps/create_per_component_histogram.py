import argparse
import matplotlib.pyplot as plt

import pandas as pd
from pymongo import MongoClient


def main(file_path: str, db: str, table: str):
    client = MongoClient()
    db_mongo = client[db]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()

    all_data = pd.DataFrame(query_fetch)
    
    grouped_by_component = all_data.groupby(by='component')
    amount_data_per_component = list()

    for group in grouped_by_component:
        amount_data_per_component.append(len(group[1]))

    pd.Series(amount_data_per_component).hist(bins=25)
    plt.title('Histograma de muestras por componente')
    plt.xlabel("Cantidad de muestras")
    plt.ylabel("Componentes")
    plt.savefig(file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='file path to create histogram in pdf', type=str, required=True)
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.file_path, cmd_args.database, cmd_args.table)
