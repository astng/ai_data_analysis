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

    muestras = {}
    for i, id_component in enumerate(list(set(all_data['component']))):
        component_group = grouped_by_component.groups[id_component]
        component_results = all_data.iloc[component_group]["iron"].dropna().reset_index(drop=True)
        for t in range(len(component_results)):
            if t not in muestras:
                muestras[t] = []
            muestras[t].append(component_results[t])
    for t in range(len(muestras.keys())):
        plt.boxplot(muestras[t], positions = [t])
        plt.xlim([0, 1814])
    plt.xlabel("numero de analisis")
    plt.title('boxplot (fierro) por muestra')
    plt.grid(True)
    plt.savefig(outfile)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfile', type=str, default="../Informe2/figs/boxplot-by-sample.pdf")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfile)
