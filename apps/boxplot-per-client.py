import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

def main(database: str, table: str, outfile: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()

    clients = []
    all_data = pd.DataFrame(query_fetch)

    grouped_by_client = all_data.groupby(by='client')
    fig = plt.figure()
    for i, id_client in enumerate(list(set(all_data['client']))):
        clients.append(id_client[-8:])
        client_group = grouped_by_client.groups[id_client]
        client_results = all_data.iloc[client_group]["iron"].dropna().reset_index(drop=True)
        plt.boxplot(client_results, positions = [i])
    plt.xlim([0, 32])
    plt.xticks(range(33),clients, rotation = 'vertical')
    plt.xlabel("cliente")
    plt.title('boxplot por cliente')
    plt.grid(True)
    fig.savefig(outfile, dpi = fig.dpi, bbox_inches = 'tight')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfile', type=str, default="../Informe2/figs/boxplot-by-client.pdf")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfile)
