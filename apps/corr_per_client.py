import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymongo import MongoClient


def main(database: str, table: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()

    all_data = pd.DataFrame(query_fetch)
    clients = {}
    some_clients = ['Minera_Escondida_Ltda', 'Minera_Esperanza', 'Komatsu_Chile', 'Minera_Antucoya']

    grouped_by_client = all_data.groupby(by='client')
    fig = plt.figure()
    cnt = 1
    for group in grouped_by_client:
        if group[0] in some_clients:
            cols = all_data.columns.difference(['client', 'component'])
            clients[group[0]] = group[1][cols].corr().abs()
            sns.set(font_scale=0.6)
            ax = fig.add_subplot(int("22"+str(cnt)))
            ax.title.set_text(group[0])
            sns.heatmap(clients[group[0]], vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
            cnt += 1
    #plt.savefig('../Informe2/figs/perclient2.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table)