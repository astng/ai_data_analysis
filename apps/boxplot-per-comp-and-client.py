import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient
import seaborn as sns
from webcolors import rgb_to_name

def main(database: str, table: str, outfolder: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()

    clients = ['Dercomaq', 'Power_Train_Technologies', 'ESO', 'Luval', 'Komatsu_Chile']
    all_data = pd.DataFrame(query_fetch)
    grouped_by_client = all_data.groupby(by='client')

    for id_client in clients:
        client_group = grouped_by_client.groups[id_client]
        client_results = all_data.iloc[client_group][["client", "iron", "component"]].dropna().reset_index(drop=True)
        counts = []
        for component in set(client_results["component"]):
            counts.append(str(component) + "- n = " + str(len(client_results["iron"][client_results["component"] == component])))
        boxes = sns.boxplot(x="client", y="iron", hue="component", data=client_results)
        leg = boxes.axes.get_legend()
        for t, l in zip(leg.texts, counts): t.set_text(l)
        plt.xlabel("cliente")
        plt.title('boxplot por componente para ' + id_client)
        plt.grid(True)
        plt.savefig(outfolder + 'bloxplot-' + id_client + '.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfolder', type=str, default="../Informe2/figs/")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfolder)
