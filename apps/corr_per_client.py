import argparse
import pandas as pd
from pymongo import MongoClient


def main(database: str, table: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()

    all_data = pd.DataFrame(query_fetch)
    nclients = 5
    clients = {}
    correlations = []

    for client in set(all_data['client']):
        clients[client] = all_data[all_data['client'] == client][all_data.columns.difference(['client', 'component'])]

    for client in list(clients.keys())[:nclients]:
        for client2 in list(clients.keys())[:nclients]:
            if client != client2:
                df1 = all_data[all_data['client'] == client][all_data.columns.difference(['client', 'component'])].dropna()
                df2 = all_data[all_data['client'] == client2][all_data.columns.difference(['client', 'component'])].dropna()
                corr = df1.corrwith(df2)
                correlations.append((corr,client,client2))
                #print(correlations[-1])
            else:
                correlations.append((1,client,client))
    print(correlations)
    #correlations.sort()
    #correlations.reverse()
    #for trio in correlations:
    #    if trio[1] != trio[2]:
    #        print(trio)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table)