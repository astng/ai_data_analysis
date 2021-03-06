import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymongo import MongoClient
import numpy as np
pd.set_option('display.max_columns', 50)

def main(database: str, table: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()

    # to pandas
    all_data = pd.DataFrame(query_fetch)
    corr_matrix = all_data.corr().abs()
    ordered_pairs = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool)).stack().sort_values(ascending=False))
    print(ordered_pairs[:5])
    ax = sns.heatmap(corr_matrix, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.savefig('heatmap.pdf')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table)
