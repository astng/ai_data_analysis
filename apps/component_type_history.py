import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

time_horizon = 150
n_types = 6
n_components = 6

def main(database: str, table: str, outfolder: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()
    types = []
    components = {}
    all_data = pd.DataFrame(query_fetch)
    grouped_by_component_type = all_data.groupby(by='component_type')
    for type in set(all_data['component_type']):
        component_group = grouped_by_component_type.groups[type]
        results = all_data.iloc[component_group]["iron"].dropna().reset_index(drop=True)
        types.append((len(results), type))
        components[type] = []
        for id_component in set(all_data['component'][all_data['component_type'] == type]):
            component_results = all_data["iron"][all_data['component'] == id_component].dropna().reset_index(drop=True)
            components[type].append((len(component_results), id_component))
    types.sort()
    types.reverse()

    for i in range(n_types):
        type = types[i][1]
        components_to_plot = list(components[type])
        components_to_plot.sort()
        components_to_plot.reverse()
        plt.figure()
        legend = []
        for _, id_component in components_to_plot[:n_components]:
            component_results = all_data["iron"][all_data['component'] == id_component].dropna().reset_index(drop=True)
            legend.append(id_component)
            plt.plot(component_results[:time_horizon])
            limits = all_data.iloc[group]["ironLSC"].dropna().reset_index(drop=True)
            protocols = all_data.iloc[group]["id_protocol"].dropna().reset_index(drop=True)
            axs[row, col].plot(limits[:time_horizon], linestyle='dashed', label='_nolegend_')
            axs[row, col].plot(protocols[:time_horizon], linestyle='dotted', label="prot " + list(set(protocols))[0]
                                                                                   + "for id_comp " + str(id_component))
        plt.legend(legend)
        plt.xlabel("muestras")
        plt.title('Analisis del nivel de fierro para id_tipo_componente ' + type)
        plt.grid(True)
        plt.savefig(outfolder + 'with_protocol_component_type' + type + '.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfolder', type=str, default="../figures/iron-plots/")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfolder)