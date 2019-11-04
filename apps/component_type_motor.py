import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

time_horizon = 150
n_components = 6


def main(database: str, table: str, outfolder: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()
    types = [39, 630, 681, 682] # 798, 799, 1236 and 1237 shows KeyError
    components = {}
    all_data = pd.DataFrame(query_fetch)
    print("from ungrouped:")
    print(all_data.columns)
    grouped_by_component_type = all_data.groupby(by='component_type')
    for type in types:
        component_group = grouped_by_component_type.groups[type]
        results = all_data.iloc[component_group]["iron"].dropna().reset_index(drop=True)
        components[type] = []
        for id_component in set(all_data['component'][all_data['component_type'] == type]):
            component_results = all_data["iron"][all_data['component'] == id_component].dropna().reset_index(drop=True)
            components[type].append((len(component_results), id_component))
    types.sort()
    types.reverse()

    for i in range(len(types)):
        type = types[i]
        components_to_plot = list(components[type])
        components_to_plot.sort()
        components_to_plot.reverse()
        plt.figure()
        legend = []
        for _, id_component in components_to_plot[:n_components]:
            component_results = all_data["iron"][all_data['component'] == id_component].dropna().reset_index(drop=True)
            legend.append("id_component: " + str(id_component))
            plt.plot(component_results[:time_horizon])
            limits = all_data["ironLSC"][all_data['component'] == id_component].dropna().reset_index(drop=True)
            protocols = all_data["id_protocol"][all_data['component'] == id_component].dropna().reset_index(drop=True)
            plt.plot(limits[:time_horizon], linestyle='dashed', label='_nolegend_')
            plt.plot(protocols[:time_horizon], linestyle='dotted')
            legend.append("prot " + str(list(set(protocols))[0]) + "for id_comp " + str(id_component))
        plt.legend(legend)
        plt.xlabel("muestras")
        plt.title('Analisis del nivel de fierro para motor diesel (id_tipo_componente: ' + str(type) + ')')
        plt.grid(True)
        plt.savefig(outfolder + 'with_protocol-dieselmotor' + str(type) + '.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfolder', type=str, default="../figures/iron-plots/")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfolder)