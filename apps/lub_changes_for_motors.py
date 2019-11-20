import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pymongo import MongoClient

time_horizon = 30
n_components = 1  # maximum of 4 components for plotting
colors_to_plot = ['k', 'g', 'b', 'y', 'r']


def main(database: str, table: str, outfolder: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()
    types = [39, 630, 681, 682]  # 798, 799, 1236 and 1237 shows KeyError
    components = {}
    all_data = pd.DataFrame(query_fetch)
    for id_type in types:
        components[id_type] = []
        for id_component in set(all_data['component'][all_data['component_type'] == id_type]):
            component_results = all_data["iron"][all_data['component'] == id_component].dropna().reset_index(drop=True)
            components[id_type].append((len(component_results), id_component))
    types.sort()
    types.reverse()

    for i in range(len(types)):
        id_type = types[i]
        components_to_plot = list(components[id_type])
        components_to_plot.sort()
        components_to_plot.reverse()
        fig, (ax1, ax2) = plt.subplots(2, sharex='all')
        legend = []
        legend2 = []
        color = 0
        for _, id_component in components_to_plot[:n_components]:
            lub_changes = all_data[all_data['component'] == id_component]['h_k_lubricante'] < \
                          all_data[all_data['component'] == id_component]['h_k_lubricante'].shift()
            lub_changes = lub_changes.reset_index()
            lub_changes = lub_changes.loc[lub_changes.h_k_lubricante, :]
            predicted_changes = all_data[all_data['component'] == id_component]['iron'] < \
                                0.6*all_data[all_data['component'] == id_component]['iron'].shift()
            predicted_changes = predicted_changes.reset_index()
            predicted_changes = predicted_changes.loc[predicted_changes.iron, :]
            component_results = all_data["iron"][all_data['component'] == id_component].dropna().reset_index(drop=True)
            legend.append("id_comp= " + str(id_component))
            ax1.plot(component_results[:time_horizon], colors_to_plot[color])
            lub_hours = all_data['h_k_lubricante'][all_data['component'] == id_component].dropna().reset_index(drop=True)
            component_results = component_results.loc[component_results.index & lub_changes.index]
            component_results = component_results.loc[np.intersect1d(component_results.index, lub_changes.index)]
            component_results = component_results.loc[component_results.index.intersection(lub_changes.index)]
            ax1.plot(component_results[:time_horizon], colors_to_plot[-1-color] + 'o')
            component_results = all_data["iron"][all_data['component'] == id_component].dropna().reset_index(drop=True)
            component_results = component_results.loc[component_results.index & predicted_changes.index]
            component_results = component_results.loc[np.intersect1d(component_results.index, predicted_changes.index)]
            component_results = component_results.loc[component_results.index.intersection(predicted_changes.index)]
            ax1.plot(component_results[:time_horizon], colors_to_plot[-2-color] + '<')
            legend.append("real changes for id_comp=" + str(id_component))
            legend.append("predicted changes for id_comp=" + str(id_component))
            ax2.plot(lub_hours[:time_horizon], colors_to_plot[color])
            legend2.append("lub-hours for id_comp=" + str(id_component))
            color += 1
        ax1.legend(legend)
        ax2.legend(legend2)
        plt.xlim(0, time_horizon)
        ax2.set_xlabel('muestras')
        ax1.set_ylabel('desgaste [ppm]')
        ax2.set_ylabel('uso lubricante [hrs]')
        fig.suptitle('Fierro para cierto id_componente de motor diesel (id_tipo_componente: ' + str(id_type) + ')')
        ax1.grid(True)
        ax2.grid(True)
        plt.savefig(outfolder + 'predicted-lub-changes-for-comptype' + str(id_type) + '.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfolder', type=str, default="../figures/predictions/")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfolder)