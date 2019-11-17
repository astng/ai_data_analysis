import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

time_horizon = 150
n_components = 5
colors_to_plot = ['r', 'g', 'b', 'y', 'k']

def main(database: str, table: str, outfolder: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()
    types = [39, 630, 681, 682] # 798, 799, 1236 and 1237 shows KeyError
    components = {}
    all_data = pd.DataFrame(query_fetch)
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
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        legend = []
        legend2 = []
        color = 0
        for _, id_component in components_to_plot[:n_components]:
            lub_changes = all_data[all_data['component'] == id_component]['h_k_lubricante'] < \
                          all_data[all_data['component'] == id_component]['h_k_lubricante'].shift()
            lub_changes = lub_changes.reset_index()
            lub_changes = lub_changes.loc[lub_changes.h_k_lubricante, :]
            component_results = all_data["iron"][all_data['component'] == id_component].dropna().reset_index(drop=True)
            legend.append("id_comp= " + str(id_component))
            ax1.plot(component_results[:time_horizon], colors_to_plot[color])
            limits = all_data["ironLSC"][all_data['component'] == id_component].dropna().reset_index(drop=True)
            protocols = all_data["id_protocol"][all_data['component'] == id_component].dropna().reset_index(drop=True)
            ax1.plot(limits[:time_horizon], colors_to_plot[color] + '--')
            legend.append("LSC for id_comp=" + str(id_component))
            #ax1.plot(protocols[:time_horizon], linestyle='dotted')
            #legend.append("prot " + str(list(set(protocols))[0]) + "for id_comp " + str(id_component))
            lub_hours = all_data['h_k_lubricante'][all_data['component'] == id_component].dropna().reset_index(drop=True)
            ax1.plot(lub_changes['h_k_lubricante'][:time_horizon], colors_to_plot[-1-color] + 'o')
            legend.append("real changes for id_comp=" + str(id_component))
            ax2.plot(lub_hours[:time_horizon], colors_to_plot[color])
            legend2.append("lub-hours for id_comp=" + str(id_component))
            color += 1
        ax1.legend(legend)
        ax2.legend(legend2)
        ax2.set_xlabel('muestras')
        ax1.set_ylabel('desgaste [ppm]')
        ax2.set_ylabel('uso lubricante [hrs]')
        #plt.title('Analisis del nivel de fierro para motor diesel (id_tipo_componente: ' + str(type) + ')')
        ax1.grid(True)
        ax2.grid(True)
        plt.savefig(outfolder + 'with_lub_changes_and_LSC-dieselmotor' + str(type) + '.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfolder', type=str, default="../figures/iron-plots/")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfolder)