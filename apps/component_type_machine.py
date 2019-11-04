import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pymongo import MongoClient

time_horizon = 150
n_machine_types = 4  # una figura por cada tipo de equipo
n_component_types = 4  # debe tener raiz cuadrada exacta
n_components = 6  # curvas en cada subplot


def main(database: str, table: str, outfolder: str):
    client = MongoClient()
    db_mongo = client[database]
    table_mongo = db_mongo[table]
    query_fetch = table_mongo.find()

    data_numbers = []
    machines = {}
    types = {}
    combinations = set()

    all_data = pd.DataFrame(query_fetch)
    grouped_by_triplet = all_data.groupby(by=['machine_type', 'component_type', 'component'])
    for group_id in grouped_by_triplet:
        machine_type, component_type, id_component = group_id[0][0], group_id[0][1], group_id[0][2]
        group = grouped_by_triplet.groups[group_id[0]]
        results = all_data.iloc[group]["iron"].dropna().reset_index(drop=True)
        data_numbers.append((len(results), machine_type, component_type, id_component))
    data_numbers.sort()
    data_numbers.reverse()
    print("done")

    for data in data_numbers:
        combinations.add(tuple(data[1:]))

    for cnt, data in enumerate(data_numbers):
        if data[1] not in machines and len(machines) < n_machine_types:
            machines[data[1]] = set()
        if len(machines[data[1]]) < n_component_types:
            machines[data[1]].add(data[2])
        for n_data, machine_type, component_type, id_component in data_numbers[cnt + 1:]:
            components = all_data["component"][all_data["component_type"] == component_type]
            if data[1] == machine_type and component_type not in machines[data[1]] and len(
                    set(components)) >= n_components and len(machines[data[1]]) < n_component_types:
                machines[data[1]].add(component_type)
            if len(machines[data[1]]) == n_component_types:
                break
        if len(machines) == n_machine_types:
            break

    for data in data_numbers:
        if data[2] not in types and tuple(data[1:]) in combinations:
            types[data[2]] = []
        elif tuple(data[1:]) in combinations:
            types[data[2]].append(data[3])  # primeros id_componente son los primeros de data_numbers (con mas datos)

    for machine_type, component_types in machines.items():
        square = int(float(n_component_types) ** 0.5)
        fig, axs = plt.subplots(square, square)
        row, col = 0, 0
        for component_type in component_types:
            legend = []
            cnt = 0
            for id_componente in types[component_type]:
                if (machine_type, component_type, id_componente) in combinations:
                    group = grouped_by_triplet.groups[(machine_type, component_type, id_componente)]
                    results = all_data.iloc[group]["iron"].dropna().reset_index(drop=True)
                    limits = all_data.iloc[group]["ironLSC"].dropna().reset_index(drop=True)
                    protocols = all_data.iloc[group]["id_protocol"].dropna().reset_index(drop=True)
                    axs[row, col].plot(results[:time_horizon])
                    axs[row, col].plot(limits[:time_horizon], linestyle='dashed', label='_nolegend_')
                    axs[row, col].plot(protocols[:time_horizon], linestyle='dotted', label="prot " +
                                                                                           list(set(protocols))[0] + "for id_comp " + str(id_component))
                    legend.append(id_componente)
                    cnt += 1
                    if cnt == n_components:
                        break
            axs[row, col].legend(legend)
            axs[row, col].grid(True)
            axs[row, col].set_title("component type: " + str(component_type))
            col += 1
            if col == square:
                col = 0
                row += 1
        fig.suptitle("machine type: " + str(machine_type))
        plt.savefig(outfolder + 'with_protocol-iron-' + 'machinetype-' + str(machine_type) + '.pdf', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, required=True,
                        help="MongoDB database used in essayed_results_database_raw_format.py")
    parser.add_argument('--outfolder', type=str, default="../figures/iron-plots/")
    parser.add_argument('--table', type=str, required=True, help="table used in essayed_results_database_raw_format.py")
    cmd_args = parser.parse_args()
    main(cmd_args.database, cmd_args.table, cmd_args.outfolder)
