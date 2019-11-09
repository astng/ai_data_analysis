import numpy as np
import pandas as pd
import MySQLdb
import argparse
from pymongo import MongoClient

pd.set_option('display.max_columns', 50)

STNG_ID_2_CHARACTERIZATION = {
    35: "karl_fisher_water",
    3: "aluminium",
    42: "antifreeze",
    17: "barium",
    18: "boron",
    11: "cadmium",
    19: "calcium",
    4: "copper",
    47: "iso_code",
    38: "1000_hit_consistency",
    37: "non_worked_consistency",
    36: "worked_consistency",
    26: "water_content",
    2: "chromium",
    40: "ph",
    30: "dilution_by_combustible",
    8: "tin",
    49: "ester_breakdown_1",
    50: "ester_breakdown_2",
    51: "ferrography",
    1: "iron",
    22: "phosphorus",
    55: "soot_percentage",
    34: "soot_absorbency",
    57: "magnetic_plug_image",
    25: "viscosity_index_image",
    29: "pq_index",
    21: "magnesium",
    12: "manganese",
    20: "molybdenum",
    6: "nickel",
    33: "nitration",
    28: "acid_total_number",
    27: "basic_total_number",
    31: "oxidation",
    54: "100_um_particles",
    53: "50_um_particles",
    52: "25_um_particles",
    46: "14_um_particles",
    45: "6_um_particles",
    44: "4_um_particles",
    7: "silver",
    5: "lead",
    14: "potassium",
    41: "freezing_point",
    39: "dripping_point",
    43: "pmcc_combustion_point",
    15: "silicon",
    13: "sodium",
    32: "sulfation",
    56: "magnetic_plug",
    48: "patch_test",
    9: "titanium",
    10: "vanadium",
    24: "100_cinematic_viscosity",
    23: "40_cinematic_viscosity",
    16: "zinc"
}


def main(user, password, db_name, table_name, mysql_db, mongo_limits_db):
    client = MongoClient()
    db_limits_name = mongo_limits_db
    table_limits_name = "essay_limits"
    db_mongo_limits = client[db_limits_name]
    table_limits = db_mongo_limits[table_limits_name]
    query_fetch = table_limits.find()
    all_limits_data = pd.DataFrame(query_fetch)

    db_mongo = client[db_name]
    table_mongo = db_mongo[table_name]
    table_mongo_rejected = db_mongo[table_name + "_rejected"]

    mysql_cn = MySQLdb.connect(host='localhost',
                               port=3306,
                               user=user,
                               passwd=password,
                               db=mysql_db)

    sql_resultado = 'select id_resultado, valor, id_ensayo, correlativo_muestra,' \
                    ' id_protocolo from trib_resultado'
    sql_muestra = 'select correlativo_muestra, id_componente, cambio_componente from trib_muestra'
    sql_componente = 'select id_componente, id_equipo, id_tipo_componente, tag from trib_componente'
    sql_equipo = 'select id_equipo, id_faena, id_tipo_equipo from trib_equipo'
    sql_faena = 'select id_faena, id_cliente from trib_faena'
    sql_cliente = 'select id_cliente, nombre_abreviado from trib_cliente'

    print("getting all data from mysql")
    data_resultado = pd.read_sql(sql_resultado, con=mysql_cn)
    data_muestra = pd.read_sql(sql_muestra, con=mysql_cn)
    data_componente = pd.read_sql(sql_componente, con=mysql_cn)
    data_equipo = pd.read_sql(sql_equipo, con=mysql_cn)
    data_faena = pd.read_sql(sql_faena, con=mysql_cn)
    data_cliente = pd.read_sql(sql_cliente, con=mysql_cn)
    data_cliente['nombre_abreviado'] = data_cliente['nombre_abreviado'].map(lambda x: x.replace(' ', '_'))
    print("finished reading data")

    dict_muestra = pd.Series(data_muestra.id_componente.values, index=data_muestra.correlativo_muestra).to_dict()
    dict_changes = pd.Series(data_muestra.cambio_componente.values, index=data_muestra.correlativo_muestra).to_dict()
    dict_componente = pd.Series(data_componente.id_equipo.values, index=data_componente.id_componente).to_dict()
    dict_tipo_componente = pd.Series(data_componente.id_tipo_componente.values, index=data_componente.id_componente).to_dict()
    dict_tag = pd.Series(data_componente.tag.values, index=data_componente.id_componente).to_dict()
    dict_equipo = pd.Series(data_equipo.id_faena.values, index=data_equipo.id_equipo).to_dict()
    dict_tipo_equipo = pd.Series(data_equipo.id_tipo_equipo.values, index=data_equipo.id_equipo).to_dict()
    dict_faena = pd.Series(data_faena.id_cliente.values, index=data_faena.id_faena).to_dict()
    dict_cliente = pd.Series(data_cliente.nombre_abreviado.values, index=data_cliente.id_cliente).to_dict()
    data_resultado = data_resultado.loc[data_resultado['id_ensayo'].notna()]
    data_resultado['id_ensayo'] = data_resultado['id_ensayo'].map(lambda x: STNG_ID_2_CHARACTERIZATION[x])
    dict_protocolo = pd.Series(data_resultado.id_protocolo.values, index=data_resultado.correlativo_muestra).to_dict()

    group_by_correlative = data_resultado.groupby(by='correlativo_muestra')
    count = 0
    for group in group_by_correlative:
        if count % 100 == 0:
            print("count ", count)
        new_row = group[1].pivot(index='correlativo_muestra', columns='id_ensayo', values='valor')

        try:
            iso_code_serie = None
            patch_test = None
            if 'patch_test' in new_row.columns:
                patch_test = new_row['patch_test']
                new_row = new_row.drop(columns=["patch_test"])

            if 'iso_code' in new_row.columns:
                iso_code_serie = new_row["iso_code"]
                new_row = new_row.drop(columns=["iso_code"])
            new_row = new_row.applymap(lambda x: float(
                x.replace(",", ".").replace("<", "").replace(">", "").replace("N/A", "nan").replace(" ", "").replace(
                    "..", ".")
                    .replace("NASD", "0.0").replace("NAD", "0.0").replace("NAS", "nan").replace("NA", "nan")
                    .replace("NSD.", "0.0").replace("NSD", "0.0").replace("/A", "nan").replace("%", "")
                    .replace("2.480.", "2.480").replace("NH/A", "nan").replace("0nan", "nan").replace("N8A", "nan")
                    .replace("N8A", "nan").replace("`", "0.0").replace("Mnan", "nan").replace("n/a", "nan")
                    .replace("N/", "nan").replace("nsd", "0.0").replace("NDS", "0.0")
                    .replace("SD", "0.0").replace("0.0.", "0.0").replace("N*A", "nan").replace("N|", "nan")
                    .replace("NSA", "0.0").replace("nanD", "nan").replace("5.638.", "5.638").replace("1.3.", "1.3")
                    .replace("nan0.0", "0.0").replace("NS", "0.0").replace("0.2.", "0.2").replace("nanS", "0.0")
                    .replace("3.618.", "3.618").replace("15.374.", "15.374").replace("|", "").replace("15.25.", "15.25")
                    .replace("/nan", "nan").replace("°C", "").replace("0.00.0", "0.0").replace("B0.0", "0.0")
                    .replace("5.671.", "5.671").replace("53.34.", "53.34").replace("11.034.", "11.034")
                    .replace("9.1.", "9.1").replace("0.0S", "0.0").replace("8nan", "0.0")
            ))
            if iso_code_serie is not None:
                new_row["iso_code"] = iso_code_serie
            if patch_test is not None:
                new_row["patch_test"] = patch_test

            records = new_row.iloc[0].to_dict()
            records.update({'correlativo_muestra': group[0]})
            try:
                if dict_muestra.get(group[0]) is None or dict_componente.get(dict_muestra[group[0]]) is None:
                    continue
                id_component = dict_componente[dict_muestra[group[0]]]
                all_protocols = np.isin(all_limits_data.id_protocol.values, group[1].id_protocolo.unique())
                df_limits = all_limits_data[all_protocols].set_index('essayed')
                limits_cleaned = df_limits.loc[:, ['LIC', 'LIM', 'LSM', 'LSC']].reset_index()\
                    .pivot_table(columns='essayed').unstack()  # unpivoting id_ensayo from essay_limits table
                limits_cleaned.index = limits_cleaned.reset_index()[['essayed', 'level_1']].sum(axis=1)
                records.update(limits_cleaned.to_dict())
                records.update({'client': dict_cliente[dict_faena[dict_equipo[id_component]]],
                                'component': id_component, 'component_type': dict_tipo_componente[id_component],
                                'tag': dict_tag[id_component],
                                'change': dict_changes[group[0]],  # no use for Afta BD
                                'machine_type': dict_tipo_equipo[dict_equipo[id_component]],
                                'id_protocol': dict_protocolo[group[0]]})
                print(records)
                table_mongo.insert(records)
            except Exception as e:
                pass
            finally:
                pass

        except ValueError as e:
            print("not possibly to upload value to mongo, value error: ", e)
            records = new_row.iloc[0].to_dict()
            table_mongo_rejected.insert(records)
        finally:
            count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", type=str, help="mysql user", required=True)
    parser.add_argument("-p", "--password", type=str, help="mysql password", required=True)
    parser.add_argument("-m", "--mysql_db", type=str, help="old code version mysql db path", required=True)
    parser.add_argument("-d", "--db", type=str, help="mongo target db", required=True)
    parser.add_argument("-t", "--table", type=str, help="mongo table name", required=True)
    parser.add_argument("-m", "--mongo_limits_db", type=str, help="mongo limtis db", required=True)

    cmd_args = parser.parse_args()
    main(cmd_args.user, cmd_args.password, cmd_args.db, cmd_args.table, cmd_args.mysql_db, cmd_args.mongo_limits_db)