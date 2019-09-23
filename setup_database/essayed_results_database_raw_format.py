import json
import random

import pandas as pd
import MySQLdb
import argparse
import numpy as np
from datetime import datetime as dt
from pymongo import MongoClient

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


def main(user, password, db_name, table_name, mysql_db):
    current_time = dt.now()
    client = MongoClient()
    db_mongo = client[db_name]
    table_mongo = db_mongo[table_name]
    table_mongo_rejected = db_mongo[table_name + "_rejected"]

    mysql_cn = MySQLdb.connect(host='localhost',
                               port=3306,
                               user=user,
                               passwd=password,
                               db=mysql_db)

    sql = 'select id_resultado, valor, id_ensayo, correlativo_muestra, id_protocolo from trib_resultado'

    print("getting all data from mysql")
    data = pd.read_sql(sql, con=mysql_cn)
    print("finished reading data")

    data = data[pd.notnull(data['id_ensayo'])]
    data['id_ensayo'] = data['id_ensayo'].map(lambda x: STNG_ID_2_CHARACTERIZATION[x])
    group_by_correlative = data.groupby(by='correlativo_muestra')
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
                x.replace(",", ".").replace("<", "").replace(">", "").replace("N/A", "nan").replace(" ", "").replace("..", ".")
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
            table_mongo.insert(records)
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
    parser.add_argument("-d", "--db", type=str, help="mongo db name", required=True)
    parser.add_argument("-t", "--table", type=str, help="mongo table name", required=True)
    cmd_args = parser.parse_args()
    main(cmd_args.user, cmd_args.password, cmd_args.db, cmd_args.table, cmd_args.mysql_db)
