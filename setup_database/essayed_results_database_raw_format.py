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

    mysql_cn = MySQLdb.connect(host='localhost',
                               port=3306,
                               user=user,
                               passwd=password,
                               db=mysql_db)

    sql = 'select id_resultado, valor, id_ensayo, correlativo_muestra, id_protocolo from trib_resultado'

    data = pd.read_sql(sql, con=mysql_cn)
    data = data[pd.notnull(data['id_ensayo'])]
    data['id_ensayo'] = data['id_ensayo'].map(lambda x: STNG_ID_2_CHARACTERIZATION[x])
    group_by_correlative = data.groupby(by='correlativo_muestra')
    count = 0
    for group in group_by_correlative:
        if count % 100 == 0:
            print("count ", count)
        new_row = group[1].pivot(index='correlativo_muestra', columns='id_ensayo', values='valor')
        records = new_row.iloc[0].to_dict()
        table_mongo.insert(records)
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
