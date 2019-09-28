import pandas as pd
import MySQLdb
import argparse

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


def main(user, password, mysql_db):

    mysql_cn = MySQLdb.connect(host='localhost',
                               port=3306,
                               user=user,
                               passwd=password,
                               db=mysql_db)

    sql = 'select * from trib_ensayo_protocolo'

    print("getting all data from mysql")
    data = pd.read_sql(sql, con=mysql_cn)
    print("finished reading data")

    print(data)
    print("columnas:",data.keys())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", type=str, help="mysql user", required=True)
    parser.add_argument("-p", "--password", type=str, help="mysql password", required=True)
    parser.add_argument("-m", "--mysql_db", type=str, help="old code version mysql db path", required=True)
    cmd_args = parser.parse_args()
    main(cmd_args.user, cmd_args.password, cmd_args.mysql_db)