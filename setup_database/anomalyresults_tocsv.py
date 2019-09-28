import pandas as pd
import MySQLdb
import argparse

pd.set_option('display.max_columns', 50)

def main(user, password, mysql_db):

    mysql_cn = MySQLdb.connect(host='localhost',
                               port=3306,
                               user=user,
                               passwd=password,
                               db=mysql_db)

    sql = 'select * from trib_resultado'

    print("getting data from tables trib_resultado and trib_muestra")
    trib_resultado = pd.read_sql(sql, con=mysql_cn)
    trib_resultado = trib_resultado[pd.notnull(trib_resultado['id_ensayo'])]
    trib_resultado['correlativo_muestra'] = trib_resultado['correlativo_muestra'].apply(lambda x: int(x))

    sql = 'select * from trib_muestra'
    trib_muestra = pd.read_sql(sql, con=mysql_cn)
    trib_muestra = trib_muestra[pd.notnull(trib_muestra['cp_14_condicion_muestra'])]
    #print(trib_muestra[['id_muestra','fecha_muestreo','correlativo_muestra']])
    #for corr_muestra in trib_muestra['correlativo_muestra'][:2]:
        #print(trib_resultado[trib_resultado['correlativo_muestra'] == corr_muestra][['correlativo_muestra','id_ensayo','id_protocolo','valor']])
    merged = trib_muestra.merge(trib_resultado, on='correlativo_muestra')
    merged.set_index('fecha_muestreo')
    merged.to_csv('dataset.csv', columns = ['correlativo_muestra', 'fecha_muestreo', 'id_lubricante', 'id_componente', 'id_protocolo_lubricante', 'id_protocolo_componente', 'id_ensayo', 'id_protocolo', 'valor', 'cp_14_condicion_muestra'])


    print("done")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", type=str, help="mysql user", required=True)
    parser.add_argument("-p", "--password", type=str, help="mysql password", required=True)
    parser.add_argument("-m", "--mysql_db", type=str, help="old code version mysql db path", required=True)
    cmd_args = parser.parse_args()
    main(cmd_args.user, cmd_args.password, cmd_args.mysql_db)