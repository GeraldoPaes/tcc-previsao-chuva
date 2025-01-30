import pandas as pd
import numpy as np
import io
import re
from pathlib import Path

def rename_cols_add_prefix(df, ignore=None, prefix='_', inplace=True):
  rename_f = lambda col : col if (ignore is not None and col in ignore) else prefix+col
  return df.rename(columns=rename_f, inplace=inplace)

# Função para renomear a coluna 'YR' para 'YEAR' (Tipo 2)
def rename_year_column(df):
    return df.rename(columns={'YR': 'YEAR'})

# Função para alterar os meses em texto para o formato numérico (Tipo 1)
def change_months(df, vname):
    months_to_int = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6, JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)
    df = df.melt(id_vars=['YEAR'], var_name='MON', value_name=vname)
    df['MON'] = df['MON'].apply(lambda x : months_to_int[x])
    return df

# Extração dos Dados (Tipo 1)
# Nesses arquivos, o formato do cabeçalho começa com 'YEAR', e depois possui colunas individuais para cada mês.

def extract_data_1(file_content):
    # Dividir o conteúdo em linhas
    lines = file_content.split('\n')
    
    # Encontrar o início dos dados (linha com YEAR, JAN, FEB, ...)
    start_index = next(i for i, line in enumerate(lines) if "YEAR" in line)
    
    # Encontrar o fim dos dados (primeira linha com 2024)
    end_index = next(i for i in range(start_index, len(lines)) if lines[i].startswith('2024'))
    
    # Extrair as linhas relevantes, incluindo o cabeçalho
    data_lines = lines[start_index:end_index+1]

    # Pré-processamento das linhas para garantir a correta separação de colunas
    processed_lines = []
    for line in data_lines:
        # Substituir -999.9 por espaços em branco
        line = re.sub(r'-999\.9', '      ', line)
        # Garantir que todos os campos sejam separados por pelo menos um espaço
        fields = re.split(r'\s+', line.strip())
        processed_line = ' '.join(fields)
        processed_lines.append(processed_line)
    
    # Criar um DataFrame
    df = pd.read_csv(io.StringIO('\n'.join(processed_lines)), sep=r'\s+', na_values=[''])
    
    return df

# Extração dos Dados (Tipo 2)
#Nesses arquivos, o formato do cabeçalho começa com 'YR MON', em vez de possuir colunas individuais para cada mês.

def extract_data_2(file_content):
    # Dividir o conteúdo em linhas
    lines = file_content.split('\n')
    
    # Encontrar o início dos dados (linha do cabeçalho)
    start_index = next(i for i, line in enumerate(lines) if "YR MON" in line)
    
    # Encontrar o fim dos dados (última linha com 2024)
    end_index = next(i for i in range(len(lines)-1, start_index, -1) if lines[i].startswith('2024'))
    
    # Extrair as linhas relevantes, incluindo o cabeçalho
    data_lines = lines[start_index:end_index+1]

    # Pré-processamento das linhas para garantir a correta separação de colunas
    processed_lines = []
    for line in data_lines:
        # Substituir -999.9 por espaços em branco
        line = re.sub(r'-999\.9', '      ', line)
        # Garantir que todos os campos sejam separados por pelo menos um espaço
        fields = re.split(r'\s+', line.strip())
        processed_line = ' '.join(fields)
        processed_lines.append(processed_line)
    
    # Criar um DataFrame
    df = pd.read_csv(io.StringIO('\n'.join(processed_lines)), sep=r'\s+', na_values=[''])

    # Identificar colunas que contêm "ANOM"
    anomaly_columns = df.filter(like='ANOM').columns

    # Remover essas colunas do DataFrame
    df = df.drop(anomaly_columns, axis=1)
    
    return df

#Processamento dos Arquivos
#Recebe todos os arquivos, separados por tipo, chama as funções para extrair os dataframes, e os salva em listas.

def process_files(file_list_1, file_list_2):
    dataframes_1 = []
    dataframes_2 = []

    # Processar arquivos do tipo 1
    for filename in file_list_1:
        with open(filename, 'r') as file:
            content = file.read()
        df = extract_data_1(content)
        dataframes_1.append(df)

    # Processar arquivos do tipo 2
    for filename in file_list_2:
        with open(filename, 'r') as file:
            content = file.read()
        df = extract_data_2(content)
        dataframes_2.append(df)

    return dataframes_1, dataframes_2

def merge_in(file_list_1, file_list_2):

    # Primeiro, processa os arquivos e gera os dataframes dos dois tipos.
    dfs1, dfs2 = process_files(file_list_1, file_list_2)

    # Altera os meses para o formato numérico e renomeia as colunas (Tipo 1).
    dfs1[0] = change_months(dfs1[0], 'TW_CP')
    dfs1[1] = change_months(dfs1[1], 'DarwinPr')
    dfs1[2] = change_months(dfs1[2], 'TW_EP')
    dfs1[3] = change_months(dfs1[3], 'TahitiPr')
    dfs1[4] = change_months(dfs1[4], 'TW_WP')

    # Renomeia as colunas YEAR dos arquivos Tipo 2.
    dfs2 = [rename_year_column(df) for df in dfs2]

    all_dfs = dfs1 + dfs2

    for i, df in enumerate(all_dfs):
      if df is not None:
        rename_cols_add_prefix(df, ignore=['YEAR', 'MON'], prefix=str(i+1)+'_', inplace=True)

    df_full_in = dfs1[0]
    for df in all_dfs[1:]:
      if df is not None:
        df_full_in = df_full_in.merge(df, on=['YEAR', 'MON'], how='inner')
    
    return df_full_in

####################################################################################################
####################################################################################################

#Extração e Tratamento dos Dados
#Pegamos os arquivos .csv desorganizados e extraímos as informações importantes para o projeto.
def extract_data_out(file_content):
    
    # Carregar o CSV supondo que todos os dados estão na primeira coluna
    df = pd.read_csv(file_content, header=None, sep='\t')
    
    # Separar a primeira coluna em múltiplas colunas usando o delimitador correto
    df = df[0].str.split(';', expand=True)
    
    # Identificar a linha que contém o cabeçalho correto
    header = df.iloc[9]
    header.name = None
    
    # Atribuir os novos cabeçalhos ao DataFrame
    df.columns = header
    
    # Remover todas as linhas anteriores ao cabeçalho e a linha do cabeçalho duplicada
    df = df.drop(index=range(0, 10))
    df.reset_index(drop=True, inplace=True)
    
    df['Data Medicao'] = pd.to_datetime(df['Data Medicao'])
    
    # create new columns for year and month
    df['YEAR'] = df['Data Medicao'].dt.year
    df['MON'] = df['Data Medicao'].dt.month
    
    # reindex columns to put 'YEAR' and 'MON' as first two columns
    df = df.reindex(columns=['YEAR', 'MON'] + list(df.columns[:-2])).drop(columns='Data Medicao')

    # Renomeia e extrai apenas as colunas necessárias
    df = df.rename(columns={'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)':'PRECIP'})
    df = df.rename(columns={'PRECIPITACAO TOTAL, MENSAL(mm)':'PRECIP'})
    df = df[['YEAR', 'MON', 'PRECIP']]
    
    return df

#Processamento dos Arquivos
#Recebe todos os arquivos, separados por tipo de estação (automática ou manual), chama as funções para extrair os dataframes, e os salva em listas.
def process_files_2(file_list_1, file_list_2):
    dataframes_1 = []
    dataframes_2 = []

    # Processar arquivos do tipo 1
    for filename in file_list_1:
        df = extract_data_out(filename)
        dataframes_1.append(df)

    # Processar arquivos do tipo 2
    for filename in file_list_2:
        df = extract_data_out(filename)
        dataframes_2.append(df)

    return dataframes_1, dataframes_2

# Função para atribuir os nomes das estações aos dataframes.
def name_station(df, vname):
    df = df.rename(columns={'PRECIP':vname})
    return df

def merge_out(file_list_auto, file_list_manual):

    # Chama a função para extrair os dataframes
    dfs_auto, dfs_manual = process_files_2(file_list_auto, file_list_manual)

    # Renomeia as colunas de acordo com a estação.
    dfs_auto[0] = name_station(dfs_auto[0], 'REC_a')
    dfs_auto[1] = name_station(dfs_auto[1], 'MAC_a')
    dfs_auto[2] = name_station(dfs_auto[2], 'NAT_a')
    dfs_auto[3] = name_station(dfs_auto[3], 'JP_a')
    dfs_auto[4] = name_station(dfs_auto[4], 'SAL_a')
    dfs_auto[5] = name_station(dfs_auto[5], 'SAL_RF_a')
    
    dfs_manual[0] = name_station(dfs_manual[0], 'NAT_m')
    dfs_manual[1] = name_station(dfs_manual[1], 'JP_m')
    dfs_manual[2] = name_station(dfs_manual[2], 'REC_m')
    dfs_manual[3] = name_station(dfs_manual[3], 'MAC_m')
    dfs_manual[4] = name_station(dfs_manual[4], 'POR_m')
    dfs_manual[5] = name_station(dfs_manual[5], 'SAL_m')
    dfs_manual[6] = name_station(dfs_manual[6], 'CAN_m')

    all_dfs = dfs_auto + dfs_manual

    df_full_out = dfs_auto[0]
    for df in all_dfs[1:]:
        df_full_out = df_full_out.merge(df, on=['YEAR', 'MON'], how='outer')
    
    # Trata os 'null', transformando-os em NaN
    df_full_out.replace('null', np.nan, inplace=True)
    
    # Força o tipo numérico em todos os valores do dataframe.
    for col in df_full_out.columns[2:]:
        df_full_out[col] = pd.to_numeric(df_full_out[col], errors='coerce')
    
    # Cria a coluna 'PRECIP', com a média da precipitação de todas as estações para aquele mês/ano.
    df_full_out['PRECIP'] = df_full_out.iloc[:, 2:].mean(axis=1, skipna=True)
    
    return df_full_out

####################################################################################################
####################################################################################################

def prep_data():
    
    # Obter caminho absoluto da pasta raiz do projeto
    PROJECT_ROOT = Path(__file__).parent.parent  # src/ -> tcc/
    
    # Caminhos base para os dados
    DADOS_IN = PROJECT_ROOT / 'dados' / 'dados_in_new'
    DADOS_OUT = PROJECT_ROOT / 'dados' / 'dados_out_new'

    # Definição dos arquivos tipo 1 e tipo 2
    file_list_1 = [
        DADOS_IN / 'cpac850.txt',
        DADOS_IN / 'darwin.txt',
        DADOS_IN / 'epac850.txt',
        DADOS_IN / 'tahiti.txt',
        DADOS_IN / 'wpac850.txt',
    ]
    
    file_list_2 = [
        DADOS_IN / 'sstoi.atl.indices.txt',
        DADOS_IN / 'sstoi.indices.txt',
    ]

    # Arquivos de estações automáticas
    file_list_auto = [
        DADOS_OUT / 'automaticas' / 'dados_A301_M_2004-12-21_2021-11-11.csv',
        DADOS_OUT / 'automaticas' / 'dados_A303_M_2003-02-24_2024-11-30.csv',
        DADOS_OUT / 'automaticas' / 'dados_A304_M_2003-02-23_2024-11-30.csv',
        DADOS_OUT / 'automaticas' / 'dados_A320_M_2007-07-20_2024-11-30.csv',
        DADOS_OUT / 'automaticas' / 'dados_A401_M_2000-05-12_2024-11-30.csv',
        DADOS_OUT / 'automaticas' / 'dados_A456_M_2018-03-09_2024-11-30.csv'
    ]
    
    # Arquivos de estações manuais
    file_list_manual = [
        DADOS_OUT / 'convencionais' / 'dados_82598_M_1970-01-01_2024-11-30.csv',
        DADOS_OUT / 'convencionais' / 'dados_82798_M_1970-01-01_2024-11-30.csv',
        DADOS_OUT / 'convencionais' / 'dados_82900_M_1970-01-01_2020-08-31.csv',
        DADOS_OUT / 'convencionais' / 'dados_82994_M_1970-01-01_2021-04-07.csv',
        DADOS_OUT / 'convencionais' / 'dados_82996_M_1970-01-01_2024-11-30.csv',
        DADOS_OUT / 'convencionais' / 'dados_83229_M_1970-01-01_2024-11-30.csv',
        DADOS_OUT / 'convencionais' / 'dados_83398_M_1970-01-01_2021-07-30.csv'
    ]

    df_in = merge_in(file_list_1, file_list_2)
    df_out = merge_out(file_list_auto, file_list_manual)

    return df_in, df_out




    