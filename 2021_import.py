import pandas as pd
import numpy as np
import csv

data_2021 = pd.read_csv('2021_data.csv')
data_2021['TEAM'] = data_2021.TEAM.str.split('\\n',expand=True)[0]
data_2021['W'] = data_2021.Rec.str.split('\\n',expand=True)[0].str.split('-',expand=True)[0].astype('float')
data_2021['ADJDE'] = data_2021.AdjDE.str.split('\\n',expand=True)[0]
data_2021['ADJOE'] = data_2021.AdjOE.str.split('\\n',expand=True)[0]
data_2021['BARTHAG'] = data_2021.Barthag.str.split('\\n',expand=True)[0]
data_2021['EFG_O'] = data_2021['EFG%'].str.split('\\n',expand=True)[0]
data_2021['EFG_D'] = data_2021['EFGD%'].str.split('\\n',expand=True)[0]
data_2021['2P_O'] = data_2021['2P%'].str.split('\\n',expand=True)[0]
data_2021['2P_D'] = data_2021['2P%D'].str.split('\\n',expand=True)[0]
data_2021['3P_O'] = data_2021['3P%'].str.split('\\n',expand=True)[0]
data_2021['3P_D'] = data_2021['3P%D'].str.split('\\n',expand=True)[0]
data_2021['ADJ_T'] = data_2021['Adj T.'].str.split('\\n',expand=True)[0]
data_2021 = data_2021.drop(columns=['Rk','Rec','AdjOE','AdjDE','Barthag','EFG%','EFGD%','2P%','2P%D','3P%','3P%D','Adj T.'])
data_2021['TOR'] = data_2021['TOR'].str.split('\\n',expand=True)[0]
data_2021['TORD'] = data_2021['TORD'].str.split('\\n',expand=True)[0]
data_2021['ORB'] = data_2021['ORB'].str.split('\\n',expand=True)[0]
data_2021['DRB'] = data_2021['DRB'].str.split('\\n',expand=True)[0]
data_2021['FTR'] = data_2021['FTR'].str.split('\\n',expand=True)[0]
data_2021['FTRD'] = data_2021['FTRD'].str.split('\\n',expand=True)[0]
data_2021['WAB'] = data_2021['WAB'].str.split('\\n',expand=True)[0]
data_2021['POSTSEASON'] = 'NA'
data_2021['WAS'] = 'NA'

data_2015_2019 = pd.read_csv('cbbBracket.csv')
full_data = pd.concat([data_2015_2019,data_2021])
full_data.to_csv('cbb-2015-2021.csv')
