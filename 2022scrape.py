import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

#Importing data
data = pd.read_csv('cbb.csv', header=0)
teams = data['TEAM'].unique()
url1 = 'https://barttorvik.com/team.php?team='
url2 = '&year=2022'
# Scraping function
url = url1+teams[0].replace(' ','+')+url2
page = requests.get(url)
schedule_table = BeautifulSoup(page.content, 'html.parser').find("table", attrs={"class": "skedtable"}).tbody
schedule_table = schedule_table.find_all("tr")
schedule_table = schedule_table[:-1]
print(schedule_table[0].find_all("td")[0].text.split('\n')[2])
#matchups = []
#for i in schedule_table:
#    matchups.append([teams[0],i.find_all("td")[5].a.text,i.find_all("td")[7].a.text[0]])

# Looping through teams
matchup_data = []
for i in teams:
    try:
        i = i.replace(' ','+')
        url = url1+i+url2
        page = requests.get(url)
        schedule_table = BeautifulSoup(page.content, 'html.parser').find("table", attrs={"class": "skedtable"}).tbody
        schedule_table = schedule_table.find_all("tr")
        schedule_table = schedule_table[:-1]
        #print(schedule_table[0].find("td", attrs={"class": "mobileout"}).a.text)
        for j in schedule_table:
            try:
                matchup_data.append([j.find_all("td")[0].text.split('\n')[2],i,j.find_all("td")[5].a.text,j.find_all("td")[7].a.text])
            except:
                pass
    except:
        print(i)
        pass
matchups = pd.DataFrame(matchup_data, columns=['date','team','opponent','result'])
matchups.to_csv('2022Matchups.csv',index=False)
