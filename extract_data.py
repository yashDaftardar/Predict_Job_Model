import re
import requests
from bs4 import BeautifulSoup
import os
import time
import csv
from sys import platform


def text(job,loc):

    if platform=='win32':
        file=os.getcwd()+'\\'+job+'\\'+loc+'\\'
    elif platform=='darwin':
        file=os.getcwd()+'/'+job+'/'+loc+'/'
    texts=os.listdir(file)
    for t in texts:
        if os.path.splitext(t)[1]=='.html':
            write_csv(get_ad_data(file,t),job)
                

def write_csv(text,job):
    with open('Job_Ads.csv','a',encoding='utf8') as fw:
        writer=csv.writer(fw,lineterminator='\n')
        writer.writerow([text,job])
    

def get_ad_data(path,file):
    if re.match(r'.?data sci[a-z]+',file,re.I):
        re_sub=r'.?data sci[a-z]+'
    elif re.match('software eng[a-z]+',file,re.I):
        re_sub=r'.?software eng[a-z]+'
    job_text,data=[],[]
    soup=get_beautifulSoup(path+file)
    job_text_list=soup.find('div',class_='jobsearch-jobDescriptionText').get_text()

    job_text_list=re.sub(re_sub,' ',job_text_list,re.I)

    job_text_list=re.sub('[^a-z0-9$\s]',' ',job_text_list.lower()).split('\n')
       
    for x in job_text_list:
        if x!='' and x.strip()!='':
            x=re.sub(re_sub,' ',x,re.I)
            job_text.append(x.strip())

    for x in job_text:
        y=x.split(' ')
        for z in y:
            if z!='' and z.strip()!='':
                data.append(z.strip())

    data=' '.join(data)
    return data
    

def get_beautifulSoup(path):
    soup=None
    with open(path,'r',encoding='utf8') as fw:
        soup=BeautifulSoup(fw.read(),'html.parser')
    return soup
    
def get_location_name(job):
    file=os.getcwd()+'\\'+ job+'\\'
    name=os.listdir(file)
    location=[]
    for n in name:
        location.append(n)
    return location

def ad_data_scrape(job):
    location=get_location_name(job)
    for loc in location:
        text(job,loc)
        print(job+'\\'+ loc +'\\'+'extracting finished')
 

ad_data_scrape('Software Engineer')
#ad_data_scrape('Data Scientists')
