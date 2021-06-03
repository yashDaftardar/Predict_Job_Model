import re
import requests
from bs4 import BeautifulSoup
import os,sys
from sys import platform
import time


seen=set()

def get_beautifulSoup(url):
    response=requests.get(url,headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE', })
    html=response.text
    soup = BeautifulSoup(html,'html.parser')
    return soup  

def get_searchpage(who,where,page):
    url="https://www.indeed.com/jobs?q="+who.replace(' ','+')+"&l="+where.replace(' ','+')+"&start="+str((page-1)*10)
    return url

def get_ad(url):
    soup=get_beautifulSoup(url)
    ads_list_indi=soup.find_all(href=re.compile(";*fccid=*"))
    linkRe=re.compile('(.{16})(.fccid=)')
    Ads_indi_url=list()
    for x in ads_list_indi:
        a=re.findall(linkRe, x['href'])[0][0]
        if a in seen:
            continue
        seen.add(a)
        Each_Ad='https://www.indeed.com/viewjob?jk='+a+'&from=serp&vjs=3'
        Ads_indi_url.append(Each_Ad)
    return Ads_indi_url
   
def get_html(url,who,where,adsnum):
    mode='{}_{}_no{}'
    for i in range(5):
        soup=get_beautifulSoup(url).find(id='viewJobSSRRoot')
        if soup:
            break
        else: 
            time.sleep(0.5)
    html_content=soup.prettify('utf-8')
    if platform=='win32':
        Dirs_Path=os.getcwd()+'\\'+who+'\\'+where+'\\'
    elif platform=='darwin':
        Dirs_Path=os.getcwd()+'/'+who+'/'+where+'/'
    if not os.path.exists(Dirs_Path):
        os.makedirs(Dirs_Path)
    with open(Dirs_Path+mode.format(who,where,adsnum)+'.html','wb') as f:
        f.write(html_content)


def scrape(who,where):
    first=get_searchpage(who,where,1)
    page_limit=get_beautifulSoup(first).find(id='searchCountPages')
    if page_limit:
        page_limit=page_limit.text
        pageRe=re.compile('(of)(.*)(jobs)')
        page_limit=int(re.findall(pageRe,page_limit)[0][1].strip().replace(',',''))
        c=1
        for x in range(1,page_limit):
            print('Page:',x)
            url1=get_searchpage(who,where,x)
            Ad_list=get_ad(url1)
            print('url num:',len(Ad_list))
            for ad in Ad_list:
                print('URL',ad)
                try:
                    get_html(ad,who,where,c)
                    c=c+1
                except:
                    print('Can\'t get data due to captcha')
                    sys.exit(1)
    else:
        print("Received hCaptcha")
        sys.exit(1)
            

#scrape('Software Engineer','Minneapolis')
scrape('Data Scientists','Cambridge')
