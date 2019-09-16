import numpy as np

import selenium
from selenium import webdriver
import requests
from bs4 import BeautifulSoup
import psycopg2

from datetime import datetime
import time

import json


# set date, load json connection args, and create a cursos

date = datetime.strftime(datetime.today(), "%Y-%m-%d")

with open('../conn.json') as fp:
    conn_kwargs = json.load(fp)

conn = psycopg2.connect(**conn_kwargs)
cur = conn.cursor()

# Search terms

queries = [['Data', 'Scientist'], ['Machine', 'Learning', 'Engineer'],
           ['Data', 'Engineer'], ['Research', 'Engineer'],
           ['Artificial', 'Intelligence', 'Engineer'],
           ['Product', 'Analyst'], ['Product', 'Scientist']]

locations = [['Seattle'], ['New', 'York,+NY&_ga='],
             ['Los', 'Angeles'], ['Chicago'],
             ['San', 'Francisco', 'Bay', 'Area%2C+CA']]



###############################################################################
##                             Helper  Functions                             ##
###############################################################################

def get_soup(url):
    chrome_options = webdriver.chrome.options.Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = (webdriver.Chrome(
        executable_path='../../../miniconda3/bin/chromedriver',
        options=chrome_options)
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    driver.close()
    return soup

def grab_job_links(base_url):
    
    soup = get_soup(base_url)
    
    urls = []
            
    for link in soup.find_all('div', {'class': 'title'}):
        try:
            partial_url = link.a.get('href')
            url = 'https://www.indeed.com' + partial_url
            urls.append(url)
            del url
            del partial_url

        except:
            continue
    
    del soup
        
    return urls

###############################################################################
##                              Master Function                              ##
###############################################################################

def get_jobs_pages(cur):

    date = str(datetime.strftime(datetime.today(), "%Y-%m-%d"))
    
    cur.execute("SELECT query, city, link FROM job_listings_pages;")
    job_listing_urls = cur.fetchall()
    
    job_listing_urls = list(set(job_listing_urls))
    
    for query, city, job_listing_url in job_listing_urls:
        
        urls = grab_job_links(job_listing_url)
        
        values = ((date, query, city, url) for url in urls)

        insert_statement = '''
        INSERT INTO jobs_pages
        (date_pulled, query, city, link)
        VALUES (%s, %s, %s, %s);
        '''

        cur.executemany(insert_statement, values)
        
        conn.commit()
        
        time.sleep(np.random.poisson(100)/50)
    
    conn.close()

get_jobs_pages(cur)