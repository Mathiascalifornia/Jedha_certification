import scrapy 
from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By 
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time


class BookingSpider(scrapy.Spider):
    name = 'Booking_Spider'
    
    def __init__(self , start_url : str , n_next_page_to_scrape : int):

        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install())) 
        self.start_url = start_url
        self.n_next_page_to_scrape = n_next_page_to_scrape

    

    def start_requests(self):
        yield scrapy.Request(url=self.start_url , callback=self.parse)

    def parse(self , response):
        names = response.xpath('//*[@class="fcab3ed991 a23c043802"]/text()')
        locs_ = response.xpath('//*[@class="f4bd0794db b4273d69aa"]/text()')
        descrs = response.xpath('//*[@class="d8eab2cf7f"]/text()')
        ratings = response.xpath('//*[@class="b5cd09854e d10a6220b4"]/text()')
        links = response.xpath('//*[@class="e13098a59f"]/@href')


        for name in names:
            name_list.append(name.get())

        for loc_ in locs_:
            if loc_.get() != 'Indiquer sur la carte':
                loc_list.append(loc_.get())

        for descr in descrs:
            descr_list.append(descr.get())

        for rating in ratings:
            ratings_list.append(rating.get())

        for link in links:
            links_list.append(link.get())



        next_page_button = '//*[@id="search_results_table"]/div[2]/div/div/div[4]/div[2]/nav/div/div[3]/button'
        self.driver.get(self.start_url) # Get the start url
        self.driver.find_element(By.XPATH , ('//*[@id="onetrust-accept-btn-handler"]')).click() # Accept conditions


        for i in range(self.n_next_page_to_scrape):
                time.sleep(0.5)
                self.driver.find_element(By.XPATH , (next_page_button)).click() # Click on the next page button
                time.sleep(0.5)
                next_url = self.driver.current_url # Get the new url
                yield scrapy.Request(next_url, callback=self.parse) # Scrape the new page
                time.sleep(0.5)

        try:
            self.driver.quit()
        except:
            pass

               

name_list = []
loc_list = []
descr_list = []
ratings_list = []
links_list = []

configure_logging({'LOG_ENABLED': False})
process = CrawlerProcess(settings={
        'LOG_LEVEL': 'ERROR',
        'USER_AGENT': 'Chrome/97.0'
    })

def get_data_by_city(city : str , n_page : int) -> pd.DataFrame:
    '''city : the city to get a hotel dataframe for
        n_page : number of page to scrape on booking.com'''
    
    process.crawl(BookingSpider , start_url=f'https://www.booking.com/searchresults.fr.html?label=gen173nr-1FCAEoggI46AdIDVgEaE2IAQGYAQ24ARfIAQzYAQHoAQH4AQKIAgGoAgO4AvuylaEGwAIB0gIkMjBiZDJlZWUtZjY0ZC00OWVlLWExZGQtMWQzN2NhYTA1NDA52AIF4AIB&aid=304142&ss={city}&ssne={city}&ssne_untouched={city}&lang=fr&sb=1&src_elem=sb&src=index&dest_id=-1456928&dest_type=city&group_adults=2&no_rooms=1&group_children=0&sb_travel_purpose=leisure&offset=200',
                n_next_page_to_scrape=n_page)
    
    process.start()

    assert len(loc_list) == len(name_list) == len(descr_list) == len(ratings_list) == len(links_list)

    return pd.DataFrame({'city' : [city for i in range(len(loc_list))] , 'name_hotel' : name_list , 'Loc' : loc_list ,
                         'Description' : descr_list , 'Rating' : ratings_list , 'Link' : links_list})

