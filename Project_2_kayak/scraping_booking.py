import scrapy 
from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
import pandas as pd , numpy as np
import time



class BookingSpider(scrapy.Spider): 

    name = 'Booking_Spider' # Mandatory
    

    def __init__(self): # To store attributes

        self.cities = list(pd.read_csv('df_with_weather.csv')['City']) # To get the right city for the right elements
        self.urls = [f'https://www.booking.com/searchresults.fr.html?label=gen173nr-1FCAEoggI46AdIDVgEaE2IAQGYAQ24ARfIAQzYAQHoAQH4AQKIAgGoAgO4AvuylaEGwAIB0gIkMjBiZDJlZWUtZjY0ZC00OWVlLWExZGQtMWQzN2NhYTA1NDA52AIF4AIB&aid=304142&ss={city}&ssne={city}&ssne_untouched={city}&lang=fr&sb=1&src_elem=sb&src=index&dest_type=city&sb_travel_purpose=leisure' for city in self.cities]


    def start_requests(self): # Mandatory function to set up the iteration logic

        # Iterate through the urls and city
        for url, city in zip(self.urls, self.cities):
            yield scrapy.Request(url=url, callback=self.parse, meta={'city': city}) # Iterate throught each url , adding the city as metadata


    def parse(self , response): # Main parse method

        # Get the city as meta-data of the responses
        city = response.meta['city']
        
        # Iterate throught the hotel object , to make sure all the elements belongs to the same hotel
        for  property_card in response.xpath('//*[@data-testid="property-card"]'): 
                
                # Define the XPAThs
                names = property_card.xpath('.//*[@class="fcab3ed991 a23c043802"]/text()')
                locs_ = property_card.xpath('.//*[@class="f4bd0794db b4273d69aa" and contains(@data-testid, "address")]/text()') 
                descrs = property_card.xpath('.//*[@class="a1b3f50dcd ef8295f3e6 f7c6687c3d"]/div[@class="d8eab2cf7f"]/text()') 
                ratings = property_card.xpath('.//*[@class="b5cd09854e d10a6220b4"]/text()') 
                links = property_card.xpath('.//*[@class="e13098a59f"]/@href')
                adress_links = property_card.xpath('.//*[@class="fc63351294 a168c6f285 e0e11a8307 a25b1d9e47"]/@href')

                # Get the individual variables
                for i , rating in enumerate(ratings): # Formating with the ratings , since if no ratings is available I am not interested in the hotel
                    # if i < len(names) and i < len(locs_) and i < len(descrs) and i < len(links): # To avoid indexerrors. Not mandatory

                        # Get all the variable as strings
                        name = names[i].get()
                        loc_ = locs_[i].get()
                        descr = descrs[i].get()
                        link = links[i].get()
                        adress_link = adress_links[i].get()

                        # Add the variables to the dictionaries
                        ratings_dict[name] = rating.get()
                        name_list.append(name)
                        loc_dict[name] = loc_
                        descr_dict[name] = descr
                        links_dict[name] = link
                        cities_dict[name] = city

                        # Follow the links to the map url , to get the address
                        yield scrapy.Request(url=adress_link, callback=self.parse_adress, cb_kwargs={'name': name})
                        


            

    # Parse the adress 
    def parse_adress(self , response , name):
            adress = response.xpath('//*[contains(@class , "hp_address_subtitle")]/text()')

            for adress_ in adress:
                adress_dict[name] = (str(adress_.get()).replace('\n' , ''))



# Instantiate the variable to store data , outside the class
name_list = []
loc_dict = {}
descr_dict = {}
ratings_dict = {}
links_dict = {}
cities_dict = {}
adress_dict = {}

# Since I don't want any log expect for fatal errors
configure_logging({'LOG_ENABLED': False})
process = CrawlerProcess(settings={
'LOG_LEVEL': 'ERROR',
'USER_AGENT': 'Chrome/97.0'
})

# Start the process
process.crawl(BookingSpider)
process.start() 


# Instantiate the dataframe
df = pd.DataFrame({'Name' : name_list})

# Add the variables
df = df.drop_duplicates(subset='Name')     
df['City'] = [cities_dict.get(name) for name in list(df['Name'])]           
df['Rating'] = [float(str(ratings_dict.get(name)).replace(',' , '.')) for name in list(df['Name'])]
df['Loc'] = [loc_dict.get(name) for name in list(df['Name'])]
df['Description'] = [descr_dict.get(name) for name in list(df['Name'])]
df['Adress'] = [adress_dict.get(name) for name in list(df['Name'])]  
df['Link'] = [links_dict.get(name) for name in list(df['Name'])]


# Save it
df.to_csv('Scraped_df.csv')