from selectorlib import Extractor
import requests
import json
import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from newsplease import NewsPlease
import re


#Script used to scrap articles of cybersecurity websites recursively, and gather them in a big cybersecurity corpus used for pretraining. 

#It is sometimes necessary to help the scrapper and show It where the corpus text is, when NewsPlease is not sufficient, see below
"""
e = Extractor.from_yaml_file('../yml_templates/text_nakedsec.yml')
user_agent = 'projet recherche datasophia'
headers = {'User-Agent': user_agent}"""
class TheFriendlyNeighbourhoodSpider(CrawlSpider):
    """
    The class of the scrapper, 
    """
    name = 'TheFriendlyNeighbourhoodSpider'
    allowed_domains = ['helpnetsecurity.com']
    start_urls = ['https://www.helpnetsecurity.com/2020/12/24/us-cybersecurity-2021-challenges/']
    custom_settings = {'LOG_LEVEL': 'INFO'}
    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )
    def parse_item(self, response):#ADAPT FOR EACH WEBSITE
        """
        This script runs each time the scrapper meets a new webpage
        """
        link = response.url
        print('lien :', link) 
        article = NewsPlease.from_url(link)
        filename = '../storage/embedding/tout_2.txt'
        with open(filename, 'a') as f:
            m = article.maintext
            f.write('\n\n'+m)