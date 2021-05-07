import pandas as pd
import numpy as np
import pickle
import partition
import requests
import os
import socket
import re 
import whois
import string
import math

from urllib.parse import urlparse, parse_qs

from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
from ip2geotools.databases.noncommercial import DbIpCity
from bs4 import BeautifulSoup
from tld import get_tld

class Preprocessing:
    ### Helping functions
    # Convert a multiple categorical to ONE-HOT 
    def to_one_hot(self, df, col:str, exclude:set=None):
        if exclude is None:
            exclude = set()

        cs = set(df[col])
        for c in cs - exclude:
            df[c.lower()] = df[col] == c
            df[c.lower()] = df[c.lower()].astype(int)
        del df[col]

    # Convert a binary categorical to ONE-HOT
    def to_num(self, df, new_name:str, col_name:str, value:str):
        df[new_name] = df[col_name] == value
        df[new_name] = df[new_name].astype(int)

    # Normalise column
    def norm_col(self, df, col:str):
        max_val = max(df[col])
        df[col] = df[col] / max_val
        df[col] = df[col].astype(int)


class DFPreprocessing: 
    def __init__(self, data_name:str, btrunc:int, atrunc:int):
        self.df = pd.read_csv(data_name)
        self.df = self.df[['url','ip_add','geo_loc','url_len','js_len','js_obf_len','tld','who_is','https','content','label']]
        self.df = self.df.truncate(before=btrunc, after=atrunc)
    
    # URL preprocessing
    def get_parsed_url (self, row):
        return urlparse(row.url)

    def get_body_len (self, row):
        parsed = self.get_parsed_url(row)
        return len(parsed.netloc)

    def get_num_args (self, row):
        parsed = self.get_parsed_url(row)
        params = parse_qs(parsed.query)
        path_components = list(filter(bool, parsed.path.split('/')))
        return len(path_components) + len(list(params))
    
    # IP addresses preprocessing
    def address_to_numeric(self, row):
        nums = row.ip_add.split('.')
        address_num = 0
        for offset in range(0, 4):
            address_num += int(nums[offset]) << (8 * (3 - offset))
        return address_num

    def replace_countries (self):
        self.df.replace(to_replace='Congo Republic', value='Congo', inplace=True)
        self.df.replace(to_replace='DR Congo', value='Congo', inplace=True)
        self.df.replace(to_replace='U.S. Virgin Islands', value='United States', inplace=True)
        self.df.replace(to_replace='Åland', value='Unknown', inplace=True)
        self.df.replace(to_replace='Sint Maarten', value='Unknown', inplace=True)
        self.df.replace(to_replace='Bonaire, Sint Eustatius, and Saba', value='Unknown', inplace=True)
        self.df.replace(to_replace='Vatican City', value='Unknown', inplace=True)
        self.df.replace(to_replace='Kosovo', value='Unknown', inplace=True)
        self.df.replace(to_replace='East Timor', value='Unknown', inplace=True)
        self.df.replace(to_replace='St Kitts and Nevis', value='Unknown', inplace=True)
        self.df.replace(to_replace='French Southern Territories', value='Unknown', inplace=True)
        self.df.replace(to_replace='Saint Helena', value='Unknown', inplace=True)
        self.df.replace(to_replace='Antarctica', value='Unknown', inplace=True)

    def country2continent (self, row):
        if row.geo_loc != 'Unknown':
            cn_a2_code = country_name_to_country_alpha2(row.geo_loc)
            cn_continent = country_alpha2_to_continent_code(cn_a2_code)
            return cn_continent
        return row.geo_loc

    def sep(self):
        x = self.df.copy()
        del x['label']
        y = pd.DataFrame(self.df['label'])
        return x, y


    def preprocess (self, save_name):
        self.df['url_body_len'] = self.df.apply(self.get_body_len, axis=1)

        self.df['url_num_args'] = self.df.apply(self.get_num_args, axis=1)

        del self.df['url']

        self.df['numeric_address'] = self.df.apply(self.address_to_numeric, axis=1)

        for bit in range(0, 32):
            self.df[f'address_bit{bit}'] = self.df['numeric_address'] & (1 << bit) != 0
            self.df[f'address_bit{bit}'] = self.df[f'address_bit{bit}'].astype(int)
        
        del self.df['ip_add'], self.df['numeric_address']

        self.replace_countries()

        self.df['con_loc'] = self.df.apply(self.country2continent, axis=1)
        del self.df['geo_loc']

        Preprocessing().to_one_hot(self.df, 'con_loc', {'?'})

        for col in self.df.columns:
            if col == 'unknown':
                del self.df['unknown']

        Preprocessing().to_num(self.df, 'is_com', 'tld', 'com')
        del self.df['tld']

        Preprocessing().to_num(self.df, 'who_is', 'who_is', 'complete')

        Preprocessing().to_num(self.df, 'https', 'https', 'yes')

        Preprocessing().to_num(self.df, 'label', 'label', 'bad')

        del self.df['content'], self.df['js_obf_len']

        # Normalize every column
        for col in self.df:
            self.df[col] = self.df[col].astype(float)
            self.df[col] = (self.df[col] / max(abs(self.df[col])))
        self.df = self.df.fillna(0.0)
        
        print(self.df.columns)
        print(self.df.head())

        # Separate x and y values
        x, y = self.sep()

        # Export data
        pair = x, y

        dataset_name = '../Datasets/' + save_name
        with open(dataset_name, 'wb') as f:
            pickle.dump(pair, f)


class SamplePreprocessing:
    def __init__(self, url:str):
        self.url = url
    
    # IP address 
    def get_ip(self):
        p_url = urlparse(self.url).netloc
        return socket.gethostbyname(p_url) 

    def tld (self):
        return get_tld(self.url, fail_silently=True)

    # HTTPS
    def get_parsed_url (self):
        return urlparse(self.url)

    def is_http (self):
        parsed = self.get_parsed_url()
        return parsed.scheme

    # Continent
    def country2continent (self, country):
        if country != 'Unknown':
            return country_alpha2_to_continent_code(country)
        return country

    def get_continent(self):
        try: 
            response = DbIpCity.get(self.ip_addr, api_key='free')
            return country2continent(response.country)
        except: 
            return 'unknown'
    
    # Content
    def get_content(self):
        try:
            r = requests.get(self.url)
            soup = BeautifulSoup(r.content, 'html.parser')
            [s.extract() for s in soup(['iframe', 'script', 'style'])]
            raw_text = soup.text
            raw_text.translate(str.maketrans('', '', string.punctuation))
            raw_text = raw_text.lower()
            raw_text = raw_text.replace("\n", "")
            return raw_text
        except:
            return "ERROR"

    # Filtered content length in KBytes
    def get_js_len_inKB(self):
        if self.content == "ERROR":
            return 10000000
        js = ''.join(str(j) for j in self.content)
        return len(js.encode('utf-8'))/1000

    # Entropy
    def get_entropy(self):
        string = self.url.strip()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
        return entropy

    # URL length
    def get_url_len (self):
        return len(self.url) 

    # URL body or host length --> DNS length
    def get_body_len (self):
        parsed = self.get_parsed_url()
        return len(parsed.netloc)

    # URL number of attributes
    def get_num_args (self):
        parsed = self.get_parsed_url()
        params = parse_qs(parsed.query)
        path_components = list(filter(bool, parsed.path.split('/')))
        return len(path_components) + len(list(params))
    
    # URL path length
    def get_path_len (self):
        parsed = self.get_parsed_url()
        return len(parsed.path)

    # URL letters length
    def get_letter_count (self):
        letters = 0
        for i in self.url:
            if i.isalpha():
                letters = letters + 1
        return letters

    # URL digits length (numbers and symbols)
    def get_digits_len (self):
        digits = 0 
        for i in self.url:
            if i.isnumeric():
                digits = digits + 1
            if i in string.punctuation:
                digits = digits + 1
        return digits

    # Convert IP address to numeric 
    def address_to_numeric(self):
        nums = self.ip.split('.')
        address_num = 0
        for offset in range(0, 4):
            address_num += int(nums[offset]) << (8 * (3 - offset))
        return address_num

    # Main preprocessing
    def preprocess(self):

        self.ip = self.get_ip()

        self.df = pd.DataFrame(data={'url': [self.url], 'ip_addr': [self.ip]})

        self.df['tld'] = self.tld()

        Preprocessing().to_num(self.df, 'is_com', 'tld', 'com')

        self.df['https'] = self.is_http()

        Preprocessing().to_num(self.df, 'https', 'https', 'https') 

        con_loc = self.get_continent()

        self.df['continent'] = con_loc

        for i in ['na', 'eu', 'oc', 'af', 'as', 'sa', 'unknown']:
            self.df[i] = 0

        Preprocessing().to_one_hot(self.df, 'continent', {'?'})

        self.content = self.get_content() 

        self.df['content'] = self.content

        self.df['js_len'] = self.get_js_len_inKB()

        self.df['entropy'] = self.get_entropy()

        self.df['url_len'] = self.get_url_len()

        self.df['url_body_len'] = self.get_body_len()

        self.df['url_num_args'] = self.get_num_args()

        self.df['url_path_len'] = self.get_path_len()

        self.df['url_letters_len'] = self.get_letter_count()

        self.df['url_digit_len'] = self.get_digits_len()

        self.df['numeric_address'] = self.address_to_numeric()

        for bit in range(0, 32):
            self.df[f'address_bit{bit}'] = self.df['numeric_address'] & (1 << bit) != 0
            self.df[f'address_bit{bit}'] = self.df[f'address_bit{bit}'].astype(int)

        # Delete unused DF columns
        print(self.df['numeric_address'])
        del self.df['url'], self.df['ip_add'], self.df['tld'], self.df['numeric_address'], self.df['content']

        # Normalize every column --> FALTA CAMBIAR ESTO
        for col in self.df: 
            self.df[col] = self.df[col].astype(float)
            if col == 'js_len':
                self.df[col] = np.minimum(self.df[col] / 855, 1)
            elif col == 'url_len':
                self.df[col] = np.minimum(self.df[col] / 721, 1)
            elif col == 'url_body_len':
                self.df[col] = np.minimum(self.df[col] / 71, 1)
            elif col == 'url_num_args':
                self.df[col] = np.minimum(self.df[col] / 20, 1)
        
        self.df = self.df.fillna(0.0)
        self.df = self.df.to_numpy()
        print(self.df[0])
        return self.df

# DFPreprocessing("../Datasets/Webpages_Classification_test_data.csv", 10000, 360000).preprocess("test_dataset.pickle")
# DFPreprocessing("../Datasets/Webpages_Classification_train_data.csv", 1, 1200000).preprocess("train_dataset.pickle")
# DFPreprocessing("../Datasets/Webpages_Classification_test_data.csv", 1, 10000).preprocess("val_dataset.pickle")

# SamplePreprocessing("http://www.collectiblejewels.com").preprocess()