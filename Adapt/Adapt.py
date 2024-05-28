import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

# Create a pipeline for named entity recognition
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Load your dataset
df = pd.read_csv('../Scrap/data.csv')

# Load valid brands and names
with open('../IA/data/category_values_df-pkl/Brand.txt') as f:
    valid_brands = [line.strip() for line in f]
    
# Format valid_brands in a dictionary {FormatBrand: Brand, etc..}
valid_brands = {re.sub(r'\W+', '', brand.lower()): brand for brand in valid_brands}

with open('../IA/data/category_values_df-pkl/Name.txt') as f:
    valid_names = [line.strip() for line in f]

valid_names = {re.sub(r'\W+', '', name.lower()): name for name in valid_names}

def match_valid_entity(entity, valid_list):
    for valid_entity in valid_list:
        if re.search(r'\b' + re.escape(entity) + r'\b', valid_entity, re.IGNORECASE):
            return valid_entity
    return 'Unknown'

import re

def isBrandInBrandList(key):
    formatted_key = re.sub(r'\W+', '', key.lower())
    
    for brand in valid_brands.keys():
        if brand.find(formatted_key) != -1 or formatted_key.find(brand) != -1:
            return valid_brands[brand]
    
    return False
    
def extract_brand_and_name(brand, description):
    entities = ner_pipeline(description)
    brand_extracted = "Unknown"
    name = "Unknown"
    for entity in entities:
        entity_type = entity.get('entity_group') or entity.get('entity')        
         
        _isBrandInBrandList = isBrandInBrandList(brand)
        if _isBrandInBrandList != False:
            brand_extracted = _isBrandInBrandList
        else :
            if entity_type in ['I-ORG', 'B-ORG']:  # Assuming ORG represents brands
                extracted_brand = entity['word']
                brand_extracted = match_valid_entity(extracted_brand, valid_brands.values())
            
        if entity_type in ['I-MISC', 'B-MISC']:  # Assuming MISC represents names
            extracted_name = entity['word']
            matched_name = match_valid_entity(extracted_name, valid_names.values())
            if matched_name != 'Unknown':
                name = matched_name
                
    return brand_extracted, name

# Create new columns for the brand and name
df['Brand_Before'] = df['Brand']
df['Brand'] = 'Unknown'
df['Name'] = 'Unknown'

def process_row(index, row):
    extracted_brand_category, name = extract_brand_and_name(row['Brand_Before'], row['Description'])
    print("Brand: ", extracted_brand_category, " <=> ", row['Brand_Before'])
    print(f"{index}/{len(df)} -> Extracted brand: {extracted_brand_category}, name: {name}")
    return index, extracted_brand_category, name

# Use ThreadPoolExecutor to parallelize the entity extraction
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_row, index, row): index for index, row in df.iterrows()}
    for future in futures:
        index, extracted_brand_category, name = future.result()
        df.at[index, 'Brand'] = extracted_brand_category
        df.at[index, 'Name'] = name

# Define the conversion rate
DKK_TO_USD = 0.15

# Convert Prices to USD
df['Asking'] = df['Asking'].str.replace(' kr', '').astype(float)
df['Asking'] = df['Asking'] * DKK_TO_USD

# Transform DataFrame
#transformed_df = df[['Asking', 'Brand', 'Name', '', '']]

# Save the transformed DataFrame
#transformed_df.to_csv('output/transformed_data_with_ner.csv', index=False)
#print("Transformation complete. Data saved to 'transformed_data_with_ner.csv'")

# Save in a new file datas where the Brand and Name are not Unknown
transformed_filter_df = df[(df['Brand'] != 'Unknown') & (df['Name'] != 'Unknown')]
transformed_filter_df.to_csv('output/transformed_data_with_ner_not_unknown.csv', index=False)