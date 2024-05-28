# Guitar Deal Finder

## Project Overview
The Guitar Deal Finder is a tool designed to identify the best deals on guitars for purchase. This project is structured into three main components:

1. `/scrap`: Scrapes guitar sale listings from the website dba.dk.
2. `/adapt`: Extracts relevant information from the listings for price prediction.
3. `/predict`: Predicts the selling price of a guitar based on the information obtained from the `/adapt` module.

### Scrap
The `/scrap` module utilizes JavaScript to fetch and parse data from dba.dk. It extracts details like price and location from the listings and additional information from each listing's detail page. This data is then written to a CSV file for further processing.

### Adapt
The `/adapt` module uses Python with Pandas, NumPy, and the Transformers library from Hugging Face to perform Named Entity Recognition (NER) on the descriptions extracted by the `/scrap` module. It identifies and validates guitar brands and model names, then augments the dataset with these details.

### Predict
The `/predict` module employs a machine learning model to estimate the selling price of a guitar based on its features extracted and processed in previous steps. It uses a pre-trained model loaded from a file to make predictions on new data.

## Installation and Usage
Ensure you have Node.js and Python installed on your system. Clone the repository, navigate to the respective module directories, and follow the instructions in the README files for each module to set up and run the scripts.

## License
This project is open-sourced under the MIT License. However, the author disclaims any liability for the use of this software for purposes not permitted by the original data sources or for any errors that may be present in the code. Users must ensure they comply with the terms of service of any websites they scrape using this tool.

**Note**: The use of this script on websites that do not permit scraping is not endorsed by the author, and users are responsible for any legal implications.