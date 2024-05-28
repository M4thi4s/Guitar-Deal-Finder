// Import required modules
import { JSDOM } from 'jsdom';
import fetch from 'node-fetch';
import { setTimeout } from 'timers/promises';
import fs from 'fs';
import csvWriter from 'csv-writer';

// Function to generate URL
function generateUrl(page) {
    return `https://www.dba.dk/musikinstrumenter/musikinstrumenter/el-og-halvakustiske-guitarer/produkt-elguitar/reg-koebenhavn-og-omegn/side-${page}/?pris=(-3000)`;
}

// Array to store data
let data = [];

// Function to clean text by removing special characters
function cleanText(text) {
    return text.replace(/[^\w\s]|[\r\n]/gi, '').replace(/\s+/g, ' ');
}

// Function to parse page and extract data
function parsePage(document) {
    const listings = document.querySelectorAll('.dbaListing');
    listings.forEach(listing => {
        const price = cleanText(listing.querySelector('.price').textContent);
        const location = listing.querySelector('.listingLink').href;
        data.push({ price, location });
    });
}

// Function to fetch and parse page
async function fetchAndParse(url) {
    const response = await fetch(url);
    console.log(response.status, response.statusText);
    const text = await response.text();
    const dom = new JSDOM(text);
    return dom.window.document;
}

// Function to fetch and parse description from detail page
async function fetchSpecifiqInfo(url) {
    const response = await fetch(url);
    console.log(response.status, response.statusText);
    const text = await response.text();
    const dom = new JSDOM(text);
    var description = cleanText(dom.window.document.querySelector('.vip-additional-text').textContent.trim());
    var asking = cleanText(dom.window.document.querySelector('.price-tag').textContent.trim());

    console.log(url)

    if(dom.window.document.querySelector('.vip-matrix-data dd:nth-child(2)').textContent != 'Elguitar')
        return null;

    var extraInfoDom = dom.window.document.querySelector('.vip-matrix-data dl').children;
    var extraInfo = {};
    for(var i=0; i<extraInfoDom.length; i+=2) {
        extraInfo[extraInfoDom[i].textContent] = extraInfoDom[i+1].textContent;
    }
    
    var brand = extraInfo['Mærke'];
    var model = extraInfo['Model'];

    if(model == null)
        return null;

    return { description, asking, brand, model };
}

// Main function
(async function() {
    // Fetch and parse first page
    const firstPageDocument = await fetchAndParse(generateUrl(1));
    const pageLinks = firstPageDocument.querySelectorAll("li .a-page-link");
    const maxPage = Math.max(...Array.from(pageLinks, link => parseInt(link.textContent)).filter(nb => !isNaN(nb)));

    // Fetch and parse all pages
    for (let i = 1; i <= maxPage; i++) {
        const document = await fetchAndParse(generateUrl(i));
        parsePage(document);
        await setTimeout(500);
        console.log(`Page ${i}/${maxPage} fetched`);
    }

    // Fetch descriptions for each listing
    for (let i = 0; i < data.length; i++) {
        let datas = await fetchSpecifiqInfo(data[i].location);
        if (datas == null) {
            data.splice(i, 1);
            i--; // Adjust index after removal
            continue;
        }
    
        data[i].description = datas.description;
        data[i].asking = datas.asking;
        data[i].brand = datas.brand;
        data[i].model = datas.model;
    
        console.log(`Info ${i + 1}/${data.length} fetched, ${data[i].asking} - ${data[i].brand} - ${data[i].model}`);
        //await setTimeout(200);
    }

    // Write data to CSV file
    const csv = csvWriter.createObjectCsvWriter({
        path: 'data.csv',
        header: [
            { id: 'price', title: 'Price' },
            { id: 'location', title: 'Location' },
            { id: 'description', title: 'Description' } ,
            { id: 'asking', title: 'Asking' },
            { id: 'brand', title: 'Brand' },
            { id: 'model', title: 'Model' }
        ]
    });

    await csv.writeRecords(data);
    console.log('Data written to data.csv');
})();