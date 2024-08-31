import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Function to scrape all links from a given URL
def scrape_links(url):
    # Send a GET request to the webpage
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve content from {url}")
        return []

    # Parse the content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all anchor tags
    anchor_tags = soup.find_all('a')

    # Extract the href attributes and convert them to absolute URLs
    links = [
        urljoin(url, tag.get('href')) for tag in anchor_tags
        if tag.get('href') and urljoin(url, tag.get('href')).startswith(url)
    ]
    return links