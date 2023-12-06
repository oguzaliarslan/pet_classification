import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import uuid
import re

def get_urls_cats() -> None:
    base_url = "https://www.petcim.com/"
    all_urls = []  

    for i in tqdm(range(2, 40), total=37):
        url = f"https://www.petcim.com/sahibinden-satilik-kedi-ilanlar-40?syf={i}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        
        elements = soup.find_all("tr", attrs={"onclick": re.compile(r"location.href='(.*?)'")})
        
        urls = [base_url + re.search(r"location.href='(.*?)'", str(element)).group(1) for element in elements]
        
        all_urls.extend(urls)

        with open('extracted_urls.txt', 'a') as file:
            for url in urls:
                file.write(url + '\n')

    print("URLs extracted and saved to 'extracted_urls.txt'")

def get_images_cats() -> None:
    directory = "images"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }

    with open('extracted_urls.txt', 'r') as file:
        urls = file.readlines()

    for url in tqdm(urls, total=(len(urls))):
        try:
            url = url.strip()
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            divs_row = soup.find_all("div", class_="satir")
            # hardcoded approach, html structure was a mess.
            last_div = divs_row[-1].find("span").text.strip()
            breed_folder = os.path.join(directory, last_div.lower().replace(' ', '_'))
            if not os.path.exists(breed_folder):
                os.makedirs(breed_folder)

            og_image_tags = soup.find_all("meta", {"name": "og:image"})
            for _, img_tag in enumerate(og_image_tags):
                image_url = img_tag.get("content")
                image = requests.get(image_url)
                img_extension = image_url.split('.')[-1]
                # randomly generated unique identifier
                img_name = f"{uuid.uuid4()}.{img_extension}" 
                img_path = os.path.join(breed_folder, img_name)

                with open(img_path, 'wb') as file:
                    file.write(image.content)
        except Exception as e:
            print(f'Error occurred: {e}. Skipping...')
            pass


def get_urls_dogs() -> None:
    base_url = "https://www.petcim.com/"
    all_urls = []  

    for i in tqdm(range(2, 40), total=37):
        url = f"https://www.petcim.com/sahibinden-satilik-kopek-ilanlar-41?syf={i}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        
        elements = soup.find_all("tr", attrs={"onclick": re.compile(r"location.href='(.*?)'")})
        
        urls = [base_url + re.search(r"location.href='(.*?)'", str(element)).group(1) for element in elements]
        
        all_urls.extend(urls)

        with open('extracted_urls_kopek.txt', 'a') as file:
            for url in urls:
                file.write(url + '\n')

    print("URLs extracted and saved to 'extracted_urls_kopek.txt'")

def get_images_dogs() -> None:
    directory = "images_kopek"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }

    with open('extracted_urls_kopek.txt', 'r') as file:
        urls = file.readlines()

    for url in tqdm(urls, total=(len(urls))):
        try:
            url = url.strip()
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")
            divs_row = soup.find_all("div", class_="satir")
            # hardcoded approach, html structure was a mess.
            last_div = divs_row[-1].find("span").text.strip()
            breed_folder = os.path.join(directory, last_div.lower().replace(' ', '_'))
            if not os.path.exists(breed_folder):
                os.makedirs(breed_folder)

            og_image_tags = soup.find_all("meta", {"name": "og:image"})
            for _, img_tag in enumerate(og_image_tags):
                image_url = img_tag.get("content")
                image = requests.get(image_url)
                img_extension = image_url.split('.')[-1]
                # randomly generated unique identifier
                img_name = f"{uuid.uuid4()}.{img_extension}" 
                img_path = os.path.join(breed_folder, img_name)

                with open(img_path, 'wb') as file:
                    file.write(image.content)
        except Exception as e:
            print(f'Error occurred: {e}. Skipping...')
            pass

if __name__ == '__main__':
    print('Getting the cat images')
    get_urls_cats()
    get_images_cats()
    print('Getting the dog images')
    get_urls_dogs()
    get_images_dogs()