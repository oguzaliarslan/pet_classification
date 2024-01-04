import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import uuid
import re
from PIL import Image
def count_images_in_folder(folder_path: str) -> int:
    image_extensions = ['png', 'jpg', 'jpeg', 'gif']
    count = sum(1 for file in os.listdir(folder_path) if file.split('.')[-1].lower() in image_extensions)
    return count

def get_breed_urls_cat() -> None:

    base_url = "https://www.petcim.com/sahibinden-satilik-kedi-ilanlar-40"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    elements = soup.find("div", {'class': 'alts'})

    list_items = elements.find_all('li')

    with open('extracted_breeds_cat.txt', 'a') as file:
        for item in list_items:
            href_value = item.find('a')['href']
            full_url = f"https://petcim.com/{href_value}"
            file.write(full_url + '\n')


def get_urls_cats(min, max) -> None:
    base_url = "https://www.petcim.com/"
    all_urls = []  
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }
    with open('extracted_breeds_cat.txt', 'r') as file:
        urls = file.readlines()
    
    for url in urls:
        url = url.strip()
        for page_num in tqdm(range(min, max), desc=f"Scraping {url}"):
            page_url = f"{url}?syf={page_num}"
            try:
                response = requests.get(page_url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                
                elements = soup.find_all("tr", attrs={"onclick": re.compile(r"location.href='(.*?)'")})
                
                scraped_urls = [base_url + re.search(r"location.href='(.*?)'", str(element)).group(1) for element in elements]
                
                all_urls.extend(scraped_urls)

                with open('extracted_urls_cats.txt', 'a') as file:
                    for scraped_url in scraped_urls:
                        file.write(scraped_url + '\n')
            except requests.HTTPError as e:
                print(f"Error accessing {page_url}: {e}")
                continue

    print("URLs extracted and saved to 'extracted_urls_cats.txt'")


def get_images_cats(image_limit=100) -> None:
    directory = "images"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }

    with open('extracted_urls_cats.txt', 'r') as file:
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

                # number of img check
                images_count = count_images_in_folder(breed_folder)
                image_limit = image_limit

                if images_count >= image_limit:
                    print(f"Image limit reached for {breed_folder}. Skipping further images.")
                    break  # Break out of the loop once the image limit is reached

                with open(img_path, 'wb') as file:
                    file.write(image.content)

        except Exception as e:
            print(f'Error occurred: {e}. Skipping...')
            pass



def get_breed_dogs_cat() -> None:

    base_url = "https://www.petcim.com/sahibinden-satilik-kopek-ilanlar-41"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    elements = soup.find("div", {'class': 'alts'})

    list_items = elements.find_all('li')

    with open('extracted_breeds_dog.txt', 'a') as file:
        for item in list_items:
            href_value = item.find('a')['href']
            full_url = f"https://petcim.com/{href_value}"
            file.write(full_url + '\n')


def get_urls_dogs(min, max) -> None:
    base_url = "https://www.petcim.com/"
    all_urls = []  
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }
    with open('extracted_breeds_dog.txt', 'r') as file:
        urls = file.readlines()
    
    for url in urls:
        url = url.strip()
        for page_num in tqdm(range(min, max), desc=f"Scraping {url}"):
            page_url = f"{url}?syf={page_num}"
            try:
                response = requests.get(page_url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                
                elements = soup.find_all("tr", attrs={"onclick": re.compile(r"location.href='(.*?)'")})
                
                scraped_urls = [base_url + re.search(r"location.href='(.*?)'", str(element)).group(1) for element in elements]
                
                all_urls.extend(scraped_urls)

                with open('extracted_urls_dogs.txt', 'a') as file:
                    for scraped_url in scraped_urls:
                        file.write(scraped_url + '\n')
            except requests.HTTPError as e:
                print(f"Error accessing {page_url}: {e}")
                continue

    print("URLs extracted and saved to 'extracted_urls_dogs.txt'")


def get_images_dogs(image_limit=100) -> None:
    directory = "images_kopek"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }

    with open('extracted_urls_dogs.txt', 'r') as file:
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

                # number of img check
                images_count = count_images_in_folder(breed_folder)
                image_limit = image_limit

                if images_count < image_limit:
                    with open(img_path, 'wb') as file:
                        file.write(image.content)
                else:
                    print(f"Image limit reached for {breed_folder}. Skipping further images.")
                    break

        except Exception as e:
            print(f'Error occurred: {e}. Skipping...')
            pass

if __name__ == '__main__':
    print('Getting the cat images')
    get_breed_urls_cat()
    get_urls_cats(2,8)
    get_images_cats()
    print('Getting the dog images')
    get_breed_dogs_cat()
    get_urls_dogs(2,8)
    get_images_dogs()