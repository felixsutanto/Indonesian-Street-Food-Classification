# src/data_scraper.py
import os
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import io

class GoogleImageScraper:
    def __init__(self, headless=True):
        """Initialize the scraper with robust Chrome driver options."""
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--log-level=3") # Suppress console noise
        self.chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        try:
            self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.chrome_options)
        except Exception as e:
            print(f"Error initializing WebDriver: {e}")
            print("Please ensure you have Google Chrome installed.")
            raise

    def _handle_consent(self):
        """Handle the cookie consent pop-up if it appears."""
        try:
            consent_button = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, '//button[.//div[contains(text(), "Accept all")]]'))
            )
            consent_button.click()
            print("  Accepted cookie consent.")
            time.sleep(1)
        except Exception:
            # print("  No cookie consent button found, or it was not clickable.")
            pass

    def scrape_images(self, search_term, category_name, num_images=120, output_dir="data/raw_images"):
        """Scrape images using updated selectors and explicit waits."""
        category_dir = os.path.join(output_dir, category_name)
        os.makedirs(category_dir, exist_ok=True)
        
        search_url = f"https://www.google.com/search?q={search_term.replace(' ', '+')}&tbm=isch"
        self.driver.get(search_url)
        self._handle_consent()

        print(f"Scrolling to load images for '{search_term}'...")
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        while scroll_attempts < 5:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                scroll_attempts += 1
            else:
                scroll_attempts = 0 # Reset if new content loaded
            last_height = new_height

        # UPDATED SELECTOR for image thumbnails
        image_thumbnails = self.driver.find_elements(By.CSS_SELECTOR, "div.H8Rx8c")
        print(f"Found {len(image_thumbnails)} potential images. Starting download process...")
        
        downloaded_count = 0
        for thumb in image_thumbnails:
            if downloaded_count >= num_images:
                break
            try:
                self.driver.execute_script("arguments[0].click();", thumb)
                time.sleep(1) # Wait for the high-res image to load
                
                # UPDATED SELECTOR for the high-resolution preview image
                actual_images = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "img.sFlh5c"))
                )
                
                for actual_image in actual_images:
                    src = actual_image.get_attribute('src')
                    if src and src.startswith('http'):
                        if self._download_image(src, category_dir, downloaded_count):
                            downloaded_count += 1
                            print(f"  Downloaded {downloaded_count}/{num_images} for {category_name}", end='\r')
                        # Once we get a valid src, break to move to the next thumbnail
                        break
            except Exception:
                continue
        
        print(f"\nSuccessfully downloaded {downloaded_count} images for '{category_name}'.")
        return downloaded_count

    def _download_image(self, url, save_dir, index):
        """Download a single image, validate it, and save."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            if img.width < 100 or img.height < 100: return False
            if img.mode != 'RGB': img = img.convert('RGB')
            
            filename = f"{index:04d}.jpg"
            img.save(os.path.join(save_dir, filename), 'JPEG', quality=95)
            return True
        except Exception:
            return False

    def close(self):
        self.driver.quit()

def main():
    """Main function to scrape all categories."""
    food_map = {
        "sate": "sate ayam madura",
        "bakso": "bakso kuah sapi",
        "martabak": "martabak manis terang bulan"
    }
    
    scraper = GoogleImageScraper(headless=True)
    try:
        for category, search_term in food_map.items():
            print(f"--- Scraping images for: {category} (using query: '{search_term}') ---")
            scraper.scrape_images(search_term, category, num_images=120)
            time.sleep(2)
    finally:
        scraper.close()
    print("\nData scraping completed!")

if __name__ == "__main__":
    main()

