from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class Crawling_Image():
    def __init__(self):
        pass

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    def set_chrome_driver(self):
        chrome_options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        return driver

    def crawling(self):
        keyword = 'cat'
        self. createFolder('./' + keyword )
        driver = self.set_chrome_driver()
        driver.implicitly_wait(3)

        # =============================================================================
        # 구글 이미지 검색 접속 및 검색어 입력
        # =============================================================================
        print(keyword, '검색')
        driver.get('https://www.google.co.kr/imghp?hl=ko')

        Keyword = driver.find_element(By.XPATH,'//*[@id="sbtc"]/div/div[2]/input')
        Keyword.send_keys(keyword)

        driver.find_element(By.XPATH,'//*[@id="sbtc"]/button').click()

        # =============================================================================
        # 스크롤
        # =============================================================================
        print(keyword + ' 스크롤 중 .............')
        elem = driver.find_element(By.TAG_NAME,"body")
        for i in range(60):
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.1)

        try:
            driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div[1]/div[4]/div[2]/input').click()
            for i in range(60):
                elem.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.1)
        except:
            pass

        # =============================================================================
        # 이미지 개수
        # =============================================================================
        links = []
        images = driver.find_elements(By.CSS_SELECTOR,"img.rg_i.Q4LuWd")
        for image in images:
            if image.get_attribute('src') != None:
                links.append(image.get_attribute('src'))

        print(keyword + ' 찾은 이미지 개수:', len(links))
        time.sleep(2)

        # =============================================================================
        # 이미지 다운로드
        # =============================================================================
        for k, i in enumerate(links):
            url = i
            start = time.time()
            urllib.request.urlretrieve(url, "./" + keyword + "_img_download/" + keyword + "_" + str(k) + ".jpg")
            print(str(k + 1) + '/' + str(len(links)) + ' ' + keyword + ' 다운로드 중....... Download time : ' + str(
                time.time() - start)[:5] + ' 초')
        print(keyword + ' ---다운로드 완료---')
        driver.close()



if __name__ == '__main__':
    Crawling_Image().crawling()