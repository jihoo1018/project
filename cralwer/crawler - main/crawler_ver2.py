from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.request
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
# 프로그램을 잠깐 멈추게 하기위한 라이브러리
import time
# url로 이미지를 다운받기 위한 라이브러리
import urllib.request
class Crawler():
    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)

    def crawler(self):
        keyword = '고양이'
        self.createFolder('./' + keyword)
        chrome_options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        # driver로 해당 페이지로 이동 : 구글 이미지로 이동
        driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")

        # 검색창 element 찾기 / 구글 이미지 input name = q
        elem = driver.find_element(By.NAME, "q")


        # 원하는 값 입력
        elem.send_keys(keyword)

        # 입력한 값 전송
        elem.send_keys(Keys.RETURN)

        SCROLL_PAUSE_TIME = 1

        # Get scroll height
        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            # Scroll down to bottom
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait to load page
            time.sleep(SCROLL_PAUSE_TIME)

            # Calculate new scroll height and compare with last scroll height
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                try:
                    driver.find_element(By.CSS_SELECTOR,".my34qd").click()
                except: break
            last_height = new_height

        # 내가 필요한 요소 선택 : 검색한 미리보기 이미지
        images = driver.find_elements(By.CSS_SELECTOR,"img.rg_i.Q4LuWd")
        count = 1
        #반복문으로 이미지요소 배열들 돌며 작업
        links = []
        for image in images:
            if image.get_attribute('src') != None:
                links.append(image.get_attribute('src'))

        print(keyword + ' 찾은 이미지 개수:', len(links))
        time.sleep(2)

        # 이미지를 url로 다운받는다.
        for k, i in enumerate(links):
            url = i
            start = time.time()
            urllib.request.urlretrieve(url, "./" + keyword + "_" + str(k) + ".jpg")
            print(str(k + 1) + '/' + str(len(links)) + ' ' + keyword + ' 다운로드 중....... Download time : ' + str(
                time.time() - start)[:5] + ' 초')
        print(keyword + ' ---다운로드 완료---')
        driver.close()


if __name__ == '__main__':
    Crawler().crawler()