import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':
    html_doc = requests.get('https://www.colorhexa.com/web-safe-colors').text
    soup = BeautifulSoup(html_doc, features="lxml")
    all_colors = soup.select('table.color-list td>a.tw, td>a.tb')
    only_colors = [color.text for color in all_colors]
    print(only_colors)

