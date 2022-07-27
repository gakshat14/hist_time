from bs4 import BeautifulSoup, Tag
import requests
import re
import pandas as pd


def get_previous_sibling_text(tag: Tag, tag_to_search_for='h2') -> str:
    # first search for h2
    previous_sibling = tag.find_previous_sibling(tag_to_search_for)
    if previous_sibling is None and tag_to_search_for != 'h3':
        # now search for h3
        return get_previous_sibling_text(tag, 'h3')
    if previous_sibling is None and tag_to_search_for == 'h3':
        return ''
    return previous_sibling.getText()


if __name__ == '__main__':
    url = "https://en.wikipedia.org/wiki/Lists_of_earthquakes"
    heading_title = 'Deadliest earthquakes by year '
    regex = re.compile(heading_title, re.IGNORECASE)
    html_doc = requests.get(url).text
    soup = BeautifulSoup(html_doc)
    all_tables = soup.select('table.sortable.wikitable')
    table_to_parse = None
    for table in all_tables:
        text = get_previous_sibling_text(table)
        print(text.lower())
        if len(re.findall(regex, text)) >= 1:
            table_to_parse = table
            break
    if table_to_parse:
        df = pd.read_html(str(table_to_parse))[0]
        df.to_csv('temp.csv', index=False)
    else:
        print('No table found')

