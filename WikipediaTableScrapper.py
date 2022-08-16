import re
from typing import List

import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup, Tag
from yachalk import chalk


class WikiTable:
    def __init__(self, url: str, table_title: str):
        print(chalk.blue('Initialising Wiki'))
        self.url = url
        self.__only_char_regex = r'[a-zA-Z]+'
        self.__square_brackets_regex = r'\[.*?\]'
        self.table_title = ' '.join(re.findall(self.__only_char_regex, table_title, flags=re.IGNORECASE))
        self.__file_name = '_'.join(self.table_title.split(" "))
        self.__regex = re.compile(self.table_title, re.IGNORECASE)
        self.required_columns = ['Date', 'Event']
        self.optional_columns = ['Magnitude', 'Location']
        self.__df: pd.DataFrame = None
        self.nlp: spacy.Language = spacy.load('en_core_web_lg')
        self.__fetch_my_table()

    def is_this_my_table(self, tag: Tag, tag_to_search_for='h2') -> bool:
        print(chalk.yellow_bright(f'Searching for {tag_to_search_for} in the table'))
        # first check if the table has caption
        caption = tag.select('caption')

        if len(caption) >= 1:
            text = caption[0].getText()
            if len(re.findall(self.__regex, text)) >= 1:
                return True

        # first search for h2
        previous_sibling = tag.find_previous_sibling(tag_to_search_for)
        if previous_sibling is not None:
            text = previous_sibling.getText()
            if len(re.findall(self.__regex, text)) >= 1:
                return True
            else:
                return self.is_this_my_table(tag, 'h3')
        elif tag_to_search_for != 'h3':
            return self.is_this_my_table(tag, 'h3')
        return False

    def __fetch_my_table(self):
        html_doc = requests.get(self.url).text
        soup = BeautifulSoup(html_doc, features="lxml")
        all_tables = soup.select('table.wikitable')
        table_to_parse = None
        for table in all_tables:
            check_if_this_is_the_table = self.is_this_my_table(table)
            if check_if_this_is_the_table:
                table_to_parse = table
                break
        if table_to_parse:
            print(chalk.green(f'table found {self.__file_name}'))
            df = pd.read_html(str(table_to_parse))[0]
            print(df.columns)
            # clean up column names
            df = df.set_axis(self.clean_my_column_name(df.columns.tolist()), axis=1)
            self.__df = df
            column_mapping = self.__get_column_mapping()
            df = df.loc[:, column_mapping]
            self.__df = df
            # finally process the columns

        else:
            print(chalk.red('No table found'))

    def clean_my_column_name(self, columns: List[str]) -> List[str]:
        return [re.split(self.__square_brackets_regex, column.strip(), flags=re.IGNORECASE)[0].strip() for column in
                columns]

    def __process_my_columns(self):
        # check if the date is a range column
        pass
        # process date column
        # clean magnitude column and remove any textual data


    def __get_column_mapping(self):
        columns_to_filter = []
        print(chalk.blue(
            'These are required columns and need to be entered. If a column is suggested then it can be left blank'))
        for required_column in self.required_columns:
            suggested_column = self.__get_column_suggestion(required_column)
            required_column = 'Date, containing year' if required_column == 'Date' else required_column
            while True:
                suggested_string = f'Suggested column is {suggested_column}' if suggested_column else ''
                column_name = str(input(
                    f'Please enter the column for {required_column}.{suggested_string}') or suggested_column)
                if column_name:
                    columns_to_filter.append(column_name)
                    break

        print(chalk.blue(
            'These are optional columns and can be left blank. These columns will be mainly used for annotation'))
        for optional_column in self.optional_columns:
            suggested_column = self.__get_column_suggestion(optional_column)
            while True:
                suggested_string = f'Suggested column is {suggested_column}' if suggested_column else ''
                column_name = str(input(
                    f'Please enter the column for {optional_column}.{suggested_string}') or suggested_column)
                if column_name:
                    columns_to_filter.append(column_name)
                break
        return columns_to_filter

    def __get_column_suggestion(self, column_name: str) -> str:
        columns = self.__df.columns
        token_1 = self.nlp(column_name.lower().strip())
        similarity_object = [(column, token_1.similarity(self.nlp(str(column).strip().lower()))) for column in
                             columns if token_1.similarity(self.nlp(str(column).strip().lower())) > 0.6]
        if len(similarity_object) >= 1:
            sorted_similarity = sorted(similarity_object, key=lambda x: x[1], reverse=True)
            return sorted_similarity[0][0]

        return ''


if __name__ == '__main__':
    url = "https://en.wikipedia.org/wiki/Lists_of_earthquakes"
    heading_title = 'Deadliest earthquakes by year'
    wt = WikiTable(url, heading_title)
