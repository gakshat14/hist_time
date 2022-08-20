import re
from typing import List

import numpy as np
import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup, Tag
from yachalk import chalk


class WikiTable:
    MY_DATE_COLUMN = 'date'
    MY_EVENT_COLUMN = 'event'
    MY_MAGNITUDE_COLUMN = 'magnitude'
    MY_LOCATION_COLUMN = 'location'

    def __init__(self, url: str, table_title: str):
        print(chalk.blue('Initialising Wiki'))
        self.url = url
        self.__only_char_regex = r'[a-zA-Z]+'
        self.__square_brackets_regex = r'\[.*?\]'
        self.__numeric_regex = r'\d+\.\d+'
        self.table_title = ' '.join(re.findall(self.__only_char_regex, table_title, flags=re.IGNORECASE))
        self.__file_name = '_'.join(self.table_title.split(" "))
        self.__regex = re.compile(self.table_title, re.IGNORECASE)
        self.required_columns = [self.MY_DATE_COLUMN, self.MY_EVENT_COLUMN]
        self.optional_columns = ['magnitude', 'location']
        self.df: pd.DataFrame = None
        self.nlp: spacy.Language = spacy.load('en_core_web_lg')
        self.__fetch_my_table()
        self.__is_range = False

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
            self.df = df
            self.__get_column_mapping()
            # column_mapping = self.required_columns + self.optional_columns
            # df = df.loc[:, column_mapping]
            self.df = df
            # self.save_as_csv()
            # finally process the columns
            self.__process_my_columns()
            self.save_as_csv()

        else:
            print(chalk.red('No table found'))

    def clean_my_column_name(self, columns: List[str]) -> List[str]:
        return [re.split(self.__square_brackets_regex, column.strip(), flags=re.IGNORECASE)[0].strip() for column in
                columns]

    def __check_is_it_a_range(self):
        try:
            print(chalk.blue('Checking if it is a date range'))
            return self.df[self.df[self.MY_DATE_COLUMN].str.contains('–')].shape[0] >= self.df.shape[0] * 0.6
        except:
            return False

    def print_delete_message_date(self, df_null_dates):
        print(chalk.yellow_bright('Following rows will be removed as the date is not in supported format.\n '
                                  'If you want all the rows to be rendered correctly please edit following rows and '
                                  'visualise it as an CSV.'))
        for index, row in df_null_dates.iterrows():
            print(f'Row with event name {row[self.MY_EVENT_COLUMN]}')

    def __process_date_range(self):
        # split the date
        df_date_split = self.df[self.MY_DATE_COLUMN].str.split('–')

        self.df['Start Date'] = pd.to_datetime([start[len(start) - 2] for start in df_date_split], errors='coerce')
        self.df['End Date'] = pd.to_datetime([end[len(end) - 2] for end in df_date_split], errors='coerce')

        # get all the dates
        df_null_dates = self.df[pd.isnull(self.df['Start Date'])]

        self.print_delete_message_date(df_null_dates)

        self.df = self.df.drop([index for index, value in df_null_dates.iterrows()])

    def __process_date(self):
        self.df[self.MY_DATE_COLUMN] = self.df[self.MY_DATE_COLUMN].astype(str)
        self.df[self.MY_DATE_COLUMN] = pd.to_datetime(self.df[self.MY_DATE_COLUMN], errors='coerce')

        print(self.df[self.MY_DATE_COLUMN])

        df_null_dates = self.df[pd.isnull(self.df[self.MY_DATE_COLUMN])]

        if len(df_null_dates) > 0:
            self.print_delete_message_date(df_null_dates)
            self.df = self.df.drop(df_null_dates)

    def __process_magnitude(self):
        print(self.df.columns)
        # first convert it into string
        self.df[self.MY_MAGNITUDE_COLUMN] = self.df[self.MY_MAGNITUDE_COLUMN].astype(str)

        # now extract all the numbers
        magnitude_values = self.df[self.MY_MAGNITUDE_COLUMN].tolist()
        only_numeric = [re.findall(self.__numeric_regex, value) for value in magnitude_values]
        # sum if we have 2 values together
        cleaned_magnitude_value = [sum(list((map(float, value)))) for value in only_numeric]

        self.df['magnitude_calc'] = cleaned_magnitude_value

        print(chalk.blue('Magnitude column processed successfully.'))

    def save_as_csv(self):
        self.df.to_csv(f'{self.__file_name}.csv', index=False)

    def __process_my_columns(self):
        # check if the date is a range column
        # process date column
        if self.__check_is_it_a_range():
            self.__process_date_range()
        else:
            self.__process_date()

        print(chalk.blue('Date column has been processed successfully'))
        print(self.df.columns)
        # clean magnitude column and remove any textual data
        not self.df[self.MY_MAGNITUDE_COLUMN].empty and self.__process_magnitude()

    def __get_column_mapping(self):
        columns_to_filter = []
        print(chalk.blue(
            'These are required columns and need to be entered. If a column is suggested, '
            'then please enter the suggested column'))
        print(chalk.yellow_bright('These cant be left blank'))
        for required_column in self.required_columns:
            suggested_column = self.__get_column_suggestion(required_column)
            column_string = 'Date, containing year' if required_column == 'date' else required_column
            while True:
                suggested_string = f'Suggested column is {suggested_column}' if suggested_column else ''
                column_name = str(input(
                    f'Please enter the column for {column_string}.{suggested_string}\n'))
                if column_name:
                    self.df[required_column] = self.df[column_name]
                    print(self.df.columns)
                    break

        print(chalk.blue(
            'These are optional columns and can be left blank. These columns will be mainly used for annotation'))
        for optional_column in self.optional_columns:
            suggested_column = self.__get_column_suggestion(optional_column)
            while True:
                suggested_string = f'Suggested column is {suggested_column}' if suggested_column else ''
                column_name = str(input(
                    f'Please enter the column for {optional_column}.{suggested_string}\n'))
                if column_name:
                    self.df[optional_column] = self.df[column_name]
                break

    def __get_column_suggestion(self, column_name: str) -> str:
        columns = self.df.columns
        token_1 = self.nlp(column_name.lower().strip())
        similarity_object = [(column, token_1.similarity(self.nlp(str(column).strip().lower()))) for column in
                             columns if token_1.similarity(self.nlp(str(column).strip().lower())) > 0.6]
        if len(similarity_object) >= 1:
            sorted_similarity = sorted(similarity_object, key=lambda x: x[1], reverse=True)
            return sorted_similarity[0][0]

        return ''


if __name__ == '__main__':
    url = "https://en.wikipedia.org/wiki/List_of_recessions_in_the_United_States"
    heading_title = 'Free Banking Era to the Great Depression'
    wt = WikiTable(url, heading_title)
