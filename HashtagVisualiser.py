import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import seaborn as sns
from textblob import TextBlob
from yachalk import chalk
from scipy.signal import find_peaks
import calendar

from Timeline import TimelineVisualiser
from Constants import ALL_COLORS

class HashtagVisualiser():
    def __init__(self, path_to_csv: str, tweet_key: str, date_time_key: str):
        print('Initializer called')
        self.df = pd.read_csv(path_to_csv)
        self.__date_time_key = date_time_key
        self.__tweets_key = tweet_key
        self.__hashtag_index = None
        self.__hashtag_color_mapping = {}
        self.__all_colors = ALL_COLORS
        self.process_my_df()

    def print_my_dataframe(self):
        print('print dataframe called')
        # if not self.df:
        #     raise ValueError('The specified dataframe does not exists')
        print(self.df)

    @staticmethod
    def __return_my_hashtag(series):
        content, date_time = series
        regex = r'#\w+'
        results = re.findall(regex, content)
        if len(results) > 0:
            return results
        return np.nan

    def __generate_hashtags(self):
        print(chalk.blue('Generating hashtag'))
        self.df['hashtags'] = self.df.apply(self.__return_my_hashtag, axis=1)
        self.df.dropna(axis=0, inplace=True)
        self.df = self.df.explode('hashtags')

    def color_my_hashcode(self, hashtag, count):
        color = ''
        if hashtag not in self.__hashtag_color_mapping:
            color = self.__all_colors[0]
            self.__hashtag_color_mapping[hashtag] = color
            del self.__all_colors[0]
            print(color)
            print(chalk.hex(color)(f'{hashtag}({count})'))
            return f"""{chalk.hex(color)(f'{hashtag}({count})')}"""
        else:
            print(chalk.hex(self.__hashtag_color_mapping[hashtag])(f'{hashtag}({count})'))
            return f"""{chalk.hex(self.__hashtag_color_mapping[hashtag])(f'{hashtag}({count})')}"""

    @staticmethod
    def __generate_sentiments(tweet):
        # remove all the RT
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        # remove hash signs
        tweet = re.sub(r'#', '', tweet)
        # remove mentions
        tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)
        # remove links
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        # remove punctuation
        tweet = re.sub(r'[' + string.punctuation + ']+', ' ', tweet)

        # finally get the sentiment score
        blob = TextBlob(tweet)
        return round(blob.sentiment.polarity, 2)

    def __get_sentiments(self):
        print(chalk.blue('Generating sentiment'))
        self.df['sentiment'] = self.df[self.__tweets_key].apply(self.__generate_sentiments)

    def process_my_df(self):
        self.df[self.__date_time_key] = pd.to_datetime(self.df[self.__date_time_key])
        self.df = self.df.drop(columns=self.df.columns.difference([self.__tweets_key, self.__date_time_key]))
        print(chalk.blue('Processing date to generate year, month and date'))
        self.__generate_hashtags()
        self.__get_sentiments()
        self.df['year'] = self.df[self.__date_time_key].dt.year
        self.df['month_year'] = self.df[self.__date_time_key].dt.to_period('M')
        self.df['date_only'] = self.df[self.__date_time_key].dt.date
        self.__hashtag_index = self.__generate_hashtag_index()
        self.df.drop(columns=self.__tweets_key, inplace=True)

    def __generate_hashtag_index(self):
        print('generating list of hashtags')
        count = 0
        hashtags = dict()

        def get_hashtag(series):
            nonlocal count
            regex = r'#\w+'
            results = re.findall(regex, series)
            for result in results:
                year = self.df.iloc[count].year
                if result not in hashtags:
                    hashtags[result] = {'index': [count], 'year': [year]}
                else:
                    hashtags[result]['index'].append(count)
                    hashtags[result]['year'].append(year)
            count += 1

        self.df[self.__tweets_key].map(get_hashtag)
        return hashtags

    def list_all_the_hashtags(self):
        print(self.df.hashtags.unique())

    def get_top_5_hashtags(self) -> list:
        sorted_hashtags = sorted(self.__hashtag_index,
                                 key=lambda hashtag: len(self.__hashtag_index[hashtag].get('index')),
                                 reverse=True)
        return sorted_hashtags[:5]

    def generate_sentiment_time_series(self, hashtag_to_plot):
        # temp_df = self.df[self.df.hashtags == hashtag]
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.lineplot(data=self.df[self.df.hashtags == hashtag_to_plot], x=self.__date_time_key, y='sentiment', ax=ax, ci=False)
        # ax.plot(temp_df[self.__date_time_key], np.zeros_like(temp_df[self.__date_time_key]))
        # ax.annotate('Neutral', xy=(1, 0.4), xytext=(1, 0.5))
        ax.set_xlabel('Date')
        ax.set_ylabel('Sentiment Score')
        plt.show()

    def generate_quantile_all(self):
        print(chalk.blue('generating grouped hashtags'))
        temp_df = self.df.copy()
        df_grouped = temp_df.loc[:, ['year', 'hashtags']].groupby(by='year').value_counts().to_frame()
        unique_year = temp_df.year.unique().tolist()
        year_quantile = {}
        for year in unique_year:
            year_quantile[year] = np.quantile(df_grouped.loc[year].values, 0.99)
        df_grouped.columns = ['count_value']
        year_event = {'year': [], 'event': []}
        for year in sorted(unique_year):
            print(year)
            temp_df = df_grouped.loc[year]
            events = [f'{key}({value.count_value})' for key, value in
                      temp_df[temp_df['count_value'] > year_quantile[year]][:10].iterrows()]
            if len(events) > 0:
                year_event['year'].append(year)
                year_event['event'].append(events)

        df_final = pd.DataFrame.from_dict(year_event)
        TimelineVisualiser(df_final, 'Usage of prominent hashtag over the years').create_my_timeline()

    def generate_quantile_specific_month(self, month=5):
        print(month)
        print('generating grouped hashtags')
        temp_df = self.df.copy()
        temp_df['month'] = temp_df.month_year.dt.month
        temp_df = temp_df[temp_df['month'] == month]
        df_grouped = temp_df.loc[:, ['month_year', 'hashtags']].groupby(by='month_year').value_counts().to_frame()
        unique_values = temp_df.month_year.unique().tolist()
        value_quantile = {}
        for value_month in unique_values:
            value_quantile[value_month] = np.quantile(df_grouped.loc[value_month].values, 0.99)
        df_grouped.columns = ['count_value']
        print(value_quantile)
        print(sorted(unique_values))
        value_event = {'month': [], 'event': []}
        for value in sorted(unique_values):
            print(value)
            temp_df = df_grouped.loc[value]
            events = [f'{key}({value.count_value})' for key, value in
                      temp_df[temp_df['count_value'] > value_quantile[value]][:10].iterrows()]
            if len(events) > 0:
                value_event['month'].append(value)
                value_event['event'].append(', '.join(events))
        df_final = pd.DataFrame.from_dict(value_event)
        TimelineVisualiser(df_final, f'Usage of prominent hashtag over {calendar.month_name[month]}',
                           timeline_key='month').create_my_timeline()

    def generate_quantile_specific_year(self, year=2015):
        print(year)
        print('generating grouped hashtags')
        temp_df = self.df.copy()
        temp_df['month'] = temp_df.month_year.dt.month
        temp_df = temp_df[temp_df['year'] == year]
        df_grouped = temp_df.loc[:, ['month', 'hashtags']].groupby(by='month').value_counts().to_frame()
        unique_values = temp_df.month.unique().tolist()
        value_quantile = {}
        for value_month in unique_values:
            value_quantile[value_month] = np.quantile(df_grouped.loc[value_month].values, 0.99)
        df_grouped.columns = ['count_value']
        print(value_quantile)
        print(sorted(unique_values))
        value_event = {'month': [], 'event': []}
        for value in sorted(unique_values):
            print(value)
            temp_df = df_grouped.loc[value]
            events = [f'{key}({value.count_value})' for key, value in
                      temp_df[temp_df['count_value'] > value_quantile[value]][:10].iterrows()]
            if len(events) > 0:
                value_event['month'].append(value)
                value_event['event'].append(', '.join(events))
        df_final = pd.DataFrame.from_dict(value_event)
        TimelineVisualiser(df_final, f'Usage of prominent hashtag over year {year}',
                           timeline_key='month').create_my_timeline()

    def generate_peak_based_timeline_all(self):
        temp_df = self.df.copy()
        df_grouped_2 = temp_df.loc[:, ['year', 'hashtags']].groupby(by=['hashtags', 'year']).value_counts().to_frame()
        df_grouped_2.columns = ['value_count']
        all_hashtags = temp_df.hashtags.unique().tolist()
        final_time_series_dict = {}
        for hash in all_hashtags:
            temp_df = df_grouped_2.loc[hash]
            peaks = find_peaks(temp_df.value_count)
            if len(peaks[0]) > 0:
                for key, value in temp_df.iloc[list(peaks[0])].iterrows():
                    if key not in final_time_series_dict:
                        final_time_series_dict[key] = [f'{hash}({value.value_count})']
                    else:
                        final_time_series_dict[key].append(f'{hash}({value.value_count})')

        final_df_dict = {'year': [], 'event': []}
        for key, items in final_time_series_dict.items():
            final_df_dict['year'].append(key)
            final_df_dict['event'].append(', '.join(
                sorted(items, key=lambda item: int(re.findall(r'[0-9]+', re.findall(r'\([0-9]+\)', item)[0])[0]),
                       reverse=True)[:10]))

        df_time_series = pd.DataFrame.from_dict(final_df_dict)
        TimelineVisualiser(df_time_series, 'Anything time series').create_my_timeline()

    def generate_peak_based_timeline_specific_month(self, month):
        temp_df = self.df.copy()
        temp_df['month'] = temp_df.month_year.dt.month
        temp_df = temp_df[temp_df['month'] == month]
        df_grouped_2 = temp_df.loc[:, ['month_year', 'hashtags']].groupby(by=['hashtags', 'month_year']).value_counts().to_frame()
        df_grouped_2.columns = ['value_count']
        all_hashtags = temp_df.hashtags.unique().tolist()
        final_time_series_dict = {}
        for hash in all_hashtags:
            temp_df = df_grouped_2.loc[hash]
            peaks = find_peaks(temp_df.value_count)
            if len(peaks[0]) > 0:
                for key, value in temp_df.iloc[list(peaks[0])].iterrows():
                    if key not in final_time_series_dict:
                        final_time_series_dict[key] = [f'{hash}({value.value_count})']
                    else:
                        final_time_series_dict[key].append(f'{hash}({value.value_count})')

        final_df_dict = {'month': [], 'event': []}
        for key, items in final_time_series_dict.items():
            final_df_dict['month'].append(key)
            final_df_dict['event'].append(', '.join(
                sorted(items, key=lambda item: int(re.findall(r'[0-9]+', re.findall(r'\([0-9]+\)', item)[0])[0]),
                       reverse=True)[:10]))

        df_time_series = pd.DataFrame.from_dict(final_df_dict)
        TimelineVisualiser(df_time_series, f'Peaks of hashtags over the month {month}', 'month').create_my_timeline()

    def generate_peak_based_timeline_specific_year(self, year=2015):
        temp_df = self.df.copy()
        temp_df['month'] = temp_df.month_year.dt.month
        temp_df = temp_df[temp_df['year'] == year]
        df_grouped_2 = temp_df.loc[:, ['month', 'hashtags']].groupby(by=['hashtags', 'month']).value_counts().to_frame()
        df_grouped_2.columns = ['value_count']
        all_hashtags = temp_df.hashtags.unique().tolist()
        final_time_series_dict = {}
        for hash in all_hashtags:
            temp_df = df_grouped_2.loc[hash]
            peaks = find_peaks(temp_df.value_count)
            if len(peaks[0]) > 0:
                for key, value in temp_df.iloc[list(peaks[0])].iterrows():
                    if key not in final_time_series_dict:
                        final_time_series_dict[key] = [f'{hash}({value.value_count})']
                    else:
                        final_time_series_dict[key].append(f'{hash}({value.value_count})')
        print()
        final_df_dict = {'month': [], 'event': []}
        for key in sorted(final_time_series_dict.keys()):
            items = final_time_series_dict[key]
            final_df_dict['month'].append(key)
            final_df_dict['event'].append(', '.join(
                sorted(items, key=lambda item: int(re.findall(r'[0-9]+', re.findall(r'\([0-9]+\)', item)[0])[0]),
                       reverse=True)[:10]))

        df_time_series = pd.DataFrame.from_dict(final_df_dict)
        TimelineVisualiser(df_time_series, f'Peaks of hashtags over the year {year}', 'month').create_my_timeline()

    def generate_time_series_hashtag(self, hashtag_to_plot: str):
        fig, ax = plt.subplots(figsize=(15, 8))
        new_df = self.df.copy()
        new_df = new_df[new_df['hashtags'] == hashtag_to_plot]
        temp = new_df.groupby('year').count().reset_index().iloc[:, [0, 1]]
        plt.plot(temp.iloc[:, 0], temp.iloc[:, 1])
        ax.set_xticks(temp.iloc[:, 0])
        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        ax.set_title(f'Usage of {hashtag_to_plot} over the years')
        plt.show()

    def get_sentiment_score(self):
        pass

if __name__ == '__main__':
    tw = HashtagVisualiser('data/tweets.csv', 'content', 'date_time')
    tw.generate_time_series_hashtag('#FallonTonight')
    # tw.generate_quantile_specific_year(2015)
    # tw.generate_peak_based_timeline_all()
    # tw.generate_peak_based_timeline_specific_month(5)
    # tw.generate_peak_based_timeline_specific_year()
