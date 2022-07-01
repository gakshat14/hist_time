import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt


class HashtagVisualiser():
    def __init__(self, path_to_csv: str, tweet_key: str, date_time_key: str):
        print('Initializer called')
        self.__df = pd.read_csv(path_to_csv)
        self.__df[date_time_key] = pd.to_datetime(self.__df[date_time_key])
        self.__df = self.__df.drop(columns=self.__df.columns.difference([tweet_key, date_time_key]))
        self.__date_time_key = date_time_key
        self.__tweets_key = tweet_key
        self.__hashtags = self.__get_the_list_of_hashtags()

    def print_my_dataframe(self):
        print('print dataframe called')
        # if not self.__df:
        #     raise ValueError('The specified dataframe does not exists')
        print(self.__df)

    def __get_the_list_of_hashtags(self):
        print('generating list of hashtags')
        count = 0
        hashtags = dict()

        def get_hashtag(series):
            nonlocal count
            regex = r'#\w+'
            results = re.findall(regex, series)
            for result in results:
                year = self.__df[self.__date_time_key][count].year
                if result not in hashtags:
                    hashtags[result] = {'index': [count], 'year': [year]}
                else:
                    hashtags[result]['index'].append(count)
                    hashtags[result]['year'].append(year)
            count += 1

        self.__df[self.__tweets_key].map(get_hashtag)
        return hashtags

    def list_all_the_hashtags(self):
        print(self.__hashtags.keys())

    def __draw_hashtag_timeline(self, temp_df: pd.DataFrame, magnitude_key: str, title: str, hashtag: str):
        unique_values = temp_df.groupby(magnitude_key).count().index.to_numpy()
        magnitude = temp_df.groupby(magnitude_key).count().iloc[:, 1].values
        # dates_to_plot = [i for i in temp_df]
        levels = np.tile([-6, 6, -3, 3, -1, 1],
                         int(np.ceil(len(unique_values) / 6)))[:len(unique_values)]

        marker_size = np.array([n ** 2 for n in magnitude]).astype(float)
        fig, ax = plt.subplots(figsize=(25, 15))
        ax.set(title=title)

        # the date_test here is the place where the graph should be
        # 0 is the min
        # levels is the maximum here.
        ax.vlines(unique_values, 0, levels, color="tab:red")

        ax.plot(unique_values, np.zeros_like(unique_values))
        ax.scatter(unique_values, np.zeros_like(unique_values), s=marker_size, edgecolors='k', c='lightgray')
        plt.legend(['', '', 'Magnitude'])

        for d, m, lev in zip(unique_values, magnitude, levels):
            ax.annotate(f'{hashtag} ({m})', xy=(d, lev), xytext=(-3, np.sign(lev) * 6), textcoords="offset points",
                        horizontalalignment="center",
                        verticalalignment="bottom" if lev > 0 else "top", wrap=True)

        ax.set_xticks(unique_values)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.yaxis.set_visible(False)
        ax.spines[["left", "top", "right"]].set_visible(False)

        ax.margins(y=0.1)
        plt.show()

    def visualise_hashtag(self, hashtag: str):
        temp_df = self.__df.iloc[self.__hashtags[hashtag]['index']]
        temp_df['year'] = temp_df['date_time'].dt.year
        temp_df['month'] = temp_df['date_time'].dt.month
        temp_df['days'] = temp_df['date_time'].dt.day

        distinct_number_of_years = temp_df['year'].nunique()
        distinct_number_of_months = temp_df['month'].nunique()
        distinct_number_of_days = temp_df['days'].nunique()

        if distinct_number_of_days >= 5:
            self.__draw_hashtag_timeline(temp_df, 'days', f'Usage of {hashtag} over different days', hashtag)
