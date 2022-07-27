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

    def get_top_5_hashtags(self) -> list:
        sorted_hashtags = sorted(self.__hashtags, key=lambda hashtag: len(self.__hashtags[hashtag].get('index')),
                                 reverse=True)
        return sorted_hashtags[:5]

    @staticmethod
    def __draw_hashtag_timeline(temp_df: pd.DataFrame, magnitude_key: str, title: str, hashtag: str):

        unique_values = temp_df.groupby(magnitude_key).count().index.to_numpy()

        magnitude = temp_df.groupby(magnitude_key).count().iloc[:, 1].values
        # dates_to_plot = [i for i in temp_df]
        marker_size = np.array([n ** 3 if n < 10 else n ** 2 if n < 50 else n ** 1.4 for n in magnitude]).astype(float)

        levels = np.tile([1, -1, 2, -2, 3, -3],
                         int(np.ceil(len(unique_values) / 6)))[:len(unique_values)]

        # creating even spaced dots
        x_axis = [x + 2 for x in range(0, len(unique_values))]

        fig, ax = plt.subplots(figsize=(25, 15))
        ax.set(title=title)

        # the date_test here is the place where the graph should be
        # 0 is the min
        # levels is the maximum here.
        ax.vlines(x_axis, 0, levels, color="tab:red")

        ax.plot(x_axis, np.zeros_like(unique_values))
        ax.scatter(x_axis, np.zeros_like(unique_values), s=marker_size, edgecolors='k', c='lightgray')
        plt.legend(['', '', 'Magnitude'])

        for d, m, lev in zip(x_axis, magnitude, levels):
            ax.annotate(f'{hashtag} ({m})', xy=(d, lev), xytext=(-3, np.sign(lev) * 6), textcoords="offset points",
                        horizontalalignment="center",
                        verticalalignment="bottom" if lev > 0 else "top", wrap=True, fontsize=13)

        ax.set_xticks(x_axis)
        ax.set_xticklabels(unique_values)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.yaxis.set_visible(False)
        ax.spines[["left", "top", "right"]].set_visible(False)

        ax.margins(y=0.1)
        plt.show()

    def visualise_hashtag(self, hashtag: str):
        temp_df = self.__df.iloc[self.__hashtags[hashtag]['index']]
        temp_df['year'] = temp_df['date_time'].dt.year
        temp_df['month_year'] = temp_df['date_time'].dt.to_period('M')
        temp_df['date_only'] = temp_df['date_time'].dt.date
        #
        distinct_number_of_years = temp_df['year'].nunique()
        distinct_number_of_months = temp_df['month_year'].nunique()
        distinct_number_of_days = temp_df['date_only'].nunique()

        if distinct_number_of_years >= 5:
            self.__draw_hashtag_timeline(temp_df, 'year', f'Usage of {hashtag} over different years', hashtag)
            return

        if distinct_number_of_months >= 5:
            self.__draw_hashtag_timeline(temp_df, 'month_year', f'Usage of {hashtag} over different months', hashtag)
            return

        if distinct_number_of_days >= 5:
            self.__draw_hashtag_timeline(temp_df, 'date_only', f'Usage of {hashtag} over different days', hashtag)
            return
