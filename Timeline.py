import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TimelineVisualiser:
    MY_DATE_COLUMN = 'date'
    MY_EVENT_COLUMN = 'event'
    MY_MAGNITUDE_COLUMN = 'magnitude'
    MY_LOCATION_COLUMN = 'location'

    def __init__(self, df: pd.DataFrame, heading: str, x_label='Dates', timeline_key='year', magnitude_legend_key='',
                 is_range=False, attribution_text=''):
        self.df = df
        self.__is_range = is_range
        self.__timeline_key = timeline_key
        self.heading = heading
        self.x_label = x_label
        self.magnitude_label = magnitude_legend_key
        self.attribution_text = attribution_text

    @staticmethod
    def __short_my_string(text: str) -> str:
        split_text = text.split(' ')
        if len(split_text) <= 2:
            return text

        fixed_string = [val + '\n' if (i + 1) % 3 == 0 else val for i, val in enumerate(split_text)]

        return ' '.join(fixed_string)

    def create_my_timeline(self):
        # marker_size = np.array([n ** 2.8 for n in magnitude]).astype(float)
        length_df = self.df.shape[0]

        magnitude = self.df['magnitude_calc'].tolist() if 'magnitude_calc' in self.df.columns else []

        magnitude_text = self.df[self.MY_MAGNITUDE_COLUMN].tolist() if 'magnitude_calc' in self.df.columns else []

        location = self.df[self.MY_LOCATION_COLUMN].tolist() if self.MY_LOCATION_COLUMN in self.df.columns else []

        event = self.df[self.MY_EVENT_COLUMN].tolist()

        date = self.df[self.__timeline_key].tolist()

        # initialise the figure
        if length_df > 50:
            fig, ax = plt.subplots(figsize=(45, 8))
        else:
            fig, ax = plt.subplots(figsize=(35, 8))
        ax.set_title(self.heading, pad=20)

        # create vertical lines
        levels = np.tile([1, -1, 2, -2],
                         int(np.ceil(length_df / 4)))[:length_df]

        # defining x_axis
        x_axis = [x * 2 for x in range(0, length_df)]

        # the date_test here is the place where the graph should be
        # 0 is the min
        # levels is the maximum here.
        ax.vlines(x_axis, 0, levels, color="tab:red", label='Events')

        y_axis = np.zeros_like(date)

        ax.plot(x_axis, y_axis, color="k", label='Baseline')

        if len(magnitude) > 0:
            multiplier = 1.7 if max(magnitude) > 10 else 2.7
            marker_size = np.array([n ** multiplier for n in magnitude], dtype=int)
            ax.scatter(x_axis, y_axis, s=marker_size, edgecolors='k', c='lightgray', label=self.magnitude_label)

        for index in range(0, len(x_axis)):
            text = f'{self.__short_my_string(event[index])} \n'
            if len(location) > 0:
                text += f'({self.__short_my_string(location[index])}) \n'
            if len(magnitude) > 0:
                text += f'({self.__short_my_string(magnitude_text[index])}) \n'
            # self.__short_my_string(text)
            ax.text(s=text, x=x_axis[index], y=levels[index],
                    horizontalalignment="center",
                    verticalalignment="bottom" if levels[index] > 0 else "top", wrap=True, fontsize=11)
            # ax.annotate(text, xy=(x_axis[index], levels[index]), xytext=(-3, np.sign(levels[index]) * 6),
            #             textcoords="offset points",
            #             horizontalalignment="center",
            #             verticalalignment="bottom" if levels[index] > 0 else "top", wrap=True, fontsize=11)

        ax.set_xticks(x_axis)
        ax.set_xticklabels(date)
        ax.set_xlabel(self.x_label)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.yaxis.set_visible(False)
        ax.spines[["left", "top", "right"]].set_visible(False)
        plt.margins(y=0.2)
        plt.subplots_adjust(bottom=0.12)
        plt.legend(bbox_to_anchor=(1.001, 0.5), loc='center right', borderaxespad=0)
        self.attribution_text and fig.text(0.50, 0.02, self.attribution_text, horizontalalignment='center', wrap=True)
        # plt.legend(loc='center right')
        plt.savefig(self.heading, bbox_inches='tight', dpi=200)
        # plt.tight_layout()
        plt.show()
