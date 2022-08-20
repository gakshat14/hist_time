import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TimelineVisualiser:
    MY_DATE_COLUMN = 'date'
    MY_EVENT_COLUMN = 'event'
    MY_MAGNITUDE_COLUMN = 'magnitude'
    MY_LOCATION_COLUMN = 'location'

    def __init__(self, df: pd.DataFrame, heading: str, timeline_key='year', is_range=False):
        self.df = df
        self.__is_range = is_range
        self.__timeline_key = timeline_key
        self.heading = heading

    @staticmethod
    def __short_my_string(text: str) -> str:
        split_text = text.split(' ')
        if len(split_text) <= 4:
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
        fig, ax = plt.subplots(figsize=(35, 15))
        ax.set(title=self.heading)

        # create vertical lines
        levels = np.tile([1, -1, 2, -2, 3, -3],
                         int(np.ceil(length_df / 6)))[:length_df]

        # defining x_axis
        x_axis = [x * 2 for x in range(0, length_df)]

        # the date_test here is the place where the graph should be
        # 0 is the min
        # levels is the maximum here.
        ax.vlines(x_axis, 0, levels, color="tab:red")

        y_axis = np.zeros_like(date)

        ax.plot(x_axis, y_axis)

        if len(magnitude) > 0:
            marker_size = np.array([n ** 2.7 for n in magnitude], dtype=int)
            ax.scatter(x_axis, y_axis, s=marker_size, edgecolors='k', c='lightgray')

        for index in range(0, len(x_axis)):
            text = f'{self.__short_my_string(event[index])} \n'
            if len(location):
                text += f'({self.__short_my_string(location[index])}) \n'
            if len(magnitude) > 0:
                text += f'({self.__short_my_string(magnitude_text[index])}) \n'
            # self.__short_my_string(text)

            ax.annotate(text, xy=(x_axis[index], levels[index]), xytext=(-3, np.sign(levels[index]) * 6),
                        textcoords="offset points",
                        horizontalalignment="center",
                        verticalalignment="bottom" if levels[index] > 0 else "top", wrap=True)

        ax.set_xticks(x_axis)
        ax.set_xticklabels(date)
        plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.yaxis.set_visible(False)
        ax.spines[["left", "top", "right"]].set_visible(False)

        ax.margins(y=0.1)
        plt.savefig(self.heading, bbox_inches='tight',dpi=100)
