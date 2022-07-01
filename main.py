from HashtagVisualiser import HashtagVisualiser

if __name__ == '__main__':
    tw = HashtagVisualiser('data/tweets.csv', 'content', 'date_time')

    # tw.print_my_dataframe()
    # tw.list_all_the_hashtags()

    tw.visualise_hashtag('#taylurking')

    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # plt.axis([0, 10, 0, 10])
    # t = ("This is a really long string that I'd rather have wrapped so that it "
    #      "doesn't go o utside of the figure, but if it's long enough it will go "
    #      "off the top or bottom!")
    # plt.text(4, 1, t, ha='left', rotation=15, wrap=True)
    # plt.text(6, 5, t, ha='left', rotation=15, wrap=True)
    # plt.text(5, 5, t, ha='right', rotation=-15, wrap=True)
    # plt.text(5, 10, t, fontsize=18, style='oblique', ha='center',
    #          va='top', wrap=True)
    # plt.text(3, 4, t, family='serif', style='italic', ha='right', wrap=True)
    # plt.text(-1, 0, t, ha='left', rotation=-15, wrap=True)
    #
    # plt.show()


