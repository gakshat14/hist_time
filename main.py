from HashtagVisualiser import HashtagVisualiser

if __name__ == '__main__':
    tw = HashtagVisualiser('data/tweets.csv', 'content', 'date_time')
    print(tw.get_top_5_hashtags())




