# wikitweesation
import argparse

from HashtagVisualiser import HashtagVisualiser
from WikipediaTableScrapper import WikiTable

parser = argparse.ArgumentParser()

parser.add_argument(dest='application_to_run', type=str,
                    help='Specifies the application you want to run.', choices=['tweet', 'wiki'])

parser.add_argument('-U', '--url', type=str, default='https://en.wikipedia.org/wiki/Lists_of_earthquakes',
                    help='Wikipedia page URL to look for table in')
parser.add_argument('-H', '--heading', type=str, default='Deadliest earthquakes by year',
                    help='Caption or heading of the table')
parser.add_argument('-f', '--tweet_file', type=str, help='Location of tweet CSV to process')

args = parser.parse_args()

if __name__ == '__main__':
    if args.application_to_run == 'wiki':
        wt = WikiTable(args.url, args.heading)
    else:
        tw = HashtagVisualiser('data/tweets.csv', 'content', 'date_time')
        print(tw.get_top_5_hashtags())
