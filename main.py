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
parser.add_argument('-m', '--month', type=int,
                    help='Month in numeric for a specific month to process irrespective of year. When paired with -y '
                         'or --year only month of that year is processed')
parser.add_argument('-y', '--year', type=int,
                    help='Year in numeric for a specific year to process. When paired with -m '
                         'or --month only month of that year is processed')
parser.add_argument('-l', '--list', type=str, help='Can be used to print all of the hashtags found')
parser.add_argument('-q', '--quantile', type=bool,
                    help='Used for generating usage of hashtags over the specified month or year. IQR is calculated')
parser.add_argument('-t', '--time-series', type=bool,
                    help='Used for generating usage of hashtags over the specified month or year. Time series '
                         'analysis is done')


args = parser.parse_args()

if __name__ == '__main__':
    if args.application_to_run == 'wiki':
        wt = WikiTable(args.url, args.heading)
        # next thing to do
    else:
        tw = HashtagVisualiser('data/tweets.csv', 'content', 'date_time')
        tw.generate_quantile_specific_month(7)
