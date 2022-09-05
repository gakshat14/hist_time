# wikitweesation
import argparse
from yachalk import chalk

from HashtagVisualiser import HashtagVisualiser
from WikipediaTableScrapper import WikiTable

parser = argparse.ArgumentParser()

parser.add_argument(dest='application_to_run', type=str,
                    help='Specifies the application you want to run.', choices=['tweet', 'wiki'])

parser.add_argument('-U', '--url', type=str, default='https://en.wikipedia.org/wiki/Lists_of_earthquakes',
                    help='Wikipedia page URL to look for the table in.')
parser.add_argument('-H', '--heading', type=str, default='Deadliest earthquakes by year',
                    help='Caption or heading of the table')
parser.add_argument('-f', '--tweet_file', type=str, default='data/tweets.csv', help='Location of tweet CSV to process')
parser.add_argument('-m', '--month', type=int,
                    help='Month in numeric for a specific month to process irrespective of year. Only the month of '
                         'that year is processed when paired with -y or --year.')
parser.add_argument('-y', '--year', type=int,
                    help='Year in numeric for a specific year to process. Only the month of that year is processed '
                         'when paired with -m or --month.')
parser.add_argument('-l', '--list', type=bool, help='Can be used to print all of the hashtags found')
parser.add_argument('-q', '--quantile', type=bool,
                    help='Used for generating usage of hashtags over the specified month or year. 99th quantile will '
                         'be calculated.')
parser.add_argument('-t', '--time-series', type=bool,
                    help='Used for generating usage of hashtags over the specified month or year. Time series '
                         'analysis is done')
parser.add_argument('-s', '--sentiment', type=str, help='Used for generating time series on the basis of sentiment '
                                                        'for a hashtag.')


args = parser.parse_args()

if __name__ == '__main__':
    if args.application_to_run == 'wiki':
        wt = WikiTable(args.url, args.heading)
        # next thing to do
    else:
        if not args.quantile and not args.time_series and not args.list and not args.sentiment:
            print(chalk.red_bright('When visualising hashtags, please specify type of statistical analysis required. '
                                   'User -q for quantiles or -t for time series.'))
            exit(-1)

        tw = HashtagVisualiser(args.tweet_file, 'content', 'date_time')
        year = args.year
        month = args.month
        notValidYear = type(year) == int and year < 1000
        notValidMonth = type(month) == int and 0 > month > 12
        # if args.quantile or args.time_series:
        #     if notValidYear:
        #         print(chalk.red_bright('When visualising hashtags, year or month is required.'))
        #         exit(-1)
        #     if notValidMonth:
        #         print(chalk.red_bright('When visualising hashtags, year or month is required.'))
        #         exit(-1)

        if args.list:
            print(', '.join(tw.df.hashtags.unique()))

        if args.quantile:
            if year and not month:
                tw.generate_quantile_specific_year(year)
            elif month and not year:
                tw.generate_quantile_specific_month(month)
            elif year and month:
                tw.generate_quantile_specific_year_month(month, year)
            else:
                tw.generate_quantile_all()

        if args.time_series:
            if year and not month:
                print(year)
                tw.generate_peak_based_timeline_specific_year(year)
            elif month and not year:
                tw.generate_peak_based_timeline_specific_month(month)
            elif year and month:
                tw.generate_peak_based_timeline_specific_month_year(month, year)
            else:
                tw.generate_peak_based_timeline_all()

        if args.sentiment:
            tw.generate_sentiment_time_series(args.sentiment)
