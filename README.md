# Visualizing Temporal References
## Introduction
A novel application to automate the creation of a timeline
visualisation of any event or set of discrete events with a temporal reference. The events here can be a list
of events extracted from online sources like Wikipedia with an eye for the humanities and social science
or events taken out of a standard time series as ”special points” (e.g. peaks, inflections, change-points,
etc) from tweets. This would enable the automatic and seamless creation of a timeline from the data,
whether it was extracted from a dataset or scraped from the internet. As a result, the application can
understand the significance of an event (if supplied) and properly visualise it. Additionally, the tool can
be used to process a batch of tweets. Hashtags can be collected using Natural Language Processing(NLP)
techniques, and following statistical analysis, the hashtags can be shown across the timeline, explaining
their usage over the year, month, and even days
## Environment Setup

We have used `conda` to manage the environments for the application. The environment is based on `Windows` and `Python3`.

To install packages from the `requirements.txt`, use the command

``conda create --name my-env-name -f requirements.txt``

## Usage

The application is builtin CLI which can be used to interact with the application.

For visualising **historical events** (wiki) you can use the following command

``python main.py wiki``

The options that can be used are:

1. **-U or –url**: It is used to specify the URL of the Wikipedia page from where we have to scrap the
table containing the list of events. it has a default value.
2. **-H or –heading**: It is used to find the table of interest. A Wikipedia page can contain multiple
tables; we can use this option to highlight which table we need, ensuring the correct table is scraped.
It has a default value

For visualising **hashtag analysis** you can use the following command

``python main.py tweet -t True``

1. **-f or –tweet file:** It is used to specify the tweet dataset file to process. The specified file should
be a CSV file.
2. **-m or –month:** It is used to specify the month for which the hashtags should be visualized. It
accepts numeric values, and input ranges from 1-12.
3. **-y or –year:** It is used to specify the year for which the hashtags should be visualized. It also
accepts numeric values.
4. **-l or –list:** It is used to print all the hashtags extracted from the tweets. It is of boolean type and
hence accepts True when passed.
5. **-q or –quantile:** It is used to specify to generate timeline visualization of hashtags after statistical
analysis of the tweet dataset by calculating 0.99 quantile. It accepts a boolean value and accepts
True when passed.
6. **-t or –time series:** It is used to specify to generate timeline visualization of hashtags after time
series analysis of the tweet dataset by finding peaks. It accepts a boolean value and accepts True
when passed.
7. **-s or –sentiment:** It is used to generate a time series visualization of the sentiment scores of
the tweets in which that hashtag was used. It accepts hashtags found in the dataset. The list of
hashtags can be printed by passing the -l or –list option.

