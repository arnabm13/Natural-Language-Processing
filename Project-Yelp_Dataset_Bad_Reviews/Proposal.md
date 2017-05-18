## Overview

We are trying to solve certain aspects of the yelp data set challenge which contains :

•	4.1M reviews and 947K tips by 1M users for 144K businesses

•	1.1M business attributes, e.g., hours, parking availability, ambience.

•	Aggregated check-ins over time for each of the 125K businesses



Our Objective :

•	Analyze review text for Restaurants, do a sentiment analysis of the review     text

•	Analyze why a user gave a good or a bad review





## Data

 The data is made available on the following website. 

  https://www.yelp.com/dataset_challenge/



•	The data is in Json format we need to change it to CSV

•	The data set is huge and to load it in memory, we require batch processing or cache

•	We need to do a horizontal filtering on rows as data is missing 

•	We need to study the data and filter only certain columns like BusinessName, UserId, Review Text, Review Stars

•	The data has non English reviews that has to be filtered, we will be taking data from USA only



## Method

·         We plan to use NLTK to understand sentiment text

·         We plan to use Scikitlearn, numpy, pandas to load data and interpret    whether the review is good or bad

·         We will be writing our own code using these libraries

## Related Work

•	https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1928601

•	https://arxiv.org/abs/1401.0864

•	http://dl.acm.org/citation.cfm?id=2631784

•	http://dl.acm.org/citation.cfm?id=2507163

•

https://pdfs.semanticscholar.org/8b2b/ada22181916196116f1711d456ea212f2b3b.pdf

•	https://pdfs.semanticscholar.org/9c85/836ffaa9dfb3523b793f0d41198d13621b6a.pdf



## Evaluation

•	Our main tools to evaluate results will be Precision, recall, RMSE

•	We will be visualizing our data based on the category and compare it with the predicted output

•	K-fold cross validation for measuring the accuracy

•	80-20 division for training and test data ratio for measuring accuracy  



