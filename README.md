## Recommendation System for Restaurants using Yelp dataset

**Project statement:**

Restaurants is one of the major industry that has close ties with humans’ necessity & offers a variety of experiences and food from specialty cuisines (e.g. Mexican, Italian etc.) to taste, décor & services. Yelp users give rating to the restaurant based on their preferences so one rating for a restaurant that all users see. Yelp has a single star rating which is not enough to address different user preferences. Customized rankings are imperatives to any business and people will benefit more from this service. So, we can build a recommendation system that can identify a user’s preferences and provide customized rankings for each individual user. 

This recommender system focuses on predicting the rating that a user would have given to a certain restaurant, which is used to rank all the restaurants including those that have not been rated by the user. 

**Data Source:**

The academic dataset (https://www.yelp.com/dataset/challenge) from yelp was downloaded and untarred.

Description of Yelp data:
1. business.json: Contains business data including location, attributes, and categories. 
2. review.json: Contains full review text data including the user_id that wrote the review and the business_id the review is written for and the review stars and usefulness. 
3. user.json: Contains the user's friend mapping and all the metadata associated with the user. 

**Data Preparation:**

The following steps are used in preparing data for analysis & prediction:
 	
1. Large Dataset: As the dataset is huge, we only took 100K samples of the observations from each dataset (businesses, users and reviews) to perform the initial EDA.
2. Combining Data: The sample data was observed to be clean and we merged the dataset based on unique keys.
3. Filtering: We filtered it down to the restaurants by selecting businesses based on category ‘Food’; for reviews/users we only included the data which was for the restaurants in the sample.
4. Visualization: We used bar charts, histograms, scatter plots and distribution plots to explore the data and understand the data patterns for the restaurants’ reviews.
5. Ratings are integers ranging between 1 and 5. The loss function to compare various methods is measured by the root mean squared error (RMSE).

**References:**

1. How the Netflix prize was won, http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/
2. Matrix factorization for recommender systems, https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf
3. Ensembling for the Netflix Prize, http://web.stanford.edu/~lmackey/papers/netflix_story-nas11-slides.pdf
4. Reviews on methods for netflix prize, http://arxiv.org/abs/1202.1112andhttp://www.grouplens.org/system/files/FnT%20CF%20Recsys%20Survey.pdf
5. Advances in Collaborative Filtering from the Netflix prize, https://datajobs.com/data-science-repo/Collaborative-Filtering-%5BKoren-and-Bell%5D.pdf
6. Python Surprise library for models, https://pypi.python.org/pypi/scikit-surprise.
7. Library for SVD, https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html
8. Library for accessign Similarity, http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html

