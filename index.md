---
title: Recommendation System for Restaurants using Yelp dataset
---

This is the home page

## Lets have fun

>here is a quote

Here is *emph* and **bold**.

Here is some inline math $\alpha = \frac{\beta}{\gamma}$ and, of-course, E rules:

$$ G_{\mu\nu} + \Lambda g_{\mu\nu}  = 8 \pi T_{\mu\nu} . $$

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

