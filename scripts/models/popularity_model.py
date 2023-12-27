import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt




event_type_strength = {
        'VIEW': 1.0,
        'LIKE': 2.0, 
        'BOOKMARK': 2.5, 
        'FOLLOW': 3.0,
        'COMMENT CREATED': 4.0,  
}

class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, interactions_df, articles_df, event_type_strength):
        self.interactions_df = interactions_df
        self.articles_df = articles_df
        self.event_type_strength = event_type_strength
        self.popularity_df, self.items_df = self.munging(self.interactions_df, self.articles_df, self.event_type_strength)
        
    def get_model_name(self):
        return self.MODEL_NAME
    
    def smooth_user_preference(self, x):
            return math.log(1+x, 2)
    
    def  munging(self, interactions_df, articles_df, event_type_strength): 
        articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']

        

        interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])


        users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()

        users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]

        interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
                    how = 'right',
                    left_on = 'personId',
                    right_on = 'personId')
        
        
            
        interactions_full_df = interactions_from_selected_users_df \
                            .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                            .apply(self.smooth_user_preference).reset_index()
        #Computes the most popular items
        item_popularity_df = interactions_full_df.groupby('contentId')['eventStrength'].sum().sort_values(ascending=False).reset_index()
        return item_popularity_df, articles_df

        
    def recommend_items(self, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)] \
                               .sort_values('eventStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df
    def set_interaction(self,new_interactions_df):
        self.interactions_df = new_interactions_df
        self.popularity_df, self.items_df = self.munging(self.interactions_df,self.articles_df)
# run the model 
# userid = -1479311724257856983
        
if __name__ == "__main__":
    interactions_df = pd.read_csv('data/users_interactions.csv')
    articles_df = pd.read_csv('data/shared_articles.csv')
    popularity_model = PopularityRecommender(interactions_df, articles_df, event_type_strength)
    print('Top 10 articles popularity\n',popularity_model.recommend_items(topn=10, verbose=True))



