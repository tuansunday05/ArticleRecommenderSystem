
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
import nltk
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

class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, articles_df=None, interaction_df = None, event_type_strength = None):
        self.articles_df = articles_df
        self.interaction_df = interaction_df
        self.event_type_strength = event_type_strength
        self.cf_predictions_df, self.items_df, self.all_user_predicted_rating_norm = self.factorization(self.articles_df, self.interaction_df)
        
    def get_model_name(self):
        return self.MODEL_NAME
    
    def smooth_user_preference(self, x):
        return math.log(1+x, 2)

    def factorization(self, articles_df, interactions_df):

        interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: self.event_type_strength[x])

        users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()


        users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]

        interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
                    how = 'right',
                    left_on = 'personId',
                    right_on = 'personId')
        
        interactions_full_df = interactions_from_selected_users_df \
                            .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                            .apply(self.smooth_user_preference).reset_index()
        self.interactions_full_df = interactions_full_df
        # print('# of unique user/item interactions: %d' % len(interactions_full_df))

        articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']

        users_items_pivot_matrix_df = interactions_full_df.pivot(index='personId', 
                                                                columns='contentId', 
                                                                values='eventStrength').fillna(0)

        users_items_pivot_matrix = users_items_pivot_matrix_df.values

        users_ids = list(users_items_pivot_matrix_df.index)
        users_items_pivot_sparse_matrix = csr_matrix(users_items_pivot_matrix)
        #The number of factors to factor the user-item matrix.
        NUMBER_OF_FACTORS_MF = 15
        #Performs matrix factorization of the original user item matrix
        #U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
        U, sigma, Vt = svds(users_items_pivot_sparse_matrix, k = NUMBER_OF_FACTORS_MF)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
        all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
        cf_preds_df = pd.DataFrame(all_user_predicted_ratings_norm, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
        return cf_preds_df, articles_df, all_user_predicted_ratings_norm
    
    def get_items_interacted(self, person_id, interactions_df):
        # Get the user's data and merge in the movie information.
        interacted_items = interactions_df.loc[person_id]['contentId']
        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])  
     
    def recommend_items(self, user_id, ignore_interacted= False, topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'recStrength'})
        if ignore_interacted == True:
            items_to_ignore = self.get_items_interacted(user_id, self.interactions_full_df.set_index('personId'))
        else:
            items_to_ignore = []
        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
                            .sort_values('recStrength', ascending = False) \
                            .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                        left_on = 'contentId', 
                                                        right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]

        return recommendations_df
    
    def set_interaction(self,new_interactions_df):
        self.cf_predictions_df, self.items_df,self.all_user_predicted_rating_norm = self.factorization(self.articles_df, new_interactions_df)
    

if __name__ == "__main__":
    # run the model 
    interactions_df = pd.read_csv('data/users_interactions.csv')
    articles_df = pd.read_csv('data/shared_articles.csv')
    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
    userid = -1479311724257856983
    cf_recommender_model = CFRecommender(articles_df, interactions_df, event_type_strength)
    print(' 20 articles for user {}',cf_recommender_model.recommend_items(userid, topn=10, verbose=True))
