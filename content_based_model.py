import numpy as np
import scipy
import pandas as pd
import math
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

event_type_strength = {
        'VIEW': 1.0,
        'LIKE': 2.0, 
        'BOOKMARK': 2.5, 
        'FOLLOW': 3.0,
        'COMMENT CREATED': 4.0,  
}

class UsersItemsProfiles:
        
    def __init__(self, articles_df, interactions_df, event_type_strength):     
        self.articles_df = articles_df
        self.interactions_df = interactions_df
        self.event_type_strength = event_type_strength
        # self.build_interaction_df()
        # self.build_tfidf_matrix()

    def update_interactions_df(self, new_interactions_df):
        self.interactions_df = new_interactions_df
        ## changes_index = ...
        ## self.update_user_profile(changes_idex)

    def smooth_user_preference(self, x):
        return math.log(1+x, 2)

    def build_interaction_df(self, person_id = None):
        if person_id is not None:
            self.interactions_df.loc[self.interactions_df['personId'] == person_id, 'eventStrength'] = self.interactions_df\
                                                                                                    .loc[self.interactions_df['personId'] \
                                                                                                    == person_id,'eventType'].apply(lambda x: \
                                                                                                    self.event_type_strength[x])
        else:
            self.interactions_df['eventStrength'] = self.interactions_df['eventType'].apply(lambda x: self.event_type_strength[x])

        users_interactions_count_df = self.interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
        users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 1].reset_index()[['personId']]
        interactions_from_selected_users_df = self.interactions_df.merge(users_with_enough_interactions_df, 
                how = 'right',
                left_on = 'personId',
                right_on = 'personId')
        
        self.interactions_full_df = interactions_from_selected_users_df \
                        .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                        .apply(self.smooth_user_preference).reset_index()
        ##
        return self.interactions_full_df

        ##  Content-based filtering
    
    def build_tfidf_matrix(self):
        #Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
        stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

        #Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
        vectorizer = TfidfVectorizer(analyzer='word',
                            ngram_range=(1, 2),
                            min_df=0.003,
                            max_df=0.5,
                            max_features=5000,
                            stop_words=stopwords_list)

        self.item_ids = self.articles_df['contentId'].tolist()
        self.tfidf_matrix = vectorizer.fit_transform(self.articles_df['title'] + "" + self.articles_df['text'])
        
        return self.item_ids, self.tfidf_matrix

    def get_item_profile(self, item_id):
        idx = self.item_ids.index(item_id)
        item_profile = self.tfidf_matrix[idx:idx+1]
        return item_profile

    def get_item_profiles(self, ids):
        item_profiles_list = [self.get_item_profile(x) for x in ids] if isinstance(ids, pd.Series) else self.get_item_profile(ids)
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles
    
    def build_items_profile(self):
        interactions_full_df = self.build_interaction_df()
        item_ids, tfidf_matrix = self.build_tfidf_matrix()
        return interactions_full_df, item_ids, tfidf_matrix


    def build_users_profile(self, person_id, interactions_indexed_df = None):
        if interactions_indexed_df is not None:
            interactions_person_df = interactions_indexed_df.loc[person_id]
        else:
            self.interactions_indexed_df = self.interactions_full_df[self.interactions_full_df['contentId'] \
                                                    .isin(self.articles_df['contentId'])].set_index('personId')
            interactions_person_df = self.interactions_indexed_df.loc[person_id]
        user_item_profiles = self.get_item_profiles(interactions_person_df['contentId'])
        
        user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1,1)
        #Weighted average of item profiles by the interactions strength
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
        user_profile_norm = sklearn.preprocessing.normalize(np.asarray(user_item_strengths_weighted_avg))
        return user_profile_norm
    
    def build_users_profiles(self): 
        self.interactions_indexed_df = self.interactions_full_df[self.interactions_full_df['contentId'] \
                                                    .isin(self.articles_df['contentId'])].set_index('personId')
        user_profiles = {}
        for person_id in self.interactions_indexed_df.index.unique():
            user_profiles[person_id] = self.build_users_profile(person_id, self.interactions_indexed_df)

        self.user_profiles = user_profiles
        return user_profiles

    def update_user_profile(self, person_id, interactions_full_df = None, interactions_indexed_df= None):
        if interactions_full_df is None and interactions_indexed_df is None:
            self.build_interaction_df(person_id)
            self.interactions_indexed_df = self.interactions_full_df[self.interactions_full_df['contentId'] \
                                                    .isin(self.articles_df['contentId'])].set_index('personId')
        user_profile = self.build_users_profile(person_id, self.interactions_indexed_df)
        self.user_profiles[person_id] = user_profile
        return self.user_profiles
    
    def update_users_profiles(self, changes_indexed_df = None):
        pass

        
class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df= None, interactions_df = None, event_type_strength = None):
        users_items_profiles = UsersItemsProfiles(items_df, interactions_df, event_type_strength)
        users_items_profiles.build_items_profile()
        users_items_profiles.build_users_profiles()
        self.item_ids = users_items_profiles.item_ids
        self.items_df = items_df
        self.interactions_df = users_items_profiles.interactions_df
        self.tfidf_matrix = users_items_profiles.tfidf_matrix
        self.user_profiles = users_items_profiles.user_profiles
        # self.interactions_full_df = users_items_profiles.interactions_full_df
        self.users_items_profiles = users_items_profiles

    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id = None, user_profile= None, topn=50):
        #Computes the cosine similarity between the user profile and all item profiles
        if user_profile is not None:
            cosine_similarities = cosine_similarity(user_profile, self.tfidf_matrix)
        else:
            cosine_similarities = cosine_similarity(self.user_profiles[person_id], self.tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(self.item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
    
    def get_similar_items_to_item_profile(self, item_id= None, user_id = None, item_profile = None, ignore_interacted = True, topn=50, verbose= True):
        #Computes the cosine similarity between the user profile and all item profiles
        if item_profile is None:
            item_profile = self.users_items_profiles.get_item_profile(item_id)
        cosine_similarities = cosine_similarity(item_profile, self.tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(self.item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        items_to_ignore = self.get_items_interacted(user_id, self.interactions_df.set_index('personId'))
        #Ignores items the user has already interacted
        if ignore_interacted == True:
            similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        else:
            similar_items_filtered = similar_items
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
                                    .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]

        return similar_items
    
    def get_items_interacted(self, person_id, interactions_df):
        # Get the user's data and merge in the movie information.
        interacted_items = interactions_df.loc[person_id]['contentId']
        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])
    
    def update_user_profile(self, person_id = None):
        self.users_items_profiles.update_user_profile(person_id= person_id)
        return self.users_items_profiles
    
    def update_interactions_df(self, new_interactions_df):
        self.users_items_profiles.update_interactions_df(new_interactions_df)
        self.interactions_df = new_interactions_df

    
    def recommend_items(self, user_id = None, user_profile = None, ignore_interacted=False, topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id, user_profile)
        items_to_ignore = self.get_items_interacted(user_id, self.interactions_df.set_index('personId'))
        #Ignores items the user has already interacted
        if ignore_interacted == True:
            similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        else:
            similar_items_filtered = similar_items
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
                                    .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df


if __name__ == "__main__":
    articles_df = pd.read_csv('data/shared_articles.csv')
    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
    interactions_df = pd.read_csv('data/users_interactions.csv')
    content_based_recommender_model = ContentBasedRecommender(articles_df, interactions_df, event_type_strength)
    # ------ update while online running
    person_id = -1479311724257856983 #-9150583489352258206 #-1479311724257856983
    content_based_recommender_model.update_interactions_df(interactions_df) # new interactions_df
    content_based_recommender_model.update_user_profile(person_id=person_id) # new interaction of person_id

    result = content_based_recommender_model.recommend_items(person_id, 
                                               ignore_interacted= True, 
                                               topn=10,
                                               verbose= True)
    print(result)
