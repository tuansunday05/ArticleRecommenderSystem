from nltk.corpus import stopwords
import pandas as pd
from content_based_model import UsersItemsProfiles, ContentBasedRecommender
from collaborative_filtering_model import CFRecommender
from apriori_model import AprioriRecommender


event_type_strength = {
        'VIEW': 1.0,
        'LIKE': 2.0, 
        'BOOKMARK': 2.5, 
        'FOLLOW': 3.0,
        'COMMENT CREATED': 4.0,  
}

class HybridRecommender:
    
    MODEL_NAME = 'Hybrid'
    
    def __init__(self, cb_rec_model, cf_rec_model, ap_rec_model, items_df, cb_ensemble_weight=1.0, cf_ensemble_weight=1.0, ap_ensemble_weight = 1.0):
        self.cb_rec_model = cb_rec_model
        self.cf_rec_model = cf_rec_model
        self.ap_rec_model = ap_rec_model
        self.cb_ensemble_weight = cb_ensemble_weight
        self.cf_ensemble_weight = cf_ensemble_weight
        self.ap_ensemble_weight = ap_ensemble_weight
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, ignore_interacted= False, topn=10, verbose=False):
        #Getting the top-1000 Content-based filtering recommendations
        cb_recs_df = self.cb_rec_model.recommend_items(user_id, ignore_interacted= ignore_interacted, verbose=verbose,
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCB'})
        
        #Getting the top-1000 Collaborative filtering recommendations
        cf_recs_df = self.cf_rec_model.recommend_items(user_id, ignore_interacted= ignore_interacted, verbose=verbose, 
                                                           topn=1000).rename(columns={'recStrength': 'recStrengthCF'})
        
        #
        ap_recs_df = self.ap_rec_model.recommend_items(user_id, ignore_interacted=ignore_interacted, verbose=verbose,
                                                        topn=1000).rename(columns={'recStrength': 'recStrengthAP'})

        #Combining the results by contentId
        recs_df = cb_recs_df.merge(cf_recs_df, how = 'outer', left_on = 'contentId', right_on = 'contentId').fillna(0.0) \
                            .merge(ap_recs_df, how = 'outer', left_on = 'contentId', right_on = 'contentId').fillna(0.0)
        
        #Computing a hybrid recommendation score based on CF and CB scores
        #recs_df['recStrengthHybrid'] = recs_df['recStrengthCB'] * recs_df['recStrengthCF'] 
        recs_df['recStrengthHybrid'] = (recs_df['recStrengthCB'] * self.cb_ensemble_weight) + (recs_df['recStrengthCF'] \
                                            * self.cf_ensemble_weight) + (recs_df['recStrengthAP'] * self.ap_ensemble_weight)

        
        #Sorting recommendations by hybrid score
        recommendations_df = recs_df.sort_values('recStrengthHybrid', ascending=False).head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')
            # print(recommendations_df)
            recommendations_df = recommendations_df.drop(['title', 'url', 'lang'], axis = 1).merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrengthHybrid', 'contentId', 'title', 'url', 'lang']]
    
        return recommendations_df
    
if __name__ == "__main__":
    articles_df = pd.read_csv('data/shared_articles.csv')
    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
    interactions_df = pd.read_csv('data/users_interactions.csv')
    ##
    users_items_profiles = UsersItemsProfiles(articles_df, interactions_df, event_type_strength)
    users_items_profiles.build_items_profile()
    users_items_profiles.build_users_profiles()

    person_id = -1479311724257856983

    uid = users_items_profiles.build_users_profile(person_id=person_id)
    content_based_recommender_model = ContentBasedRecommender(articles_df, users_items_profiles)
    
    ##

    cf_recommender_model = CFRecommender(articles_df, interactions_df, event_type_strength)

    ##
    apriori_recommender_model = AprioriRecommender(articles_df, interactions_df, event_type_strength)

    ##

    hybrid_recommender_model = HybridRecommender(content_based_recommender_model, cf_recommender_model, apriori_recommender_model, articles_df,
                                             cb_ensemble_weight=1.0, cf_ensemble_weight=100, ap_ensemble_weight=1.0)
    
    result = hybrid_recommender_model.recommend_items(user_id = person_id, ignore_interacted= True, topn = 10, verbose=True)
    print(result)

