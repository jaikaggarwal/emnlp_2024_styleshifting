from project_utils import *
from community_embeddings import *

class UserEngagement:

    def __init__(self, data):
        self.posting_data = data.all_data
        self.manospheric_author_data = self.augment_posting_data(data.manospheric_author_data)
        self.mano_in_mano = self.augment_posting_data(data.mano_in_mano)
        self.mano_outside_mano = data.mano_outside_mano.reset_index()
        self.com_embeddings = CommunityEmbeddings()
        self.subculture_to_focused_authors = None
        self.author_to_focus = None
    
    def augment_posting_data(self, data):
        data = data.reset_index()
        data['category'] = data['subreddit'].apply(lambda x: SUB_TO_SUBCULTURE.get(x.lower(), x))
        data['is_manosphere'] = data['subreddit'].apply(lambda x: "MANOSPHERE" if x.lower() in MANOSPHERE else x)
        return data

    def descriptive_statistics(self):
        agg_unique = self.mano_in_mano.groupby("category").nunique()
        agg_sum = self.mano_in_mano.groupby("category").sum()
        subculture_stats = pd.concat((agg_unique[['author', 'subreddit']],  agg_sum['id']), axis=1)
        subculture_stats.columns = ['num_authors', 'num_subreddits', 'num_posts']
        return subculture_stats.loc[SUBCULTURES]


    # User focus functions
    def compute_focus(self, data, groupby_column, threshold=0.5):
        posts_per_subreddit = data.groupby(["author", groupby_column]).sum()
        # Find the proportion of posts per subreddit (for each author)
        author_distribution = posts_per_subreddit.groupby(level=0).transform(lambda x: x/x.sum())
        # Find authors with more than 50% of their posts in the same subreddit
        focused_authors = author_distribution[author_distribution['id'] > threshold].reset_index()
        return focused_authors
        
    def compute_platform_wide_focus(self, compute_per_subculture, only_manospheric_focuses=True, threshold=0.5):
        if compute_per_subculture:
            focused_authors = self.compute_focus(self.manospheric_author_data, groupby_column = "category", threshold=threshold)
            manosphere_focused_authors = focused_authors[focused_authors['category'].isin(SUBCULTURES)]
        else:
            focused_authors = self.compute_focus(self.manospheric_author_data, groupby_column = "is_manosphere", threshold=threshold)
            manosphere_focused_authors = focused_authors[focused_authors['is_manosphere'] == "MANOSPHERE"]
        
        if not only_manospheric_focuses:
            return focused_authors
        else:
            return manosphere_focused_authors

    def compute_manosphere_internal_focus(self, set_focused_authors, only_manospheric_focuses=True):
        focused_authors = self.compute_focus(self.mano_in_mano, groupby_column="category")
        subculture_focused_authors = focused_authors[focused_authors['category'].isin(SUBCULTURES)]
        if set_focused_authors:
            self.subculture_to_focused_authors = subculture_focused_authors.groupby('category')['author'].agg(list).to_dict()
            self.author_to_focus = subculture_focused_authors.set_index("author")['category'].to_dict()
            Serialization.save_obj(self.author_to_focus, "author_to_focus_aug_23") #latest is after removing posts that are too small after removing punctuation
        
        if not only_manospheric_focuses:
            return focused_authors
        return subculture_focused_authors
    
    def compute_manosphere_focus(self):
        pass


    def compute_subculture_focus(self, set_focused_authors=True):
        pw_focus = self.compute_platform_wide_focus(compute_per_subculture=True)
        mi_focus = self.compute_manosphere_internal_focus(set_focused_authors=set_focused_authors)
        mano_focus = self.compute_manosphere_focus()

        subculture_statistics = self.descriptive_statistics()
        authors_per_subculture = subculture_statistics['num_authors']

        pw_focus_prop = pw_focus.groupby("category").count()['author'] / authors_per_subculture
        mi_focus_prop = mi_focus.groupby("category").count()['author'] / authors_per_subculture

        all_focus_stats = pd.concat((pw_focus_prop, mi_focus_prop), axis=1)
        all_focus_stats.columns = ['platform_wide', 'manosphere_internal']

        return all_focus_stats.loc[SUBCULTURES]


    # User crossposting patterns
    def compute_first_order_crossposting(self):

        if (self.subculture_to_focused_authors is None) or (self.author_to_focus is None):
            self.compute_manosphere_internal_focus(set_focused_authors=True)
            
        at_least_10_outside_mano = self.mano_outside_mano[self.mano_outside_mano['id'] >= 10]
        
        
        num_authors_per_sub = at_least_10_outside_mano.groupby("subreddit").count()['author']
        num_posts_per_sub = at_least_10_outside_mano.groupby("subreddit").sum()['id']
        first_order_stats = pd.concat((num_authors_per_sub, num_posts_per_sub), axis=1)
        first_order_stats.columns = ['num_authors', 'num_posts']
        
        at_least_10_outside_mano['focus'] = at_least_10_outside_mano['author'].apply(lambda x: self.author_to_focus.get(x, np.nan))
        at_least_10_outside_mano = at_least_10_outside_mano.dropna()
        num_subculture_per_sub = at_least_10_outside_mano.groupby(["subreddit", "focus"]).count()
        print("at least 10 focused")
        num_subculture_per_sub = num_subculture_per_sub[num_subculture_per_sub['id'] >= 0].reset_index()
        subcultures_per_sub = num_subculture_per_sub.groupby("subreddit")['focus'].agg(list).to_dict()
        
        first_order_stats['subcultures_with_at_least_30_authors'] = first_order_stats.index.map(lambda x: subcultures_per_sub.get(x, np.nan))
        first_order_stats = first_order_stats.dropna()
        first_order_stats['num_subcultures'] = first_order_stats['subcultures_with_at_least_30_authors'].apply(len)

        first_order_stats['in_top_10K_most_active'] = first_order_stats.index.map(lambda x: x.lower() in self.com_embeddings.sub_to_idx)
        first_order_stats['manosphere_sim'] = first_order_stats.index.map(lambda x: self.com_embeddings.compute_similarity_to_manosphere(x))

        return first_order_stats

    def compute_higher_order_crossposting(self, top_k=30):
        subculture_to_most_similar = {}
        for subculture in SUBCULTURES:
            most_similar = self.com_embeddings.compute_subculture_embedding_similarity(subculture, SUB_TO_SUBCULTURE, top_k)
            subculture_to_most_similar[subculture] = most_similar
        return pd.DataFrame.from_dict(subculture_to_most_similar, orient='index').T
