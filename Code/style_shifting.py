from project_utils import *
from nltk import word_tokenize
import nltk


class StyleShifting:

    def __init__(self, host_subreddit, posting_dist, user_engagement):
        self.subreddit = host_subreddit
        self.posting_dist = posting_dist
        self.author_to_focus = user_engagement.author_to_focus
        self.first_post_in_mano = self.load_first_mano_post_data()


    def load_first_mano_post_data(self):
        first_mano_post = pd.read_csv(ABRIDGED_DATA_DIR + "manospheric_first_posts.csv")
        first_post_in_mano = first_mano_post.set_index("author")['created_utc'].to_dict()
        return first_post_in_mano
    

    def load_subreddit_data(self, subreddit):
        """
        Extract all data that was written after a Manospheric author's first post in the Manosphere.
        Convert negative scores to 0.
        Split the data up into replies and parents.
        Keep only the replies by people who have posted at least 10 times on the subreddit.
        Keep only the parents that correspond to those replies.
        """


        data = pd.read_csv(TOTAL_SAMPLE_FULL_DATA_DIR + f"{subreddit}.csv")
        # If written by a Manospheric author, only keep if after their first post on the Manosphere
        data['to_keep'] = data.apply(lambda x: (x['author'] not in self.first_post_in_mano) or (self.first_post_in_mano[x['author']] <= x['created_utc']), axis=1)
        data = data[data['to_keep']]
        data = split_parent_identifier(data)
 
        data['score'] = data['score'].fillna(-10)
        data['score'] = data['score'].apply(lambda x: 0 if x <=0 else x)

        print(data.shape)
        print(data['author'].nunique())
        data = data[data['author'] != "autotldr"]
        print(data.shape)
        print(data['author'].nunique())

        valid_reply_ids = pd.read_csv(TOTAL_SAMPLE_IDS_ONLY_DIR + f"{subreddit}_valid_reply_ids.txt", header=None)[0].tolist()
        valid_parent_ids = pd.read_csv(TOTAL_SAMPLE_IDS_ONLY_DIR + f"{subreddit}_valid_parent_ids.txt", header=None)[0].tolist()

        replies = data[data['id'].isin(valid_reply_ids)]
        parents = data[data['id'].isin(valid_parent_ids)]
        # Further filter replies to ensure that the parents are present in this dataset, as they may have been removed when getting all posts after the cutoff date
        replies = replies[replies['cropped_id'].isin(parents['id'])]
        _, valid_replies_df = groupby_threshold(replies, "count", is_index_level=False, groupby_column="author", threshold_column="id", threshold=10, to_print=False)
        valid_parents_df = parents[parents['id'].isin(valid_replies_df['cropped_id'])]

        return valid_replies_df, valid_parents_df


    def load_data(self):
        self.manosphere_replies_df, self.manosphere_parents_df = self.load_subreddit_data("manosphere")
        self.valid_manospheric_authors = intersect_overlap(self.posting_dist.valid_manospheric_authors, self.manosphere_replies_df['author'].unique())
        self.host_replies_df, self.host_parents_df  = self.load_subreddit_data(self.subreddit)
        self.no_manospheric_host_replies_df = self.host_replies_df[~self.host_replies_df['author'].isin(self.first_post_in_mano)]
        self.valid_baseline_authors = self.no_manospheric_host_replies_df['author'].unique()
        print(f"Number of {self.subreddit} posts total: {self.host_replies_df.shape[0]}")
        print(f"Number of {self.subreddit} posts total after removing Manospheric individuals: {self.no_manospheric_host_replies_df.shape[0]}")

        print("Any authors who have replied in the Manosphere at least 10 times|| RQ1 Baseline data")
        intersect_overlap(self.manosphere_replies_df['author'].unique(), self.posting_dist.baseline_author_data.index.get_level_values(0).unique())
        print("Any authors who have replied in the Manosphere at least 10 times || RQ1 Manospheric authors")
        intersect_overlap(self.manosphere_replies_df['author'].unique(), self.posting_dist.valid_manospheric_authors)
        print(f"Any authors who have replied in the Manosphere at least 10 times || All authors who have posted on r/{self.subreddit} at least 10 times")
        intersect_overlap(self.manosphere_replies_df['author'].unique(), self.host_replies_df['author'].unique())
        print(f"Any authors who have replied in the Manosphere at least 10 times || All non-Manospheric authors who have posted on r/{self.subreddit} at least 10 times")
        intersect_overlap(self.manosphere_replies_df['author'].unique(), self.valid_baseline_authors)


    def load_subculture_data(self, to_save=False):
        manosphere_replies_copy = self.manosphere_replies_df.copy()
        manosphere_replies_copy = manosphere_replies_copy[manosphere_replies_copy['author'].isin(self.posting_dist.valid_manospheric_authors)]
        manosphere_replies_copy['focus'] = manosphere_replies_copy['author'].apply(lambda x: self.author_to_focus.get(x, np.nan))
        manosphere_replies_copy['sub_focus_match'] = manosphere_replies_copy.apply(lambda x: SUB_TO_SUBCULTURE[x['subreddit'].lower()] == x['focus'], axis=1)
        author_in_subculture = manosphere_replies_copy[manosphere_replies_copy['sub_focus_match']]

        _, self.active_authors_in_subculture = groupby_threshold(author_in_subculture, "count", False, "author", "id", 10, False)
        if to_save:
            Serialization.save_obj(self.active_authors_in_subculture, "stylistic_differences_within_manosphere_aug_23")


    def get_active_on_host(self):
       
        self.manosphere_on_host_text = self.host_replies_df[self.host_replies_df['author'].isin(self.valid_manospheric_authors)]
        self.baseline_on_host_text = self.host_replies_df[self.host_replies_df['author'].isin(self.valid_baseline_authors)]

        self.manosphere_on_host_posting_dist = self.manosphere_on_host_text.groupby("author").count()
        self.baseline_on_host_posting_dist = self.baseline_on_host_text.groupby("author").count()

        print(f"Manospheric authors on {self.subreddit} || Baseline authors on {self.subreddit}")
        intersect_overlap(self.manosphere_on_host_text['author'].unique(), self.baseline_on_host_text['author'].unique())
        print(f"Manospheric authors on Manosphere || Baseline authors on {self.subreddit}")
        intersect_overlap(self.manosphere_replies_df['author'].unique(), self.baseline_on_host_text['author'].unique())



    def generate_test_set(self, load_from_scratch):

        # Extract an equivalent number of control authors with similar posting volume
        # This is for the **test set **
        if load_from_scratch:
            # Extract data
            baseline_on_host_copy = self.baseline_on_host_posting_dist.copy()
            new_baseline = []
            for _, user in tqdm(self.manosphere_on_host_posting_dist.iterrows(), total=len(self.manosphere_on_host_posting_dist)):
                try:
                    sample = baseline_on_host_copy[baseline_on_host_copy['id'] == user['id']]
                    sub_sample = sample.sample(random_state=48)
                except:
                    #  
                    diffs = np.abs(baseline_on_host_copy['id'] - user['id'])
                    sample = diffs[diffs == diffs.min()]
                    sample = baseline_on_host_copy[baseline_on_host_copy.index.isin(sample.index)]
                    # sample = baseline_on_host_copy[baseline_on_host_copy['id'].between(0.8*user['id'], 1.2*user['id'])]
                    sub_sample = sample.sample(random_state=48)
                baseline_on_host_copy = baseline_on_host_copy.drop(labels=sub_sample.index)
                new_baseline.append(sub_sample)
                
            controlled_baseline_on_host = pd.concat(new_baseline)
            Serialization.save_obj(self.manosphere_on_host_posting_dist, f"{self.subreddit}_manospheric_test_authors") # TODO: CHANGE SAVE NAME f"rq3_manospheric_posters_{self.subreddit}")
            Serialization.save_obj(controlled_baseline_on_host,  f"{self.subreddit}_baseline_test_authors")#, # TODO: CHANGE SAVE NAME  f"rq3_controls_{self.subreddit}")
            Serialization.save_obj(dict(zip(self.manosphere_on_host_posting_dist.index, controlled_baseline_on_host.index)), f"{self.subreddit}_matched_authors_v2")
        else:
            print("Loading pre-saved testing data")
            self.manosphere_on_host_posting_dist = Serialization.load_obj(f"{self.subreddit}_manospheric_test_authors")
            controlled_baseline_on_host = Serialization.load_obj(f"{self.subreddit}_baseline_test_authors")

        self.controlled_baseline_on_host_posting_dist = controlled_baseline_on_host
        self.controlled_baseline_on_host_text =  self.baseline_on_host_text[self.baseline_on_host_text['author'].isin(controlled_baseline_on_host.index)]

    def generate_validation_set(self, load_from_scratch):
        if load_from_scratch:
            validation_data = self.manosphere_replies_df[self.manosphere_replies_df['author'].isin(self.manosphere_on_host_posting_dist.index)]
            Serialization.save_obj(validation_data, f"{self.subreddit}_validation_set")
        else:
            print("Loading pre-saved validation data")
            validation_data = Serialization.load_obj(f"{self.subreddit}_validation_set")
        self.validation_data = validation_data

    def generate_parent_set(self):
        manospheric_test_data = self.manosphere_on_host_text
        baseline_test_data = self.controlled_baseline_on_host_text
        total = pd.concat((baseline_test_data, manospheric_test_data, self.validation_data))
        all_parent_posts = pd.concat((self.host_parents_df, self.manosphere_parents_df))
        hosted_parent_posts = all_parent_posts[all_parent_posts['id'].isin(set(total['cropped_id']))]
        self.attested_parent_data = hosted_parent_posts


    def filter_training_set_pool(self):

        baseline_test_group = set(self.controlled_baseline_on_host_posting_dist.index)
        remaining_baseline_users = set(self.valid_baseline_authors).difference(baseline_test_group)
        unused_baseline_posts = self.no_manospheric_host_replies_df[self.no_manospheric_host_replies_df['author'].isin(remaining_baseline_users)]
        # unused_baseline_posts = unused_baseline_posts[~unused_baseline_posts['author'].isin(self.attested_parent_data['author'].unique())]

        # For the training set, we only get authors who have posted within their focus
        manosphere_test_group = set(self.manosphere_on_host_posting_dist.index)
        remaining_manosphere_users = set(self.valid_manospheric_authors).difference(manosphere_test_group)       
        unused_manosphere_posts = self.active_authors_in_subculture[self.active_authors_in_subculture['author'].isin(remaining_manosphere_users)]
        # unused_manosphere_posts = unused_manosphere_posts[~unused_manosphere_posts['author'].isin(self.attested_parent_data['author'].unique())]
        intersect_overlap(unused_manosphere_posts['author'].unique(), unused_baseline_posts['author'].unique())
        
        return unused_manosphere_posts, unused_baseline_posts


    def generate_training_set(self, load_from_scratch):

        unused_manosphere_posts, unused_baseline_posts = self.filter_training_set_pool()

        # Step 2: Sample a set of Manospheric users, as below        
        self.sample_manosphere_training_set(unused_manosphere_posts)

        # Step 3: Sample a baseline training set using an appropriate sampling function, also according to score
        self.sample_baseline_authors(unused_baseline_posts, load_from_scratch=load_from_scratch)

    

    def sample_manosphere_training_set(self, unused_manosphere_posts):
        # Once we have the remaining Manospheric authors, we need to 
        # sample from them to generate a training distribution. Unlike before
        # We can't just sample 1000 authors. Instead, we need to set a minimum
        # for one subset of the Manosphere (e.g. 30), and then sample the rest
        # proportionately to how many there are in our test set.

        # Also, we need to only get author's posts within their subculture. [Done in sample_manosphere]

        remaining_authors = pd.DataFrame(unused_manosphere_posts.groupby("author")['score'].mean()) 
        remaining_authors['focus'] = remaining_authors.index.map(lambda x: self.author_to_focus.get(x))
        remaining_authors = remaining_authors.dropna(subset=['focus']).reset_index()

        test_author_focus_dist = self.manosphere_on_host_posting_dist.index.map(self.author_to_focus).dropna()
        subculture_author_dist = test_author_focus_dist.value_counts() / test_author_focus_dist.shape[0]
        users_per_subculture = np.ceil(subculture_author_dist/subculture_author_dist.min() * 50).astype(int).to_dict()

        manosphere_training_set = remaining_authors.groupby("focus").apply(lambda x: x.sample(n=users_per_subculture[x.name], weights=x['score'], random_state=2)).reset_index(drop=True)
        self.manosphere_training_data = self.active_authors_in_subculture[self.active_authors_in_subculture['author'].isin(manosphere_training_set['author'])]
        return 


    def sample_baseline_authors(self, unused_baseline_posts, load_from_scratch):
        # Potentially re-use sampling from previous part?
        # Create control for ML classifier

        # TODO: Do we want to sample baseline authors by posting volume, or by score??
        # TODO: We can't do both very well simultaneously
        if load_from_scratch:
            baseline_posts_by_author = unused_baseline_posts.groupby("author").count().sort_values(by='body')
            manosphere_training_set_by_author = self.manosphere_training_data.groupby("author").count().sort_values(by='body')

            baseline_scores_by_author = unused_baseline_posts[['author', 'score']].groupby("author").mean()
            baseline_posts_per_author_clone = baseline_posts_by_author.copy()

            mano_authors = []
            controls = []
            posting_volume_vals = manosphere_training_set_by_author['body'].unique()
            for val in tqdm(posting_volume_vals):
                n = manosphere_training_set_by_author[manosphere_training_set_by_author['body'] == val].shape[0]
                try:
                    sub_ask = baseline_posts_per_author_clone[baseline_posts_per_author_clone['body'] == val]
                    sub_ask_scores = baseline_scores_by_author.loc[sub_ask.index]['score']
                    to_add = sub_ask.sample(n, random_state=48, weights=sub_ask_scores)
                except:
                    # sub_ask = baseline_posts_per_author_clone[baseline_posts_per_author_clone['body'].between(0.8*val, 1.2*val)]
                    # sub_ask_scores = baseline_scores_by_author.loc[sub_ask.index]['score']
                    # to_add = sub_ask.sample(n, random_state=48, weights=sub_ask_scores)
                    baseline_posts_per_author_clone['diffs'] = np.abs(baseline_posts_per_author_clone['id'] - val)
                    to_add = baseline_posts_per_author_clone.sort_values(by=['diffs', 'score'],  ascending=[True,False]).head(n)

                baseline_posts_per_author_clone = baseline_posts_per_author_clone.drop(labels=to_add.index)
                controls.append(to_add)
                mano_authors.append(manosphere_training_set_by_author[manosphere_training_set_by_author['body'] == val])


            sample_ask_agg = pd.concat(controls)
            sample_mano = pd.concat(mano_authors)

            Serialization.save_obj(dict(zip(sample_ask_agg.index, sample_mano.index)), f"{self.subreddit}_matched_training_authors")
            print(f"Number of Manospheric authors {manosphere_training_set_by_author.shape[0]} across {manosphere_training_set_by_author.sum()['body']} posts")
            print(f"Number of baseline authors {sample_ask_agg.shape[0]} across {sample_ask_agg.sum()['body']} posts")
            baseline_training_data = unused_baseline_posts[unused_baseline_posts['author'].isin(sample_ask_agg.index)]
            
            Serialization.save_obj(baseline_training_data, f"{self.subreddit}_baseline_training_set")
            Serialization.save_obj(self.manosphere_training_data, f"{self.subreddit}_manospheric_training_set")

        else:
            baseline_training_data = Serialization.load_obj(f"{self.subreddit}_baseline_training_set")
            self.manosphere_training_data = Serialization.load_obj(f"{self.subreddit}_manospheric_training_set")
        








    def compile_data(self):
        manospheric_test_data = self.manosphere_on_host_text
        baseline_test_data = self.controlled_baseline_on_host_text

        manospheric_training_data = Serialization.load_obj(f"{self.subreddit}_manospheric_training_set")
        baseline_training_data = Serialization.load_obj(f"{self.subreddit}_baseline_training_set")

        validation_data = Serialization.load_obj(f"{self.subreddit}_validation_set")

        baseline_training_data['label'] = 0
        manospheric_training_data['label'] = 1

        baseline_test_data['label'] = 0
        manospheric_test_data['label'] = 1
        validation_data['label'] = 2

        baseline_training_data['is_training'] = 1
        manospheric_training_data['is_training'] = 1

        baseline_test_data['is_training'] = 0
        manospheric_test_data['is_training'] = 0
        validation_data['is_training'] = 0

        total = pd.concat((baseline_test_data, manospheric_test_data, validation_data))
        
        all_parent_posts = pd.concat((self.host_parents_df, self.manosphere_parents_df))
        hosted_parent_posts = all_parent_posts[all_parent_posts['id'].isin(set(total['cropped_id']))]
        adjusted_total = total[total['cropped_id'].isin(set(hosted_parent_posts['id']))]

        adjusted_total['is_parent'] = 0
        hosted_parent_posts['is_parent'] = 1
        hosted_parent_posts['is_training'] = 0

        test_and_validation = pd.concat((adjusted_total, hosted_parent_posts))
        all_data = pd.concat((baseline_training_data, manospheric_training_data, test_and_validation))
        print(all_data.shape)

        Serialization.save_obj(all_data.reset_index(drop=True), f"{self.subreddit}_all_subreddit_data")