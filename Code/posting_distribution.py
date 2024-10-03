from project_utils import *

class PostingDistribution:

    def __init__(self, load_from_start, load_presaved, print_output=True):
        self.load_from_start = load_from_start
        self.load_presaved = load_presaved
        self.print_output = print_output



    def raw_load_posting_distribution(self, load_from_scratch):
        if load_from_scratch: # Takes about 4 minutes to run
            for year in ["2014", "2015", "2016", "2017"]:
                replies = Serialization.load_obj(f"RC_reddit_reply_sample_dist_{year}")
                parent_comments = Serialization.load_obj(f"RC_reddit_parent_sample_dist_{year}")
                parent_submissions = Serialization.load_obj(f"RS_reddit_parent_sample_dist_{year}")
                all_dist = replies.add(parent_comments, fill_value=0).add(parent_submissions, fill_value=0)
                print(all_dist.sum())
                Serialization.save_obj(all_dist, f"full_{year}_posting_distribution")

            y_2014 = Serialization.load_obj("full_2014_posting_distribution")
            y_2015 = Serialization.load_obj("full_2015_posting_distribution")
            y_2016 = Serialization.load_obj("full_2016_posting_distribution")
            y_2017 = Serialization.load_obj("full_2017_posting_distribution")

            # Takes about 4 min to run
            first_two = y_2014.add(y_2015, fill_value=0)
            last_two = y_2016.add(y_2017, fill_value=0)
            all_years = first_two.add(last_two, fill_value=0)
            
            print("Saving...")
            Serialization.save_obj(all_years, "all_data_reddit_sample_dist_all_years")
        else:
            # Takes 32 seconds
            print("Loading from serialized object...")
            # all_years = Serialization.load_obj("all_data_reddit_sample_dist_all_years")
            all_years = Serialization.load_obj("aggregate_posting_distribution")

        return all_years



    def extract_active_authors(self, posting_distribution, minimum_num_posts=100):
        # Get people who posted at least 100 times across the dataset (takes 50 seconds) (cumulatively about 2m 05s)
        posts_per_author = posting_distribution.groupby(level=0).sum()
        freq_posters = posts_per_author[posts_per_author['id'] >= minimum_num_posts].index
        freq_overall = posting_distribution[posting_distribution.index.get_level_values(0).isin(freq_posters.tolist())]
        Serialization.save_obj(freq_overall, "active_posters_2014_2017")
        return freq_overall


    def load_data(self):
        self.all_data = self.load_posting_distribution(self.load_presaved, self.load_from_start)
        self.ever_posted_manosphere_data = pd.read_csv(TOTAL_SAMPLE_FULL_DATA_DIR + f"manosphere.csv").groupby(['author', 'subreddit']).count()
        ever_posted_manosphere_min_100_posts = self.all_data[self.all_data.index.get_level_values(1).str.lower().isin(MANOSPHERE)]
        # Valid Manospheric authors are those who have posted at least 100 posts on Reddit, and at least 10 of those on the Manosphere
        self.valid_manospheric_authors, self.manosphere_in_manosphere = groupby_threshold(ever_posted_manosphere_min_100_posts, operation="sum", is_index_level=True, groupby_column=0, threshold_column="id", threshold=10, to_print=False)
        self.baseline_author_data = self.all_data[~self.all_data.index.get_level_values(0).isin(self.ever_posted_manosphere_data.index.get_level_values(0))]
        self.manospheric_author_data = self.all_data[self.all_data.index.get_level_values(0).isin(self.valid_manospheric_authors)]
        
        print("Baseline and Manosphere Author Overlap")
        intersect_overlap(self.baseline_author_data.index.get_level_values(0).unique(), self.valid_manospheric_authors)
        if self.print_output:
            print(f"Number of authors that have posted in the Manosphere: {self.ever_posted_manosphere_data.index.get_level_values(0).nunique()}")
            print(f"Number of posts in the Manosphere: {self.ever_posted_manosphere_data['id'].sum()}")
            print(f"Number of authors who have posted in the Manosphere at least 10 times: {len(self.valid_manospheric_authors)}")

            print(f"Baseline group statistics:")
            compute_posting_distribution_statistics(self.baseline_author_data)
            
            print(f"Manospheric group statistics:")
            compute_posting_distribution_statistics(self.manospheric_author_data)

        # Ensure there is no overlap between the user groups
        
        self.mano_in_mano = self.manospheric_author_data[self.manospheric_author_data.index.get_level_values(1).str.lower().isin(MANOSPHERE)].reset_index()
        self.mano_outside_mano = self.manospheric_author_data[~self.manospheric_author_data.index.get_level_values(1).str.lower().isin(MANOSPHERE)].reset_index()

    def load_posting_distribution(self, load_presaved, load_from_start):
        if load_presaved:
            final_out =  Serialization.load_obj("active_posters_2014_2017")
        else:
            all_years = self.raw_load_posting_distribution(load_from_scratch=load_from_start)
            
            if self.print_output:
                compute_posting_distribution_statistics(all_years)
            final_out = self.extract_active_authors(all_years)
        if self.print_output:
            print("Statistics for all authors with at least 100 posts/comments")
            compute_posting_distribution_statistics(final_out)
        return final_out