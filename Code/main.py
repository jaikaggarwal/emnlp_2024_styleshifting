from project_utils import *
from posting_distribution import PostingDistribution
from style_shifting import StyleShifting
from engagement_patterns import UserEngagement
from feature_extraction import *
from scipy.stats import wilcoxon
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr

posting_dist = PostingDistribution(load_from_start=False, load_presaved=True)
posting_dist.load_data()

user_engagement = UserEngagement(posting_dist)
subculture_stats = user_engagement.descriptive_statistics()

# Takes 50 seconds
overall_focus_values = user_engagement.compute_platform_wide_focus(compute_per_subculture=False)
subculture_focus_values = user_engagement.compute_subculture_focus(set_focused_authors=True)
first_order = user_engagement.compute_first_order_crossposting()
higher_order = user_engagement.compute_higher_order_crossposting()

for subreddit_of_interest in ["adviceanimals", "videos", "pics", "funny", "wtf", "gaming"]:#["worldnews", "todayilearned"]: #['news', 'movies', 'askmen', 'askreddit']: #askmen
    style_obj = StyleShifting(subreddit_of_interest, posting_dist, user_engagement)
    style_obj.load_data()
    style_obj.load_subculture_data()
    style_obj.get_active_on_host()
    style_obj.generate_test_set(load_from_scratch=True)
    style_obj.generate_validation_set(load_from_scratch=True)
    style_obj.generate_parent_set()
    style_obj.generate_training_set(load_from_scratch=True)
    style_obj.compile_data()

# for subreddit_of_interest in ["worldnews", "todayilearned"]:#['news', 'movies', 'askmen', 'askreddit']: #askmen
#     augment_dataset(subreddit_of_interest)