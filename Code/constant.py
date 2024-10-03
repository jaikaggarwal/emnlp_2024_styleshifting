RAW_POST_DIR= '/ais/hal9000/datasets/reddit/data_dumps_submissions/'
RAW_COMMENT_DIR= '/ais/hal9000/datasets/reddit/data_dumps/'
ROOT_DIR= '/ais/hal9000/datasets/reddit/icwsm/'
RAW_REPLY_DIR = ROOT_DIR + "all_replies_raw/"
REPLY_DIR = ROOT_DIR + 'all_replies/'
PARENT_DIR = ROOT_DIR + 'all_parents/'
ABRIDGED_DATA_DIR = ROOT_DIR + "abridged_data/"
TMP_DIR = ROOT_DIR + "tmp_data/"
TOTAL_SAMPLE_DIR = ROOT_DIR + "total_reddit_sample/"
TOTAL_SAMPLE_FULL_DATA_DIR = TOTAL_SAMPLE_DIR + "full_data/"
TOTAL_SAMPLE_FULL_DATA_WITH_LENGTH_DIR = TOTAL_SAMPLE_DIR + "full_data_with_length/"

TOTAL_SAMPLE_IDS_ONLY_DIR = TOTAL_SAMPLE_DIR + "ids_only/"
VALID_PARENTS_DIR = ROOT_DIR + "all_validated_parents/"
DATA_BY_SUBREDDIT_DIR = ROOT_DIR + "data_by_subreddit/"

POST_FIELDS_TO_KEEP = ['author', 'selftext', 'title', 'body', 'controversiality', 'created_utc', 'id', 'score', 'subreddit', 'author_flair_text', 'author_flair_css_class']
COMMENT_FIELDS_TO_KEEP = ['author', 'body', 'controversiality', 'created_utc', 'id', 'parent_id', 'score', 'subreddit', 'author_flair_text', 'author_flair_css_class']

SERIALIZATION_DIR = #TODO
PERSPECTIVE_API_DIR =  #TODO
LEXICON_FILE = #TODO
MANOSPHERE_TAXONOMY_FILE = #TODO
YEARS = ["2014", "2015", "2016", "2017"]
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

ZSTD_MAX_WINDOW_SIZE = 2147483648

EMBEDDINGS = '~/AutismHateSpeech/Data/reddit-master-vectors.tsv'
EMBEDDING_METADATA = '~/AutismHateSpeech/Data/reddit-master-metadata.tsv'
