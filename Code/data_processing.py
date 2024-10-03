from project_utils import *
import json
from multiprocessing import Pool
import gc
import csv
import string
from nltk import word_tokenize
import fasttext
import random

# Currently, we have a list of all the comments that meet our selection criteria that come from a 10% sample of Reddit
# We also have the parent comments and parent posts of all of these comments, whether or not they come from a 10% sample of Reddit

# Now, we need to find all ids that are in both the parent comments dataset [DONE]
# We also need to remove duplicates posts. [DONE]
# We also need to remove all children that do not have parents in our dataset [DONE]

# While reading, we also need to track the first post per author
# While reading, we also need to track the first Manospheric post per author

# Step 1: Read through each file in REPLY_DIR and PARENT_DIR line by line
# Step 2: For each JSON object, only keep the fields listed above. If RS file, add t0_top to parent_id field
# Step 3: Then, add the JSON object to a dictionary of {subreddit: [json]}
# The subreddit keys are determined by the top 100 subreddits by #authors in community embedding metadata, as well as one for all Manosphere related subreddits, and one for the remainder of the platform
# Whenever one list reaches 1M entries, write all that data to a file in the appropriate directory
# Step 2: and save to CSV file in directory BASED ON SUBREDDIT_MONTH (enables extremely efficient parallelization and grouping)


fields_to_keep = ['author', 'body', 'controversiality', 'created_utc', 'id', 'parent_id', 'score', 'subreddit']


metadata = pd.read_csv(EMBEDDING_METADATA, delimiter='\t')
metadata = metadata.sort_values(by='total_users', ascending=False)
top_100 = metadata.head(100)['community'].tolist()
middle_1000 = metadata.head(1000).tail(900)['community'].tolist()
top_100 = [sub.lower() for sub in top_100]
middle_1000 = [sub.lower() for sub in middle_1000]

for sub in top_100 + ['middle_1000', 'miscellaneous', 'manosphere']:
    out_dir = f"{DATA_BY_SUBREDDIT_DIR}{sub}/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


PRETRAINED_FASTTEXT_MODEL_PATH = #TODO
fasttext_model = fasttext.load_model(PRETRAINED_FASTTEXT_MODEL_PATH)

def extra_preprocessing(post):
    new_body_no_punctuation = post.translate(str.maketrans('', '', string.punctuation))
    if len(word_tokenize(new_body_no_punctuation)) < 5:
        return False
    else:
        return True
    

def is_english(data):
    sentence_list = [line['body'] for line in data]
    labels, probs = fasttext_model.predict(sentence_list)
    new_data = []
    for label, line in zip(labels, data):
        if label[0].endswith("en"):
            new_data.append(line)
    return new_data
        
def reorganize_wrapper():
    reply_files = sorted(os.listdir(REPLY_DIR))
    parent_files = sorted(os.listdir(PARENT_DIR))

    reply_files = [{"dir": REPLY_DIR, "file": file, "prefix": "RC"} for file in reply_files]
    parent_comment_files = [{"dir": PARENT_DIR, "file": file, "prefix": "RC"}  for file in parent_files if file.startswith("RC")]
    parent_submission_files = [{"dir": PARENT_DIR, "file": file, "prefix": "RS"} for file in parent_files if file.startswith("RS")]

    all_files = reply_files + parent_comment_files + parent_submission_files

    for i in range(12):
        curr_files = all_files[12*i:12*(i+1)]
        with Pool(processes=12) as p:
            r = list(tqdm(p.imap(resort_data_files, curr_files), total=len(curr_files)))
        for sub_to_data, file in zip(r, curr_files):
            print(file['file'])
            sorted_subs = sorted(sub_to_data.keys())
            for sub in tqdm(sorted_subs):
                out_dir = f"{DATA_BY_SUBREDDIT_DIR}{sub}/"
                with open(out_dir + file['file'][:-5] + ".csv", "w", newline="") as f:
                    title = fields_to_keep
                    cw = csv.DictWriter(f, title, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    cw.writeheader()
                    cw.writerows(sub_to_data[sub])


def resort_data_files(data_parameters):
    directory = data_parameters['dir']
    filename = data_parameters['file']
    prefix = data_parameters['prefix']

    sub_to_data = defaultdict(list)
    counter = 0
    print(filename)
    with open(directory + filename, "r") as f_in:
        for line in f_in:
            counter += 1
            data = json.loads(line)

            to_keep = extra_preprocessing(data['body'])
            if not to_keep:
                continue

            new_line = {field: data.get(field, "t0_parent") for field in fields_to_keep} #comments always have parents
            new_line['subreddit'] = new_line['subreddit'].lower()
            if new_line['subreddit'] in MANOSPHERE:
                sub_to_data['manosphere'].append(new_line)
            elif new_line['subreddit'] in top_100:
                sub_to_data[new_line['subreddit']].append(new_line)
            elif new_line['subreddit'] in middle_1000:
                sub_to_data['middle_1000'].append(new_line)
            else:
                sub_to_data['miscellaneous'].append(new_line)
            #TODO: REMOVE
            # if counter == 1000:
            #     break
    
    # print("Saving CSVs for " + filename)
    sub_to_data = dict(sub_to_data)
    return {sub: is_english(sub_to_data[sub]) for sub in sub_to_data}


        
        # if not os.path.exists(out_dir):
        #     os.makedirs(out_dir)
        # pd.DataFrame(sub_to_data[sub]).to_csv(out_dir + filename[:-5] + ".csv")
    
    gc.collect()


def create_final_sample(sub):
    if os.path.exists(TOTAL_SAMPLE_FULL_DATA_DIR + sub + ".csv"):
        return
    in_dir = f"{DATA_BY_SUBREDDIT_DIR}{sub}/"
    all_files = sorted(os.listdir(in_dir))
    dfs = []
    print(sub)
    for file in tqdm(all_files):
        dfs.append(pd.read_csv(in_dir + file, usecols=['id', 'parent_id']))
    df = pd.concat(dfs)
    df.replace({"t0-parent": "t0_parent"}, inplace=True)
    del dfs
    gc.collect()

    # Get all comments whose parent id is in the set of ids
    df = split_parent_identifier(df)
    valid_replies = df[df['cropped_id'].isin(df['id'])]
    valid_reply_ids = valid_replies['id'].tolist()
    valid_parent_ids = valid_replies['cropped_id'].tolist()

    write_ids_to_file(TOTAL_SAMPLE_IDS_ONLY_DIR, valid_reply_ids, f"{sub}_valid_reply_ids.txt")
    write_ids_to_file(TOTAL_SAMPLE_IDS_ONLY_DIR, valid_parent_ids, f"{sub}_valid_parent_ids.txt")

    valid_post_ids = set(valid_reply_ids + valid_parent_ids)
    df = df[df['id'].isin(valid_post_ids)]
    print(f"# rows before dropping duplicates: {df.shape[0]}")
    df = df.drop_duplicates(subset=['id'])
    print(df.shape[0])
    print(f"# rows after dropping duplicates: {df.shape[0]}")

    tracked_ids = {id: False for id in df['id'].unique()}

    with open(TOTAL_SAMPLE_FULL_DATA_DIR + sub + ".csv", "w", newline="") as f_out:
        title = fields_to_keep
        cw = csv.DictWriter(f_out, title, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        cw.writeheader()
        for file in tqdm(all_files):
            with open(in_dir + file, "r", newline="") as f_in:
                cr = csv.DictReader((line.replace("\0", " ") for line in f_in), fieldnames=fields_to_keep)
                for row in cr:
                    if row['id'] in tracked_ids:
                        if not tracked_ids[row['id']]:
                            row['parent_id'] = "t0_parent" if row['parent_id'] == 't0-parent' else row['parent_id']
                            cw.writerow(row)
                            tracked_ids[row['id']] = True


    

def create_final_sample_wrapper():
    for sub in top_100 + ['middle_1000', 'miscellaneous', 'manosphere']:
        create_final_sample(sub)



def extract_first_manospheric_post_per_author():
    data = pd.read_csv(TOTAL_SAMPLE_FULL_DATA_DIR + "manosphere.csv", usecols=['author', 'created_utc'])
    data.groupby("author").min().to_csv(ABRIDGED_DATA_DIR + "manospheric_first_posts.csv")


def compute_posting_distribution():
    
    first_mano_post = pd.read_csv(ABRIDGED_DATA_DIR + "manospheric_first_posts.csv")
    first_post = first_mano_post.set_index("author")['created_utc'].to_dict()

    all_aggs = None
    for sub in tqdm(top_100 + ['middle_1000', 'miscellaneous', "manosphere"]):
        data = pd.read_csv(TOTAL_SAMPLE_FULL_DATA_DIR + f"{sub}.csv", usecols=['author', 'subreddit', 'id', 'created_utc', 'length'])
        # Insert post length information here!
        data = data[data['length'] >= XXX]
        data['to_keep'] = data.apply(lambda x: (x['author'] not in first_post) or (first_post[x['author']] <= x['created_utc']), axis=1)
        data = data[data['to_keep']]
        agg = pd.DataFrame(data.groupby(["author", "subreddit"])['id'].count())
        if all_aggs is None:
            all_aggs = agg
        else:
            all_aggs = all_aggs.add(agg, fill_value=0)
        
        
    Serialization.save_obj(all_aggs, "aggregate_posting_distribution")


def extract_lengths(subreddit):
    df = pd.read_csv(TOTAL_SAMPLE_FULL_DATA_DIR + f"{subreddit}.csv")
    df['length'] = df['body'].apply(lambda x: len(word_tokenize(x.translate(str.maketrans('', '', string.punctuation)))))
    return (subreddit, df)

def extract_post_length_info():

    smaller_subreddits = top_100[::-1][:99]
    with Pool(12) as p:
        r = list(tqdm(p.imap(extract_lengths, smaller_subreddits), total=len(smaller_subreddits)))
    
    for subreddit, df in r:
        df.to_csv(TOTAL_SAMPLE_FULL_DATA_WITH_LENGTH_DIR + f"{subreddit}.csv")


    for sub in tqdm(["askreddit", 'manosphere', 'middle_1000', 'miscellaneous']):
        with open(TOTAL_SAMPLE_FULL_DATA_WITH_LENGTH_DIR + sub + ".csv", "w", newline="") as f_out:
            title = fields_to_keep + ['length']
            cw = csv.DictWriter(f_out, title, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            cw.writeheader()
            with open(TOTAL_SAMPLE_FULL_DATA_DIR + sub + ".csv", "r", newline="") as f_in:
                cr = csv.DictReader((line.replace("\0", " ") for line in f_in), fieldnames=fields_to_keep)
                for row in cr:
                    row['length'] = len(word_tokenize(row['body'].translate(str.maketrans('', '', string.punctuation))))
                    cw.writerow(row)
    

if __name__ == "__main__":
    extract_post_length_info()
    # reorganize_wrapper()
    # create_final_sample_wrapper()
    # extract_first_manospheric_post_per_author()
    # compute_posting_distribution()







