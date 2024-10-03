from core_utils import *
from project_utils import split_parent_identifier
import zstd
from nltk import word_tokenize
import argparse
import json
from multiprocessing import Pool
import gc
import string


def text_is_nan(text):
    if text is np.nan:
        return True


def format_filtered_response(to_keep, to_consider, object):
    """
    to_keep: if False, we immediately discard the data
    to_consider: this is a flag we include for data less than 5 tokens, in case we would like to 
    include it in the future. This is mostly for short post titles that we may want to include.
    object: JSON object of the particular submission/comment
    """
    return {
        "to_keep": to_keep,
        "to_consider": to_consider,
        "object": object
    }


def apply_filters(new_line, prefix):
    """
    Determine whether the post can be kept in our dataset. Note that
    a submission's text is the concatenation of its title and body.
    """

    if type(new_line['subreddit']) == float:
        return format_filtered_response(to_keep=False, to_consider=False, object=new_line)
    # Step 1: Remove if submission title or comment body are nan
 
    # If post is a submission / original post
    if prefix == "RS":
        title_text = new_line['title']
        if text_is_nan(title_text):
            return format_filtered_response(to_keep=False, to_consider=False, object=new_line)
        else:
            title_text = title_text.strip()
            # Add punctuation to the end of the title if it doesn't have any
            if not (title_text[-1] in ["!", ".", "?"]):
                title_text = title_text + "."
        body_text = new_line['selftext']
        if text_is_nan(body_text):
            body_text = ""
        if body_text.lower() in ['[deleted]', '[removed]']:
            return format_filtered_response(to_keep=False, to_consider=False, object=new_line)
        # A submission's text is the concatenation of its title and body.
        new_line_text = (title_text + " " + body_text).strip()

    # If post is a comment
    else:
        new_line_text = new_line['body']
        if text_is_nan(new_line_text):
            return format_filtered_response(to_keep=False, to_consider=False, object=new_line)
        if new_line_text.lower() in ['[deleted]', '[removed]']:
            return format_filtered_response(to_keep=False, to_consider=False, object=new_line)
        
    # Remove all bots, moderators, deleted, removed authors
    # TODO: BETTER REMOVE BOTS
    if new_line['author'] in ["AutoModerator", "[deleted]", "[removed]"]:
        to_keep = False
    elif new_line['author'].lower().endswith("bot"):
        to_keep = False
    else:
        to_keep = True

    # Now we can check to see if the body is long enough
    if to_keep:
        new_body = preprocess(new_line_text)
        new_body_no_punctuation = new_body.translate(str.maketrans('', '', string.punctuation))
        if len(word_tokenize(new_body_no_punctuation)) < 5:
            return format_filtered_response(True, False, new_line)
        new_line['body'] = new_body 
        new_line['subreddit'] = new_line['subreddit'].lower()
        return format_filtered_response(True, True, new_line)
    else:
        return format_filtered_response(False, False, new_line)




def extract_data(extraction_parameters):

    year = extraction_parameters['year']
    month = extraction_parameters['month']
    prefix = extraction_parameters['prefix']

    if prefix == "RS":
        data_dir = RAW_POST_DIR
        fields_to_keep = POST_FIELDS_TO_KEEP
    else: #prefix == "RC"
        data_dir = RAW_COMMENT_DIR
        fields_to_keep = COMMENT_FIELDS_TO_KEEP

    zst_files = [f'{prefix}_{year}-{month}.zst']
    for filename in zst_files:
        # Trackers
        print(filename) # Which file
        counter = 0 # How many bytes we've seen
        loops = 0 # How many windows we've decompressed
        line_counter = 0 # How many AskReddit lines we've seen (used for sampling)
        a = time.time() # Overall time

        with open(f"{RAW_REPLY_DIR}{filename[:-4]}_counts.json", "w") as f_out:
            with open(data_dir + filename, 'rb') as fh:
                dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW_SIZE)
                with dctx.stream_reader(fh) as reader:
                    previous_line = ""
                    while True:
                        chunk = reader.read(2**24)  # 16mb chunks
                        counter += 2**24
                        loops += 1
                        if loops % 2000 == 0:
                            print(f"{counter/10**9:.2f} GB")
                            print(f"{(time.time()-a)/60:.2f} minutes passed")
                        if not chunk:
                            break

                        string_data = chunk.decode('utf-8')
                        lines = string_data.split("\n")

                        for i, line in enumerate(lines[:-1]):
                            if i == 0:
                                line = previous_line + line
                            line = json.loads(line)

                            line_counter += 1
                            if line_counter % 10 != 0: # Sample 10% of our data
                                continue

                            # Add additional filters
                            new_line = {field: line.get(field, np.nan) for field in fields_to_keep}
                            filtered_object = apply_filters(new_line, prefix)
                            if not filtered_object['to_keep']:
                                continue

                            new_line = filtered_object['object']
                            new_line['to_consider'] = filtered_object['to_consider']

                            f_out.write(json.dumps(new_line))
                            f_out.write("\n")
                            # do something with the object here
                        previous_line = lines[-1]



def extract_data_by_id(extraction_parameters):
    """
    In this function, we don't include any sampling. We search from the ENTIRE dataset. We can check if they are in the sample later.

    Make sure ids is a set
    """
    
    
    year = extraction_parameters['year']
    month = extraction_parameters['month']
    prefix = extraction_parameters['prefix']
    ids = extraction_parameters['ids']
    
    assert type(ids) == set

    if prefix == "RS":
        data_dir = RAW_POST_DIR
        fields_to_keep = POST_FIELDS_TO_KEEP
    else: #prefix == "RC"
        data_dir = RAW_COMMENT_DIR
        fields_to_keep = COMMENT_FIELDS_TO_KEEP

    zst_files = [f'{prefix}_{year}-{month}.zst']
    tracked_ids = []

    for filename in zst_files:
        if os.path.exists(f"{PARENT_DIR}{filename[:-4]}_parent_counts.json"):
            df = pd.read_json(f"{PARENT_DIR}{filename[:-4]}_parent_counts.json", lines=True)
            return df['id'].tolist()
        # Trackers
        print(filename) # Which files
        counter = 0 # How many bytes we've seen
        loops = 0 # How many windows we've decompressed
        line_counter = 0 # How many AskReddit lines we've seen (used for sampling)
        a = time.time() # Overall time

        with open(f"{PARENT_DIR}{filename[:-4]}_parent_counts.json", "w") as f_out:
            with open(data_dir + filename, 'rb') as fh:
                dctx = zstd.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW_SIZE)
                with dctx.stream_reader(fh) as reader:
                    previous_line = ""
                    while True:
                        chunk = reader.read(2**24)  # 16mb chunks
                        counter += 2**24
                        loops += 1
                        if loops % 2000 == 0:
                            print(f"{counter/10**9:.2f} GB")
                            print(f"{(time.time()-a)/60:.2f} minutes passed")
                        if not chunk:
                            break

                        string_data = chunk.decode('utf-8')
                        lines = string_data.split("\n")
                        for i, line in enumerate(lines[:-1]):
                            if i == 0:
                                line = previous_line + line
                            line = json.loads(line)
                            line_counter += 1

                            # Add additional filters
                            if line['id'] not in ids:
                                continue
                            tracked_ids.append(line['id'])
                            new_line = {field: line.get(field, np.nan) for field in fields_to_keep}
                            filtered_object = apply_filters(new_line, prefix)
                            if not filtered_object['to_keep']:
                                continue

                            if not filtered_object['to_consider']:
                                continue

                            new_line = filtered_object['object']

                            f_out.write(json.dumps(new_line))
                            f_out.write("\n")

                        previous_line = lines[-1]

    return tracked_ids



# def extract_parent_comments():
#     all_ids = []
#     for year in ["2017"]:#"2014", "2015", "2016",
#         print(year)
#         ids = Serialization.load_obj(f"reddit_sample_dist_{year}_parent_ids")
#         all_ids.extend(ids)
#         print(len(all_ids))
#     all_ids = list(set(all_ids))
#     post_ids = set([id[3:] for id in all_ids if id.startswith("t3")])
#     comment_ids = set([id[3:] for id in all_ids if id.startswith("t1")])
#     del all_ids

#     print("Extracting")
#     for year in ["2017"]:#"2014", "2015", 
#         for quarter in quarter_to_data:
#             extract_data_by_id(year, quarter, post_ids, "RS")
#             extract_data_by_id(year, quarter, comment_ids, "RC")



def extract_data_wrapper(year, prefix):
    extraction_parameter_dicts = [{"year": year, "prefix": prefix, "month": month} for month in MONTHS]
    with Pool(processes=12) as p:
        r = list(tqdm(p.imap(extract_data, extraction_parameter_dicts), total=len(extraction_parameter_dicts)))



def extract_data_by_id_wrapper(year, prefix, ids):
    ids = set(ids)
    extraction_parameter_dicts = [{"year": year, "prefix": prefix, "month": month, "ids": ids} for month in MONTHS]
    with Pool(processes=12) as p:
        r = list(tqdm(p.imap(extract_data_by_id, extraction_parameter_dicts), total=len(extraction_parameter_dicts)))
    return set(flatten(r))


def retain_only_to_consider(file):
    if os.path.exists(REPLY_DIR + file):
        return
    with open(RAW_REPLY_DIR + file, "r", encoding='latin-1') as f_in:
        with open(REPLY_DIR + file, "w") as f_out:
            for line in f_in:
                doc = json.loads(line)
                if doc['to_consider']:
                    f_out.write(json.dumps(doc))
                    f_out.write("\n")

    # df = pd.read_json(RAW_REPLY_DIR + file, lines=True)
    # df[df['to_consider']].to_json(REPLY_DIR + file, orient='records', lines=True)

def retain_only_to_consider_wrapper():
    reply_files = os.listdir(RAW_REPLY_DIR)
    with Pool(processes=12) as p:
        r = list(tqdm(p.imap(retain_only_to_consider, reply_files), total=len(reply_files)))



def extract_all_parents():
    """
    Find the parents of all comments in our dataset. We start with the most recent year
    of data, and extract all parents for that set of comments. We remove the ids for the
    parents that have already been found. For the next most recent year, we append the new set
    of parent ids to those that have not yet been found, and repeat the procedure. This limits
    the amount of memory required to store the huge number of ids we have.
    """
    reply_files = os.listdir(REPLY_DIR)
    year_to_reply_files = {year: sorted([file for file in reply_files if year in file]) for year in YEARS}
    
    remaining_parents_in_comment_data = []
    remaining_parents_in_submission_data = []

    # Iterate backward through the years
    for year in YEARS[::-1]:
        if os.path.exists(ABRIDGED_DATA_DIR + f"{year}_posting_data.csv"):
            print(f"Loading presaved {year} posting data")
            df = pd.read_csv(ABRIDGED_DATA_DIR + f"{year}_posting_data.csv", usecols=['parent_id'])
            for id_val in tqdm(df['parent_id'].tolist()):
                parent_type, parent_id = id_val.split("_")
                if parent_type == "t1":
                    remaining_parents_in_comment_data.append(parent_id)
                else:
                    remaining_parents_in_submission_data.append(parent_id)
            del df
            gc.collect()
        else:
            print(f"Computing {year} posting data from scratch")
            year_id_data = []
            for reply_file in tqdm(year_to_reply_files[year]):
                with open(REPLY_DIR + reply_file, "r", encoding='latin-1') as f:
                    for line in f:
                        doc = json.loads(line)
                        ## Saving a subset of the data to make posting concentration calculations easier
                        year_id_data.append([doc['author'], doc['subreddit'], doc['id'], doc['parent_id']])

                        parent_type, parent_id = doc['parent_id'].split("_")
                        if parent_type == "t1":
                            remaining_parents_in_comment_data.append(parent_id)
                        else:
                            remaining_parents_in_submission_data.append(parent_id)
            pd.DataFrame(year_id_data, columns = ['author', 'subreddit', 'id', 'parent_id']).to_csv(ABRIDGED_DATA_DIR + f"{year}_posting_data.csv")
            del year_id_data
            gc.collect()
        
        write_ids_to_file(remaining_parents_in_comment_data, f"remaining_parents_in_comments_{year}_pre_extraction.txt")
        write_ids_to_file(remaining_parents_in_submission_data, f"remaining_parents_in_submissions_{year}_pre_extraction.txt")
        
        completed_comment_ids = extract_data_by_id_wrapper(year, "RC", remaining_parents_in_comment_data)
        completed_submission_ids = extract_data_by_id_wrapper(year, "RS", remaining_parents_in_submission_data)

        remaining_parents_in_comment_data = list(set(remaining_parents_in_comment_data).difference(completed_comment_ids))
        remaining_parents_in_submission_data = list(set(remaining_parents_in_submission_data).difference(completed_submission_ids))

        write_ids_to_file(remaining_parents_in_comment_data, f"remaining_parents_in_comments_{year}_post_extraction.txt")
        write_ids_to_file(remaining_parents_in_submission_data, f"remaining_parents_in_submissions_{year}_post_extraction.txt")

        del completed_comment_ids
        del completed_submission_ids
        gc.collect()
        

        

if __name__ == "__main__":
    
    # for year in YEARS:
    #     extract_data_wrapper(year, "RC")
    
    # retain_only_to_consider_wrapper()
    extract_all_parents()