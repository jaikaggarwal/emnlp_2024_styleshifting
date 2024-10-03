import googleapiclient
from googleapiclient import discovery
import json
from core_utils import *

API_KEY = # ADD API KEY

ATTRIBUTE_DICT = {
  # Toxicity features
  'TOXICITY': {},
  'SEVERE_TOXICITY': {},
  'IDENTITY_ATTACK': {},
  'INSULT': {},
  'THREAT': {},
  'INFLAMMATORY': {},
  'LIKELY_TO_REJECT': {},
  'SEXUALLY_EXPLICIT': {},
  'OBSCENE': {},
  'LIKELY_TO_REJECT': {},

  # Aggressiveness features
  'ATTACK_ON_AUTHOR': {},
  'ATTACK_ON_COMMENTER': {},

  # Content quality
  'INCOHERENT': {},
  'UNSUBSTANTIAL': {},
  'SPAM': {},

  # Other features I want to include
  'FLIRTATION': {},
  'PROFANITY': {},
}

def load_existing_ids():
    dirs = os.listdir(PERSPECTIVE_API_DIR)
    all_ids = []
    for dir in tqdm(dirs):
        curr_files = os.listdir(PERSPECTIVE_API_DIR + dir + "/")
        curr_ids = [file[ : file.index(".")] for file in curr_files]
        all_ids.extend(curr_ids)
    return set(all_ids)

def load_batch_info():
    dirs = os.listdir(PERSPECTIVE_API_DIR)
    batch_nums = [int(dir[dir.index("_") + 1:]) for dir in dirs]
    curr_batch = max(batch_nums)
    curr_dir = f"batch_{curr_batch}"
    curr_batch_size = len(os.listdir(PERSPECTIVE_API_DIR))
    return curr_batch, curr_dir, curr_batch_size

def perspective_api_call(data):

    client = discovery.build(
      "commentanalyzer",
      "v1alpha1",
      developerKey=API_KEY,
      discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
      static_discovery=False,
    )

    rate_limit = 180
    start_time = time.time()
    n_queries_this_minute = 0

    output_dir = PERSPECTIVE_API_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading IDs...")
    all_ids = load_existing_ids()
    print(f"Number of IDs: {len(all_ids)}")
    print("Loading batches...")
    curr_batch, curr_dir, curr_batch_size = load_batch_info()
    for i, line in tqdm(data.iterrows(), total=data.shape[0]):

        if line['id'] in all_ids:
            continue

        output_path = f"{output_dir}/{curr_dir}/{line['id']}.perspective_attributes.json"

        analyze_request = {
            'comment': { 'text': line["body"] },
            'requestedAttributes': ATTRIBUTE_DICT
        }
        response = None
        while response is None:
            try:
                n_queries_this_minute += 1
                response = client.comments().analyze(body=analyze_request).execute()
                time.sleep(55 / rate_limit) #make it slightly faster to allow for 3QPS
            except googleapiclient.errors.HttpError as e:
                print('bad response code:', e)
                if e.resp.status == 429:
                    # this shouldn't happen, since we only query rate_limit
                    # times per minute. Including this just in case.
                    time.sleep(1)
                else:
                    time.sleep(1)
                    break
        
        all_ids.add(line['id'])
        curr_batch_size += 1
        if curr_batch_size > 10000:
            curr_batch += 1
            curr_dir = f"batch_{curr_batch}"
            curr_batch_size = 0
            print(f"Now starting batch #{curr_batch}")
            if not os.path.exists(f"{output_dir}/{curr_dir}/"):
                os.makedirs(f"{output_dir}/{curr_dir}/")

        if response is None:
            with open(output_path, 'w') as f: pass
            print(f"skipping line: {line}")
            continue

        with open(output_path, 'w') as f:
            json.dump(response, f)

        if n_queries_this_minute == rate_limit:
            print("Exceeded limit")
            time_elapsed = time.time() - start_time
            if time_elapsed < 60:
                time.sleep(60 - time_elapsed)
            start_time = time.time()
            n_queries_this_minute = 0


def extract_toxicity_scores(data):
    output_dir = PERSPECTIVE_API_DIR
    dirs = os.listdir(output_dir)
    id_to_score = {}
    unfinished_files = []
    for curr_dir in tqdm(dirs):
        files = os.listdir(output_dir + curr_dir)
        for file in files:
            try:
                with open(output_dir + curr_dir + "/" + file, "r") as curr:
                    curr_id = file[:file.find(".")]
                    try:
                        data = json.load(curr)
                        score = data['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value']
                        id_to_score[curr_id] = score
                    except:
                        id_to_score[curr_id] = np.nan
            except:
                print(f"Missed file: {output_dir + curr_dir + '/' + file}")
                unfinished_files.append(output_dir + curr_dir + "/" + file)

    for file in unfinished_files:
        with open(file, "r") as curr:
            curr_id = file[:file.find(".")]
            try:
                data = json.load(curr)
                score = data['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value']
                id_to_score[curr_id] = score
            except:
                id_to_score[curr_id] = np.nan
    return id_to_score