# from project_utils import *
from core_utils import *
from perspective_api import *
from nltk import word_tokenize, sent_tokenize
import nltk
import re
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sentence_transformers.SentenceTransformer import torch as pt

pt.cuda.set_device(1)
print(pt.cuda.is_available())
print(pt.__version__)
model = SentenceTransformer('bert-large-nli-mean-tokens')


# TTR Feature
def get_TTR(post):
    post = post.lower()
    post = re.sub(r'\W', ' ', post)
    tokens = word_tokenize(post)
    types = nltk.Counter(tokens)
    if len(tokens) == 0:
        print(post)
        ttr = 0
    else:
        ttr = len(types)/len(tokens)
    return ttr


# For valence feature
def goodness_of_fit(model, true, X):
        y_predicted = model.get_prediction(X)
        pred_vals = y_predicted.summary_frame()['mean']
        fit_val = r2_score(true, pred_vals)
        print(fit_val)
        return fit_val



def init_valence():
    df_vad = pd.read_csv(LEXICON_FILE, delimiter='\t', header=0)
    df_vad = df_vad.dropna().reset_index(drop=True)
    df = df_vad[['Word', 'Valence']]
    valence = np.array(df['Valence'].tolist())

    df['valence_scaled'] = df['Valence']*4 + 1
    df['v_group'] = df['valence_scaled'].apply(np.ceil)
    vad_embeddings = Serialization.load_obj("vad")

    print("LOADING VALENCE MODEL")
    valence_model = Serialization.load_obj('valence_model')
    goodness_of_fit(valence_model, valence, vad_embeddings)
    return valence_model
        


def infer_valence(valence_model, to_encode):
    pt.cuda.empty_cache()
    embeddings = model.encode(to_encode, show_progress_bar=True)
    v_predictions = valence_model.predict(embeddings)
    assert(len(v_predictions) == len(embeddings))
    return v_predictions

def valence_wrapper(original_data):
    
    valence_model = init_valence()

    data = original_data[['id', 'body']]
    data['split_body'] = data['body'].progress_apply(sent_tokenize)
    data = data.explode("split_body")

    sub_dfs = np.array_split(data, 10)
    all_valence_scores = []
    for df in sub_dfs:
        valence_scores = infer_valence(valence_model, df['split_body'].tolist())
        all_valence_scores.extend(valence_scores)
    data['valence'] = all_valence_scores
    mean_valences = data.groupby("id")['valence'].mean()
    return mean_valences.loc[original_data['id']].tolist()
    



# For politeness
def infer_politeness(politeness_model, to_encode):
    pt.cuda.empty_cache()
    embeddings = model.encode(to_encode, show_progress_bar=True)
    probs = politeness_model.predict_proba(embeddings)
    log_odd_vals = np.log((probs + 1e-8)/((1-probs) + 1e-8))
    return log_odd_vals[:, 1].tolist()

def politeness_wrapper(data):
    politeness_model = Serialization.load_obj('wikipedia_politeness_classifier')
    sub_dfs = np.array_split(data, 10)
    all_politeness_scores = []
    for df in sub_dfs:
        politeness_scores = infer_politeness(politeness_model, df['body'].tolist())
        all_politeness_scores.extend(politeness_scores)
    return all_politeness_scores

def toxicity_wrapper(data):
    perspective_api_call(data)
    return extract_toxicity_scores(data)
# For Perspective API



def augment_dataset():
    filename = #TODO Taken out for privacy
    data = pd.read_csv(filename)
    print(data.groupby("subreddit")[['author', 'id']].nunique())
    data['ttr'] = data['body'].progress_apply(get_TTR)
    data['valence'] = valence_wrapper(data)
    Serialization.save_obj(data, f"hate_speech_all_subreddit_data_pre_liwc")
    data['politeness'] = politeness_wrapper(data)
    Serialization.save_obj(data, f"hate_speech_march_5_subreddit_data_pre_liwc")

    data = Serialization.load_obj(f"hate_speech_march_5_subreddit_data_pre_liwc")
    # Perspective API
    print(f"Data Size: {len(data)}")
    id_to_toxicity_score = toxicity_wrapper(data)
    data['toxicity'] = data['id'].progress_apply(lambda x: id_to_toxicity_score.get(x, np.nan))

    file_output_name = # TODO
    data.to_csv(file_output_name)



if __name__ == "__main__":
    print("Starting")
    augment_dataset()
