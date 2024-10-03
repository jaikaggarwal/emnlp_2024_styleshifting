from core_utils import *


def find_controls(treatment, control, NUM_QUANTILES):
    sub = treatment.groupby("speaker").count()
    control_sub = control.groupby("speaker").count()
    sub_size = sub.shape[0]//(NUM_QUANTILES-1)
    q = [int(np.quantile(sub['text'], quantile)) for quantile in np.linspace(0, 1, NUM_QUANTILES)]
    dfs = []
    take = 0
    for i in range(0, NUM_QUANTILES-1):
        take += sub_size
        min_q = control_sub[control_sub['text'] >= q[i]]
        bound_q = min_q[min_q['text'] < q[i+1]]
        if len(bound_q) < take:
            take = take - len(bound_q) + 1
            dfs.append(bound_q)
        else:
            dfs.append(bound_q.sample(take))
            take = 0
    matched_control = pd.concat(dfs)
    print(len(sub))
    print(len(matched_control))
    print(sub.sum()['text'])
    print(matched_control.sum()['text'])
    return matched_control.index.tolist()


def process_subreddit_name(sub):
    sub = sub.strip("/")
    return sub[sub.rindex("/") + 1: ].lower()



def compute_posting_distribution_statistics(distribution):
    print(f"""{distribution['id'].sum()} posts over {distribution.index.get_level_values(0).nunique()} authors and {distribution.index.get_level_values(1).nunique()} subreddits\n""")


def split_parent_identifier(df):
    df['submission_type'] = df['parent_id'].apply(lambda x: x.split("_")[0])
    df['cropped_id'] = df['parent_id'].apply(lambda x: x.split("_")[1])
    return df


def load_manosphere_subreddits():
    manosphere_taxonomy = pd.read_csv(MANOSPHERE_TAXONOMY_FILE)
    manosphere_taxonomy = manosphere_taxonomy[manosphere_taxonomy['Women subreddit?']!= "Y"]
    manosphere_taxonomy['Subreddit'] = manosphere_taxonomy['Subreddit'].apply(lambda x: process_subreddit_name(x))
    sub_to_manosphere_category = manosphere_taxonomy[['Subreddit', 'Category after majority agreement']].set_index("Subreddit").to_dict()['Category after majority agreement']
    print(f"Number of Manosphere subreddits: {len(sub_to_manosphere_category)}")
    return sub_to_manosphere_category



def manosphere_subculture_distribution(sub_to_manosphere_category):
    return pd.DataFrame.from_dict(sub_to_manosphere_category, orient='index').reset_index().groupby(0).count()



def write_ids_to_file(directory, ids, file_name):
    print(f"Writing to file: {file_name}")
    with open(directory + file_name, "w") as f:
        for id in ids:
            f.write(id)
            f.write("\n")


SUB_TO_SUBCULTURE = load_manosphere_subreddits()
MANOSPHERE = list(SUB_TO_SUBCULTURE.keys())
SUBCULTURES = list(set(SUB_TO_SUBCULTURE.values()))