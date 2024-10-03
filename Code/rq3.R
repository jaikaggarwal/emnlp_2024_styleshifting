source("base_functions.R")

# Load Features
syntactic_features = scan("../Data/syntactic_features.txt", character(), quote = "")
semantic_features = scan("../Data/semantic_features.txt", character(), quote = "")
uncivil_features = scan("../Data/uncivil_features.txt", character(), quote = "")

semantic_features = semantic_features[! semantic_features %in% c("Authentic", "Clout", "Tone", "Analytic")]

# Load Data
subreddits = c("askreddit", "news", "worldnews", "todayilearned", "askmen", "movies", "politics", "technology","adviceanimals", "videos", "pics", "funny", "wtf", "gaming")

df = load_data("../Data/testing_data/", 'feature_switching', "emnlp")

# Create cross-validation folds
i_folds <- createMultiFolds(df$label, k=2, times = 5)

female_model = train(
  form = label ~ female + female_parent,
  data = df,
  trControl = trainControl(method = "repeatedcv", number = 2, repeats=5, index=i_folds, savePredictions = "all"),
  method = "glm",
  family = "binomial"
)
summary(female_model)


male_model = train(
  form = label ~ male + male_parent,
  data = df,
  trControl = trainControl(method = "repeatedcv", number = 2, repeats=5, index=i_folds, savePredictions = "all"),
  method = "glm",
  family = "binomial"
)
summary(male_model)



toxicity_model = train(
  form = label ~ toxicity + toxicity_parent,
  data = df,
  trControl = trainControl(method = "repeatedcv", number = 2, repeats=5, index=i_folds, savePredictions = "all"),
  method = "glm",
  family = "binomial"
)
summary(toxicity_model)




politeness_model = train(
  form = label ~ politeness + politeness_parent,
  data = df,
  trControl = trainControl(method = "repeatedcv", number = 2, repeats=5, index=i_folds, savePredictions = "all"),
  method = "glm",
  family = "binomial"
)
summary(politeness_model)




valence_model = train(
  form = label ~ valence + valence_parent,
  data = df,
  trControl = trainControl(method = "repeatedcv", number = 2, repeats=5, index=i_folds, savePredictions = "all"),
  method = "glm",
  family = "binomial"
)
summary(valence_model)