source("base_functions.R")

# Load Features
syntactic_features = scan("syntactic_features.txt", character(), quote = "")
semantic_features = scan("semantic_features.txt", character(), quote = "")
uncivil_features = scan("uncivil_features.txt", character(), quote = "")

train_on_feature_subset <- function(dataset, feature_subset, folds){
  subset_training_data <- subset(dataset, select=feature_subset)
  subset_mod = training_wrapper(subset_training_data, folds)
  subset_mod
}




# Load Data
subreddit = 'full_sample_14_subreddits'
all_features_training_data <- load_data("../Data/training_data/", subreddit, "training")[c("label", semantic_features, syntactic_features, uncivil_features)]
all_features_training_data <- subset(all_features_training_data, select=-c(Authentic, Clout, Tone, Analytic))


# Create cross-validation folds
i_folds <- createMultiFolds(all_features_training_data$label, k=2, times = 5)

semantic_features = semantic_features[! semantic_features %in% c("Authentic", "Clout", "Tone", "Analytic")]
new_syntactic_features = syntactic_features[! syntactic_features %in% c("shehe")]

# Train full model
all_features_mod = training_wrapper(all_features_training_data, i_folds)
confusion_matrix_wrapper(all_features_mod)
plot_variable_importance(all_features_training_data, all_features_mod, "Baseline", 20)


# Only uncivil variables
only_uncivil_model <- train_on_feature_subset(all_features_training_data, c(uncivil_features, c("label")), i_folds)
confusion_matrix_by_fold(only_uncivil_model)


# Only syntactic variables
only_syntactic_model <- train_on_feature_subset(all_features_training_data, c(syntactic_features, c("label")), i_folds)
confusion_matrix_by_fold(only_syntactic_model)


# Only male/female variables
male_female_model <- train_on_feature_subset(all_features_training_data, c("label", "female", "male"), i_folds)
confusion_matrix_by_fold(male_female_model)

# Final features
final_model <- train_on_feature_subset(all_features_training_data, c(syntactic_features, uncivil_features, c("label", "female", "male")), i_folds)
confusion_matrix_by_fold(final_model)



