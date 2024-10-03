library(lme4)
library("reticulate")
library(plyr)
library(dplyr)
library(caret)
library(glmnet)
library(car)
library(tidyverse)
library(ggpattern)
library(rwa)
library(stringr)

training_wrapper <- function(curr_data, curr_folds){
  train(
    form = label ~ .,
    data = curr_data,
    trControl = trainControl(method = "repeatedcv", number = 2, repeats=5, index=curr_folds, savePredictions = "all"),
    method = "glm",
    family = "binomial"
  )
}

set_plot_dimensions <- function(width_choice, height_choice) {
  options(repr.plot.width=width_choice, repr.plot.height=height_choice)
}


plot_variable_importance <- function(curr_data, curr_model, subreddit, num_features){
  x = varImp(curr_model)$importance %>% 
    as.data.frame() %>%
    rownames_to_column() %>%
    arrange(desc(Overall))
  
  most_important_vars = x[1:num_features, ]
  
  xticks = most_important_vars$rowname
  
  colours = c()
  for (feature in xticks){
    if (feature %in% semantic_features){
      colours = c(colours, "dark green")
    } else if (feature %in% syntactic_features){
      colours = c(colours, "purple")
    } else {
      colours = c(colours, "orange")
    }
  }
  
  print(most_important_vars)
  a = data.frame(summary(curr_model)$coefficients)
  coefs = a[most_important_vars$rowname, ]
  signs = sign(coefs$Estimate)
  print(signs)
  most_important_vars$sign = signs
  most_important_vars$rowname = factor(most_important_vars$rowname, levels=most_important_vars$rowname)
  most_important_vars$sign = factor(most_important_vars$sign)
  
  
  ggplot(most_important_vars, aes(x=rowname, y=Overall, fill=sign)) + 
    scale_x_discrete(name = 'Predictors') + #, limits=rev) +
    scale_y_continuous(name = 'Feature Importance', limits = c(0, 100)) +
    scale_fill_manual(values=c("#cc0000",
                               "#1155cc"), labels=c(subreddit, 'Manosphere')) +
    # coord_flip() + 
    geom_bar(stat='identity', position=position_dodge(.6)) +
    theme(text = element_text(size=30), axis.text.x = element_text(angle=90, vjust = 0.2, hjust=0.95, colour = colours), legend.position=c(.83,.85), axis.title.x = element_text(margin = margin(t = 10, r = 0, b = 0, l = 0))) +
    labs(fill="Group") #+
    # scale_fill_discrete(labels=c(subreddit, 'Manosphere'))
}




write_predictions <- function(curr_model, curr_data, subreddit, feature_set){
  probabilities <- curr_model$finalModel %>% predict(subset(curr_data, select = -c(label)), type = "response")
  curr_data$probs = probabilities
  write.csv(curr_data, sprintf("../Data/regression_predictions/%s_%s_testing_data_predictions.csv", tolower(subreddit), feature_set))
}

load_data <- function(folder, subreddit, split_type){
  dataset = read.csv(sprintf("%s%s_%s_data.csv", folder, tolower(subreddit), split_type))
  dataset$label = factor(dataset$label)
  dataset
}


confusion_matrix_wrapper <- function(model){
  training_pred <- model$pred$pred
  expected <- model$pred$obs
  accuracy <- mean(training_pred == expected)
  print(accuracy)
  print(confusionMatrix(training_pred, expected, mode="everything", positive = "1"))
}

confusion_matrix_by_fold <- function(model){
  rfConfusionMatrices <- list()
  a = model$pred
  unique_folds = unique(a$Resample)
  counter = 0
  tprs = c()
  tnrs = c()
  for (f in unique_folds){
    counter = counter+1
    curr = a[a$Resample == f, ]
    training_pred <- curr$pred
    expected <- curr$obs
    conf = confusionMatrix(training_pred, expected, mode="everything", positive = "1")
    tprs = c(tprs, conf$byClass[1])
    tnrs = c(tnrs, conf$byClass[2])
    # print(conf)
    # rfConfusionMatrices[[counter]] <- conf
  }
  print(tprs)
  print(mean(tprs))
  print(tnrs)
  print(mean(tnrs))
  # print("HERE")
  # rfConfusionMatrixMean <- Reduce('+', rfConfusionMatrices) / 10
  # print(rfConfusionMatrixMean)
}