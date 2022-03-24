# Aux
setwd("GitHub/1915_project-whatagwup")
source("Auxilliary.R")

# load or install packages
get.package(c("glmnet", "Matrix", "ROCR", "caret", "glmnetUtils", "randomForest",
              "mlbench", "ranger", "xgboost", "corrplot", "RColorBrewer",
              "ROSE"))

# load
dat_carav <- as.data.frame(lapply(read.csv("caravan-insurance-challenge.csv"), as.character),                                                 stringsAsFactors = T)
# split 
dat_carav <- split(dat_carav, dat_carav$ORIGIN) |> setNames(c("test", "train"))


# sapply(dat_carav[["train"]], class)

# remove cvonst col from df

lapply(dat_carav, function(x){
  x[, "ORIGIN"] <- NULL
  x
}) -> dat_carav

# sum(apply(dat_carav[["train"]], 2, function(x) any(is.na(x))))



links <- c("logit", "probit", "cauchit", "cloglog")
lambdas <- numeric(length(links)) |> setNames(links)




# control
ctrl <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 2)

# tuning grid
# tgrid <- expand.grid("mtry" = seq(10, 50, 5),
#                      "splitrule" = "gini",
#                      "min.node.size" = 1:5)

# ext
tgrid <- expand.grid("mtry" = seq(5, 60, 5),
                     "splitrule" = "gini",
                     "min.node.size" = 1:7)
#"num.trees" = seq(500, 1000, 100))

# # train
# rf_carav <- caret::train(CARAVAN ~ ., data = dat_carav[["train"]], method = "ranger",
#                   metric = "Kappa", tuneGrid = tgrid, trControl = ctrl)
# 
# # safe
# saveRDS(rf_carav, "Parameters/tree.RDS")

# read results
param_tree <- readRDS("Parameters/tree.RDS")
besTrf <- param_tree$bestTune

# best tree model
ranger::ranger(CARAVAN ~ ., data = dat_carav[["train"]],
               mtry = besTrf[["mtry"]], splitrule = besTrf[, "splitrule"],
               min.node.size = besTrf$min.node.size, probability = T) -> besTrf_mod

# predict
Pred_val_mod_rf <- predict(besTrf_mod, dat_carav[["test"]])$predictions[, 2]

opt.para <- read.csv("Parameters/optimal_parameters_elastic_net.csv")

# RoC curve 
Map(function(l, a, lin){
  
  # estimate "best" model
  fit <- glmnetUtils::glmnet(CARAVAN ~., data = dat_carav[["train"]], family = binomial(link = lin),
                             alpha = a, lambda = l)
  
  # CE on test data
  pred.vals <- predict(fit, dat_carav[["test"]], type = "response")
  
  # return
  return(list(pred.vals, fit))
  
}, opt.para[2, -1], opt.para[3, -1], links) -> glm_probs

# ext from list
glm_fits <- lapply(glm_probs, "[[", 2)
glm_prob_net <- lapply(glm_probs, "[[", 1)
glm_and_forest_pred_prob <- c(glm_prob_net, list(Pred_val_mod_rf)) |> 
  setNames(c("logit", "probit", "cauchit", "cloglog", "forest"))

# preliminary for plotting
prelim <- Eval_Curve_prel(c(glm_prob_net, list(Pred_val_mod_rf)), 
                          as.numeric(dat_carav[["test"]][, "CARAVAN"]) - 1)
# only Glmnet
plot_ROC_net <- prelim[1:4]

# names for legend and colors
Links <- c("Normal", "Logistic", "Cauchy", "Complementary Log-Log")
col <-  c("red1", rgb(51, 51, 178, maxColorValue = 255),"forestgreen","turquoise4")

# # RoC Plot 
# Eval_Curve(plot_ROC_net, col = col, leg_text = Links)
# 
# 
# 
# # Ps Curve
# Eval_Curve(plot_ROC_net, col = col, leg_text = Links, RoC = FALSE, 
#            act_label = as.numeric(dat_carav[["test"]][, "CARAVAN"]) - 1)

# confusion matrices
# knitr::kable(
#   lapply(glm_prob_net, function(x){
#     table(as.numeric(x > 0.5), 
#           as.numeric(dat_carav[["test"]][, "CARAVAN"]) - 1) 
#   }))



# # RoC
# Eval_Curve(prelim[3:5], col = col[2:4], leg_text = c("Cauchit", "Cloglog", "Forest"))
# 
# 
# # PR
# Eval_Curve(prelim[3:5], col = col[2:4], leg_text = c("Cauchit", "Cloglog", "Forest"),
#            RoC = FALSE, act_label = as.numeric(dat_carav[["test"]][, "CARAVAN"]) - 1)
# 


# abs amount of pos and neg
n <- sum(as.numeric(dat_carav[["train"]][, "CARAVAN"] == 1)) 
N <- nrow(dat_carav[["train"]]) - n

# vec
meth <- c("over", "under", "both")
dat_carav_ts <- list()

# over under and both
# lapply doesnt work here for whatever reason ??
for(i in 1:3){
  
  # sample
  dat_carav_ts[[i]] <- ROSE::ovun.sample(CARAVAN ~ ., data = dat_carav[["train"]], 
                                         method = meth[i], seed = 33)$data
  
}  

# Menardi and Torelli algo
c(dat_carav_ts, list(ROSE::ROSE(CARAVAN ~ ., data = dat_carav[["train"]], seed = 33)$data)) |>
  setNames(nm = c("over", "under", "both", "menarelli")) -> dat_carav_ts 

# train rf for all datasets

# control
ctrl <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 2)

# ext
tgrid <- expand.grid("mtry" = seq(50, 85, 5),
                     "splitrule" = "gini",
                     "min.node.size" = 3:9)
#"num.trees" = seq(500, 1000, 100))

# read and fit
lapply(c("both", "over", "under"), function(n){

  # read
  best_tune <- readRDS(paste0("Parameters/Resampling/Forest/", n, ".RDS"))$bestTune
  
  # fit
  # best tree model
  ranger::ranger(CARAVAN ~ ., data = dat_carav[["train"]],
                 mtry = best_tune[["mtry"]], splitrule = best_tune[, "splitrule"],
                 min.node.size = best_tune$min.node.size, probability = T) -> besTrf_mod
  
  # predict
  predict(besTrf_mod, dat_carav[["test"]])$predictions[, 2]
  
}) -> rf_carav_ts_res

# eval
prelim_ovun_rf <- Eval_Curve_prel(rf_carav_ts_res, 
                                  as.numeric(dat_carav[["test"]][, "CARAVAN"]) - 1)

# add orig tree
orig_ovun <- c(prelim_ovun_rf, list(prelim[[5]])) |> setNames(c("both", "over", "under", "orig"))

saveRDS(orig_ovun, "Parameters/Resampling/Forest/Forest_Plot_Prelim.RDS")

# add col 
col <- c(col, "#BB650B")

# # RoC
# Eval_Curve(orig_ovun, col = col[1:5], leg_text = c("over", "under", "both", "menarelli", "Orig"))
# 
# 
# # PR
# Eval_Curve(orig_ovun, col = col[1:5], leg_text = c("over", "under", "both", "menarelli", "Orig"),
#            RoC = FALSE, act_label = as.numeric(dat_carav[["test"]][, "CARAVAN"]) - 1)

# Glms and forest
ttt <- c(glm_and_forest_pred_prob, rf_carav_ts_res, 
         list(readRDS('DEV_files/predXGB.RDS')[, 2]) |> setNames("XGB"))

# calc per
sapply(ttt, function(x) 
  COIL_per(x, as.numeric(dat_carav[["test"]][,"CARAVAN"]) - 1))[-9] -> COIL_res
