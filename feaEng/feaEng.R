# Aux
source("../Auxilliary.R")

# load or install packages
get.package(c("glmnet", "Matrix", "ROCR", "caret", "glmnetUtils", "randomForest",
              "mlbench", "ranger", "xgboost", "corrplot", "RColorBrewer",
              "ROSE"))


# FE --------------------------------------------------------------------

# load
dat_carav <- as.data.frame(lapply(read.csv("../caravan-insurance-challenge.csv"), as.character), stringsAsFactors = T)

# kick socio dem features
dat_carav <- dat_carav[,c(1:6, 44:87)]

# add interaction terms
dat_carav$interCar <- factor(paste(as.character(dat_carav$PPERSAUT), as.character(dat_carav$APERSAUT), sep = ''))
dat_carav$interFire <- factor(paste(as.character(dat_carav$PBRAND), as.character(dat_carav$ABRAND), sep = ''))

# split 
dat_carav <- split(dat_carav, dat_carav$ORIGIN) |> setNames(c("test", "train"))

sapply(dat_carav[["train"]], class)

# remove cvonst col from df

lapply(dat_carav, function(x){
  x[, "ORIGIN"] <- NULL
  x
}) -> dat_carav




# GLM ---------------------------------------------------------------------


params <- read.csv("./../Parameters/optimal_parameters_elastic_net.csv")

glm.cloglog <- glmnetUtils::glmnet(CARAVAN ~ ., data = dat_carav[["train"]],
                                   alpha = params[3, 4], lambda = params[2, 4],
                                   family = binomial(link = "cloglog"))

predvals <- predict(glm.cloglog, dat_carav[["test"]], type = "response")

saveRDS(predvals, file = 'PredValsGLM_FeatEng.RDS')

# RF ----------------------------------------------------------------------

# control
ctrl <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 2)


# ext
tgrid <- expand.grid("mtry" = seq(5, 35, 10),
                     "splitrule" = "gini",
                     "min.node.size" = 1:4)

rf_carav <- caret::train(CARAVAN ~ ., data = dat_carav[["train"]], method = "ranger",
                         metric = "Kappa", tuneGrid = tgrid, trControl = ctrl)



rf_model <- ranger::ranger(CARAVAN ~ ., data = dat_carav[["train"]],
                           mtry = rf_carav$bestTune[, "mtry"],
                           splitrule = "gini",
                           min.node.size = rf_carav$bestTune[, "min.node.size"],
                           probability = TRUE)

# predict
predvalsRF <- predict(rf_model, dat_carav[["test"]], )$predictions[, 2]

saveRDS(predvalsRF, file = 'PredValsRF_FeatEng.RDS')

# XGB ---------------------------------------------------------------------


#independent variable
y <- as.numeric(dat_carav$train$CARAVAN) - 1

#dependent variables
X <- Matrix::sparse.model.matrix(CARAVAN ~ ., data = dat_carav$train)

#imbalanced data
#spw <- sum(y==0) / sum(y)

#intialize (unplausibly) high starting performance
bestPer <- 10^6 

#initialize number of iterations
interations <- 500

#CV to set HP - we interate over the parameter grid
for(i in 1:interations){
  
  #the parameters for the CV are set here
  param <- list(objective = 'binary:logistic', #binary prediction problem
                
                booster = 'gbtree', #trees are the simple learners
                
                eval_metric = 'error', # proposed by xbg (alter.: error:=1-accuracy)
                
                eta = runif(1, .01, .6), #default: .3
                
                gamma = runif(1), #default: 0
                
                lambda = runif(1, .01, 2), #default: 1
                
                max_depth = sample(2:10, 1)) #default: 6
  
  #scale_pos_weight = spw)
  
  
  #max number of interations within each XGB-Model (B in the slides)
  cv.nround <-  1000
  
  #4-fold cross-validation
  cv.nfold <-  4
  
  #set seed for the CV
  seed.number  <-  sample.int(10000, 1)
  set.seed(seed.number)
  
  
  #CV step
  mdcv <- xgb.cv(data = as.matrix(X), label = as.matrix(y), params = param,
                 nfold = cv.nfold, nrounds = cv.nround,
                 verbose = F, early_stopping_rounds = 8, maximize = FALSE)
  
  
  #index of best interation
  best.iteration  <-  mdcv$best_iteration
  
  #rmse of best interation
  performanceInspamle <- mdcv$evaluation_log$test_error_mean[best.iteration]
  
  
  if(performanceInspamle < bestPer){
    
    #update bestPer
    bestPer <- performanceInspamle
    
    #save hyperparameters
    bestpara <- param
    
    #save nrounds as hyperparameter
    bnrounds <- best.iteration 
  }
  
  # print counter (nice if you run it in R but sub optimal for knitting)
  print(paste(str(interations-i), 'interations remained', sep=' '))
  
}


#train the model with the parameters set above
txgb <- xgboost(data = as.matrix(X),
                label = as.matrix(y),
                params = bestpara,
                nrounds = bnrounds)

#dependent variable for test
y.test <- as.numeric(dat_carav$test$CARAVAN)-1

#rgressors for Test data
X.test <- Matrix::sparse.model.matrix(CARAVAN ~ ., data = dat_carav$test)


#predict
pred.xgb <- predict(txgb, newdata = as.matrix(X.test))

#save
saveRDS(pred.xgb, file = 'PredValsXGB_FeatEng.RDS')



# diesdas -----------------------------------------------------------------

#load 
glm <- readRDS('PredValsGLM_FeatEng.RDS')
rf <- readRDS('PredValsRF_FeatEng.RDS')
xgb <- readRDS('PredValsXGB_FeatEng.RDS')
prelim_raw <- readRDS("./../Parameters/RawModelPrelim.RDS")
act_label <- readRDS("./../Parameters/act_label.RDS")

#new list
ecpALL <- Eval_Curve_prel(act_label = list(glm, rf, xgb), pred_val = act_label)
names(ecpALL) <- c('glm', 'rf', 'xgb')

ecpFE <- list(GLM=list(glm = prelim_raw$cloglog, glmFE = ecpALL$glm), RF=list(rf = prelim_raw$RF, rfFE = ecpALL$rf), XGB=list(xgb = prelim_raw$XGB, xgbFE = ecpALL$xgb))

save(ecpFE, file = 'ecpFE.RData')

Eval_Curve(E_Curve_Prel = ecpFE$XGB, col = c(1,2), leg_text = names(ecpFE$XGB), RoC = F)

#feaEng data

