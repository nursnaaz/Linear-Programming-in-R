#############################################################################################################
######################################GAMMA DATA with H20 DEEP LEARNING#############################################################
#############################################################################################################
# setting the working directory
rm(list=ls(all=TRUE))

# Set working directory
setwd(choose.dir())


path_local <- getwd()
path_data <- paste0(path_local, "/gamma_data.csv")

#Loading library of h20
library(h2o)

#Initialinzing H20
h2o.init(ip='localhost', port = 54321, max_mem_size = '1g',nthreads = -1)

#Importing data directly to h20
gama.hex = h2o.importFile(path = path_data)
summary(gama.hex)
str(gama.hex)


##Preprocessing - all the data is in  numeric and target is of factor

#Splitting the data 80:20
splits <- h2o.splitFrame(gama.hex, 0.8, seed=1234)

##x and y attributes
y = "class"
x = setdiff(colnames(gama.hex), y)


# Prepare the parameters for the for H2O glm grid search
lambda_opts = list(list(1), list(.5), list(.1), list(.01), 
                   list(.001), list(.0001), list(.00001), list(0))
alpha_opts = list(list(0), list(.25), list(.5), list(.75), list(1))

hyper_parameters = list(lambda = lambda_opts, alpha = alpha_opts)

print("################################################################")
print("Starting the Grid search with glm")
# Build H2O GLM with grid search
grid_GLM <- h2o.grid("glm", 
                     hyper_params = hyper_parameters, 
                     grid_id = "grid_GLM.hex",
                     y = y, 
                     x = x,
                     training_frame = splits[[1]], 
                     family = "binomial")

# Remove unused R objects
rm(lambda_opts, alpha_opts, hyper_parameters)

# Get grid summary
summary(grid_GLM)

# Fetch GBM grid models
grid_GLM_models <- lapply(grid_GLM@model_ids, 
                          function(model_id) { h2o.getModel(model_id) })

for (i in 1:length(grid_GLM_models)) 
{ 
  print(sprintf("regularization: %-50s auc: %f", grid_GLM_models[[i]]@model$model_summary$regularization, h2o.auc(grid_GLM_models[[i]])))
}

# Function to find the best model with respective to AUC
find_Best_Model <- function(grid_models){
  best_model = grid_models[[1]]
  best_model_AUC = h2o.auc(best_model)
  for (i in 2:length(grid_models)) 
  {
    temp_model = grid_models[[i]]
    temp_model_AUC = h2o.auc(temp_model)
    if(best_model_AUC < temp_model_AUC)
    {
      best_model = temp_model
      best_model_AUC = temp_model_AUC
    }
  }
  return(best_model)
}

# Find the best model by calling find_Best_Model Function
best_GLM_model = find_Best_Model(grid_GLM_models)

rm(grid_GLM_models)

# Get the auc of the best GBM model
best_GLM_model_AUC = h2o.auc(best_GLM_model)

# Examine the performance of the best model
best_GLM_model

# View the specified parameters of the best model
best_GLM_model@parameters

# Important Variables.
h2o.varimp(best_GLM_model)

###################################################################################################################
##################################COMMENT#######################################
##############################################################################################################
## Based on the Best Model 
# the alpha = 0.75
# Lamda = 1e-04
# The Regression formula is Lamda(alpha(Lasso)+(1-alpha)(Ridge))
# Here the alpha is 0 so it has Elastic Regression
# with 1e-04(0.75(Lasso) + 0.25(Ridge))


# Predict on same training data set
predict.hex = h2o.predict(best_GLM_model, 
                          newdata = splits[[1]][,setdiff(names(splits[[1]]), "class")])

#Binding the data
data_GLM = h2o.cbind(splits[[1]][,"class"], predict.hex)


# Copy predictions from H2O to R
pred_GLM = as.data.frame(data_GLM)

# Hit Rate and Penetration calculation
conf_Matrix_GLM = table(pred_GLM$class, pred_GLM$predict) 

Accuracy = (conf_Matrix_GLM[1,1]+conf_Matrix_GLM[2,2])/sum(conf_Matrix_GLM)

#Accuracy = 77.3



# Predict on same test data set

predict.hex = h2o.predict(best_GLM_model, 
                          newdata = splits[[2]][,setdiff(names(splits[[2]]), "class")])

data_GLM = h2o.cbind(splits[[2]][,"class"], predict.hex)

# Copy predictions from H2O to R
pred_GLM = as.data.frame(data_GLM)

# Hit Rate and Penetration calculation
conf_Matrix_GLM = table(pred_GLM$class, pred_GLM$predict) 

Accuracy = (conf_Matrix_GLM[1,1]+conf_Matrix_GLM[2,2])/sum(conf_Matrix_GLM)

# Accuracy for test is 77.53

##################################################################################
###############COMMENT##############################################################
#Building Autoencoder to extract new features
####################################################

#Attributes
y = "class"
x = setdiff(colnames(splits[[1]]), y)
#Autoencoder algorithm model
aec <- h2o.deeplearning(x = x, autoencoder = T, 
                        training_frame=splits[[1]],
                        activation = "Tanh",
                        hidden = c(20),
                        epochs = 100)

# Extract features from train data
features_train <- as.data.frame(h2o.deepfeatures(data = splits[[1]][,x], object = aec))


# Extract features from test data
features_test <- as.data.frame(h2o.deepfeatures(data = splits[[2]][,x], object = aec))

train_old <- as.data.frame(splits[[1]])
test_old <-  as.data.frame(splits[[2]])

# remove temp variables
rm(x,y,aec)

# add extracted features with original data to train the model
train_new <-data.frame(train_old,features_train)
test_new <-data.frame(test_old,features_test)

#remove train,test
rm(train_old,test_old)

# Build the classification model using randomForest to find important features 
require(randomForest)
rf_DL <- randomForest(class ~ ., data=train_new, keep.forest=TRUE, ntree=30)

# importance of attributes
round(importance(rf_DL), 2)
importanceValues = data.frame(attribute=rownames(round(importance(rf_DL), 2)),MeanDecreaseGini = round(importance(rf_DL), 2))
row.names(importanceValues)=NULL
importanceValues = importanceValues[order(-importanceValues$MeanDecreaseGini),]
# Top 17 Important attributes
Top17ImpAttrs = as.character(importanceValues$attribute[1:17])

Top17ImpAttrs
#Subset the impoertant attributes
train_Imp = subset(train_new,select = c(Top17ImpAttrs,"class"))
test_Imp = subset(test_new,select = c(Top17ImpAttrs,"class"))
#remove unwanted data
rm(train_new,test_new)


#############################################################
#######################COMMENT#############################
print("grid with important features")
##Building with the important attributes with Grid in h2o

train.hex <- as.h2o(train_Imp)
test.hex <- as.h2o(test_Imp)

y = "class"
x = setdiff(colnames(train.hex), y)

# Prepare the parameters for the for H2O glm grid search
lambda_opts = list(list(1), list(.5), list(.1), list(.01), 
                   list(.001), list(.0001), list(.00001), list(0))
alpha_opts = list(list(0), list(.25), list(.5), list(.75), list(1))

hyper_parameters = list(lambda = lambda_opts, alpha = alpha_opts)

# Build H2O GLM with grid search
grid_GLM <- h2o.grid("glm", 
                     hyper_params = hyper_parameters, 
                     grid_id = "grid_GLM_IMP.hex",
                     y = y, 
                     x = x,
                     training_frame = train.hex, 
                     family = "binomial")

# Remove unused R objects
rm(lambda_opts, alpha_opts, hyper_parameters)

# Get grid summary
summary(grid_GLM)

# Fetch GBM grid models
grid_GLM_models <- lapply(grid_GLM@model_ids, 
                          function(model_id) { h2o.getModel(model_id) })

for (i in 1:length(grid_GLM_models)) 
{ 
  print(sprintf("regularization: %-50s auc: %f", grid_GLM_models[[i]]@model$model_summary$regularization, h2o.auc(grid_GLM_models[[i]])))
}

# Function to find the best model with respective to AUC
find_Best_Model <- function(grid_models){
  best_model = grid_models[[1]]
  best_model_AUC = h2o.auc(best_model)
  for (i in 2:length(grid_models)) 
  {
    temp_model = grid_models[[i]]
    temp_model_AUC = h2o.auc(temp_model)
    if(best_model_AUC < temp_model_AUC)
    {
      best_model = temp_model
      best_model_AUC = temp_model_AUC
    }
  }
  return(best_model)
}

# Find the best model by calling find_Best_Model Function
best_GLM_model = find_Best_Model(grid_GLM_models)

rm(grid_GLM_models)

# Get the auc of the best GBM model
best_GLM_model_AUC = h2o.auc(best_GLM_model)

# Examine the performance of the best model
best_GLM_model

# View the specified parameters of the best model
best_GLM_model@parameters

# Important Variables.
h2o.varimp(best_GLM_model)


# Predict on same training data set
predict.hex = h2o.predict(best_GLM_model, 
                          newdata = test.hex[,setdiff(names(test.hex), "class")])

data_GLM = h2o.cbind(test.hex[,"class"], predict.hex)

# Copy predictions from H2O to R
pred_GLM = as.data.frame(data_GLM)


# Hit Rate and Penetration calculation
conf_Matrix_GLM = table(pred_GLM$class, pred_GLM$predict) 

Accuracy = (conf_Matrix_GLM[1,1]+conf_Matrix_GLM[2,2])/sum(conf_Matrix_GLM)

#Accuracy = 82.6

