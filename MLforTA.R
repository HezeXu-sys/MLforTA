library(caret)
library(DALEX)
library(DALEXtra)
library(ggplot2)
library(randomForest)
library(kernlab)
library(xgboost)
library(pROC)
library(fs)

library(glmnet)
setwd("E://cohort/ML/")

#读取数据
data <- read.csv(file = "./data_step1.csv")
data <- data[,-1]
data <- data[,c("abortion_sum",setdiff(names(ML_need),"abortion_sum"))]
colnames(data)[1] <- "Type"
data$Type <- as.factor(data$Type)

#划分测试集训练集
set.seed(123)
inTrain <- createDataPartition(y = data$Type,p = 0.8,list = F)
train <- data[inTrain,]
test <- data[-inTrain,]

#欠采样
balanced_data <- ovun.sample(
  Type ~ ., 
  data = train,
  method = "under",   # 指定欠采样
  N = 2*sum(data$Type == 1),            # 目标样本量（少数类样本数的2倍）
)$data
train <- balanced_data
balanced_data <- ovun.sample(
  Type ~ ., 
  data = test,
  method = "under",   # 指定欠采样
  N = 2*sum(data$Type == 1),            # 目标样本量（少数类样本数的2倍）
)$data
test <- balanced_data

dummies <- model.matrix(
  ~  occupation+conception_method - 1,  # -1 表示不保留截距项
  data = train
) %>% 
  as.data.frame()
dummies <- dummies[,-1]
train <- train %>%
  dplyr::select(-education, -occupation, -annual_income,-conception_method) %>%  # 删除原列
  cbind(dummies) 
dummies <- model.matrix(
  ~  occupation+conception_method - 1,  # -1 表示不保留截距项
  data = test
) %>% 
  as.data.frame()
dummies <- dummies[,-1]
test <- test %>%
  dplyr::select(-education, -occupation, -annual_income,-conception_method) %>%  # 删除原列
  cbind(dummies) 

#lasso
library(glmnet)
y <- train$Type
x <- as.matrix(train[,-1])
fit <- glmnet(x,y,family = "binomial",alpha = 1)
cvfit <- cv.glmnet(x,y,family = "binomial")
plot(fit,label = F)
plot(cvfit)
coef(cvfit,s="lambda.1se")
coef(cvfit, s = "lambda.min")

#boruta
#install.packages("Boruta")
library(Boruta)
Boruta_select <- Boruta(Type ~ ., 
                        data = train, 
                        ntree = 500)
attStats(Boruta_select)
par(mar = c(14, 4, 4, 2) + 0.1)
plot(Boruta_select,las = 2,xlab = NA,line = 1) 
par(mar = c(6, 4, 4, 2) + 0.1)
plotImpHistory(Boruta_select)
Boruta_confirm <- getConfirmedFormula(Boruta_select)

#RFE
ctrl <- rfeControl(
  functions = rfFuncs,  # 使用随机森林评估特征重要性
  method = "cv",        # 交叉验证
  number = 10,           # 5折交叉验证
  verbose = FALSE
)
rfe_results <- rfe(
  x = x,  # 特征矩阵（排除标签列）
  y = y, # 标签列
  sizes = c(1:4,6,10,20,30,40,50),        # 测试的特征子集大小
  rfeControl = ctrl
)
print(rfe_results)
plot(rfe_results, type=c("g", "o"))


library(ggplot2)
#RF随机森林模型
control <- trainControl(method = "repeatedcv",number = 5,savePredictions = T)
# Random Forest
mod_rf = train(Type ~ .,
               data = train, method='rf', trControl = control)
# Generalized linear model (i.e., Logistic Regression)
mod_glm = train(Type ~ .,
                data = train, method="glm", family = "binomial", trControl = control)
# Support Vector Machines
mod_svm <- train(Type ~.,
                 data = train, method = "svmRadial", prob.model = TRUE, tuneLength = 1,trControl=control)
#GBM
mod_gbm = train(Type ~ .,
                data = train, method='gbm', trControl = control)
#KNN
mod_knn = train(Type ~ .,
                data = train, method='knn', trControl = control)
#nnet
mod_nnet = train(Type ~ .,
                 data = train, method='nnet', trControl = control)
#lasso
mod_lasso = train(Type ~ .,
                  data = train, method='glmnet', trControl = control)
#DT 有bug
mod_dt = train(Type ~ .,
               data = train, method='rpart', trControl = control)

#定义预测函数
p_fun <- function(object, newdata){
  predict(object, newdata=newdata, type="prob")[,2]
}
#分组设置对照组为0
yTest = as.numeric(test$Type)
yTest <- yTest-1

explainer_rf  <- DALEX::explain(mod_rf, label = "RF",
                                data = test, y = yTest,
                                predict_function = p_fun,
                                verbose = FALSE)
explainer_glm <- DALEX::explain(mod_glm, label = "GLM",
                                data = test, y = yTest,
                                predict_function = p_fun,
                                verbose = FALSE)

explainer_svm <- DALEX::explain(mod_svm, label = "SVM",
                                data = test, y = yTest,
                                predict_function = p_fun,
                                verbose = FALSE)

explainer_gbm <- DALEX::explain(mod_gbm, label = "GBM",
                                data = test, y = yTest,
                                predict_function = p_fun,
                                verbose = FALSE)
explainer_knn <- DALEX::explain(mod_knn, label = "KNN",
                                data = test, y = yTest,
                                predict_function = p_fun,
                                verbose = FALSE)
explainer_nnet <- DALEX::explain(mod_nnet, label = "NNET",
                                 data = test, y = yTest,
                                 predict_function = p_fun,
                                 verbose = FALSE)
explainer_lasso <- DALEX::explain(mod_lasso, label = "LASSO",
                                  data = test, y = yTest,
                                  predict_function = p_fun,
                                  verbose = FALSE)
explainer_dt <- DALEX::explain(mod_dt, label = "DT",
                               data = test, y = yTest,
                               predict_function = p_fun,
                               verbose = FALSE)
mp_rf = model_performance(explainer_rf)
mp_glm = model_performance(explainer_glm)
mp_svm  = model_performance(explainer_svm)
mp_gbm = model_performance(explainer_gbm)
mp_knn = model_performance(explainer_knn)
mp_nnet = model_performance(explainer_nnet)
mp_lasso = model_performance(explainer_lasso)
mp_dt = model_performance(explainer_dt)


#绘制残差反向累积分布图
p1 <- plot(mp_rf,mp_glm,mp_gbm,mp_knn,mp_nnet,mp_lasso,mp_dt)
p1 <- plot(mp_rf,mp_glm,mp_gbm,mp_knn,mp_nnet,mp_lasso,mp_dt,mp_svm,geom = "boxplot")

plot(mp_glm,mp_gbm,mp_knn)
vi_rf <- variable_importance(explainer_rf, loss_function = loss_root_mean_square)
vi_glm <- variable_importance(explainer_glm, loss_function = loss_root_mean_square)
vi_svm <- variable_importance(explainer_svm, loss_function = loss_root_mean_square)
plot(vi_rf, vi_glm, vi_svm)

vip_rf  <- model_parts(explainer = explainer_rf,  B = 50, N = NULL)
vip_glm  <- model_parts(explainer = explainer_glm,  B = 50, N = NULL)
vip_svm <- model_parts(explainer = explainer_svm, B = 50, N = NULL)

plot(vip_rf, vip_glm, vip_svm, max_vars = 4, show_boxplots = FALSE) +
  ggtitle("Mean variable-importance over 50 permutations", "")


p2 <- plot(mp_rf,mp_glm,mp_gbm,mp_knn,mp_nnet,mp_lasso,mp_dt,mp_svm, geom = "prc") 
p <- plot(mp_rf,mp_glm,mp_gbm,mp_knn,mp_nnet,mp_lasso,mp_dt,mp_svm, geom = "roc")#+labs(x = "1-Specificity",y = "Sensitivity")
auc_values <- sapply(list(mp_rf, mp_glm, mp_gbm, mp_knn, mp_nnet, mp_lasso, mp_dt), 
                     function(x) x$measures$auc[1])
#
importance_rf <- variable_importance(
  explainer_rf,
  loss_function = loss
)

par(mar = c(6, 2, 2, 2) + 0.1)
plot(mp_rf,mp_glm,mp_gbm,mp_knn,mp_nnet,mp_lasso,mp_dt,mp_svm, geom = "roc")

vi_rf <- variable_importance(explainer_rf, loss_function = loss_root_mean_square)
vi_gbm <- variable_importance(explainer_glm, loss_function = loss_root_mean_square)
vi_svm <- variable_importance(explainer_svm, loss_function = loss_root_mean_square)
plot(vi_rf,max_vars = 10, show_boxplots = FALSE)
plot(vi_rf,vi_gbm,vi_svm,max_vars = 10, show_boxplots = FALSE)


ale_rf  <- model_profile(explainer_rf, variable = "view_screen_time", type = "categorical")
pdp_rf <- model_profile(explainer_rf, variable = "view_screen_time", type = "accumulated")
plot(pdp_rf)
plot(ale_rf)
plot(pdp_classif_rf, pdp_classif_glm, pdp_classif_svm)

shap_rf <- predict_parts(explainer = explainer_rf,
                         new_observation = train[15,-1],
                         type = "shap",
                         B = 25
)
plot(shap_rf)
shap_rf <- predict_parts(explainer = explainer_rf,
                         new_observation = train[15,-1],
                         type = "shap",
                         B = 25
)
plot(shap_rf)

library(fastshap)
library(ranger)
library(dplyr)
library(tidyverse)
library(shapviz)
t1 <- train
t1 <- rename(t1,survived = Type)

(rfo <- ranger(survived ~ ., data = t1, probability = TRUE))

case_1 <- t1[2378,-1]
case_1 <- t1[15,-1]

pfun <- function(object, newdata) {  
  unname(predict(object, data = newdata,type = 'response')$predictions[, "1"])
}
(jack.prob <- pfun(rfo, newdata = case_1))

(baseline <- mean(pfun(rfo, newdata = t1)))

(difference <- jack.prob - baseline)
X <- test
X <- rename(X,survived = Type)
X <- subset(X, select = -survived)  # 只含预测变量

# 建立解释器
(ex.jack <- fastshap::explain(rfo, X = X, pred_wrapper = pfun, newdata = case_1))

(ex.jack <- fastshap::explain(rfo, X = X, pred_wrapper = pfun, newdata = case_1,
                              nsim = 1000))

(ex.jack.adj <- fastshap::explain(rfo, X = X, pred_wrapper = pfun, newdata = case_1,
                                  nsim = 100, adjust = TRUE))
#fastshap::explain()


shv <- shapviz(ex.jack.adj, X = case_1, baseline = baseline)
sv_waterfall(shv)

ex.t1 <- fastshap::explain(rfo, X = X, pred_wrapper = pfun, nsim = 100, adjust = TRUE,
                           shap_only = FALSE)
tibble::as_tibble(ex.t1$shapley_values)

shv.global <- shapviz(ex.t1)
sv_importance(shv)  
sv_importance(shv.global, kind = "beeswarm")