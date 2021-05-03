# objetivo          --------------------------------------------------------
"1 melhorar modelo de classificacao"
"2 probabilidade, crime ocorrer, cliente ser bom pagador, recessão, paciente ter doenca, planta viver ao aplicar herbicida"
"3 confirmar se o pre processamento interefe no desempenho do modelo"

# pacotes           --------------------------------------------------------
library(tidymodels)
library(tidyverse)
library(janitor)
library(ggpubr)
library(funModeling)
library(ggalluvial)
library(vip)

# dados             --------------------------------------------------------

full_titanic <- 
  read.csv("train.csv", na.strings = c(""," ")) %>% 
  clean_names() %>% 
  mutate(is_train = TRUE) %>% 
  mutate(across(c("sex","embarked","survived","pclass"), as.factor))


new_data_titanic <- 
  read.csv("test.csv", na.strings = c(""," ")) %>% 
  clean_names() %>% 
  mutate(survived = NA, is_train = FALSE) %>% 
  mutate(across(c("sex","embarked","survived","pclass"), as.factor))


split_titanic <- initial_split(full_titanic, prop = 3/4, strata = survived)

train_titanic <- training(split_titanic)
test_titanic <- testing(split_titanic)


set.seed(123)
resample_titanic <- bootstraps(train_titanic, strata= survived)


# exploracao        --------------------------------------------------------

# correlacao
train_titanic %>% 
  mutate(survived = as.numeric(survived)) %>% 
  select_if(is.numeric) %>% 
  GGally::ggscatmat(color = "survived", corMethod = "spearman")+
  theme_pubclean()


train_titanic %>% 
  ggplot(aes(x=fare))+
  geom_histogram()


# age 
train_titanic %>% 
  ggplot(aes(x=age, fill = factor(survived)))+
  geom_density(alpha =0.5)

# age, sex and class  
train_titanic %>% 
  ggplot(aes(x=pclass, fill= factor(survived)))+
  geom_bar(stat = "count")+
  facet_grid(~sex)

# age sex and class  
train_titanic %>% 
  ggplot(aes(x=age, y=sex))+
  geom_jitter(aes(color = factor(survived)))+
  facet_wrap(~pclass)

# sex, class e survived 
train_titanic %>% 
  group_by(sex, survived, pclass) %>% 
  summarise(qtd = n()) %>% 
  ggplot(aes(axis1=sex, axis2=pclass, axis3 = survived, y=qtd, fill= sex))+
  geom_alluvium()+
  geom_stratum()+
  geom_text(stat = "stratum", aes(label = after_stat(stratum)))+
  scale_x_discrete(limits = c("sex", "pclass","survived"))

# fare
train_titanic %>% 
  mutate(fare = fare) %>% 
  ggplot(aes(x=fare, y=pclass))+
  geom_jitter(aes(color = factor(survived)))

# parch sib_sp 
train_titanic %>% 
  mutate(family_size = parch + sib_sp +1) %>% 
  ggboxplot(x="survived", y="family_size", fill= "survived", palette = "uchicago")+
  stat_compare_means()

# title
train_titanic %>% 
  mutate(title = str_extract(name, "[A-z]*\\.")) %>% tabyl(title) %>% arrange(desc(n))


# pre processamento --------------------------------------------------------

# especificacao
rec_titanic <- 
  recipe(survived~., data = train_titanic) %>% 
  step_impute_mode(embarked) %>% 
  step_impute_knn(age, fare, neighbors = 3, impute_with = c("sex","pclass","embarked","parch","sib_sp")) %>% 
  step_mutate(child = ifelse(age <=14, 1,0)) %>% 
  step_mutate(woman = ifelse(age >14 & sex=="female",1,0)) %>% 
  step_mutate_at(c("child","woman"), fn= as.factor) %>% 
  step_mutate(family_size = parch + sib_sp +1) %>% 
  step_mutate(title = str_extract(name,"[A-z]*\\.")) %>% 
  step_other(title , threshold = 0.044) %>%
  update_role(passenger_id, name, ticket, cabin, new_role = "id") %>% 
  themis::step_downsample(survived)

# modelo            --------------------------------------------------------
mdl_spec_xgb_titanic <- 
  boost_tree() %>% 
  set_mode("classification") %>% 
  set_engine("xgboost") %>% 
  set_args(trees = 300,
           tree_depth = tune(),
           min_n = tune(),
           loss_reduction = tune(),
           sample_size = tune(),
           mtry = tune(),
           learn_rate = tune())


# workflow          ----------------------------------------------------------------
wkfl_xgb_titanic <- 
  workflow() %>% 
  add_recipe(rec_titanic) %>% 
  add_model(mdl_spec_xgb_titanic, formula = survived ~ child + woman + title + family_size)

# tune              --------------------------------------------------------------------
grid_xgb_titanic <- 
  grid_latin_hypercube(
    tree_depth(),
    min_n(),
    loss_reduction(),
    sample_size = sample_prop(),
    finalize(mtry(), train_titanic),
    learn_rate(),
    size = 10 )

tune_xgb_titanic <- 
  tune_grid(object = wkfl_xgb_titanic,
            resamples = resample_titanic,
            grid = grid_xgb_titanic,
            control  = control_grid(save_pred = TRUE))

tune_xgb_titanic %>% show_best(metric = "roc_auc")

best_grid_xgb_titanic <- tune_xgb_titanic %>% select_best(metric = "roc_auc")

best_grid_xgb_titanic


final_wkfl_xgb_titanic <- 
  finalize_workflow(wkfl_xgb_titanic, best_grid_xgb_titanic)

final_wkfl_xgb_titanic


# validation        --------------------------------------------------------------
final_vld_xgb_titanic <- 
  last_fit(final_wkfl_xgb_titanic, split =  split_titanic) 

final_vld_xgb_titanic %>% collect_metrics()


# final_model       -------------------------------------------------------------
final_wkfl_xgb_titanic %>% 
  fit(data = full_titanic) %>% 
  pull_workflow_fit() %>% 
  vip(geom = "point")

final_mdl_fit_xgb_titanic <- 
  final_wkfl_xgb_titanic %>% 
  fit(data  = full_titanic)


# submissao         --------------------------------------------------------
submission <- 
  data.frame(PassengerId = new_data_titanic$passenger_id,
             Survived    = predict(final_mdl_fit_xgb_titanic,
                                   new_data = new_data_titanic)) %>% 
  rename(Survived = .pred_class)


write.csv(submission, file = "titanic_kaglle_v10.csv", row.names = FALSE)

