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

# dados             --------------------------------------------------------

# importacao
train_titanic <- read.csv("train.csv", na.strings = c("", " ")) %>% clean_names()
test_titanic  <- read.csv("test.csv" , na.strings = c("", " ")) %>% clean_names()

# padronizacao de  colunas
train_titanic <- train_titanic %>% mutate(is_train = TRUE)
test_titanic  <- test_titanic %>% mutate(is_train = FALSE, survived = NA)

# juntar os conjuntos
split_titanic <-bind_rows(train_titanic, test_titanic)

# validacao
split_titanic %>% df_status()
split_titanic %>% visdat::vis_dat()

# exploracao        --------------------------------------------------------

# correlacao
train_titanic %>% 
  select_if(is.numeric) %>% 
  GGally::ggscatmat(color = "survived", corMethod = "spearman")+
  theme_pubclean()
  
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
recipe(survived~., data = split_titanic) %>% 
  step_mutate_at(c("sex","embarked","survived","pclass"), fn = as.factor) %>% 
  step_modeimpute(embarked) %>% 
  step_knnimpute(age, fare, neighbors = 3, impute_with = c("sex","pclass","embarked","parch","sib_sp")) %>% 
  step_mutate(child = ifelse(age <=14, 1,0)) %>% 
  step_mutate(woman = ifelse(age >14 & sex=="female",1,0)) %>% 
  step_mutate_at(c("child","woman"), fn= as.factor) %>% 
  step_mutate(family_size = parch + sib_sp +1) %>% 
  step_mutate(title = str_extract(name,"[A-z]*\\.")) %>% 
  step_other(title , threshold = 0.044) %>%
  update_role(passenger_id, name, ticket, cabin, new_role = "id")

# preparacao
prep_titanic <- prep(rec_titanic, retain = TRUE)

# aplicacao
split_titanic_trans <- bake(prep_titanic, new_data = NULL)
train_titanic_trans <- split_titanic_trans %>% filter(is_train == TRUE)
test_titanic_trans  <- split_titanic_trans %>% filter(is_train == FALSE)


train_titanic_trans_down <- 
  recipe(survived~., data = train_titanic_trans) %>% 
  themis::step_downsample(survived) %>% 
  prep() %>% 
  juice()

train_titanic_trans_down %>% ggplot(aes(x=survived))+geom_bar(stat = "count")

train_titanic_trans_up <- 
  recipe(survived~., data = train_titanic_trans) %>% 
  themis::step_upsample(survived) %>% 
  prep() %>% 
  juice()

train_titanic_trans_up %>% ggplot(aes(x=survived))+geom_bar(stat = "count")



# modelo            --------------------------------------------------------
# especificacao
set.seed(123)
mdl_spec_rf_titanic <- 
  rand_forest() %>% 
  set_mode("classification") %>% 
  set_engine("randomForest") 

# treinar 
mdl_fit_rf_titanic <- 
  mdl_spec_rf_titanic %>% 
  fit(survived ~ child + woman + title + family_size,  data = train_titanic_trans_down)

mdl_fit_rf_titanic


# validacao         --------------------------------------------------------

# reamostragem 
resample_titanic <- bootstraps(train_titanic_trans_down, strata = survived)

# validacao reamostragem 
mdl_fit_resample_rf_titanic <- 
  mdl_spec_rf_titanic %>% 
  fit_resamples(survived ~ child + woman + title + family_size,
                resample_titanic,
                metrics = metric_set(roc_auc, accuracy, sens),
                control = control_resamples(save_pred = TRUE))

# analise dos resultados reamostragem
mdl_fit_resample_rf_titanic %>% collect_metrics()
mdl_fit_resample_rf_titanic %>% unnest(.predictions) %>% conf_mat(survived, .pred_class)


# submissao         --------------------------------------------------------
submission <- 
  data.frame(PassengerId = test_titanic_trans$passenger_id,
             Survived    = predict(mdl_fit_rf_titanic,
                                   new_data = test_titanic_trans)) %>% 
  rename(Survived = .pred_class)
  

write.csv(submission, file = "titanic_kaglle_v5.csv", row.names = FALSE)
























data("credit_data")


credit_data %>% tabyl(Status)
