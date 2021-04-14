# demanda           -----------------------------------------------------------------
"prever consumo de combustivel com base no tipo ou caracteristica"
"https://fueleconomy.gov/"

# pacotes           -----------------------------------------------------------------
library(tidyverse)
library(tidymodels)


# dados             -------------------------------------------------------------------
mtcars %>% slice_sample(n=10)

split_mtcars <- initial_split(mtcars, prop = 3/4)

train_mtcars <- training(split_mtcars)

# exploracao        --------------------------------------------------------------
mtcars %>% ggplot(aes(x=wt, y=mpg, color = vs))+ geom_point()

mtcars %>% ggplot(aes(x=disp, y=mpg))+ geom_point()


# pre processamento -------------------------------------------------------
rec_lm_mtcars <-  recipe(mpg~., data = train_mtcars) %>%  step_ns(disp)

rec_lm_mtcars

# modelo            ------------------------------------------------------------------

mdl_spec_lm_mtcars <- linear_reg() %>% set_engine("lm")


# workflow          ----------------------------------------------------------------
wkfl_lm_mtcars <- workflow() %>% add_model(mdl_spec_lm_mtcars) %>% add_recipe(rec_lm_mtcars)

wkfl_lm_mtcars

# tuning            ------------------------------------------------------------------
# validacao         ---------------------------------------------------------------
final_vld_fit_lm_mtcars <- last_fit(object = mdl_spec_lm_mtcars, 
                                    preprocessor = rec_lm_mtcars,
                                    split = split_mtcars)

final_vld_fit_lm_mtcars %>% collect_metrics()

final_mdl_fit_lm_mtcars <- fit(wkfl_lm_mtcars, data = mtcars)

final_mdl_fit_lm_mtcars

# predicao          ----------------------------------------------------------------

novo_dado <- mtcars %>% slice(20:32)

novo_dado

predict(final_mdl_fit_lm_mtcars, new_data = novo_dado) %>% bind_cols(novo_dado)

