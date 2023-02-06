# load packages 
library(pacman)
p_load(tidyverse, data.table)

# read the dataset
sdata3 = read_csv('docs/math/esl/data/sdata3_training.csv')

sdata3 <- sdata3 %>%
            group_by(class) %>%
            mutate(
                x1_sub_mean = x1 - mean(x1),
                x2roll_sub_mean = x2roll - mean(x2roll)
                )
head(sdata3)
dim(sdata3)

write_csv(sdata3, 'docs/math/esl/data/sdata3_training_R.csv')

