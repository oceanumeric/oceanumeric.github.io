# install and load essential packages
# install and import libraries 
package_list <- c(
  "tidyverse",  # this package is essential 
  "janitor", # useful functions for cleaning imported data
  "biblionetwork", # creating edges
  "tidygraph", # for creating networks
  "ggraph", # plotting networks
  "ggdark",  # dark theme 
  "flowCore",  # flow network
  "vite",
  "networkflow"
)
for (p in package_list) {
  if (p %in% installed.packages() == FALSE) {
    install.packages(p, dependencies = TRUE)
  }
  library(p, character.only = TRUE)
}


# read the dataset
data <- read_csv("/Users/Michael/Desktop/Core/innomgmt1.csv", skip=1)
for (i in c(2:3)) {
    path <- "/Users/Michael/Desktop/Core/innomgmt"
    file_path <- paste(path, as.character(i), ".csv", sep="")
    temp <- read_csv(file_path, skip=1)
    # combine the dataset and 
    data <- rbind(data, temp)
}

# clean the column names (lower case with _)
data <- clean_names(data)
names(data)[5] <- "source"
names(data)[11] <- "organization" 
names(data)[12] <- "country"
names(data)[13] <- "url"
print(names(data))

data <- data[!duplicated(data$title), ]
print(dim(data))  # 5658 (total articles) - 32 = 5626

# create a table by mapping each author ot is affiliation
article_affiliation <- data %>%
                       select(publication_id, organization, country) %>%
                       separate_rows(c("organization","country"), sep=";")
head(article_affiliation)  # what a beautiful table (see publication_id)