library(data.table)
library(magrittr)
library(knitr)  # for kable, a function to print a table in different formats
library(ggplot2)  # only use it for advanced visualization
# for dplyr, it is recommended to call it as dplyr::filter, dplyr::select, etc.

########## --------- import data --------- ###########

# read data
dt <- fread("https://raw.githubusercontent.com/oceanumeric/data-science-go-small/main/Tutorial_01/survey_responses.csv")


# data structure
str(dt)  # 24 observations of 11 variables
head(dt)  # check the first 6 rows
summary(dt)

# set variable names
dt %>%
    setnames("Timestamp", "timeStamp") %>% # use %>% to chain functions
    str()

# change variable names from the second column to the last column
# with format: "q2", "q3", "q4", ...
dt %>%
    setnames(2:ncol(dt), paste0("q", 2:ncol(dt) - 1)) %>%
    str()

########## --------- check missing values --------- ###########

# Missing values are tricky as they can be represented in different ways
# such as "", "Na", "NA", "na", "N/A", "n/a", "NA/NaN", "NA/NaN/Na", etc.

# check missing values and print it as a table
sapply(dt, function(x) sum(is.na(x))) %>% kable()

# there is only one missing value in q2?
sapply(dt, function(x) sum(x =="")) %>% kable()
sapply(dt, function(x) sum(x == "Na")) %>% kable()

sapply(dt, function(x) sum(x =="", na.rm = TRUE)) %>% kable()
sapply(dt, function(x) sum(x == "Na", na.rm = TRUE)) %>% kable()

# replace missing values (represented as strings) with NA
dt[dt == ""] <- NA
dt[dt == "Na"] <- NA

sapply(dt, function(x) sum(is.na(x))) %>% kable()


########## --------- column operations --------- ###########

# check structure of dt again
str(dt)

# convert timeStamp to date format by adding a new column
# called dateTime to dt
dt %>%
    .[, dateTime := as.POSIXct(timeStamp, format = "%m/%d/%Y %H:%M:%S")] %>%
    str()

# create a new column called year
dt %>%
    .[, year := format(dateTime, "%Y")] %>%
    # you can also convert it to numeric
    # .[, year := as.numeric(format(dateTime, "%Y"))] %>%
    str()

# extract columns year q2 q3
dt %>%
    .[, .(year, q2, q3)] %>%
    str()

# select q9: 9. Did you try ChatGPT?
# convert it to lower case
dt %>%
    .[, .(q9 = tolower(q9))] %>% 
    str()

# select columns year, and from q3 to q10
# one needs to know the column index
dt %>%
    .[, c(13, 4:11)] %>%
    str()

# return a vector
dt %>%
    .[, (q7)]


# select columns with names starting with "q"
dt %>%
    .[, .SD, .SDcols = grep("^q", names(dt))] %>%
    str()

# or using patterns to select columns
dt %>%
    .[, .SD, .SDcols = patterns("^q")] %>%
    str()

# we can also do operations on multiple columns

# select columns based on their data types == int
dt %>%
    .[, .SD, .SDcols = is.integer] %>% str()

# convert values from q2 to q10 to lower case if they are characters
# return a new data.table
dt %>%
    .[, .SD, .SDcols = is.character] %>%
    .[, lapply(.SD, tolower), .SDcols = patterns("^q")] %>%
    str()


########## --------- row operations --------- ###########
str(dt)

# one row operation, without any row indices
# summarize q1 - chr, yes, no
# treat it as factor, using table() to count the number of each level

table(dt$q1)

# or convert it to factor first
dt %>%
    .[, .(q1 = factor(q1))] %>% 
    table() 

# better presentation
dt %>%
    .[, .(q1 = factor(q1))] %>% 
    .[, .(count = table(q1))] %>%
    kable()

# calculate the percentage of each level
dt %>%
    .[, .(q1 = factor(q1))] %>% 
    .[, .(count = table(q1))] %>%
    .[, share := count.N / sum(count.N)] %>%
    kable()

# plot it as a bar chart
# set options for the size
options(repr.plot.width = 8, repr.plot.height = 5)
dt %>%
    with(table(q1)) %>%
    barplot(main = "Did you learn regression model before?")

# plot share instead of count
dt %>%
    with(table(q1)/nrow(dt)) %>%
    barplot(main = "Did you learn regression model before?")


# more advanced visualization with ggplot2
dt %>%
    .[, .(q1 = factor(q1))] %>% 
    .[, .(count = table(q1))] %>%
    .[, share := count.N / sum(count.N)] %>%
    ggplot(aes(x = count.q1, y = share)) +
    geom_col(fill = "#6F6CAE") +
    geom_text(aes(label = round(share, 2)), vjust = -0.5) +
    labs(title = "Did you learn regression model before?")