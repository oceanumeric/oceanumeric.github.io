library(data.table)
library(magrittr)
library(knitr)  # for kable, a function to print a table in different formats
library(ggplot2)  # only use it for advanced visualization
# for dplyr, it is recommended to call it as dplyr::filter, dplyr::select, etc.

########## --------- import data --------- ###########

# read data
dt <- fread("https://raw.githubusercontent.com/oceanumeric/data-science-go-small/main/Lecture01/survey_responses.csv")


# data structure
str(dt)  # 24 observations of 11 variables
head(dt)  # check the first 6 rows
summary(dt)

# set variable names
dt %>%
    setnames("Timestamp", "timesStamp") %>% # use %>% to chain functions
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


########## --------- data transformation --------- ###########

# check structure of dt again
str(dt)

# convert timeStamp to date format by adding a new column
# called dateTime to dt
dt %>%
    .[, dateTime := as.POSIXct(timesStamp, format = "%m/%d/%Y %H:%M:%S")] %>%
    str()

# create a new column called year
dt %>%
    .[, year := format(dateTime, "%Y")] %>%
    # you can also convert it to numeric
    # .[, year := as.numeric(format(dateTime, "%Y"))] %>%
    str()

# select columns and return a new data.table
dt %>%
    .[, .(year, q2, q3)] %>%
    str()

# select columns year, and from q3 to q10
dt %>%
    .[, .(year, q3:q10)] %>%
    str()

# select q9: 9. Did you try ChatGPT?
# convert it to lower case
dt %>%
    .[, .(q9 = tolower(q9))] %>% 
    str()



# we can also do operations on multiple columns
# convert values from q2 to q10 to lower case if they are characters
# return a new data.table
dt %>%
    .[, lapply(.SD, tolower), .SDcols = 2:ncol(dt)] %>%
    str()
