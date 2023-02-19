# load packages
library(pacman)
p_load(
    tidyverse, data.table, dtplyr, reshape2,
    archive, kableExtra, SPARQL, janitor,
    png, webp, Cairo, rsvg, rbenchmark,
    httr, jsonlite, fedmatch)
# set option
options(dplyr.summarise.inform = FALSE)
options(repr.plot.width = 6, repr.plot.height = 4)

gray_scale <- c("#F3F4F8", "#D2D4DA",  "#bcc0ca",
                "#D3D3D3", "#2B2B2B",
                "#B3B5BD", "#838383",
                "#9496A1", "#7d7f89", "#777986",
                "#656673", "#5B5D6B", "#4d505e",
                "#404352", "#2b2d3b", "#282A3A",
                "#1b1c2a", "#191a2b", "#141626", "#101223")

# check working directory
getwd()


########## ------- Create a merged dataset ----------- ###############
projects <- fread("./data/cordis_h2020/project.csv")
org <- fread("./data/cordis_h2020/organization.csv")
sv_tags <- fread("./data/cordis_h2020/euroSciVoc.csv")

# transform projects by selecting some columns
names(projects)
names(projects)[1] <- "projectID"

projects %>%
    .[, startYear := as.numeric(format(startDate, "%Y"))] %>%
    .[, endYear := as.numeric(format(endDate, "%Y"))] %>%
    .[, c(
        "projectID", "startYear", "endYear",
        "status", "title", "topics",
        "fundingScheme", "grantDoi")] -> projects_subset


# view organization
org %>%
    .[sample(.N, 3)]


# view sv_tas and # extract topics
sv_tags %>%
    .[, c(1:4)] %>%
    .[, svLevel1 := tstrsplit(
        euroSciVocPath, "/", fixed = TRUE, keep = c(2))] %>%
    .[, svLevel2 := tstrsplit(
        euroSciVocPath, "/", fixed = TRUE, keep = c(3))] -> sv_tags_levels

# group by title
sv_tags_levels %>%
    .[, c(1, 4)] %>%
    .[, lapply(.SD, paste0,  collapse = "/"), by = projectID] -> sv_tags_title

# merge and get unique
sv_tags_title %>%
    .[sv_tags_levels, on = .(projectID)] %>%
    unique(by = "projectID") %>%
    .[, c(1:4, 6, 7)] -> sv_tags_sum


# merge projects_sub, org and sv_tags_levels 
projects_subset <- merge(projects_subset, sv_tags_sum,
                         by = "projectID", all.x = TRUE)
horizon <- merge(org, projects_subset, by = "projectID", all.x = TRUE)



# one project is normally applied by many organizations 
horizon %>%
    .[sample(.N, 5)]
