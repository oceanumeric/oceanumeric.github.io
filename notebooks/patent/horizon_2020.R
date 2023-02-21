# load packages
library(pacman)
p_load(
    tidyverse, data.table, dtplyr, reshape2,
    kableExtra, SPARQL, janitor,
    png, webp, Cairo, rsvg, rbenchmark,
    httr, jsonlite, fedmatch
)
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


# on projects level
projects_subset %>%
    .[, .(svLevel2)] %>%
    .[, .N, by=svLevel2] %>%
    .[order(-rank(N))] %>%
    head(10) %>%
    kable("pipe", align = "cl")

# one project is normally applied by many organizations 
horizon %>%
    .[sample(.N, 5)]

names(horizon)


# filter out German firms
horizon %>%
    .[, c(1, 2, 3, 5, 7, 8:12, 23, 26, 27, 32, 33, 37)] %>%
    .[, totalCost := gsub(",", ".", totalCost)] %>%
    .[, totalCost := as.numeric(totalCost)] %>%
    .[country == "DE" & activityType == "PRC"] -> horizon_de_firms


dim(horizon_de_firms)  # 7395
names(horizon_de_firms)
horizon_de_firms[sample(.N, 5)]


# WARNINGS: one firms might have several projects 
horizon_de_firms %>%
    unique(by = "organisationID") %>% dim()  # 3339


########## ------ Match names with Orbis dataset ------ #########
orbis_de <- fread("./data/orbis_de_firms.csv")
dim(orbis_de)

orbis_de %>%
    .[, .(bvdid, name_internat, category_of_company,
                city_internat, postcode)] %>%
    # clean postcode
    .[, postcode := gsub("\\.0", "", postcode)] %>%
    # create another ID
    .[, orbisID := .I] %>%
    .[, cleaned_names := clean_strings(name_internat)] -> foo1

names(foo1)

horizon_de_firms %>%
    unique(by = "organisationID") %>% 
    .[, horizonID := .I] %>%
    .[, cleaned_names := clean_strings(name)] -> foo2

dim(foo2)
names(foo2)


######## ----------- match fuzzy ------ #######
match_fuzzy <- merge_plus(data1 = foo1, data2 = foo2,
                        by.x = "cleaned_names", by.y = "cleaned_names",
                        match_type = "fuzzy",
                        unique_key_1 = "orbisID",
                        unique_key_2 = "horizonID")

names(match_fuzzy)
match_fuzzy$match_evaluation

match_fuzzy$matches %>%
    .[postcode == postCode] %>%
    unique(by = "organisationID") -> horbis_de

horbis_de %>%
    dim()  # 1525

# view match
horbis_de %>%
    .[, .(city_internat, postcode, cleaned_names_1,
                    cleaned_names_2, postCode, city)] %>%
    .[sample(.N, 5)]


horbis_de %>%
    .[category_of_company == "VERY LARGE COMPANY"] %>%
    .[SME == FALSE] %>%
    dim()  # 420


######## ----------- save the final dataset ------ #######
fwrite(horbis_de, "./data/horbis_de.csv")


###
###
###
### ----------- match with patent datasets ------ #######
han_names <- fread("./data/202208_HAN_NAMES.txt")



dim(horbis_de)
names(horbis_de)

# function to clean string for patents
convert_string <- function(x){

    if (x %like% "aktiengesellschaft") {
        x <- gsub("aktiengesellschaft", "ag", x)
    }

    if (x %like% "electronic") {
        x <- gsub("electronic", "elect", x)
    }

    if (x %like% "and co kg") {
        x <- gsub("and co kg", "", x)
    }

    if (x %like% "international") {
        x <- gsub("international", "int", x)
    }

    if (x %like% "technologies") {
        x <- gsub("technologies", "tech", x)
    }

    if (x %like% "technologie") {
        x <- gsub("technologie", "tech", x)
    }

    if (x %like% "engineering") {
        x <- gsub("engineering", "eng", x)
    }

    return(x)
}


horbis_de %>%
    .[, c(3:7, 10:25)] %>%
    .[, horbisID := .I] %>%
    .[, cleaned_names := clean_strings(name_internat)] %>%
    .[, cleaned_names := lapply(cleaned_names,
                                convert_string)] -> horbis_foo


han_names %>%
    .[Person_ctry_code == "DE"] %>%
    .[, patentID := .I] %>%
    .[, cleaned_names := clean_strings(Clean_name)] -> patent_foo


patent_match <- merge_plus(data1 = horbis_foo, data2 = patent_foo,
                        by.x = "cleaned_names", by.y = "cleaned_names",
                        match_type = "fuzzy",
                        unique_key_1 = "horbisID",
                        unique_key_2 = "patentID")

names(patent_match)
patent_match$match_evaluation

patent_match$matches %>%
    .[, c(3:28)] %>%
    unique(by = "organisationID") -> horbis_de_han

dim(horbis_de_han)
names(horbis_de_han)
head(horbis_de_han)

horbis_de_han %>%
    .[category_of_company == "VERY LARGE COMPANY"] %>%
    .[SME == FALSE] %>%
    dim()  # 283


######## ----------- save the final dataset ------ #######
fwrite(horbis_de_han, "./data/horbis_de_han.csv")





## ---------------- Merge patents  --------- #######
horbis_de_han <- fread("./data/horbis_de_han.csv")
han_patents <- fread("./data/202208_HAN_PATENTS.txt")


dim(horbis_de_han)
names(horbis_de_han)


horbis_de_han %>%
    .[sample(.N, 5)]


horbis_de_han %>%
    .[, HAN_ID] -> all_han_ids


length(all_han_ids)

# get patents
han_patents %>%
    .[HAN_ID %in% all_han_ids] %>%
    dim()


# create a table with names and patents
han_patents %>%
    .[HAN_ID %in% all_han_ids] -> all_matches_patents

dim(all_matches_patents)
names(all_matches_patents)
head(all_matches_patents)



horbis_de_patents <- merge(horbis_de_han,
                        all_matches_patents,
                        by = "HAN_ID",
                        all.x = TRUE
                        )

dim(horbis_de_patents)
names(horbis_de_patents)
head(horbis_de_patents)


# same with the case study I did for Airbus
horbis_de_patents %>%
    .[HAN_ID == 60513] %>%
    dim()  # 1218 


horbis_de_patents %>%
    .[sample(.N, 5)]


# firm level
horbis_de_patents %>%
    unique(by = "organisationID") %>%
    .[sample(.N, 5)]


horbis_de_patents %>%
    .[, .N, by = .(name_internat, svLevel2)] %>%
    .[order(-rank(N))] %>%
    head(10)


######## ----------- save the final dataset ------ #######
fwrite(horbis_de_patents, "./data/horbis_de_patents.csv")



## --------------- extract patent information --------- #######


horbis_de_patents <- fread("./data/horbis_de_patents.csv")

horbis_de_patents %>%
    .[44833, c(HAN_ID, Patent_number)]

# align datasets
horbis_epo <- fread("./data/horbis_de_epo.csv")

dim(horbis_epo) # 44833 
tail(horbis_epo[, c(1:10)], 1)
horbis_epo[44833, c(1:10)]



horbis_epo2 <- fread("./data/horbis_de_epo2.csv")
horbis_epo2 <- horbis_epo2[c(1:2095)]

head(horbis_epo2[, c(1:10)], 1)
horbis_de_patents %>%
    .[44834, c(HAN_ID, Patent_number)]

tail(horbis_epo2[, c(1:10)], 1)
horbis_de_patents %>%
    .[46928, c(HAN_ID, Patent_number)]


horbis_epo3 <- fread("./data/horbis_de_epo3.csv")
horbis_epo3 <- horbis_epo3[c(1:1785)]

head(horbis_epo3[, c(1:10)], 1)
horbis_de_patents %>%
    .[46929, c(HAN_ID, Patent_number)]


################## EPO Linked data

foo <- fread("./epodata/horbis_de_linked.csv")
dim(foo)

tail(foo)

foo <- fread("./data/horbis_de_epo4.csv")

dim(foo)

tail(foo)