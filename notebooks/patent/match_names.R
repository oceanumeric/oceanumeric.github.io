# load packages
library(pacman)
p_load(
    tidyverse, data.table, dtplyr, reshape2,
    archive, kableExtra, SPARQL, janitor,
    png, webp, Cairo, rsvg,
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

# read the dataset
de_firms <- fread("./data/orbis_de_matched_l.csv")
han_names <- fread("./data/202208_HAN_NAMES.txt")
han_person <- fread("./data/202208_HAN_PERSON.txt")
han_patents <- fread("./data/202208_HAN_PATENTS.txt")


# dimensions of those dataset
dim(de_firms)  # 246, 9
dim(han_names)  # 4191007, 3
dim(han_patents)  # 18355687, 5


de_firms %>%
    .[, .(bvdid, name_internat)] %>%
    # create another ID
    .[, ID := .I] %>%
    .[sample(.N, 5)]


test_name <- toupper("siemens industry software")
han_names %>%
    .[Person_ctry_code == "DE"] %>%
    .[Clean_name %like% test_name]


# test fed-match package
# reference: https://rb.gy/sibi6b

# clean names
de_firms %>%
    .[, .(bvdid, name_internat)] %>%
    .[, ID := .I] %>%
    .[, cleaned_names := clean_strings(name_internat)] -> foo1


han_names %>%
    .[Person_ctry_code == "DE"] %>%
    .[, cleaned_names := clean_strings(Clean_name)] -> foo2


han_person %>%
    .[Person_ctry_code == "DE"] %>%
    .[, H_ID := .I] %>%
    .[, cleaned_names := clean_strings(Person_name_clean)] -> foo3



merge_foo <- merge_plus(data1 = foo1, data2 = foo2,
                        by.x = "cleaned_names", by.y = "cleaned_names",
                        match_type = "fuzzy",
                        unique_key_1 = "ID",
                        unique_key_2 = "HAN_ID")


names(merge_foo)

merge_foo$match_evaluation %>% head()

dim(merge_foo$matches)
dim(merge_foo$data1_nomatch)

merge_foo$matches %>%
    .[sample(.N, 5)]


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

# not matched
merge_foo$data1_nomatch %>%
    .[sample(.N, 5)] %>%
    .[, h := lapply(cleaned_names, convert_string)] %>%
    head()

# corporate words
corporate_words %>%
    .[sample(.N, 5)] %>%
    kable("pipe", align = "cc")


test_name <- toupper("applied materials gmbh")
han_names %>%
    .[Person_ctry_code == "DE"] %>%
    .[Clean_name %like% test_name]


# match again
merge_foo$data1_nomatch %>%
    .[, cleaned_names := lapply(cleaned_names, convert_string)] -> foo4


han_patents %>%
    .[HAN_ID == 160981] %>%
    dim()


merge_foo2 <- merge_plus(data1 = foo4, data2 = foo2,
                        by.x = "cleaned_names", by.y = "cleaned_names",
                        match_type = "fuzzy",
                        unique_key_1 = "ID",
                        unique_key_2 = "HAN_ID")


# plot hypergeometric distribution
m_seq <- seq(0, 60, by = 1)
prob <- 1 - phyper(1, m_seq, 82 - m_seq, k = 5)

options(repr.plot.width = 6, repr.plot.height = 3)
data.frame(m = m_seq, prob = prob) %>%
    ggplot(aes(x = m, y = prob)) +
        geom_line(color=gray_scale[15], linetype = "twodash") +
        theme_bw() +
        labs(x = "Vlaue of m", y = "Probability") +
        theme(
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(color = "black")
        ) -> p1

ggsave(
    p1, 
    file='/home/jovyan/work/docs/images/blog/match_hypergeom_prob.png',
    width=6, height=3, type="cairo-png",
    device = grDevices::png, 
    dpi=300, bg='transparent')


######### ------ Combine two matches together ------- #########

merge_foo2$matches <- merge_foo2$matches[, nm_names := NULL]

all_matches <- rbind(merge_foo$matches, merge_foo2$matches)

all_matches %>%
    .[sample(.N, 5)]


######### ------ get all patents for matched firms ------- #########

# get HAN_ID
all_matches %>%
    .[, HAN_ID] -> all_han_ids


# get patents
han_patents %>%
    .[HAN_ID %in% all_han_ids] %>%
    .[sample(.N, 5)]