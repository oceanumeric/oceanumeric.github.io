# load packages
library(pacman)
p_load(
    tidyverse, data.table, dtplyr, reshape2,
    archive, kableExtra, SPARQL, janitor,
    png, webp, Cairo, rsvg,
    httr, jsonlite)
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
han_patents <- fread("./data/202208_HAN_PATENTS.txt")


de_firms %>%
    .[sample(.N, 5)]


han_names %>%
    .[Person_ctry_code == "DE"] %>%
    .[sample(.N, 5)]
