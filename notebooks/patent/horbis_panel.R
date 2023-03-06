library(pacman)
p_load(
    tidyverse, data.table, dtplyr, reshape2,
    kableExtra, SPARQL, janitor, readxl,
    png, webp, Cairo, rsvg, rbenchmark,
    httr, jsonlite, fedmatch, patchwork,
    corrplot, tidygraph, ggraph, igraph,
    treemap, splitstackshape
)
# set option
options(dplyr.summarise.inform = FALSE)
options(jupyter.plot_mimetypes = "image/png")
options(repr.plot.width = 8, repr.plot.height = 5)

gray_scale <- c("#F3F4F8", "#D2D4DA",  "#bcc0ca",
                "#D3D3D3", "#B3B5BD", "#838383",
                "#9496A1", "#7d7f89", "#777986",
                "#656673", "#5B5D6B", "#4d505e",
                "#404352", "#2b2d3b", "#2B2B2B", "#282A3A",
                "#1b1c2a", "#191a2b", "#141626", "#101223")
color_scale <- c("#BB4444", "#EE9988", "#FFFFFF",
                            "#77AADD", "#4477AA")
blue_scale <- c("#DBE7F5", "#95BCE3",
                "#699BCB", "#6597C7", "#4879A9",
                "#206BAD", "#11549B")

# check working directory
getwd()


### --- Construct panel dataset
horbis_df <- fread("./pyrio/horbis_dataset.csv")

dim(horbis_df)
names(horbis_df)

## applicationYear > 2010 

horbis_df %>%
    .[applicationYear >= 2010] %>%
    .[, .N, by = applicationYear] 


# calculate inventive age:
horbis_df %>%
    .[, .N, by = .(HAN_ID, bvdid, name_internat, applicationYear)] %>%
    .[order(rank(name_internat))] %>%
    .[, .N, by = .(HAN_ID, bvdid, name_internat)] %>%
    .[order(-rank(N))] -> inventive_age



horbis_df %>%
    .[applicationYear >= 2010] -> horbis_df2010


## group by firm 
horbis_df2010 %>% dim()  # 138807

horbis_df2010 %>%
    .[, .N, by = bvdid] %>%
    dim()  # 633 firms 


horbis_df2010 %>%
    .[, .N, by = name_internat] %>%
    .[order(-rank(N))] %>%
    head()


### panel: patent count, applications and granted ones 
horbis_df2010 %>%
    .[, .(patAppCount = .N, patGrantCount = uniqueN(.SD[granted == 1])),
            by = .(HAN_ID, bvdid, name_internat, applicationYear)] %>%
    head()

# patent scope: number of top IPC classes

horbis_df2010 %>%
    .[, ipc_temp := gsub("\\[|\\]", "", ipcClass)] %>%
    .[, ipc_temp := gsub("'", "", ipc_temp)] %>%
    separate_longer_delim(ipc_temp, delim = ",") %>%
    as.data.table() %>% 
    .[, ipcTop := tstrsplit(ipc_temp, "/", keep = c(1))] %>%
    .[, ipcTop := gsub(" ", "", ipcTop)] %>%
    .[, .N, by = .(patentNumber, ipcTop)] %>%
    .[, .N, by = patentNumber] %>%
    setnames(c("N"), c("patentScope")) -> pt_scope

# strsplit is slow 
# start <- Sys.time()
# horbis_df2010 %>%
#     .[, ipc_temp := gsub("\\[|\\]", "", ipcClass)] %>%
#     .[, ipc_temp := gsub("'", "", ipc_temp)] %>%
#     .[, strsplit(ipc_temp, ","), by = patentNumber] %>%
#     dim()
# end <- Sys.time()
# end - start


# uniqueN is slow 
# start <- Sys.time()
# horbis_df2010 %>%
#     .[, ipc_temp := gsub("\\[|\\]", "", ipcClass)] %>%
#     .[, ipc_temp := gsub("'", "", ipc_temp)] %>%
#     separate_longer_delim(ipc_temp, delim = ",") %>%
#     as.data.table() %>% 
#     .[, ipcTop := tstrsplit(ipc_temp, "/", keep = c(1))] %>%
#     .[, ipcTop := gsub(" ", "", ipcTop)] %>%
#     .[, .N, by = .(patentNumber, ipcTop)] %>%
#     .[, .(patentScope = uniqueN(.SD)), by = patentNumber] %>%
#     head()
# end <- Sys.time()
# end - start  # 3.3 seconds
    

horbis_df2010$patentScope <- pt_scope$patentScope


# average patent scope for firm each year
horbis_df2010 %>%
    .[, .(patAppCount = .N,
                    patGrantCount = uniqueN(.SD[granted == 1]),
                    avrgPatScope = round(mean(patentScope), 2)),
            by = .(HAN_ID, bvdid, name_internat, applicationYear)] %>%
    head()


# Herfindahl-Hirschman concentration index (HHI) Garcia-Vega 2006 

