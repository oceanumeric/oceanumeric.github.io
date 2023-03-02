library(pacman)
p_load(
    tidyverse, data.table, dtplyr, reshape2,
    kableExtra, SPARQL, janitor, readxl,
    png, webp, Cairo, rsvg, rbenchmark,
    httr, jsonlite, fedmatch, patchwork,
    corrplot, tidygraph, ggraph, igraph,
    treemap
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
setwd("./work/notebooks/patent")

## ---------- Understand OECD Patent Database --------- #######

horbis_de_patents <- fread("./data/horbis_de_patents.csv")
horbis_de_patents$idx <- as.integer(rownames(horbis_de_patents))

dim(horbis_de_patents)  # 256813
names(horbis_de_patents)
head(horbis_de_patents)


## read triadic patent families
tpf_epo <- fread("./data/202208_TPF_EPO.txt")

dim(tpf_epo)
names(tpf_epo)
head(tpf_epo)


tpf_uspto <- fread("./data/202208_TPF_USPTO.txt")

dim(tpf_uspto)
names(tpf_uspto)
head(tpf_uspto)

# from tpf_uspto and tpf_epo, one can find title and class and granted

## read linked patents collection
horbis_linked <- fread("./epodata/horbis_de_ep.csv")

dim(horbis_linked)
names(horbis_linked)
head(horbis_linked)



# han patents again
han_names <- fread("./data/202208_HAN_NAMES.txt")
han_patents <- fread("./data/202208_HAN_PATENTS.txt")


han_names %>%
    .[Person_ctry_code == "DE"] %>%
    .[Clean_name %like% "^BAYER"] %>%
    .[, lapply(HAN_ID, get_dim)]


get_dim <- function(han_id) {

    han_patents %>%
    .[HAN_ID == han_id] %>%
    dim() -> foo
    if (foo[1] >= 100) {
        print(han_id)
        print("----")
        print(foo[1])
    }
}



# BMW, HAN_ID = 236744
options(repr.plot.width = 7, repr.plot.height = 4)
horbis_linked %>%
    .[HAN_ID == 236744] %>%
    .[granted == 1] %>%
    .[, temp := tstrsplit(applicationDate, ", ", fixed = TRUE, keep = c(2))] %>%
    .[, applicationYear := tstrsplit(temp, " ", fixed = TRUE, keep = c(3))] %>%
    .[, temp := NULL] %>%
    .[, temp := tstrsplit(grantDate, ", ", fixed = TRUE, keep = c(2))] %>%
    .[, grantYear := tstrsplit(temp, " ", fixed = TRUE, keep = c(3))] %>%
    .[, temp := NULL] %>%
    .[, .N, by = applicationYear] %>%
    .[order(-rank(N))] %>% head(15) %>%
    .[order(applicationYear)] %>%
    ggplot(aes(x = applicationYear, y = N)) +
    geom_col(fill = gray_scale[4], color = gray_scale[9]) + 
    theme_bw() + 
    theme(
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.line = element_line(color = "black"),
        axis.text.x=element_text(face='bold', color='black', size=10),
        axis.title.x=element_text(size=12, face='bold'),
        axis.text.y=element_text(face='bold', color='black', size=10),
        axis.title.y=element_text(size=12, face='bold'),
        ) + 
    labs(title = "Trend of patent applications")


## Combine tempdata

temp1 <- fread("./tempdata/epo_temp_batch1.csv")
temp2 <-  fread("./tempdata/epo_temp_batch2.csv")

temp <- rbind(temp1, temp2)


for (i in 2:8) {
    file <- paste(
        "./tempdata/epo_temp_batch", i,
        ".csv", sep = "")
    csv <- fread(file)
    temp <- rbind(temp, csv)
}

csv91 <- fread("./tempdata/epo_temp_batch9_pre.csv")

temp <- rbind(temp, csv91)

csv92 <- fread("./tempdata/epo_temp_batch9.csv")

temp <- rbind(temp, csv92)


for (i in 10:25) {
    print(i)
    file <- paste(
        "./tempdata/epo_temp_batch", i,
        ".csv", sep = "")
    csv <- fread(file)
    temp <- rbind(temp, csv)
}

dim(temp)

dim(horbis_de_patents)


tail(temp, 1)

horbis_de_patents %>%
    .[Patent_number == "EP2458292"]


get_idx <- function(patentNumber) {
    horbis_de_patents %>%
    .[, c(30, 31)] %>%
    .[Patent_number == patentNumber] %>%
    .[, c(2)] -> idx

    return(idx$idx)
}

get_idx("EP2458292")

temp %>%
    .[, idx2 := lapply(patentNumber, get_idx)]


fwrite(temp, "./pyrio/temp.csv")
fwrite(horbis_de_patents, "./pyrio/horbis.csv")


temp2 <- fread("./pyrio/temp2.csv")

temp2 %>%
    .[, idx2 := as.integer(idx2)]


temp <- fread("./pyrio/temp.csv")
foo <- horbis_de_patents[1:252990, ]
setnames(foo, "Patent_number", "patentNumber")

# delete empty ones
temp %>%
    .[patentNumber != ""] -> temp

foo %>%
    .[patentNumber != ""] -> foo



setkey(foo, "patentNumber")
setkey(temp, "patentNumber")

foo_merged <- merge(foo, temp, all.x = TRUE)


foo_merged %>%
    .[!is.na(kindCode) & !is.na(familyID)] -> foo_merged


dim(foo_merged)
names(foo_merged)

foo_merged %>%
    unique(by = "bvdid") %>%
    dim()  # 518 firms


foo_merged %>%
    .[,
        applicationYear := substr(as.character(applicationDate), 1, 4)
                    ] %>%
    .[, applicationYear := as.integer(applicationYear)]



foo_merged %>%
    .[, .N, by = startYear]  # start from 2014 


foo_merged %>%
    .[applicationYear >= 2010] %>%
    unique(by = "familyID") %>%
    dim()  # 135624/ unique 70605 


foo_merged %>%
    .[applicationYear >= 2010] -> foo_merged_2010


foo_merged_2010 %>%
    .[Publn_auth != "EP"] %>%
    .[, .N, by = .(name_internat)] %>%
    .[order(-rank(N))] %>% head(20)


foo_merged_2010 %>%
    .[Publn_auth != "EP"] %>%
    .[, .N, by = .(applicationYear)] %>%
    .[order(-rank(N))] %>% head(20)




foo_merged2 <- fread("./pyrio/foo_merged2.csv")

dim(foo_merged2)  # 443173
names(foo_merged2)
head(foo_merged2)


foo_merged2 %>%
    .[,
        applicationYear := substr(as.character(applicationDate), 1, 4)
                    ] %>%
    .[, applicationYear := as.integer(applicationYear)]


foo_merged2 %>%
    .[applicationYear >= 2010] -> foo_merged2_2010


foo_merged2 %>%
    .[applicationYear <= 2014] %>%
    .[, .(count1 = .N), by = .(ipcTop, name_internat)] %>%
    .[, `:=`(totalPatent = sum(count1)), by = .(name_internat)] %>%
    .[order(-rank(count1))] %>%
    .[, ipcShare := round(count1 / totalPatent, 4)] %>%
    head(10)


foo_merged2 %>%
    .[applicationYear > 2014 & applicationYear <= 2019] %>%
    .[, .N, by = .(ipcTop, name_internat)] %>%
    .[order(-rank(N))] %>%
    head(10)