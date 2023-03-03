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


for (i in 10:29) {
    print(i)
    file <- paste(
        "./tempdata/epo_temp_batch", i,
        ".csv", sep = "")
    csv <- fread(file)
    temp <- rbind(temp, csv)
}

dim(temp)
tail(temp, 1)


temp %>%
    unique(by = "patentNumber") %>% dim()


dim(horbis_de_patents)

horbis_de_patents %>%
    unique(by = "Patent_number") %>% dim()

horbis_de_patents %>%
    .[Patent_number == "EP2833367"]


foo <- horbis_de_patents
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
    dim()  # 634 firms 


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



fwrite(foo_merged, "./pyrio/foo_merged.csv")

foo_merged2 <- fread("./pyrio/foo_merged2.csv")

dim(foo_merged2)  # 469654
setnames(foo_merged2, "V1", "pgidx")
names(foo_merged2)
head(foo_merged2)


foo_merged2 %>%
    .[,
        applicationYear := substr(as.character(applicationDate), 1, 4)
                    ] %>%
    .[, applicationYear := as.integer(applicationYear)]


foo_merged2 %>%
    .[applicationYear >= 2010] -> foo_merged2_2010


## complexity of patents
options(repr.plot.width = 10, repr.plot.height = 4)

Cairo(width = 10, height = 4,
      file = "../../docs/blog/images/patent_ipc_complexity.png",
      type = "png", bg = "transparent", dpi = 300, units = "in")

par(mfrow = c(1, 2))

foo_merged2_2010 %>%
    unique(by = "pgidx") %>%
    unique(by = "familyID") %>%
    .[,
        .(average_ipc_count = mean(ipcCount, na.rm = TRUE)),
        by = applicationYear] %>%
        plot(type = "b", xlab = "Year",
        ylab = "Mean of IPC Class Counts",
        main = "Patent Applications"
        )


foo_merged2_2010 %>%
    unique(by = "pgidx") %>%
    unique(by = "familyID") %>%
    .[granted == 1] %>%
    .[,
        .(average_ipc_count = mean(ipcCount, na.rm = TRUE)),
        by = applicationYear] %>%
        plot(type = "b", xlab = "Year",
        ylab = "Mean of IPC Class Counts",
        main = "Granted Patents"
        )

dev.off()


# total patent 103413
foo_merged2_2010 %>%
    .[applicationYear <= 2014] %>%
    .[, .(count1 = .N), by = .(ipcTop, name_internat)] %>%
    .[, `:=`(totalPatent = sum(count1)), by = .(name_internat)] %>%
    .[order(-rank(count1))] %>%
    .[, ipcShare := round(count1 / totalPatent, 4)] %>%
    .[, ipcTop] -> ipcTop_1014


# total patent 128421
foo_merged2_2010 %>%
    .[applicationYear > 2014 & applicationYear <= 2019] %>%
    .[, .(count1 = .N), by = .(ipcTop, name_internat)] %>%
    .[, `:=`(totalPatent = sum(count1)), by = .(name_internat)] %>%
    .[order(-rank(count1))] %>%
    .[, ipcShare := round(count1 / totalPatent, 4)] %>%
    .[, ipcTop] -> ipcTop_1519


# get new classes
setdiff(ipcTop_1519, ipcTop_1014)
setdiff(ipcTop_1014, ipcTop_1519)


foo_merged2_2010 %>%
    .[applicationYear <= 2014] %>%
    .[, .(count1 = .N), by = .(ipcTop, name_internat)] %>%
    .[, `:=`(totalPatent = sum(count1)), by = .(name_internat)] %>%
    .[order(-rank(count1))] %>%
    .[, ipcShare := round(count1 / totalPatent, 4)] %>%
    head()

foo_merged2_2010 %>%
    .[applicationYear > 2014 & applicationYear <= 2019] %>%
    .[, .(count1 = .N), by = .(ipcTop, name_internat)] %>%
    .[, `:=`(totalPatent = sum(count1)), by = .(name_internat)] %>%
    .[order(-rank(count1))] %>%
    .[, ipcShare := round(count1 / totalPatent, 4)] %>%
    .[name_internat == "Basf SE"] %>%
    .[ipcTop %like% "^G06"]



ict_ipc <- fread("./pyrio/ict_ipc_tags.csv")

ict_ipc_top <- unique(ict_ipc$ipc_top)

length(ict_ipc_top)

ict_ipc_top

ai_ipc <- c("G06F", "G06K", "H04N", "G10L",
        "G06T", "A61B", "H04M", "G01N", "G06F",
        "H04R", "G06N", "G01S", "H04L", "G06Q",
        "H04B", "G09G", "G02B", "G11B", "G08B",
        "G01B", "G05D", "H04N", "B25J", "H04N")

is_ict <- function(ipc_class) {
    if (ipc_class %in% ict_ipc_top) {
        return(1)
    } else {
        return(0)
    }
}

is_ai <- function(ipc_class) {
    if (ipc_class %in% ai_ipc) {
        return(1)
    } else {
        return(0)
    }
}

foo_merged2_2010 %>%
    .[, isICT := lapply(ipcTop, is_ict)] %>%
    head(2)

foo_merged2_2010 %>%
    .[, isAI := lapply(ipcTop, is_ai)] %>%
    head(2)


# share of ICT patents for year
foo_merged2_2010 %>%
    unique(by = "familyID") %>%
    .[isICT == 1] %>%
    unique(by = "pgidx") %>%
    .[, .N, by = name_internat] %>%
    .[order(-rank(N))] %>%
    head(10)

# share of AI patents
foo_merged2_2010 %>%
    .[Publn_auth != "EP"] %>%
    unique(by = "familyID") %>%
    .[isAI == 1] %>%
    unique(by = "pgidx") %>%
    .[, .N, by = name_internat] %>%
    .[order(-rank(N))] %>%
    head(10)

# plot share of ICT or AI patents 
foo_merged2_2010 %>%
    unique(by = "familyID") %>%
    .[isAI == 1] %>%
    unique(by = "pgidx") %>%
    .[, .N, by = applicationYear] -> ai_patents_by_year


foo_merged2_2010 %>%
    unique(by = "familyID") %>%
    unique(by = "pgidx") %>%
    .[, .N, by = applicationYear] -> total_patent_by_year


ai_patents <- merge(ai_patents_by_year,
                        total_patent_by_year,
                        by = "applicationYear")

ai_patents %>%
    setnames(c("N.x", "N.y"), c("AI_patent", "Total_patent")) %>%
    .[, share := round(AI_patent / Total_patent, 4) * 100] %>%
    head()


options(repr.plot.width = 8, repr.plot.height = 5)

Cairo(width = 8, height = 5,
      file = "../../docs/blog/images/patent_ai_trend.png",
      type = "png", bg = "transparent", dpi = 300, units = "in")
ai_patents %>%
    .[, c(1, 4)] %>%
    plot(type = "b", xlab = "Year", ylab = "Share of AI Patents",
            main = "Trend of AI Patents (%)")

dev.off()


# AI patents 2010 to 2014
foo_merged2_2010 %>%
    .[applicationYear <= 2014] %>%
    unique(by = "familyID") %>%
    .[isAI == 1] %>%
    unique(by = "pgidx") %>%
    .[, .N, by = .(bvdid, name_internat, svLevel2, totalCost)] %>%
    .[order(-rank(N))] %>%
    setnames("N", "AI_Patents") -> ai_patents_1014


foo_merged2_2010 %>%
    .[applicationYear <= 2014] %>%
    unique(by = "familyID") %>%
    unique(by = "pgidx") %>%
    .[, .N, by = .(bvdid, name_internat, svLevel2, totalCost)] %>%
    .[order(-rank(N))] %>%
    setnames("N", "Total_Patents") -> total_patents_1014


foo_merged2_2010 %>%
    .[applicationYear > 2014 & applicationYear <= 2019] %>%
    unique(by = "familyID") %>%
    .[isAI == 1] %>%
    unique(by = "pgidx") %>%
    .[, .N, by = .(bvdid, name_internat, svLevel2, totalCost)] %>%
    .[order(-rank(N))] %>%
    setnames("N", "AI_Patents") -> ai_patents_1519



foo_merged2_2010 %>%
    .[applicationYear > 2014 & applicationYear <= 2019] %>%
    unique(by = "familyID") %>%
    unique(by = "pgidx") %>%
    .[, .N, by = .(bvdid, name_internat, svLevel2, totalCost)] %>%
    .[order(-rank(N))] %>%
    setnames("N", "Total_Patents") -> total_patents_1519



dim(ict_patents_1014)
dim(ict_patents_1519)

ai_patents_compare <- merge(ai_patents_1014,
                            ai_patents_1519, by = "bvdid")

head(ai_patents_compare)

ai_patents_compare %>%
    .[AI_Patents.x > 20] %>%
    .[, ai_growth := AI_Patents.y - AI_Patents.x] %>%
    .[, c(1:5, 10)] %>%
    .[, ai_growth_rate := round(ai_growth / AI_Patents.x, 4)] %>%
    with(plot(AI_Patents.x, ai_growth_rate))


# ai patents, scale and share
ai_patents_1519_merge <- merge(ai_patents_1519,
                                total_patents_1519[, c(1, 5)],
                                by = "bvdid")

ai_patents_1519_merge %>%
    .[, ai_patents_share := round(AI_Patents/Total_Patents, 4)*100] %>%
    .[order(-rank(AI_Patents))] %>%
    head(20) %>%
    .[, c(2, 5, 7)] %>%
    kable("pipe", align = "lcc")



# get all AI patents
foo_merged2_2010 %>%
    .[Publn_auth != "EP"] %>%
    .[applicationYear > 2014 & applicationYear <= 2020] %>%
    unique(by = "familyID") %>%
    .[isAI == 1] %>%
    unique(by = "pgidx") -> ai_patents_1520

fwrite(ai_patents_1520, "./pyrio/ai_p1520.csv")

ai_patents_1520 %>%
    .[abstract %like% "neural network"] %>%
    .[, .N, by = .(bvdid, name_internat, svLevel2, totalCost)] %>%
    .[order(-rank(N))]


ai_patents_1520 %>%
    .[abstract %like% "machine learning"] %>%
    .[, .N, by = .(bvdid, name_internat, svLevel2, totalCost)] %>%
    .[order(-rank(N))]


ai_patents_1520 %>%
    .[abstract %like% "object recogni"] %>%
    .[, .N, by = .(bvdid, name_internat, svLevel2, totalCost)] %>%
    .[order(-rank(N))]


dim(ai_patents_1520)