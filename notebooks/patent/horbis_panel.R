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

dim(horbis_df)  # 236636
names(horbis_df)


## step I - keep one patent for one family ID with priority
## US > WO > EP
horbis_df %>%
    .[order(-rank(Publn_auth))] %>%
    unique(by = "familyID") %>%
    .[, .N, by = Publn_auth]


horbis_df %>%
    .[order(-rank(Publn_auth))] %>%
    unique(by = "familyID") -> horbis_unique


## applicationYear > 2010 

horbis_unique %>%
    .[applicationYear >= 2010] %>%
    .[, .N, by = applicationYear] 


# calculate inventive age:
horbis_unique %>%
    .[, .N, by = .(HAN_ID, bvdid, name_internat, applicationYear)] %>%
    .[order(rank(name_internat))] %>%
    .[, .N, by = .(HAN_ID, bvdid, name_internat)] %>%
    .[order(-rank(N))] -> inventive_age



horbis_unique %>%
    .[applicationYear >= 2010] -> horbis_2010


## group by firm 
horbis_2010 %>% dim()  # 78914

horbis_2010 %>%
    .[, .N, by = bvdid] %>%
    dim()  # 632 firms 


horbis_2010 %>%
    .[, .N, by = name_internat] %>%
    .[order(-rank(N))] %>%
    head()


### panel: patent count, applications and granted ones 
horbis_2010 %>%
    .[, .(patAppCount = .N, patGrantCount = uniqueN(.SD[granted == 1])),
            by = .(HAN_ID, bvdid, name_internat, applicationYear)] -> pp_count


# merge the dataset
inventive_age %>%
    .[, c(1, 4)] -> age_temp

setkey(age_temp, "HAN_ID")
setkey(pp_count, "HAN_ID")

horbis_panel1 <- merge(pp_count, age_temp, by = "HAN_ID", all.x = TRUE)

horbis_panel1 %>%
    setnames(c("N"), c("inventive_age"))

# patent scope: number of top IPC classes

horbis_2010 %>%
    .[, ipc_temp := gsub("\\[|\\]", "", ipcClass)] %>%
    .[, ipc_temp := gsub("'", "", ipc_temp)] %>%
    .[, ipc_temp := gsub(" ", "", ipc_temp)] %>%
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
    

horbis_2010$patentScope <- pt_scope$patentScope


# average patent scope for firm each year (top ipc 4-digit)
horbis_2010 %>%
    .[, .(patAppCount = .N,
                    patGrantCount = uniqueN(.SD[granted == 1]),
                    avrgPatScope = round(mean(patentScope), 2)),
            by = .(HAN_ID, bvdid, name_internat, applicationYear)] -> avrg_ps

avrg_ps %>%
    .[, c(1, 4, 7)] -> ps_temp


setkeyv(horbis_panel1, c("HAN_ID", "applicationYear"))
setkeyv(ps_temp, c("HAN_ID", "applicationYear"))

horbis_panel2 <- merge(horbis_panel1, ps_temp,
                    by = c("HAN_ID", "applicationYear"), all.x = TRUE)


# total patent scope for firms at each year
# count number of unique ipc top tags for the whole year 
horbis_2010 %>%
    .[, ipc_temp := gsub("\\[|\\]", "", ipcClass)] %>%
    .[, ipc_temp := gsub("'", "", ipc_temp)] %>%
    .[, ipc_temp := gsub(" ", "", ipc_temp)] %>%
    separate_longer_delim(ipc_temp, delim = ",") %>%
    as.data.table() %>%
    .[, .N, by = .(HAN_ID, bvdid, name_internat, applicationYear, ipc_temp)] %>%
    .[, ipcTop := tstrsplit(ipc_temp, "/", keep = c(1))] %>%
    .[, ipcTop := gsub(" ", "", ipcTop)] %>%
    .[, .N, by = .(HAN_ID, bvdid, name_internat, applicationYear, ipcTop)] %>%
    setnames("N", "ipcCount") %>%
    .[, .N, by = .(HAN_ID, bvdid, name_internat, applicationYear)] %>%
    head()


# Herfindahl-Hirschman concentration index (HHI) Garcia-Vega 2006 
# based on ipc top tags (5 digits)
horbis_2010 %>%
    .[, ipc_temp := gsub("\\[|\\]", "", ipcClass)] %>%
    .[, ipc_temp := gsub("'", "", ipc_temp)] %>%
    .[, ipc_temp := gsub(" ", "", ipc_temp)] %>%
    separate_longer_delim(ipc_temp, delim = ",") %>%
    as.data.table() %>%
    .[, .N, by = .(HAN_ID, bvdid, name_internat,
                        applicationYear, ipc_temp)] %>%
    .[, ipcTop := tstrsplit(ipc_temp, "/", keep = c(1))] %>%
    .[, ipcTop := gsub(" ", "", ipcTop)] %>%
    .[, .N, by = .(HAN_ID, bvdid, name_internat,
                            applicationYear, ipcTop)] %>%
    setnames("N", "ipcCount") %>%
    .[, `:=` (tt_ipc = sum(ipcCount), tt_ipc_scope = uniqueN(.SD)),
                by = .(HAN_ID,  bvdid, name_internat,
                                            applicationYear)] %>%
    .[, hh_ratio := (ipcCount / tt_ipc) ** 2] %>%
    # option 1 
    # .[, HHI := round(sum(hh_ratio), 4),
    #             by = .(HAN_ID,  bvdid, name_internat, applicationYear)] %>%
    # head()
    # option 2
    .[, .(patHHI = round(sum(hh_ratio), 4),
                tt_ipc_scope = unique(tt_ipc_scope)),
                by = .(HAN_ID,  bvdid, name_internat,
                                        applicationYear)] -> pat_hhi_ttscope

pat_hhi_ttscope %>%
    .[, c(1, 4:6)] -> ttscope_temp

setkeyv(ttscope_temp, c("HAN_ID", "applicationYear"))
setkeyv(horbis_panel2, c("HAN_ID", "applicationYear"))


# merge again
horbis_panel3 <- merge(horbis_panel2, ttscope_temp,
                            by = c("HAN_ID", "applicationYear"), all.x = TRUE)


# patent citations 
epo_cit_counts <- fread("./data/202208_EPO_CIT_COUNTS.txt")
us_cit_counts <- fread("./data/202208_USPTO_CIT_COUNTS.txt")

epo_cit_counts %>%
    .[, c(1, 13:15, 30)] %>%
    head()


horbis_2010 %>%
    .[, .N, by = Publn_auth] 


# we use EPO patents first
# as we could access their citations 
horbis_df %>%
    .[order(rank(Publn_auth))] %>%
    unique(by = "familyID") %>%
    .[applicationYear >= 2010] %>%
    .[Publn_auth == "EP"] -> horbis_2010_ep


epo_cit_counts %>%
    .[, c(1, 13:15, 30)] %>%
    setnames(c("EP_Pub_nbr", "Total_Pat_Cits",
                "Total_NPL_Cits", "Total_Cits",
                "Total_cits_Recd"),
                c("patentNumber", "backward_pat_cits",
                        "backward_NPL_cits",
                        "backward_total_cits",
                        "forward_cits")
            ) -> epo_cit_sub

dim(epo_cit_sub)
head(epo_cit_sub)

setkey(horbis_2010_ep, "patentNumber")
setkey(epo_cit_sub, "patentNumber")

horbis_2010_ep_with_cits <- merge(horbis_2010_ep, epo_cit_sub,
                                                    all.x = TRUE)

horbis_2010_ep_with_cits %>%
    dim()  # 40808 x 50


# USA 
horbis_df %>%
    .[order(rank(Publn_auth))] %>%
    unique(by = "familyID") %>%
    .[applicationYear >= 2010] %>%
    .[Publn_auth == "US"] -> horbis_2010_us

dim(horbis_2010_us)


us_cit_counts %>%
    .[, c(1, 5:7, 10)] %>%
    setnames(
        c("US_Pub_nbr", "US_Pat_Cits",
                "US_NPL_Cits", "Total_Cits", "Total_cits_Recd"),
        c("patentNumber", "backward_pat_cits",
                        "backward_NPL_cits",
                        "backward_total_cits",
                        "forward_cits")
    ) -> us_cit_sub

setkey(horbis_2010_us, "patentNumber")
setkey(us_cit_sub, "patentNumber")


horbis_2010_us_with_cits <- merge(horbis_2010_us,
                                        us_cit_sub, all.x = TRUE)

horbis_2010_us_with_cits %>% dim()  # 22745, 50


# create wo
horbis_df %>%
    .[order(rank(Publn_auth))] %>%
    unique(by = "familyID") %>%
    .[applicationYear >= 2010] %>%
    .[Publn_auth == "WO"] -> horbis_2010_wo

dim(horbis_2010_wo)  # 15210, 46

wo_len <- dim(horbis_2010_wo)[1]

wo_temp <- data.table(
    backward_pat_cits = c(rep(0, wo_len)),
    backward_NPL_cits = c(rep(0, wo_len)),
    backward_total_cits = c(rep(0, wo_len)),
    forward_cits = c(rep(0, wo_len))
)

horbis_2010_wo_with_zeros <- cbind(horbis_2010_wo, wo_temp)

dim(horbis_2010_wo_with_zeros)  # 15210 50


horbis_2010_cits <- rbind(
    horbis_2010_us_with_cits,
    horbis_2010_ep_with_cits,
    horbis_2010_wo_with_zeros
)


dim(horbis_2010_cits)
names(horbis_2010_cits)

# replace NA with 0s
cit_names <- c("patentNumber", "backward_pat_cits",
                "backward_NPL_cits", "backward_total_cits",
                "forward_cits")
for (j in cit_names){
    set(horbis_2010_cits,
            which(is.na(horbis_2010_cits[[j]])),
            j, 0)
}

# calculate total citations for each year
horbis_2010_cits %>%
    .[, .(
        total_backward_pat = sum(backward_pat_cits),
        total_backward_npl = sum(backward_NPL_cits),
        total_backward = sum(backward_total_cits),
        total_forward = sum(forward_cits)
    ),
    by = .(HAN_ID, bvdid, name_internat, applicationYear)] %>%
    .[order(rank(applicationYear))] -> tt_citations


tt_citations %>%
    .[, c(1, 4:8)] -> tt_cits_temp

setkeyv(tt_cits_temp, c("HAN_ID", "applicationYear"))
setkeyv(horbis_panel3, c("HAN_ID", "applicationYear"))

horbis_panel4 <- merge(horbis_panel3, tt_cits_temp,
                            by = c("HAN_ID", "applicationYear"), all.x = TRUE)


# other patent quality
epo_pq <- fread("./data/202208_OECD_PATENT_QUALITY_EPO_INDIC.txt")

dim(epo_pq)
names(epo_pq)
head(epo_pq, 2)

epo_pq %>%
    .[sample(.N, 5)]


epo_pq %>%
    .[filing >= 2000] %>%
    .[app_nbr == "EP20190179482"] %>%
    head()


uspto_pq <- fread("./data/202208_OECD_PATENT_QUALITY_USPTO_INDIC.txt")

dim(uspto_pq)
names(uspto_pq)

head(uspto_pq)

horbis_2010_us %>%
    .[, c("patentNumber")] %>%
    head()

uspto_pq %>%
    .[filing >= 2000] %>%
    .[pub_nbr == "US10000918"]


# merge patent quality
epo_pq %>%
    .[filing >= 2000] %>%
    .[, c(2, 4, 7, 11, 17, 19:24)] %>%
    setnames(c("app_nbr"), c("application_epodoc")) -> epo_pq_2000

names(epo_pq_2000)
dim(epo_pq_2000)



names(horbis_2010_ep_with_cits)




setkey(horbis_2010_ep_with_cits, "application_epodoc")
setkey(epo_pq_2000, "application_epodoc")



horbis_2010_ep_with_cits_qi <- merge(horbis_2010_ep_with_cits,
                                        epo_pq_2000, all.x = TRUE)

dim(horbis_2010_ep_with_cits_qi)
names(horbis_2010_ep_with_cits_qi)

horbis_2010_ep_with_cits_qi %>%
    .[, c(45:60)] %>%
    head()


horbis_2010_ep_with_cits_qi %>%
    .[app_nbr == "EP20100000439"] %>%
    .[, c(45:50)]


uspto_pq %>%
    .[filing >= 2000] %>%
    .[, c(2, 4, 7, 11, 15:21)] %>% 
    setnames(c("pub_nbr"), c("patentNumber"))-> uspto_pq_2000

dim(uspto_pq_2000)
names(uspto_pq_2000)

head(uspto_pq_2000)


setkey(horbis_2010_us_with_cits, "patentNumber")
setkey(uspto_pq_2000, "patentNumber")


horbis_2010_us_with_cits_qi <- merge(horbis_2010_us_with_cits,
                                        uspto_pq_2000, all.x = TRUE)

dim(horbis_2010_us_with_cits_qi)
names(horbis_2010_us_with_cits_qi)


horbis_2010_us_with_cits_qi %>%
    head()


# merge with wo
wo_len <- dim(horbis_2010_wo_with_zeros)[1]
wo_temp2 <- data.table(
    tech_field = c(rep(0, wo_len)),
    family_size = c(rep(0, wo_len)),
    claims = c(rep(0, wo_len)),
    breakthrough = c(rep(0, wo_len)),
    generality = c(rep(0, wo_len)),
    originality = c(rep(0, wo_len)),
    radicalness = c(rep(0, wo_len)),
    renewal = c(rep(0, wo_len)),
    quality_index_4 = c(rep(0, wo_len)),
    quality_index_6 = c(rep(0, wo_len))
)

horbis_2010_wo_qi_with_zeros <- cbind(horbis_2010_wo_with_zeros,
                                                wo_temp2)

dim(horbis_2010_wo_qi_with_zeros)  # 15210 50


# save the final dataset
horbis_2010_patent_quality <- rbind(
    horbis_2010_ep_with_cits_qi,
    horbis_2010_us_with_cits_qi,
    horbis_2010_wo_qi_with_zeros
)


dim(horbis_2010_patent_quality)
names(horbis_2010_patent_quality)


horbis_2010_patent_quality %>%
    .[sample(.N, 5)]

# fwrite(horbis_2010_patent_quality, "./horbis_2010_final.csv")

# patent quality panel

names(horbis_2010_patent_quality)

# calculate average with na.rm = T 
horbis_2010_patent_quality %>%
    .[, .(
        avrg_tech_field = mean(tech_field, na.rm = TRUE),
        avrg_family_size = mean(family_size, na.rm = TRUE),
        avrg_claims = mean(claims, na.rm = TRUE),
        avrg_breakthrough = mean(breakthrough, na.rm = TRUE),
        avrg_generality = mean(generality, na.rm = TRUE),
        avrg_originality = mean(originality, na.rm = TRUE),
        avrg_radicalness = mean(radicalness, na.rm = TRUE),
        avrg_renewal = mean(renewal, na.rm = TRUE),
        avrg_quality_idx4 = mean(quality_index_4, na.rm = TRUE),
        avrg_quality_idx6 = mean(quality_index_6, na.rm = TRUE)
    ), by = .(HAN_ID, bvdid, name_internat, applicationYear)] -> pq_temp


pq_temp %>%
    .[, c(1, 4:14)] -> pq_temp2 


setkeyv(pq_temp2, c("HAN_ID", "applicationYear"))
setkeyv(horbis_panel4, c("HAN_ID", "applicationYear"))

horbis_panel5 <- merge(horbis_panel4, pq_temp2,
                            by = c("HAN_ID", "applicationYear"), all.x = TRUE)


fwrite(horbis_panel5, "./horbis_patent_panel_indicator.csv")


# add horizon 2020 info
horbis_df %>%
    .[, .(avrg_funding = mean(totalCost, na.rm = TRUE)),
                by = .(HAN_ID, bvdid, name_internat, startYear)] %>%
    setnames(c("startYear"), c("applicationYear"))-> funding 
    
funding %>%
    .[, c(1, 4, 5)] -> funding_temp 


setkeyv(funding_temp, c("HAN_ID", "applicationYear"))
setkeyv(horbis_panel5, c("HAN_ID", "applicationYear"))



horbis_panel6 <- merge(horbis_panel5, funding_temp,
                             by = c("HAN_ID", "applicationYear"), all.x = TRUE)


## Financial information Panel 
horbis_fins <- fread("./horbis_de_fins.csv")

dim(horbis_fins)
names(horbis_fins)

head(horbis_fins)

horbis_fins %>%
    unique(by = "bvdid") %>%
    dim()

sapply(horbis_fins, function(x) sum(is.na(x))/6295) %>%
    kable("pipe")


horbis_fins %>%
    .[closdate_year >= 2010] %>%
    .[, .N, by = closdate_year] 


horbis_fins %>%
    .[closdate_year >= 2010] -> horbis_fins_2010


dim(horbis_fins_2010)


sapply(horbis_fins_2010, function(x) sum(is.na(x))/6295) %>%
    kable("pipe")


horbis_fins_2010 %>%
    .[, c(1, 3:23)] -> horbis_fins_2010


horbis_nace <- fread("./horbis_de_nace.csv")

dim(horbis_nace)
names(horbis_nace)
head(horbis_nace)

# add nace code

horbis_nace %>%
    .[, c(1, 2)] -> nace_temp


setkey(nace_temp, "bvdid")
setkey(horbis_fins_2010, "bvdid")


horbis_fins_2010 <- merge(horbis_fins_2010, nace_temp, by = "bvdid", all.x = TRUE)


# drop duplicated
names(horbis_fins_2010)

horbis_fins_2010[!duplicated(horbis_fins_2010[c(1, 2)]), ] %>%
    .[, .N, by = closdate_year]

horbis_fins_2010 %>%
    unique(by = c("bvdid", "closdate_year")) -> horbis_panel7


head(horbis_panel7)

horbis_panel7$applicationYear <- as.integer(horbis_panel7$closdate_year)
horbis_panel7$closdate_year <- NULL 

head(horbis_panel7)

# final merge
setkeyv(horbis_panel6, c("bvdid", "applicationYear"))
setkeyv(horbis_panel7, c("bvdid", "applicationYear"))





horbis_panel_final <- merge(horbis_panel7, horbis_panel6,
                            by = c("bvdid", "applicationYear"), all.x = TRUE)



### final dataset
head(horbis_panel_final)
dim(horbis_panel_final)
names(horbis_panel_final)

fwrite(horbis_panel_final, "./horbis_panel_final.csv")

