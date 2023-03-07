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
            by = .(HAN_ID, bvdid, name_internat, applicationYear)] %>%
    head()

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
            by = .(HAN_ID, bvdid, name_internat, applicationYear)] %>%
    head()



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
    .[, .(HHI = round(sum(hh_ratio), 4),
                tt_ipc_scope = unique(tt_ipc_scope)),
                by = .(HAN_ID,  bvdid, name_internat,
                                        applicationYear)] %>%
    head()


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

dim(horbis_2010_wo)  # 15210 50


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
    .[order(rank(applicationYear))] %>%
    head()


# other patent quality



