# R code for the lab 1 
library(tidyverse)
library(data.table)
# set working directory
print(setwd("docs/math/causal/data/lab1/"))

# read the dataset and combine
year <- c(2003:2013)

# initialize the data frame
sqf_data <- NULL

# measure time
start_time <- Sys.time()

for (y in year) {
    print(y)
    path <- paste("sqf-", as.character(y), ".csv", sep="")
    df <- fread(path)
    colnames(df) <- tolower(colnames(df))

    if (y == 2006) {
        # change column names
        df %>%
        rename("stname" = "strname", "stinter" = "strintr",
        "rescode" = "rescod", "premtype" = "premtyp",
        "premname" = "prenam", "dettypcm" = "dettyp_c",
        "addrnum" = "adrnum", "addrpct" = "adrpct",
        "detailcm" = "details_") -> df
        #drop the detail1_ column in the 2006 data
        df$detail1_ <- NULL
    }

    if (y <= 2010) {
        if (y == 2006) {
            df %>% mutate(forceuse = NA, linecm = NA) -> df
        } else {
            df %>% mutate(forceuse = NA, wepfound = NA) -> df
        }
    } else {
        df %>% mutate(wepfound = NA) -> df
    }
    # remove columns
    df %>% select(-ser_num, -datestop, -recstat, -trhsloc, -perobs,
    -crimsusp, -perstop, -explnstp, -arstoffn, -sumoffen, -compyear,
    -comppct, -officrid, -adtlrept, -pf_other, -radio,-ac_rept,
    -ac_inves, -rf_vcrim, -rf_othsw, -ac_proxm, -rf_attir, -rf_vcact,
    -ac_evasv, -ac_assoc, -rf_rfcmp, -ac_cgdir, -rf_verbl, -rf_knowl,
    -ac_stsnd, -ac_other, -sb_hdobj, -sb_outln, -sb_admis, -sb_other,
    -repcmd, -revcmd, -rf_furt, -rf_bulg, -offverb, -offshld,
    -wepfound, -dettypcm, -detailcm, -dob, -ht_feet, -ht_inch,
    -weight, -haircolr, -eyecolor, -build, -othfeatr, -addrtyp,
    -rescode, -premtype, -premname, -addrnum, -stname, -stinter,
    -aptnum, -city, -state, -zip, -sector, -beat, -post, -xcoord,
    -ycoord, -crossst, -forceuse, -linecm ) -> df

    sqf_data <- rbind(sqf_data, df)
    print(paste(path, "Combined", sep=": "))
}
end_time <- Sys.time()  # around 50 seconds, it takes 300 seconds in Python

print(end_time-start_time)  # around 17 seconds with 32 CPUs

# save the dataset
save(sqf_data, file="sqf_03_13.RData")


# -------- create the dataset for regression ----------- # 

# at this stage, we will not differentiate the kind of force used

load("sqf_03_13.RData")

sqf_data %>%
    drop_na() %>% 
    mutate(any_force_used = paste(pf_hands, pf_grnd, pf_wall,
                                    pf_drwep, pf_ptwep,pf_hcuff,
                                    pf_baton, pf_pepsp, sep = ""),
                  wepnfnd = paste(contrabn, asltweap, pistol,
                                 riflshot, knifcuti,machgun,
                                 othrweap, sep = "")) %>%
    select(any_force_used,race,sex,age,inout,ac_incid,
            ac_time,timestop, offunif,typeofid,othpers,cs_bulge,
            cs_cloth,cs_casng,cs_lkout, cs_descr,cs_drgtr,
            cs_furtv,cs_vcrim, cs_objcs,cs_other,wepnfnd,pct,year)%>%
    filter(race != "X", race != " ", race != "I", race != "U",
             typeofid != " ") %>%
    mutate(any_force_used = as.factor(if_else(grepl("Y",any_force_used),
                                                    1, 0)),
         sex = as.factor(sex),
         inout = as.factor(if_else(grepl("I",inout), 1, 0)),
         timestop = as.numeric(gsub(":","",timestop)),
         daytime = as.factor(ifelse(600<=timestop & timestop<=1700,1,0)),
         ac_incid = as.factor(if_else(grepl("Y",ac_incid), 1, 0)),
         ac_time = as.factor(if_else(grepl("Y",ac_time), 1, 0)),
         offunif = as.factor(if_else(grepl("Y",offunif), 1, 0)),
         othpers = as.factor(if_else(grepl("Y",othpers), 1, 0)),
         cs_objcs = as.factor(if_else(grepl("Y",cs_objcs), 1, 0)),
         cs_descr = as.factor(if_else(grepl("Y",cs_descr), 1, 0)),
         cs_casng = as.factor(if_else(grepl("Y",cs_casng), 1, 0)),
         cs_lkout = as.factor(if_else(grepl("Y",cs_lkout), 1, 0)),
         cs_cloth = as.factor(if_else(grepl("Y",cs_cloth), 1, 0)),
         cs_drgtr = as.factor(if_else(grepl("Y",cs_drgtr), 1, 0)),
         cs_furtv = as.factor(if_else(grepl("Y",cs_furtv), 1, 0)),
         cs_vcrim = as.factor(if_else(grepl("Y",cs_vcrim), 1, 0)),
         cs_bulge = as.factor(if_else(grepl("Y",cs_bulge), 1, 0)),
         cs_other = as.factor(if_else(grepl("Y",cs_other), 1, 0)),
         wepnfnd = as.factor(if_else(grepl("Y",wepnfnd), 1, 0)),
         race = recode_factor(race, "W" = "White", "B" = "Black", 
                              "Q" ="Hispanic", "A" = "Asian",
                              "Z" = "Other"),
         year = as.factor(year),
         pct = as.factor(pct)) -> sqf_reg_data


save(sqf_reg_data, file="sqf_reg_data.RData")