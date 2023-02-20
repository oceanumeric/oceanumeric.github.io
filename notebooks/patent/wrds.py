# %% [markdown]
# # Orbis Query

# %%
import os
import wrds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_formats = ['svg']

# %%
db = wrds.Connection(wrds_username='michael_goethe')

# %% [markdown]
# ## Meaning of Variables
# 
# - bvdid: country iso code + number, such as BE0404057854, US990382107
# - addrtype: address type, such as `Trading address`, `Address abroad`
# 
# How to find German firms from the database? Query based on two conditions:
# 
# - bvdid starts with 'DE'
# 
# Note: 
# 
# - In SQL, we do `WHERE bvdid LIKE 'DE%'`
# - In Python, we need `WHERE bvdid LIKE 'DE%%'`

# %% [markdown]
# ## Germany Large Firms
# 
# - database: bvd_orbis_large
# - table: bvd.ob_w_company_id_table_l

# %%
query = """
SELECT bvdid, name_native, name_internat, prevname, akaname, slegalf, category_of_company,
addr_native, addr_internat, addr2_native, addr2_internat, city_native, city_internat,
postcode, country, ctryiso, 
listed, mainexch, website FROM bvd.ob_w_company_id_table_l
WHERE bvdid LIKE 'DE%%'
"""
delf = db.raw_sql(query)

# %%
delf.shape

# %%
delf.head()

# %% [markdown]
# let's check whether we look identify the company or not. For company 'BC Components Holding GmbH', 
# I did not find its website directly even though it is a very large company. However, the name
# was found on website of Vishay, called 'VISHAY BCCOMPONENTS'. 
# 
# Here is its short introduction.
# 
# BCcomponents (Beyschlag Centralab components), a leading manufacturer of passive electronic components, emerged from Philips Electronics Components division in January 1999. Building upon the tradition of excellence associated with Beyschlag, Philips, and Centralab, BCcomponents carried out, in close cooperation with customers, a continuous process of product innovation and improvement. This tradition of excellence included the development of several products that have become industry standards, such as SMD Mini-MELF resistors (branded Vishay Beyschlag) and a range of aluminum capacitors with industry-leading temperature capabilities. BCcomponents earned the status of preferred supplier to many of the world's leading electronics companies.
# 
# Vishay acquired BCcomponents in December 2002. The former BCcomponents product portfolio is now divided into Vishay Beyschlag and Vishay BCcomponents.
# 

# %% [markdown]
# __Another Example__: EKO Handelsunion GmbH
# 
# No company was found with searching for the name string. However, Google Map gives _EKO Stahl (ArcelorMittal Eisenhüttenstadt GmbH)_ when you
# type address. And Wikipedia gives:
# 
# EKO Stahl is a steelworks in Eisenhüttenstadt, Brandenburg, Germany. It was established by the East German government in the early 1950s on a greenfield site, initially producing only pig iron. The name was changed in 1961 __from Eisenhuttenkombinat 'J.W. Stalin' to Eisenhüttenkombinat Ost (EKO)__.
# 
# Cold rolling facilities were added in 1974, and basic oxygen steelmaking in 1984. After German reunification in 1990 the state-owned plant was privatised, and Belgian firm Cockerill-Sambre acquired it in stages from 1994 to 1998. Hot rolling facilities were added in the 1990s. Through mergers and takeovers, the owning comping since 2006 has been ArcelorMittal, and as of 2016 the plant is known as __ArcelorMittal Eisenhüttenstadt_.

# %% [markdown]
# __How do we solve this inconsistency issue__?
# 
# - we can use [Google Map API](https://github.com/googlemaps/google-maps-services-python) as it gives the right name for whoever registers the address
# - read this paper: [Matching patents to compustat firms, 1980–2015: Dynamic reassignment, name changes, and ownership structures](https://doi.org/10.1016/j.respol.2021.104217)

# %%
# check legal information from ob_legal_info_l
# for EKO DE3250055160
query = """
SELECT bvdid, akaname, prevname, historicaldate, category_of_company, ctryiso FROM bvd.ob_legal_info_l
WHERE bvdid = 'DE3250055160'
"""
db.raw_sql(query)

# %%
# check ob_additional_company_info_l
query = """
SELECT bvdid, akaname, prevname, akaname_internat, prevname_internat, category_of_company, 
ctryiso FROM bvd. ob_additional_company_info_l
WHERE bvdid = 'DE325005516'
"""
db.raw_sql(query)  # nothing returned 

# %%
delf.tail()

# %%
delf['listed'].value_counts()

# %% [markdown]
# ## German Medium Firms
# 
# - database: bvd_orbis_medium
# - table: ob_w_company_id_table_m

# %%
query = """
SELECT bvdid, name_native, name_internat, prevname, akaname, slegalf, category_of_company,
addr_native, addr_internat, addr2_native, addr2_internat, city_native, city_internat,
postcode, country, ctryiso, 
listed, mainexch, website FROM bvd.ob_w_company_id_table_m
WHERE bvdid LIKE 'DE%%'
"""
demf = db.raw_sql(query)

# %%
demf.shape

# %%
demf.head()

# %%
# together we have 506719 firms

delf.shape[0] + demf.shape[0]

# %% [markdown]
# ## Save the datasets

# %%
delf.to_csv('delf.csv', index=False)

# %%
demf.to_csv('demf.csv', index=False)

# %% [markdown]
# ## Retrieving Financial Data

# %%
query = """
SELECT bvdid, closdate, empl, df_employees, rd, rdop, opre, orig_currency, plbt, prma, toas FROM bvd.ob_ind_g_fins_eur_l
WHERE bvdid = 'DE4250103573'
"""
db.raw_sql(query)

# %%
query = """
SELECT bvdid, closdate, empl, df_employees, rd, rdop, opre, orig_currency, plbt, prma, toas FROM bvd.ob_ind_g_fins_eur_l
WHERE bvdid = 'DE6070415061'
"""
db.raw_sql(query)

# %%
query = """
SELECT bvdid, closdate, empl, df_employees, rd, rdop, opre, orig_currency, plbt, prma, toas FROM bvd.ob_ind_g_fins_eur_l
WHERE bvdid = 'DE6070415061'
"""
db.raw_sql(query)

# %%
query = """
SELECT bvdid, closdate, empl, df_employees, rd, rdop, opre, orig_currency, plbt, prma, toas FROM bvd.ob_ind_g_fins_eur_l
WHERE bvdid = 'DE2010000478'
"""
db.raw_sql(query)

# %% [markdown]
# ## Chinese Firms

# %%
query = """
SELECT bvdid, name_native, name_internat, prevname, akaname, slegalf, category_of_company,
addr_native, addr_internat, addr2_native, addr2_internat, city_native, city_internat,
postcode, country, ctryiso, 
listed, mainexch, website FROM bvd.ob_w_company_id_table_l
WHERE bvdid LIKE 'FR%%'
"""
frlf = db.raw_sql(query)

# %%
frlf.shape

# %%
frlf.sample(5)

# %%
frlf['listed'].value_counts()

# %%
query = """
SELECT bvdid, closdate, turn, empl, rd, toas, fias, tfas, 
ifas, grma, gros, loan, ltdb, opre, cf, cfop FROM bvd.ob_w_ind_g_fins_cfl_usd_l
WHERE bvdid = 'FR512053216'
"""
db.raw_sql(query)

# %%
query = """
SELECT bvdid, name_native, name_internat, prevname, akaname, slegalf, category_of_company,
addr_native, addr_internat, addr2_native, addr2_internat, city_native, city_internat,
postcode, country, ctryiso, 
listed, mainexch, website FROM bvd.ob_w_company_id_table_l
WHERE bvdid LIKE 'GB%%'
"""
gblf = db.raw_sql(query)

# %%
gblf.shape

# %%
gblf.sample(5)

# %%
query = """
SELECT bvdid, closdate, turn, empl, rd, toas, fias, tfas, 
ifas, grma, gros, loan, ltdb, opre, cf, cfop FROM bvd.ob_w_ind_g_fins_cfl_usd_l
WHERE bvdid = 'GB04067036'
"""
db.raw_sql(query)

# %% [markdown]
# ## Number of Firms for Different Countries

# %%
ctry = {
    'DE': 0, 'GB':0, 'FR':0, 'IE':0, 'NL':0, 'IT':0, 'ES':0, 'BE':0, 'PL':0, 'CH':0, 
}

# %%
for k, y in ctry.items():
    qtemp = f"""
    SELECT bvdid, country, ctryiso FROM bvd.ob_w_company_id_table_l
    WHERE bvdid LIKE '{k}%%'
    """
    dtemp = db.raw_sql(qtemp)
    size = dtemp.shape[0]
    ctry[k] = size

# %%
ctry

# %%
ctry_m = {
    'DE': 0, 'GB':0, 'FR':0, 'IE':0, 'NL':0, 'IT':0, 'ES':0, 'BE':0, 'PL':0, 'CH':0, 
}

# %%
for k, y in ctry_m.items():
    qtemp = f"""
    SELECT bvdid, country, ctryiso FROM bvd.ob_w_company_id_table_m
    WHERE bvdid LIKE '{k}%%'
    """
    dtemp = db.raw_sql(qtemp)
    size = dtemp.shape[0]
    ctry_m[k] = size

# %%
ctry_m

# %%
# bar plot
labels = list(ctry_m.keys())
num_l = list(ctry.values())
num_m = list(ctry_m.values())

# %%
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, num_l, width, label='Large Firms', color='#404352')
rects2 = ax.bar(x + width/2, num_m, width, label='Medium Firms', color='#B3B5BD')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of Firms')
ax.set_title('BvD Orbis Database')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3);

# %%



