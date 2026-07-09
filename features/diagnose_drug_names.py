import pandas as pd
from google.cloud import bigquery
from config import GCP_PROJECT, PHYSIONET_PROJECT, DATA_DIR

client = bigquery.Client(project=GCP_PROJECT)
cohort = pd.read_parquet(f"{DATA_DIR}cohort.parquet").head(50)
stay_ids = cohort.stay_id.tolist()
hadm_ids = cohort.hadm_id.tolist()

# Check inputevents drug name variety
query = f"""
SELECT DISTINCT LOWER(d.label) AS drug_name
FROM `{PHYSIONET_PROJECT}.mimiciv_3_1_icu.inputevents` ie
JOIN `{PHYSIONET_PROJECT}.mimiciv_3_1_icu.d_items` d ON ie.itemid = d.itemid
WHERE ie.stay_id IN UNNEST(@stay_ids)
"""
job_config = bigquery.QueryJobConfig(
    query_parameters=[bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids)]
)
iv_names = client.query(query, job_config=job_config).to_dataframe()
print("Sample IV drug names (first 50 patients):")
print(iv_names.head(30))

# Check prescriptions drug name variety
query2 = f"""
SELECT DISTINCT LOWER(p.drug) AS drug_name
FROM `{PHYSIONET_PROJECT}.mimiciv_3_1_hosp.prescriptions` p
WHERE p.hadm_id IN UNNEST(@hadm_ids)
"""
job_config2 = bigquery.QueryJobConfig(
    query_parameters=[bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids)]
)
rx_names = client.query(query2, job_config=job_config2).to_dataframe()
print("\nSample Rx drug names (first 50 patients):")
print(rx_names.head(30))

# Specifically check if 'vancomycin' appears anywhere in either list
print("\nAny 'vanco' matches in IV names?", iv_names.drug_name.str.contains("vanco", na=False).sum())
print("Any 'vanco' matches in Rx names?", rx_names.drug_name.str.contains("vanco", na=False).sum())
