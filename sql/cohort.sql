WITH first_stays AS (
  SELECT
    ie.subject_id,
    ie.hadm_id,
    ie.stay_id,
    ie.intime,
    ie.outtime,
    TIMESTAMP_DIFF(ie.outtime, ie.intime, MINUTE) / 60.0 AS los_hours,
    p.anchor_age AS age,
    ROW_NUMBER() OVER (PARTITION BY ie.subject_id ORDER BY ie.intime) AS stay_rank
  FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
  JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
    ON ie.subject_id = p.subject_id
  WHERE TIMESTAMP_DIFF(ie.outtime, ie.intime, MINUTE) / 60.0 >= 24
),
prior_aki AS (
  SELECT DISTINCT subject_id
  FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
  WHERE icd_code LIKE 'N17%'
),
cohort AS (
  SELECT
    fs.subject_id,
    fs.hadm_id,
    fs.stay_id,
    fs.intime,
    fs.outtime,
    fs.los_hours,
    fs.age
  FROM first_stays fs
  WHERE fs.stay_rank = 1
    AND fs.age >= 18
    AND fs.los_hours >= 24
    AND fs.subject_id NOT IN (SELECT subject_id FROM prior_aki)
)
SELECT * FROM cohort;