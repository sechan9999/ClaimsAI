"""
Generate synthetic patient claims data for the ClaimsAI prototype.

Produces four tables (written as Parquet under data/generated/):
  - patients.parquet
  - providers.parquet
  - claims.parquet            <- the "observed" fact table (what the app/user sees)
  - claims_fraud_labels.parquet  <- ground-truth labels, held out from the app,
                                     used only to score the anomaly detector.

Design notes:
  - Schema and code lists are illustrative synthetic data, not real PHI.
  - Five fraud patterns are injected at known rates so the detection module
    (fraud/detection.py) can be validated against a ground truth.
  - Column names are written in upper snake case to mirror how this would
    land in Snowflake (SCREAMING_SNAKE_CASE identifiers).
"""
from __future__ import annotations

import os
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

from reference_codes import ICD10_CODES, CPT_CODES, PAYERS, SPECIALTIES, US_STATES

SEED = 42
N_PATIENTS = 4000
N_PROVIDERS = 120
N_CLAIMS = 40000
FRAUD_RATE = 0.035  # ~3.5% of claims carry an injected fraud pattern

OUT_DIR = os.path.join(os.path.dirname(__file__), "generated")


def _rng():
    return np.random.default_rng(SEED)


def generate_patients(fake: Faker, rng) -> pd.DataFrame:
    rows = []
    for i in range(N_PATIENTS):
        dob = fake.date_of_birth(minimum_age=0, maximum_age=95)
        rows.append(
            {
                "PATIENT_ID": f"PT{i:06d}",
                "AGE": max(0, (datetime.now().date() - dob).days // 365),
                "GENDER": rng.choice(["F", "M"], p=[0.51, 0.49]),
                "STATE": rng.choice(US_STATES),
                "ENROLLMENT_START": fake.date_between(start_date="-5y", end_date="-1y"),
            }
        )
    return pd.DataFrame(rows)


def generate_providers(fake: Faker, rng) -> pd.DataFrame:
    rows = []
    for i in range(N_PROVIDERS):
        rows.append(
            {
                "PROVIDER_ID": f"PR{i:05d}",
                "PROVIDER_NAME": fake.company() + " " + rng.choice(["Clinic", "Medical Group", "Health Partners"]),
                "NPI": fake.numerify("##########"),
                "SPECIALTY": rng.choice(SPECIALTIES),
                "STATE": rng.choice(US_STATES),
            }
        )
    return pd.DataFrame(rows)


def _base_claim_amount(rng, cpt_row) -> float:
    _, _, lo, hi, _ = cpt_row
    return float(rng.uniform(lo, hi))


def generate_claims(fake: Faker, rng, patients: pd.DataFrame, providers: pd.DataFrame):
    patient_ids = patients["PATIENT_ID"].to_numpy()
    provider_ids = providers["PROVIDER_ID"].to_numpy()
    n_providers = len(provider_ids)

    # A small subset of providers will be the source of most injected fraud,
    # which is realistic: FWA tends to cluster at specific billing entities.
    n_fraud_providers = max(3, int(n_providers * 0.05))
    fraud_provider_ids = rng.choice(provider_ids, size=n_fraud_providers, replace=False)

    claims = []
    labels = []

    start_date = datetime(2024, 1, 1)
    n_fraud_target = int(N_CLAIMS * FRAUD_RATE)
    fraud_claim_idx = set(
        rng.choice(N_CLAIMS, size=n_fraud_target, replace=False).tolist()
    )

    for i in range(N_CLAIMS):
        is_fraud_slot = i in fraud_claim_idx
        icd = ICD10_CODES[rng.integers(0, len(ICD10_CODES))]
        cpt = CPT_CODES[rng.integers(0, len(CPT_CODES))]

        if is_fraud_slot:
            provider_id = rng.choice(fraud_provider_ids)
        else:
            provider_id = rng.choice(provider_ids)

        patient_id = rng.choice(patient_ids)
        service_date = start_date + timedelta(days=int(rng.integers(0, 540)))
        submission_lag = int(rng.integers(1, 30))
        billed = _base_claim_amount(rng, cpt)

        fraud_type = None
        if is_fraud_slot:
            pattern = rng.choice(
                ["upcoding", "phantom_duplicate", "unbundling", "excess_frequency", "outlier_amount"]
            )
            if pattern == "upcoding":
                billed *= rng.uniform(3.0, 6.0)
            elif pattern == "outlier_amount":
                billed *= rng.uniform(4.0, 9.0)
            elif pattern == "unbundling":
                billed *= rng.uniform(1.8, 2.6)
            elif pattern == "phantom_duplicate":
                billed *= rng.uniform(0.9, 1.1)  # amount looks normal; duplication is the signal
            elif pattern == "excess_frequency":
                billed *= rng.uniform(0.9, 1.2)  # amount looks normal; frequency is the signal
            fraud_type = pattern

        claim_id = f"CL{i:07d}"
        claims.append(
            {
                "CLAIM_ID": claim_id,
                "PATIENT_ID": patient_id,
                "PROVIDER_ID": provider_id,
                "PAYER": rng.choice(PAYERS),
                "DIAGNOSIS_CODE": icd[0],
                "DIAGNOSIS_DESC": icd[1],
                "PROCEDURE_CODE": cpt[0],
                "PROCEDURE_DESC": cpt[1],
                "PLACE_OF_SERVICE": cpt[4],
                "SERVICE_DATE": service_date.date(),
                "SUBMISSION_DATE": (service_date + timedelta(days=submission_lag)).date(),
                "BILLED_AMOUNT": round(billed, 2),
                "PAID_AMOUNT": round(billed * rng.uniform(0.55, 0.95), 2),
                "CLAIM_STATUS": rng.choice(["Paid", "Paid", "Paid", "Denied", "Pending"]),
            }
        )
        labels.append(
            {"CLAIM_ID": claim_id, "IS_FRAUD": is_fraud_slot, "FRAUD_TYPE": fraud_type}
        )

    claims_df = pd.DataFrame(claims)
    label_by_id = {l["CLAIM_ID"]: l for l in labels}

    # Inject literal duplicate/excess-frequency rows for the relevant fraud types,
    # since those patterns are about repetition, not amount.
    dup_rows = []
    dup_labels = []
    for _, row in claims_df.iterrows():
        lbl = label_by_id[row["CLAIM_ID"]]
        if lbl["FRAUD_TYPE"] == "phantom_duplicate":
            for k in range(rng.integers(1, 3)):
                new_id = f"{row['CLAIM_ID']}_D{k}"
                dup = row.copy()
                dup["CLAIM_ID"] = new_id
                dup_rows.append(dup)
                dup_labels.append({"CLAIM_ID": new_id, "IS_FRAUD": True, "FRAUD_TYPE": "phantom_duplicate"})
        elif lbl["FRAUD_TYPE"] == "excess_frequency":
            for k in range(rng.integers(3, 8)):
                new_id = f"{row['CLAIM_ID']}_F{k}"
                dup = row.copy()
                dup["CLAIM_ID"] = new_id
                dup["SERVICE_DATE"] = row["SERVICE_DATE"] + timedelta(days=int(k))
                dup["SUBMISSION_DATE"] = dup["SERVICE_DATE"] + timedelta(days=int(rng.integers(1, 5)))
                dup_rows.append(dup)
                dup_labels.append({"CLAIM_ID": new_id, "IS_FRAUD": True, "FRAUD_TYPE": "excess_frequency"})

    if dup_rows:
        claims_df = pd.concat([claims_df, pd.DataFrame(dup_rows)], ignore_index=True)
        labels.extend(dup_labels)

    labels_df = pd.DataFrame(labels)
    return claims_df, labels_df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fake = Faker()
    Faker.seed(SEED)
    rng = _rng()

    patients = generate_patients(fake, rng)
    providers = generate_providers(fake, rng)
    claims, labels = generate_claims(fake, rng, patients, providers)

    patients.to_parquet(os.path.join(OUT_DIR, "patients.parquet"), index=False)
    providers.to_parquet(os.path.join(OUT_DIR, "providers.parquet"), index=False)
    claims.to_parquet(os.path.join(OUT_DIR, "claims.parquet"), index=False)
    labels.to_parquet(os.path.join(OUT_DIR, "claims_fraud_labels.parquet"), index=False)

    print(f"patients:  {len(patients):>7,}")
    print(f"providers: {len(providers):>7,}")
    print(f"claims:    {len(claims):>7,}  ({labels['IS_FRAUD'].sum():,} flagged fraud, "
          f"{labels['IS_FRAUD'].mean()*100:.2f}%)")
    print(f"written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
