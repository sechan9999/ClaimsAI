"""
Reference code tables: a small realistic slice of ICD-10 diagnosis codes and
CPT/HCPCS procedure codes with typical allowed-amount ranges.

This is illustrative reference data (not a full code set) used to generate
believable synthetic claims. In a real Snowflake deployment these would be
dimension tables (DIM_ICD10, DIM_CPT) loaded from CMS/NCHS code files.
"""

# (code, description, category)
ICD10_CODES = [
    ("E11.9", "Type 2 diabetes mellitus without complications", "endocrine"),
    ("I10", "Essential (primary) hypertension", "cardiovascular"),
    ("M54.5", "Low back pain", "musculoskeletal"),
    ("J45.909", "Unspecified asthma, uncomplicated", "respiratory"),
    ("F41.1", "Generalized anxiety disorder", "behavioral"),
    ("K21.9", "Gastro-esophageal reflux disease without esophagitis", "digestive"),
    ("N39.0", "Urinary tract infection, site not specified", "genitourinary"),
    ("Z00.00", "General adult medical examination without abnormal findings", "wellness"),
    ("I25.10", "Atherosclerotic heart disease of native coronary artery", "cardiovascular"),
    ("E78.5", "Hyperlipidemia, unspecified", "endocrine"),
    ("M17.11", "Unilateral primary osteoarthritis, right knee", "musculoskeletal"),
    ("C50.911", "Malignant neoplasm of unspecified site of right female breast", "oncology"),
    ("S06.0X0A", "Concussion without loss of consciousness, initial encounter", "injury"),
    ("O80", "Encounter for full-term uncomplicated delivery", "obstetric"),
    ("R07.9", "Chest pain, unspecified", "symptom"),
]

# (code, description, typical_low, typical_high, place_of_service_bias)
CPT_CODES = [
    ("99213", "Office/outpatient visit, established patient, low complexity", 75, 150, "office"),
    ("99214", "Office/outpatient visit, established patient, moderate complexity", 110, 220, "office"),
    ("99204", "Office/outpatient visit, new patient, moderate complexity", 160, 320, "office"),
    ("80053", "Comprehensive metabolic panel", 15, 60, "lab"),
    ("85025", "Complete blood count with differential", 10, 45, "lab"),
    ("71046", "Chest X-ray, 2 views", 35, 120, "imaging"),
    ("72148", "MRI lumbar spine without contrast", 400, 1600, "imaging"),
    ("93000", "Electrocardiogram, routine", 20, 75, "office"),
    ("29881", "Knee arthroscopy with meniscectomy", 900, 3200, "surgical"),
    ("47562", "Laparoscopic cholecystectomy", 2500, 9500, "surgical"),
    ("99283", "Emergency department visit, moderate severity", 250, 700, "emergency"),
    ("99285", "Emergency department visit, high severity", 600, 2200, "emergency"),
    ("J1745", "Infliximab injection, per 10mg", 150, 900, "infusion"),
    ("97110", "Therapeutic exercise, physical therapy, 15 min", 30, 90, "therapy"),
    ("59400", "Routine obstetric care, vaginal delivery", 1800, 4500, "obstetric"),
]

PAYERS = ["Medicare", "Medicaid", "Aetna", "UnitedHealthcare", "Cigna", "BCBS", "Humana"]

SPECIALTIES = [
    "Family Medicine", "Internal Medicine", "Orthopedic Surgery", "Cardiology",
    "Emergency Medicine", "Obstetrics & Gynecology", "Radiology", "Physical Therapy",
    "Oncology", "Psychiatry",
]

US_STATES = ["GA", "FL", "NC", "SC", "TN", "AL", "TX", "NY", "CA", "IL"]
