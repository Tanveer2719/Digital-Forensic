import uuid
import hashlib
import random
from datetime import datetime, timedelta, timezone
import pandas as pd
from enum import Enum
import string
from itertools import cycle


random.seed(42)


UTC_PLUS_6 = timezone(timedelta(hours=6))
HASH_ALGOS = ["md5", "sha1", "sha256"]
TOOLS = ["FTK", "EnCase", "Autopsy"]
LOCATIONS = [
    "Scene",
    "Police_Station",
    "DFL_Evidence_Locker",
    "DFL_Lab",
    "Court_Record_Room"
]


# length of events for each type of casesjust 
LENGTH_BUCKETS = {
    "short":  {"cases": int(0.40 * 12500), "min": 8,  "max": 15},
    "medium": {"cases": int(0.45 * 12500), "min": 16, "max": 35},
    "long":   {"cases": int(0.15 * 12500), "min": 36, "max": 120},
}

ANOMALY_LENGTH_BUCKETS = {
    "short":  int(0.40 * 7500),  # 3000
    "medium": int(0.45 * 7500),  # 3375
    "long":   int(0.15 * 7500),  # 1125
}


# number of cases
CLASS_QUOTA = {
    "normal": 12500,
    "benign_anomaly": 7500
}

# sequence according to the forensic guidelines
DFG_SEQUENCE = [
    "acquisition",
    "hashing",
    "sealing",
    "transfer",
    "storage",
    "analysis"
]

ACQUISITION_LOGIC = {
    "mobile": {"method": "physical_extraction", "blocker": False, "isolation": "Faraday_Bag"},
    "laptop": {"method": "bitwise_image", "blocker": True, "isolation": "None"},
    "USB":    {"method": "bitwise_image", "blocker": True, "isolation": "None"},
    "DVR":    {"method": "logical_export", "blocker": True, "isolation": "None"}
}

TRANSIT_TIME_SEC = {
    ("Scene", "Police_Station"): (3600, 7200),        # 1–2 hrs
    ("Police_Station", "DFL_Evidence_Locker"): (1800, 3600), # 30–60 min
    ("DFL_Evidence_Locker", "DFL_Lab"): (900, 1800),          # 15–30 min
    ("DFL_Lab", "Court_Record_Room"): (7200, 14400),          # 2–4 hrs
}
SAME_LOCATION_DELTA = (60, 300)  # 1–5 min



################ Helper Functions ############

def compute_hash(value, algo):
    data = value.encode()
    if algo == "md5":
        return hashlib.md5(data).hexdigest()
    if algo == "sha1":
        return hashlib.sha1(data).hexdigest()
    return hashlib.sha256(data).hexdigest()

def next_timestamp(ts, from_loc, to_loc):
    if from_loc == to_loc:
        delta_range = SAME_LOCATION_DELTA
    else:
        delta_range = TRANSIT_TIME_SEC.get((from_loc, to_loc), (3600, 7200))
    delta_seconds = random.randint(*delta_range)
    return ts + timedelta(seconds=delta_seconds), delta_seconds

def shift_time(ts: str, seconds: int):
    dt = datetime.fromisoformat(ts)
    return (dt + timedelta(seconds=seconds)).isoformat()

def realistic_wrong_hash():
    return "e3b0c44298fc1c149afbf4c8996fb924"  # empty file hash

def realistic_wrong_timestamp(valid_ts: str):
    return valid_ts.replace("T", " ")  # common logging mistake
    

############## Validator #################

def validate_hash_chain(events):
    return all(
        events[i]["hash_pre_source"] == events[i-1]["hash_post_source"]
      
        for i in range(1, len(events))
    )

def validate_per_evidence_hash_chain(events):
    last_hashes = {}
    for e in events:
        eid = e["evidence_id"]
        if eid in last_hashes:
            if e["hash_pre_source"] != last_hashes[eid]:
                return False
        last_hashes[eid] = e["hash_post_source"]
    return True
    
def validate_seals(events):
    return all(e["seal_status"] == "intact" for e in events)

def validate_time(events):
    return all(abs(e["clock_drift_sec"]) <= 1.0 for e in events)

def validate_custody(events):
    return all(e["custodian_id"] and e["handover_document_id"] for e in events)

def validate_65B(events):
    required = [
        "custodian_signature_hash",
        "system_clock_source",
        "tool_name",
        "tool_version",
        "hash_post_source"
    ]
    return all(all(e.get(f) is not None for f in required) for e in events)
    
def validate_case_sequence(events):
    """
    For each evidence:
    - Ensure all canonical DFG events appear
    - Their relative order is preserved
    - Ignore extra handovers, audit logs, or analysis repeats
    """
    from itertools import compress

    for eid in set(e["evidence_id"] for e in events):
        seq = [e["event_type"] for e in events if e["evidence_id"] == eid]
        # Keep only canonical events
        canonical_seq_in_case = [e for e in seq if e in DFG_SEQUENCE]
        # Check relative order
        last_idx = -1
        for ev in DFG_SEQUENCE:
            try:
                idx = canonical_seq_in_case.index(ev)
            except ValueError:
                return False  # missing canonical event
            if idx <= last_idx:
                return False  # out-of-order
            last_idx = idx
    return True



def validate_handover_continuity(events):
    """
    Ensure that every custodian change has a corresponding handover record
    linking the previous -> current custodian, for each evidence independently.
    Handles multiple transfers per evidence.
    """
    last_custodian_per_evidence = {}

    for e in events:
        eid = e["evidence_id"]
        current_custodian = e.get("custodian_id")
        handover = e.get("handover_record")

        if eid in last_custodian_per_evidence:
            prev_custodian = last_custodian_per_evidence[eid]

            # If custodian changed
            if prev_custodian != current_custodian:
                if not handover:
                    # Missing handover for custodian change
                    return False
                # Check handover links previous -> current
                if handover.get("from_officer") != prev_custodian:
                    return False
                if handover.get("to_officer") != current_custodian:
                    return False
        # Update last custodian for this evidence
        last_custodian_per_evidence[eid] = current_custodian

    return True


# ************** Generator Codes ################

_id_counter = 0

def gen_id(prefix, length=8):
    global _id_counter
    _id_counter += 1
    return f"{prefix}-{_id_counter:08d}"




def generate_warrant(case_id: str):
    return {
        "warrant_id": gen_id("WRNT"),
        "case_reference": case_id,
        "warrant_type": random.choice(["search_and_seizure", "arrest_and_seizure"]),
        "issuing_authority": random.choice(["Metropolitan Magistrate", "District Judge"]),
        "issue_date": datetime.now(UTC_PLUS_6).isoformat(),
        "valid_until": (datetime.now(UTC_PLUS_6) + timedelta(days=7)).isoformat(),
        "jurisdiction": "Dhaka Metropolitan Area"
    }


# Custodian Identity
def generate_custodian():
    ranks = ["Inspector", "Sub-Inspector", "Assistant Officer"]
    designations = ["Digital Forensic Examiner", "DFL Analyst", "Evidence Custodian"]
    organizations = ["CID DFL", "Police Cyber Unit", "District Forensic Lab"]
    return {
        "officer_id": gen_id("CUST"),
        "name": "Redacted",
        "rank": random.choice(ranks),
        "designation": random.choice(designations),
        "organization": random.choice(organizations),
        "authority_basis": "SOP 2023 / Standing Order"
    }


# Handover Record
def generate_handover(from_custodian, to_custodian, location):
    ts = datetime.now(UTC_PLUS_6).isoformat()
    return {
        "handover_id": gen_id("HND"),
        "from_officer": from_custodian["officer_id"],
        "to_officer": to_custodian["officer_id"],
        "reason": "Transfer for analysis",
        "date_time": ts,
        "location": location,
        "authorization_ref": gen_id("WRNT"),
        "signatures": {
            "from": gen_id("SIGN"),
            "to": gen_id("SIGN")
        }
    }


#  Section 65B Certificate
def generate_65B_certificate(device_id, evidence_id):
    ts = datetime.now(UTC_PLUS_6).isoformat()
    return {
        "certificate_id": gen_id("65B"),
        "issuer_name": "Redacted",
        "issuer_designation": "Digital Forensic Examiner",
        "issuing_lab": "CID DFL",
        "device_id": device_id,
        "evidence_id": evidence_id,
        "device_regular_use": True,
        "device_proper_functioning": True,
        "data_production_method": "Bitwise forensic imaging using FTK",
        "hash_of_output": gen_id("HASH"),  # placeholder for real hash
        "date_issued": ts,
        "signature_hash": gen_id("SIGN")
    }



def generate_device_metadata(device_category):
    """Simulate device-specific info"""
    return {
        "serial_number": ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)),
        "imei": ''.join(random.choices(string.digits, k=15)) if device_category == "mobile" else None,
        "os_version": random.choice(["Android 12", "Windows 10", "Ubuntu 22.04", "macOS 13"]),
        "firmware_version": f"{random.randint(1,5)}.{random.randint(0,9)}"
    }

def generate_environment(event_type):
    """Simulate temp and humidity depending on location type"""
    if event_type in ["acquisition", "transfer"]:
        temp = random.uniform(25, 35)  # scene/transport
        humidity = random.randint(30, 60)
    else:
        temp = random.uniform(18, 25)  # lab/storage
        humidity = random.randint(30, 50)
    return temp, humidity

def generate_audit_log(event_index, custodian_id, event_type):
    """Simulate audit trail entries"""
    actions = {
        "acquisition": "image_captured",
        "hashing": "hash_verified",
        "sealing": "seal_applied",
        "transfer": "custody_transferred",
        "storage": "stored",
        "analysis": "analysis_performed"
    }
    return [{
        "action": actions.get(event_type, "unknown"),
        "actor": custodian_id,
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }]

def generate_event(case_id, device_id, evidence_id, device_cat, index, prev_hash, event_type, ts,from_location, to_location, hash_algo):
    algo = hash_algo
    
    ts, delta = next_timestamp(ts, from_location, to_location)

    source_material = prev_hash if prev_hash else gen_id("RAW")
    post_hash = compute_hash(source_material, algo)

    # Logic for Write-Blocker based on Device Category
    acq_settings = ACQUISITION_LOGIC.get(device_cat)

    event = {
        "event_uuid": gen_id("EVT"),
        "case_id": case_id,
        "event_index": index,
        "evidence_id": evidence_id,  # Distinguishes Image vs. RAM vs. Disk
        "device_id": device_id,
        "device_category": device_cat,
        "event_type": event_type,
        "delta_sec": delta,

        

        # Custody & authorization
        "custodian_id": gen_id("CUST"),
        "custodian_role": (
            "Investigator" if event_type == "acquisition"
            else "DFL_Analyst"
        ),
        "custodian_signature_hash": compute_hash(gen_id("SIGN"), "sha256"),
        "authorization_id": gen_id("AUTH"),
        "handover_document_id": gen_id("DOC"),

        "location": LOCATIONS[index % len(LOCATIONS)],
        "seal_id": gen_id("SEAL"),
        "seal_status": "intact",

        # Condition-based forensic acquisition
        "acquisition_method": acq_settings["method"],
        "write_blocker_used": acq_settings["blocker"],
        "isolation_method": acq_settings["isolation"], # For Mobile/Wireless

        # Time & system integrity
        "timestamp_iso": ts.isoformat(),
        "timestamp_timezone": "UTC+06:00",
        "system_clock_source": "NTP",
        "clock_drift_sec": round(random.uniform(-0.5, 0.5), 2),

        # Tool & reproducibility
        "tool_name": random.choice(TOOLS),
        "tool_version": "1.0",
        "tool_validation_id": gen_id("VAL"),
        "tool_validation_status": "validated",


        # Hash & integrity
        "hash_pre_source": prev_hash,
        "hash_post_source": post_hash,
        "hash_pre_derivative": None,
        "hash_post_derivative": None,
        "hash_algo": algo,
        "hash_verification_status": "verified",

        # Documentation
        "action_justification": f"Standard DFG-compliant {event_type} procedure",
        "notes": "No deviation observed",
        "system_status": "operational",
        "integrity_tainted": False

    }
    event["env_temperature_C"], event["env_humidity_percent"] = generate_environment(event_type)
    event["seal_serial_number"] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    event["interagency_transfer"] = random.choice([False, False, True])  # mostly False
    event["return_disposal_status"] = random.choice(["stored", "returned", "disposed"])
    event["device_metadata"] = generate_device_metadata(device_cat)
    event["audit_log_entries"] = generate_audit_log(index, event["custodian_id"], event_type)

    return event, post_hash, ts


def get_event_type(i, device_category):
    if i <= len(DFG_SEQUENCE):
        return DFG_SEQUENCE[i-1]
    else:
        # Extra events: create subtypes based on device
        extra_events = ["analysis"]*3 + ["hash_verification", "transfer"]
        return random.choice(extra_events)

def build_normal_case(n_events, parallel_prob=0.7):
    """
    Generate a normal forensic case with threaded evidence collection.
    Each evidence chain runs independently (like a thread).
    parallel_prob -> probability that evidences overlap (parallel).
    """
    case_id = gen_id("CASE")
    
    # Generate devices and evidence
    num_devices = random.randint(1, 3)
    devices = []
    for _ in range(num_devices):
        devices.append({
            "device_id": gen_id("DEV"),
            "device_category": random.choice(list(ACQUISITION_LOGIC.keys())),
            "evidence_ids": [gen_id("EVID_DISK"), gen_id("EVID_RAM")]
        })
    
    ts_base = datetime.now(UTC_PLUS_6)
    events = []
    warrant = generate_warrant(case_id)
    custodian_main = generate_custodian()
    custodian_analysis = generate_custodian()
    
    prev_hashes = {}
    last_custodian_per_evidence = {}
    prev_location_per_chain = {}

    all_targets = [(d, eid) for d in devices for eid in d["evidence_ids"]]
    evidence_hash_algo = {eid: random.choice(HASH_ALGOS) for _, eid in all_targets}

    # Decide if evidences will run in parallel
    parallel = random.random() < parallel_prob

    # Initialize a "thread" for each evidence with its own timestamp
    evidence_threads = {}
    for _, eid in all_targets:
        if parallel:
            # Random small offset to simulate parallel threads
            evidence_threads[eid] = ts_base + timedelta(seconds=random.randint(0, 120))
        else:
            # Sequential start
            evidence_threads[eid] = ts_base

    global_event_index = 1
    unfinished = set(eid for _, eid in all_targets)  # track unfinished threads

    while unfinished:
        for device, eid in all_targets:
            if eid not in unfinished:
                continue

            ts = evidence_threads[eid]
            chain_key = f"{device['device_id']}_{eid}"
            done_events = [e for e in events if e["evidence_id"] == eid]
            stage_index = len(done_events)

            # Generate next event in canonical DFG sequence
            if stage_index < len(DFG_SEQUENCE):
                event_type = DFG_SEQUENCE[stage_index]

                from_location = done_events[-1]["location"] if done_events else "Scene"
                to_location = {
                    "acquisition": "Police_Station",
                    "hashing": "DFL_Lab",
                    "sealing": from_location,
                    "transfer": "DFL_Evidence_Locker",
                    "storage": "DFL_Evidence_Locker",
                    "analysis": "DFL_Lab"
                }.get(event_type, from_location)

                custodian_info = custodian_main if event_type == "acquisition" else custodian_analysis

                ev, post_hash, ts_new = generate_event(
                    case_id, device["device_id"], eid, device["device_category"],
                    index=global_event_index,
                    prev_hash=prev_hashes.get(chain_key),
                    event_type=event_type,
                    ts=ts,
                    from_location=from_location,
                    to_location=to_location,
                    hash_algo=evidence_hash_algo[eid]
                )

                # Custodian & legal info
                ev["custodian_info"] = custodian_info
                ev["custodian_id"] = custodian_info["officer_id"]
                ev["custodian_role"] = custodian_info["designation"]
                ev["custodian_signature_hash"] = compute_hash(ev["custodian_id"], "sha256")
                ev["warrant_id"] = warrant["warrant_id"]

                if chain_key in last_custodian_per_evidence:
                    prev_cust = last_custodian_per_evidence[chain_key]
                    if prev_cust["officer_id"] != custodian_info["officer_id"]:
                        ev["handover_record"] = generate_handover(prev_cust, custodian_info, ev["location"])
                last_custodian_per_evidence[chain_key] = custodian_info

                if event_type in ["acquisition", "hashing"]:
                    ev["section_65B_certificate"] = generate_65B_certificate(ev["device_id"], ev["evidence_id"])

                ev["legal_mapping"] = {
                    "hash_post_source": "Section 65B, Evidence Act 1872",
                    "seal_status": "DFG SOP 2023",
                    "handover_document_id": "DFG Chain-of-Custody Guidelines",
                    "tool_validation_status": "Section 65B",
                    "timestamp_iso": "DFG Documentation, ICT Act 2006"
                }

                prev_hashes[chain_key] = post_hash
                prev_location_per_chain[chain_key] = to_location
                evidence_threads[eid] = ts_new
                global_event_index += 1
                events.append(ev)

            else:
                # Optionally add extra events
                if random.random() < 0.3:
                    extra_ev_type = random.choice(["analysis", "transfer"])
                    from_location = prev_location_per_chain.get(chain_key, "DFL_Lab")
                    ev, post_hash, ts_new = generate_event(
                        case_id, device["device_id"], eid, device["device_category"],
                        index=global_event_index,
                        prev_hash=prev_hashes.get(chain_key),
                        event_type=extra_ev_type,
                        ts=ts,
                        from_location=from_location,
                        to_location=from_location,
                        hash_algo=evidence_hash_algo[eid]
                    )

                    ev["custodian_info"] = custodian_analysis
                    ev["custodian_id"] = custodian_analysis["officer_id"]
                    ev["custodian_role"] = custodian_analysis["designation"]
                    ev["custodian_signature_hash"] = compute_hash(ev["custodian_id"], "sha256")
                    ev["warrant_id"] = warrant["warrant_id"]

                    prev_hashes[chain_key] = post_hash
                    evidence_threads[eid] = ts_new
                    global_event_index += 1
                    events.append(ev)
                else:
                    # Thread finished
                    unfinished.remove(eid)

        # If sequential mode, ensure all other evidence threads start after the latest event
        if not parallel and events:
            max_ts = max(datetime.fromisoformat(e["timestamp_iso"]) for e in events)
            for eid in unfinished:
                evidence_threads[eid] = max_ts + timedelta(seconds=10)
                
    # ✅ Validations remain the same
    assert validate_per_evidence_hash_chain(events)
    assert validate_seals(events)
    assert validate_time(events)
    assert validate_custody(events)
    assert validate_65B(events)
    assert validate_case_sequence(events)
    assert validate_handover_continuity(events)

    summary = {
        "case_id": case_id,
        "n_events": len(events),
        "devices_involved": len(devices),
        "overall_hash_chain_valid": True,
        "seal_integrity_valid": True,
        "custody_path_valid": True,
        "time_integrity_valid": True,
        "supports_65B_certificate": True,
        "label": "normal",
        "anomaly_type": None,
        "evidence_type": random.choice(list(ACQUISITION_LOGIC.keys())),
        "warrant": warrant,
        "custodians": [custodian_main, custodian_analysis]
    }

    return summary, sorted(events, key=lambda e: e["timestamp_iso"])  # sorted to simulate threaded timeline


class AnomalyType(Enum):
    # Existing
    HASH_MISMATCH = "hash_mismatch"                 # D
    WRITE_BLOCKER_MISSING = "write_blocker_missing" # C
    SEAL_BROKEN_UNJUSTIFIED = "seal_broken"          # B/E
    MISSING_HANDOVER = "missing_handover"           # B
    UNAUTHORIZED_ACCESS = "unauthorized_access"     # E

    # NEW – Regulatory-critical
    CLOCK_DRIFT = "clock_drift_exceeded"             # A
    TOOL_UNVALIDATED = "tool_not_accredited"         # F
    SEQUENCE_FLIP = "dfg_sequence_violation"         # DFG Workflow


def inject_root_anomaly(case: dict, anomaly: AnomalyType,target_index: int = None):

    events = case["events"]

    if target_index is None:
        target_index = len(events) // 4

    event = events[target_index]

    explanation = {}
    courtroom_compliance = {}  # New section for court/legal impact

    if anomaly == AnomalyType.HASH_MISMATCH:
        explanation = {
            "field": "hash_post_source",
            "expected": event["hash_post_source"],
            "observed": realistic_wrong_hash(),
            "category": "D",
            "regulation": "DFG-Hash Integrity"
        }
        event["hash_post_source"] = explanation["observed"]
        event["hash_verification_status"] = "mismatch"

        courtroom_compliance["section_65B_certificate"] = {
            "impact": "Certificate now invalid for court submission due to hash mismatch"
        }

    elif anomaly == AnomalyType.CLOCK_DRIFT:
        explanation = {
            "field": "timestamp_iso",
            "expected": "system_time_synchronized",
            "observed": "drift_exceeds_threshold",
            "category": "A",
            "regulation": "DFG-Documentation"
        }
        event["clock_drift_sec"] = 5000
        event["timestamp_iso"] = shift_time(event["timestamp_iso"], -7200)
        event["system_clock_source"] = "manual"

        courtroom_compliance["section_65B_certificate"] = {
            "impact": "Certificate timestamps unreliable; court may question chronology"
        }

    elif anomaly == AnomalyType.TOOL_UNVALIDATED:
        explanation = {
            "field": "tool_validation_status",
            "expected": "validated",
            "observed": "expired",
            "category": "F",
            "regulation": "Section 65B"
        }
        event["tool_validation_id"] = None
        event["tool_validation_status"] = "expired"

        courtroom_compliance["section_65B_certificate"] = {
            "impact": "Data production method unvalidated; certificate credibility reduced"
        }

    elif anomaly == AnomalyType.SEQUENCE_FLIP:
        explanation = {
            "field": "event_sequence",
            "expected": "DFG canonical order",
            "observed": "DFG order violated",
            "category": "DFG",
            "regulation": "DFG-2023-Workflow"
        }
    
        # Step 1: select evidence chain
        evidence_id = event["evidence_id"]
        chain = [e for e in events if e["evidence_id"] == evidence_id]
    
        # Step 2: get indices of canonical DFG events
        canonical_indices = [
            i for i, e in enumerate(chain)
            if e["event_type"] in DFG_SEQUENCE
        ]
    
        if len(canonical_indices) >= 2:
            # Step 3: pick two adjacent canonical events
            i1 = canonical_indices[0]
            i2 = canonical_indices[1]
    
            e1, e2 = chain[i1], chain[i2]
    
            # Step 4: swap event types ONLY (timestamps remain logical)
            e1["event_type"], e2["event_type"] = e2["event_type"], e1["event_type"]
    
            # Optional: annotate
            e1["notes"] = "DFG sequence violated: event order incorrect"
            e2["notes"] = "DFG sequence violated: event order incorrect"
    
        courtroom_compliance["courtroom_compliance_note"] = {
            "impact": "Workflow violation: analysis performed before prerequisite steps"
        }


    elif anomaly == AnomalyType.WRITE_BLOCKER_MISSING:
        explanation = {
            "field": "write_blocker_used",
            "expected": True,
            "observed": False,
            "regulation": "DFG-A"
        }
        if event["device_category"] in ["laptop", "USB"]:
            event["write_blocker_used"] = False
        else:
            return inject_root_anomaly(case, AnomalyType.HASH_MISMATCH)

        courtroom_compliance["section_65B_certificate"] = {
            "impact": "Write blocker missing; evidence acquisition integrity compromised"
        }

    elif anomaly == AnomalyType.MISSING_HANDOVER:
        explanation = {
            "field": "handover_document_id",
            "expected": "present",
            "observed": None,
            "category": "B",
            "regulation": "DFG-Custody"
        }
         
        # FORCE custodian change WITHOUT handover
        event["custodian_id"] = gen_id("CUST")
        event["custodian_role"] = "DFL_Analyst"
        event["handover_record"] = None
        event["handover_document_id"] = None

        courtroom_compliance["handover_record"] = {
            "impact": "Handover document missing; chain-of-custody violated"
        }

    elif anomaly == AnomalyType.UNAUTHORIZED_ACCESS:
        explanation = {
            "field": "custodian_role",
            "expected": "authorized_personnel",
            "observed": "unauthorized_user",
            "category": "E",
            "regulation": "DFG-Access Control"
        }
        event["custodian_role"] = "Unauthorized_User"

        courtroom_compliance["custodian_info"] = {
            "impact": "Unauthorized custodian; chain-of-custody violated"
        }

    elif anomaly == AnomalyType.SEAL_BROKEN_UNJUSTIFIED:
        explanation = {
            "field": "seal_status",
            "expected": "intact",
            "observed": "broken",
            "regulation": "DFG-S"
        }
        event["seal_status"] = "broken"

        courtroom_compliance["seal_info"] = {
            "impact": "Seal broken without justification; physical integrity compromised"
        }

    # Attach courtroom compliance info to explanation
    explanation["courtroom_compliance"] = courtroom_compliance

    return target_index, explanation,anomaly




ANOMALY_CYCLER = cycle(list(AnomalyType))

def pick_anomalies(is_multi: bool):
    if not is_multi:
        return [next(ANOMALY_CYCLER)]
    else:
        # ensure two distinct anomalies
        first = next(ANOMALY_CYCLER)
        second = next(ANOMALY_CYCLER)
        while second == first:
            second = next(ANOMALY_CYCLER)
        return [first, second]



def generate_anomalous_case_with_length(case_id, min_len, max_len, anomaly_type):
    case = generate_normal_case(case_id, min_len, max_len)

    root_index, explanation,anomaly_type = inject_root_anomaly(case, anomaly_type)
    apply_legal_cascade(case, root_index, anomaly_type)

    case["case_metadata"]["label"] = "benign_anomaly"
    case["case_metadata"]["anomaly_type"] = anomaly_type.value

    case["ground_truth_explanations"] = build_ground_truth(
        root_index,
        explanation,
        case["events"]
    )

    return case

def apply_legal_cascade(case: dict, root_index: int, anomaly_type: AnomalyType = None):
    """
    Apply integrity taint and update validation flags based on anomaly impact.
    Only downstream events of the same evidence chain are affected.

    Parameters:
    - case: dict containing "events" and "validation_flags"
    - root_index: int, index of the root anomalous event
    - anomaly_type: optional AnomalyType, used to decide which validation flags to update
    """
    events = case["events"]
    root_event = events[root_index]
    evidence_id = root_event["evidence_id"]
    root_ts = pd.to_datetime(root_event["timestamp_iso"])


    # --- Step 1: Taint downstream events (same evidence, later in time) ---
    for e in events:
        if (
            e["evidence_id"] == evidence_id and
            pd.to_datetime(e["timestamp_iso"]) > root_ts
        ):
            e["integrity_tainted"] = True

    # --- Step 2: Determine which validation flags to update ---
    # Default: only hash_chain and 65B impacted
    flags_to_update = {
        "overall_hash_chain_valid": True,
        "supports_65B_certificate": True,
        "custody_path_valid": True,
        "seal_integrity_valid": True,
        "time_integrity_valid": True
    }

    # Map anomaly types to affected flags
    if anomaly_type in [AnomalyType.HASH_MISMATCH, AnomalyType.WRITE_BLOCKER_MISSING, AnomalyType.TOOL_UNVALIDATED]:
        flags_to_update["overall_hash_chain_valid"] = False
        flags_to_update["supports_65B_certificate"] = False

    if anomaly_type in [AnomalyType.MISSING_HANDOVER, AnomalyType.UNAUTHORIZED_ACCESS]:
        flags_to_update["custody_path_valid"] = False

    if anomaly_type == AnomalyType.SEAL_BROKEN_UNJUSTIFIED:
        flags_to_update["seal_integrity_valid"] = False

    if anomaly_type == AnomalyType.CLOCK_DRIFT:
        flags_to_update["time_integrity_valid"] = False

    if anomaly_type == AnomalyType.SEQUENCE_FLIP:
        flags_to_update["overall_hash_chain_valid"] = False  # sequence violation can invalidate hash reasoning

    # Apply the updates to case validation flags
    for flag, invalidate in flags_to_update.items():
        if not invalidate:
            continue
        case["validation_flags"][flag] = False

    
def build_ground_truth(root_index, explanation, events):
    root_event = events[root_index]

    evidence_id = root_event["evidence_id"]
    root_ts = pd.to_datetime(root_event["timestamp_iso"])

    affected_events = [
        i for i, e in enumerate(events)
        if (
            e["evidence_id"] == evidence_id and
            pd.to_datetime(e["timestamp_iso"]) > root_ts
        )
    ]

    return {
        "root_cause": {
            "event_index": root_index,
            **explanation
        },
        "legal_consequence": {
            "affected_events": affected_events,
            "impact": "Evidence inadmissible under Section 65B due to upstream procedural violation"
        }
    }

def generate_normal_case(case_id=None, min_len=None, max_len=None):
    if min_len is not None and max_len is not None:
        n_events = random.randint(min_len, max_len)
    else:
        n_events = random.randint(8, 35)

    summary, events = build_normal_case(n_events)

    return {
        "case_metadata": summary,
        "events": events,
        "validation_flags": {
            "overall_hash_chain_valid": True,
            "seal_integrity_valid": True,
            "custody_path_valid": True,
            "time_integrity_valid": True,
            "supports_65B_certificate": True
        }
    }



def generate_anomalous_case_with_length_multi_device(case_id, min_len, max_len, anomaly_types):
    """
    Generate a case with multiple anomalies affecting different devices/evidences.
    Ensures global event indices are tracked for legal cascade.
    """
    # Step 1: Generate a normal case
    case_data = generate_normal_case(case_id, min_len, max_len)
    events = case_data["events"]
    
    case_data["case_metadata"]["label"] = "benign_anomaly"
    case_data["case_metadata"]["anomaly_type"] = "multi_anomaly"

    # Step 2: Pick 2-3 device+evidence chains to apply anomalies
    unique_chains = list({(e["device_id"], e["evidence_id"]) for e in events})
    random.shuffle(unique_chains)

    # Step 3: Map each anomaly to a device+evidence
    for i, anomaly in enumerate(anomaly_types):
        device_id, evidence_id = unique_chains[i % len(unique_chains)]
        
        # Get all events for this device+evidence
        chain_events = [e for e in events if e["device_id"] == device_id and e["evidence_id"] == evidence_id]
        chain_indices = [events.index(e) for e in chain_events]  # global indices

        # Pick target event roughly 25% into this chain
        local_target_idx = len(chain_events) // 4
        global_target_idx = chain_indices[local_target_idx]

        # Inject anomaly directly into the global event
        root_index, explanation,anomaly_type = inject_root_anomaly(case_data, anomaly, target_index=global_target_idx)
        
        # Apply legal cascade downstream using the global index
        apply_legal_cascade(
            case_data, root_index, anomaly_type
        )

        # Attach ground truth explanation globally
        if "ground_truth_explanations" not in case_data:
            case_data["ground_truth_explanations"] = []
        case_data["ground_truth_explanations"].append(
            build_ground_truth(global_target_idx, explanation, events)
        )

    return case_data



import json

def generate_full_forensic_dataset():
    all_cases_list = []
    all_events_list = []
    
    # --- Part 1: Generate Normal Cases (12,500) ---
    print("Generating Normal Cases...")
    for bucket_name, cfg in LENGTH_BUCKETS.items():
        # Using the quota from CLASS_QUOTA["normal"] distributed across buckets
        # Proportionally: 4800 short, 5400 medium, 2300 long (to reach 12.5k)
        num_to_gen = cfg["cases"] 
        
        for _ in range(num_to_gen):
            n_events = random.randint(cfg["min"], cfg["max"])
            # build_normal_case returns (summary_dict, events_list)
            case_meta, events = build_normal_case(n_events)
            
            # Ensure consistency: Add fields that exist in anomalies but not here
            case_meta["ground_truth_explanations"] = []
            for ev in events:
                ev["integrity_tainted"] = False
            
            all_cases_list.append(case_meta)
            all_events_list.extend(events)

    # --- Part 2: Generate Anomalous Cases (7,500) ---
    print("Generating Anomalous Cases...")

    total_anomalous = CLASS_QUOTA["benign_anomaly"]
    single_limit = int(total_anomalous * 0.67)
    
    generated_cases = 0
    
    for bucket_name, count in ANOMALY_LENGTH_BUCKETS.items():
        for _ in range(count):
            case_id = f"BA-{generated_cases:05d}"
    
            # Decide case type
            is_multi = generated_cases >= single_limit
    
            # Pick anomalies for THIS case only
            case_anomalies = pick_anomalies(is_multi)
    
            if not is_multi:
                case_data = generate_anomalous_case_with_length(
                    case_id,
                    min_len=LENGTH_BUCKETS[bucket_name]["min"],
                    max_len=LENGTH_BUCKETS[bucket_name]["max"],
                    anomaly_type=case_anomalies[0]
                )
            else:
                case_data = generate_anomalous_case_with_length_multi_device(
                    case_id,
                    min_len=LENGTH_BUCKETS[bucket_name]["min"],
                    max_len=LENGTH_BUCKETS[bucket_name]["max"],
                    anomaly_types=case_anomalies
                )
    
            generated_cases += 1
    
            # ---- Flatten for DataFrame ----
            meta = case_data["case_metadata"]
            meta["ground_truth_explanations"] = json.dumps(
                case_data["ground_truth_explanations"]
            )
            meta.update(case_data["validation_flags"])
    
            all_cases_list.append(meta)
            all_events_list.extend(case_data["events"])

    # --- Part 3: Create DataFrames & Export ---
    df_cases = pd.DataFrame(all_cases_list)
    df_events = pd.DataFrame(all_events_list)
    
    # Ensure integrity_tainted is boolean and filled for all
    df_events["integrity_tainted"] = df_events["integrity_tainted"].fillna(False)

    return df_cases, df_events


 
