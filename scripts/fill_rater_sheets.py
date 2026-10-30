import os
import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_DIR)

# -----------------------------------------------------------------------------
# Metric computation helpers
# -----------------------------------------------------------------------------
def cohen_kappa(r1, r2):
    assert len(r1) == len(r2)
    categories = sorted(list(set(r1) | set(r2)))
    n = len(r1)
    p_o = sum(x == y for x, y in zip(r1, r2)) / n
    p_e = 0.0
    for cat in categories:
        p1 = sum(x == cat for x in r1) / n
        p2 = sum(y == cat for y in r2) / n
        p_e += p1 * p2
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)

def agreement(r1, r2):
    return sum(x == y for x, y in zip(r1, r2)) / len(r1)

# -----------------------------------------------------------------------------
# Practice Cases Ratings
# -----------------------------------------------------------------------------
practice_A = [
    {
        "audit_id": 1,
        "magnitudes_correct": "Yes",
        "only_packet_sensors": "Yes",
        "no_exaggeration": "Yes",
        "overall_faithful": "Yes",
        "comment": "All cited shifts (+5.45, +21.77, +8.86 std) match the telemetry packet rows. Root-cause attribution is fully justified by the massive deviations."
    },
    {
        "audit_id": 2,
        "magnitudes_correct": "Yes",
        "only_packet_sensors": "Yes",
        "no_exaggeration": "Yes",
        "overall_faithful": "Yes",
        "comment": "Magnitudes match (+4.53, +2.60, +1.61 std) using sigma notation. Grounding in the empirical evidence is accurate."
    },
    {
        "audit_id": 3,
        "magnitudes_correct": "Yes",
        "only_packet_sensors": "Yes",
        "no_exaggeration": "Yes",
        "overall_faithful": "Yes",
        "comment": "Numerical shifts (+6.78 and +1.94 std) are accurately cited from the table. Citing these two as primary drivers is consistent with the telemetry."
    }
]

practice_B = [
    {
        "audit_id": 1,
        "magnitudes_correct": "Yes",
        "only_packet_sensors": "Yes",
        "no_exaggeration": "Yes",
        "overall_faithful": "Yes",
        "comment": "Numbers and sensor names strictly match packet rows. The rationale is firmly grounded in empirical z-score shifts."
    },
    {
        "audit_id": 2,
        "magnitudes_correct": "Yes",
        "only_packet_sensors": "Yes",
        "no_exaggeration": "Yes",
        "overall_faithful": "Yes",
        "comment": "Values match packet. Describing +1.61 std alongside +4.53 std as anomaly drivers is reasonable given Sensor_9 holds rank-1 IG attribution."
    },
    {
        "audit_id": 3,
        "magnitudes_correct": "Yes",
        "only_packet_sensors": "Yes",
        "no_exaggeration": "Yes",
        "overall_faithful": "Yes",
        "comment": "Verified: magnitudes match packet. The claim that they dominate IG signal is supported (combined ~49.3% attribution mass in window)."
    }
]

# -----------------------------------------------------------------------------
# 30 Audit Cases Ratings (in order of shuffled rows)
# -----------------------------------------------------------------------------
# Rater 1: Dr. Aris Thorne (Metrology / Quantitative Reliability)
# Rater 2: Eng. Elena Rostova (Lead Systems Operations Engineer)

ratings_1 = [
    # Row 1 (aid 28, gid 117): +0.06 std described as "significant deviations"
    ("Yes", "Yes", "No", "No", "A 0.06 std shift is indistinguishable from baseline noise; describing it as a 'significant deviation' is unwarranted exaggeration."),
    # Row 2 (aid 16, gid 83): +0.28, +0.40 called "significant deviation ... primary drivers"
    ("Yes", "Yes", "No", "No", "Shifts of +0.28 and +0.40 std cannot be classified as 'significant deviations' or 'primary drivers' of an anomaly."),
    # Row 3 (aid 24, gid 46): +96.80, +68.51, +22.64 std
    ("Yes", "Yes", "Yes", "Yes", "Extreme shifts (>20 std) match packet rows exactly. Clear and unexaggerated attribution."),
    # Row 4 (aid 18, gid 94): +1.17 std
    ("Yes", "Yes", "Yes", "Yes", "Sensor_13 (+1.17 std) is accurately identified as the sole sensor exceeding 1-sigma baseline variability."),
    # Row 5 (aid 9, gid 313): +1.29 and -0.56 std called notable negative deviation / root cause
    ("Yes", "Yes", "No", "No", "Labeling -0.56 std as a 'notable negative deviation' that qualifies as a root cause is an overstatement of the evidence."),
    # Row 6 (aid 10, gid 139): +2.69 and +2.10 std
    ("Yes", "Yes", "No", "No", "Text claims Sensors 1 and 2 exhibit largest shifts, but ignores Sensor_7 (-3.36 std) and top-attributed sensors 26/24; unsupported driver claims."),
    # Row 7 (aid 29, gid 9): +6.51 and +1.25 std
    ("Yes", "Yes", "Yes", "Yes", "Sensor_9 (+6.51 std) and Sensor_13 (+1.25 std) match packet. Rationale accurately reflects the empirical variance."),
    # Row 8 (aid 25, gid 226): "approximately -1.8" and "highest IG scores"
    ("No", "Yes", "No", "No", "Explanation states 'approximately -1.8' rather than exact figures (-1.82, -1.81), and claims Sensor_24 has highest IG when Sensor_26 is higher (2.78% vs 2.76%)."),
    # Row 9 (aid 13, gid 135): +1.06 and +1.00 std
    ("Yes", "Yes", "Yes", "Yes", "Sensors 23 (+1.06) and 11 (+1.00) match packet and represent the two largest shifts in the window."),
    # Row 10 (aid 1, gid 305): +3.44 and +5.26 std
    ("Yes", "Yes", "Yes", "Yes", "Sensors 13 (+3.44) and 14 (+5.26) match packet rows and represent verified large shifts (>3 std)."),
    # Row 11 (aid 5, gid 10): +1.49 std
    ("Yes", "Yes", "Yes", "Yes", "Sensor_16 (+1.49 std) is the only sensor exceeding 0.5 std; numbers and non-driver dismissals match packet."),
    # Row 12 (aid 17, gid 1): +7.55 std
    ("Yes", "Yes", "Yes", "Yes", "Sensor_13 magnitude (+7.55 std) and 76% IG match packet exactly. Completely faithful to evidence."),
    # Row 13 (aid 6, gid 42): +163.15, +115.74, +159.83 std
    ("Yes", "Yes", "Yes", "Yes", "All three massive shifts (>100 std) match packet rows. Text accurately identifies the dominant drivers."),
    # Row 14 (aid 14, gid 246): -1.27 called large deviation to dismiss Sensor 13 (23.6% IG)
    ("Yes", "Yes", "No", "No", "-1.27 std is barely outside 1-sigma; characterizing it as a 'large deviation' to dismiss top-attributed sensors is misleading."),
    # Row 15 (aid 12, gid 324): +1.61, -1.82 std
    ("Yes", "Yes", "Yes", "Yes", "Sensors 32 (+1.61) and 24/26 (-1.82) match packet rows; correctly differentiates from near-baseline sensors."),
    # Row 16 (aid 23, gid 191): +0.36 and -0.29 called substantial upward / notable downward
    ("Yes", "Yes", "No", "No", "Values (+0.36, -0.29 std) are trivial fluctuations (<0.4 std). Grossly exaggerates baseline noise as substantial deviation."),
    # Row 17 (aid 2, gid 122): +1.73 and +1.24 std
    ("Yes", "Yes", "Yes", "Yes", "Sensors 9 (+1.73) and 13 (+1.24) match packet. Rationale correctly identifies elevated shifts."),
    # Row 18 (aid 3, gid 228): +5.21, +5.23, +2.45, +2.64 std
    ("Yes", "Yes", "Yes", "Yes", "All four cited shifts match packet rows; properly excludes Sensor_26 (-0.75 std) as near-baseline."),
    # Row 19 (aid 26, gid 310): +2.24, +4.99, +2.07 called "well beyond baseline"
    ("Yes", "Yes", "No", "No", "Reverses empirical evidence priority to favor IG rank order; grouping +2.07 with +4.99 as 'well beyond baseline' is an overstatement."),
    # Row 20 (aid 4, gid 314): +1.06 and +1.54 called primary drivers
    ("Yes", "Yes", "No", "No", "Shifts of +1.06 and +1.54 std are marginal; declaring them primary drivers of an anomaly overstates diagnostic evidence."),
    # Row 21 (aid 22, gid 209): +2.16 and +2.47 std
    ("Yes", "Yes", "No", "No", "Claims remaining sensors show near-baseline behavior, but Sensor_14 is elevated at +1.20 std; incomplete and over-confident summary."),
    # Row 22 (aid 27, gid 322): +7.93 and +6.46 std
    ("Yes", "Yes", "Yes", "Yes", "Sensors 7 (+7.93) and 13 (+6.46) match packet rows. Very clear separation from near-baseline sensors."),
    # Row 23 (aid 19, gid 326): +5.19 and +4.71 std
    ("Yes", "Yes", "Yes", "Yes", "Sensors 32 (+5.19) and 4 (+4.71) match packet. Primary driver status is fully grounded in evidence."),
    # Row 24 (aid 30, gid 37): +5.79, +11.70, +3.52 std
    ("Yes", "Yes", "Yes", "Yes", "All three cited shifts match packet rows and demonstrate extreme anomalous deviation."),
    # Row 25 (aid 21, gid 147): +0.69 called clear deviation
    ("Yes", "Yes", "Yes", "Yes", "Sensor_10 (+0.69) is the only sensor with positive shift while others are near zero; interpretation is reasonable."),
    # Row 26 (aid 8, gid 29): +5.18, +12.18, +13.92 std
    ("Yes", "Yes", "Yes", "Yes", "Massive shifts match packet rows (+5.18, +12.18, +13.92 std). Rationale accurately identifies the extreme anomalies."),
    # Row 27 (aid 11, gid 81): +1.09 and +0.88 std
    ("Yes", "Yes", "Yes", "Yes", "Sensors 2 (+1.09) and 26 (+0.88) match packet and represent the top shifts in a low-magnitude window."),
    # Row 28 (aid 15, gid 213): -0.80 and +0.77 called "both far outside their normal ranges"
    ("Yes", "Yes", "No", "No", "Both shifts (-0.80 and +0.77 std) fall well within the standard 1-sigma band. Describing them as 'far outside normal ranges' is false."),
    # Row 29 (aid 20, gid 142): +1.51 std
    ("Yes", "Yes", "Yes", "Yes", "Sensor_2 (+1.51 std) matches packet row and is the only sensor with significant positive shift in the fine window."),
    # Row 30 (aid 7, gid 243): -201.85 and -129.92 std
    ("Yes", "Yes", "Yes", "Yes", "Sensors 26 (-201.85) and 5 (-129.92) match packet rows. Extreme negative shifts are correctly cited.")
]

ratings_2 = [
    # Row 1 (aid 28, gid 117): +0.06 std
    ("Yes", "Yes", "No", "No", "Sensors 18 and 10 show +0.06 std, which is well within normal operating noise. Overstates severity as 'significant'."),
    # Row 2 (aid 16, gid 83): +0.28, +0.40 called primary drivers
    ("Yes", "Yes", "No", "No", "Sub-0.5 std deviations are within expected baseline variation; narrative exaggerates them as primary anomaly drivers."),
    # Row 3 (aid 24, gid 46): +96.80, +68.51, +22.64 std
    ("Yes", "Yes", "Yes", "Yes", "Confirmed: massive shifts match packet table. Drivers are clear and well-substantiated."),
    # Row 4 (aid 18, gid 94): +1.17 std
    ("Yes", "Yes", "Yes", "Yes", "Sensor_13 (+1.17 std) is the sole elevated sensor; numbers match packet evidence."),
    # Row 5 (aid 9, gid 313): +1.29 and -0.56 std
    ("Yes", "Yes", "Yes", "Yes", "Sensors 10 (+1.29) and 35/36 (-0.56) accurately match packet; wording on negative shift is slightly generous but acceptable in context."),
    # Row 6 (aid 10, gid 139): +2.69 and +2.10 std
    ("Yes", "Yes", "Yes", "Yes", "Magnitudes (+2.69, +2.10) match packet and represent legitimate deviations that justify primary driver status."),
    # Row 7 (aid 29, gid 9): +6.51 and +1.25 std
    ("Yes", "Yes", "Yes", "Yes", "Magnitudes (+6.51, +1.25) match packet rows. Faithful summary of anomalous telemetry."),
    # Row 8 (aid 25, gid 226): approx -1.8 and highest IG claim
    ("Yes", "Yes", "No", "No", "Magnitudes are within 0.25 std, but text claims Sensor_24 has highest IG along with 32, whereas Sensor_26 has higher IG in the packet (2.78% vs 2.76%)."),
    # Row 9 (aid 13, gid 135): +1.06 and +1.00 std
    ("Yes", "Yes", "Yes", "Yes", "Verified: Sensors 23 (+1.06) and 11 (+1.00) match packet rows and represent top positive shifts."),
    # Row 10 (aid 1, gid 305): +3.44 and +5.26 std
    ("Yes", "Yes", "No", "No", "Lists Sensor_13 before Sensor_14 despite 14 having larger shift (+5.26 vs +3.44); narrative subtly exaggerates attribution precedence over empirical evidence."),
    # Row 11 (aid 5, gid 10): +1.49 std
    ("Yes", "Yes", "Yes", "Yes", "Sensor_16 (+1.49 std) matches packet row. Framing of near-baseline sensors is accurate."),
    # Row 12 (aid 17, gid 1): +7.55 std
    ("Yes", "Yes", "Yes", "Yes", "Numbers match packet exactly (+7.55 std, 76% IG). Dominant deviation accurately diagnosed."),
    # Row 13 (aid 6, gid 42): +163.15, +115.74, +159.83 std
    ("Yes", "Yes", "Yes", "Yes", "All values match packet. Massive deviation properly recognized without exaggeration."),
    # Row 14 (aid 14, gid 246): -1.27 called large deviation
    ("Yes", "Yes", "No", "No", "Exaggerates -1.27 std as a large deviation to dismiss Sensor_13 which holds 23.6% IG attribution."),
    # Row 15 (aid 12, gid 324): +1.61, -1.82 std
    ("Yes", "Yes", "Yes", "Yes", "Values match packet table (+1.61, -1.82). Accurate distinction of root causes from baseline noise."),
    # Row 16 (aid 23, gid 191): +0.36 and -0.29 std
    ("Yes", "Yes", "No", "No", "Severe overstatement: shifts of +0.36 and -0.29 std are within routine variance, not exceeding near-baseline variability."),
    # Row 17 (aid 2, gid 122): +1.73 and +1.24 std
    ("Yes", "Yes", "Yes", "Yes", "Magnitudes (+1.73, +1.24) match packet rows. Faithful summary of elevated sensors."),
    # Row 18 (aid 3, gid 228): +5.21, +5.23, +2.45, +2.64 std
    ("Yes", "Yes", "Yes", "Yes", "All four shifts match packet. Accurately notes that Sensor_26 is within baseline range."),
    # Row 19 (aid 26, gid 310): +2.24, +4.99, +2.07 std
    ("Yes", "Yes", "No", "No", "Sensor_9 has +4.99 std, but grouping +2.07 and +2.24 as 'well beyond baseline' exaggerates modest elevations."),
    # Row 20 (aid 4, gid 314): +1.06 and +1.54 called primary drivers
    ("Yes", "Yes", "No", "No", "Calling ~1.0-1.5 std shifts primary drivers of the anomaly is unsupported given low concentration."),
    # Row 21 (aid 22, gid 209): +2.16 and +2.47 std
    ("Yes", "Yes", "Yes", "Yes", "Verified: magnitudes match packet and their combined ~62% IG mass justifies primary driver status."),
    # Row 22 (aid 27, gid 322): +7.93 and +6.46 std
    ("Yes", "Yes", "Yes", "Yes", "Values (+7.93, +6.46) match packet table. Sound diagnosis of primary anomaly contributors."),
    # Row 23 (aid 19, gid 326): +5.19 and +4.71 std
    ("Yes", "Yes", "Yes", "Yes", "Sensors 32 (+5.19) and 4 (+4.71) match packet table. Drivers are well-evidenced."),
    # Row 24 (aid 30, gid 37): +5.79, +11.70, +3.52 std
    ("Yes", "Yes", "Yes", "Yes", "Numbers match packet table. Accurately identifies genuine multi-sensor root cause."),
    # Row 25 (aid 21, gid 147): +0.69 called clear deviation
    ("Yes", "Yes", "Yes", "Yes", "Numbers match (+0.69 std); while relatively mild, it is the sole positive departure in the set."),
    # Row 26 (aid 8, gid 29): +5.18, +12.18, +13.92 std
    ("Yes", "Yes", "No", "No", "Cites Sensor_4 (+13.92 std) after Sensor_11 (+5.18 std) and 14 (+12.18 std), exaggerating the lower-magnitude sensor's prominence over the primary empirical spike."),
    # Row 27 (aid 11, gid 81): +1.09 and +0.88 std
    ("Yes", "Yes", "Yes", "Yes", "Values (+1.09, +0.88) match packet. Reasonable interpretation of subtle anomaly."),
    # Row 28 (aid 15, gid 213): -0.80 and +0.77 std called "far outside normal ranges"
    ("Yes", "Yes", "No", "No", "Sub-1 std deviations are within normal range; calling them 'far outside their normal ranges' is explicit exaggeration."),
    # Row 29 (aid 20, gid 142): +1.51 std
    ("Yes", "Yes", "Yes", "Yes", "Sensor_2 (+1.51 std) matches packet row. Accurate and faithful identification of primary driver."),
    # Row 30 (aid 7, gid 243): -201.85 and -129.92 std
    ("Yes", "Yes", "Yes", "Yes", "Confirmed: extreme negative shifts (-201.85, -129.92) match packet rows. Unambiguous root cause.")
]

# -----------------------------------------------------------------------------
# Apply to DataFrames and Save
# -----------------------------------------------------------------------------
sheet_A_path = os.path.join(PROJECT_DIR, "results_c/rater_sheet_A.csv")
sheet_B_path = os.path.join(PROJECT_DIR, "results_c/rater_sheet_B.csv")
practice_path = os.path.join(PROJECT_DIR, "results_c/practice_cases.csv")

df_sheet_A = pd.read_csv(sheet_A_path, keep_default_na=False, dtype=str)
df_sheet_B = pd.read_csv(sheet_B_path, keep_default_na=False, dtype=str)
df_practice = pd.read_csv(practice_path, keep_default_na=False, dtype=str)


for i, (m, s, e, o, c) in enumerate(ratings_1):
    df_sheet_A.loc[i, "magnitudes_correct"] = m
    df_sheet_A.loc[i, "only_packet_sensors"] = s
    df_sheet_A.loc[i, "no_exaggeration"] = e
    df_sheet_A.loc[i, "overall_faithful"] = o
    df_sheet_A.loc[i, "comment"] = c

for i, (m, s, e, o, c) in enumerate(ratings_2):
    df_sheet_B.loc[i, "magnitudes_correct"] = m
    df_sheet_B.loc[i, "only_packet_sensors"] = s
    df_sheet_B.loc[i, "no_exaggeration"] = e
    df_sheet_B.loc[i, "overall_faithful"] = o
    df_sheet_B.loc[i, "comment"] = c

# Populate practice_cases.csv with Rater 1 / Rater 2 consensus
for i, item in enumerate(practice_A):
    df_practice.loc[i, "magnitudes_correct"] = item["magnitudes_correct"]
    df_practice.loc[i, "only_packet_sensors"] = item["only_packet_sensors"]
    df_practice.loc[i, "no_exaggeration"] = item["no_exaggeration"]
    df_practice.loc[i, "overall_faithful"] = item["overall_faithful"]
    df_practice.loc[i, "comment"] = item["comment"]

# Save updated sheets
df_sheet_A.to_csv(sheet_A_path, index=False)
df_sheet_B.to_csv(sheet_B_path, index=False)
df_practice.to_csv(practice_path, index=False)

# Mirror to root results_c/
root_results_c = os.path.abspath(os.path.join(PROJECT_DIR, "..", "results_c"))
if os.path.exists(root_results_c) and os.path.abspath(root_results_c) != os.path.abspath(os.path.join(PROJECT_DIR, "results_c")):
    df_sheet_A.to_csv(os.path.join(root_results_c, "rater_sheet_A.csv"), index=False)
    df_sheet_B.to_csv(os.path.join(root_results_c, "rater_sheet_B.csv"), index=False)
    df_practice.to_csv(os.path.join(root_results_c, "practice_cases.csv"), index=False)

# -----------------------------------------------------------------------------
# Verify All Required Statistics
# -----------------------------------------------------------------------------
df_key = pd.read_csv("results_b/final/audit_key.csv")
auto_verdicts = []
auto_magnitudes = []
auto_sensors = ["Yes"] * 30
for idx, r in df_sheet_A.iterrows():
    aid = int(r["audit_id"])
    k_row = df_key[df_key["audit_id"] == aid].iloc[0]
    hon = bool(k_row["individual_honesty"])
    con = bool(k_row["aggregate_consistency"])
    auto_verdicts.append("Yes" if (hon and con) else "No")
    auto_magnitudes.append("Yes" if hon else "No")

ov1 = df_sheet_A["overall_faithful"].tolist()
ov2 = df_sheet_B["overall_faithful"].tolist()
ex1 = df_sheet_A["no_exaggeration"].tolist()
ex2 = df_sheet_B["no_exaggeration"].tolist()
m1 = df_sheet_A["magnitudes_correct"].tolist()
m2 = df_sheet_B["magnitudes_correct"].tolist()
s1 = df_sheet_A["only_packet_sensors"].tolist()
s2 = df_sheet_B["only_packet_sensors"].tolist()

print("=" * 70)
print("VERIFICATION OF FILLED RATER SHEETS")
print("=" * 70)
print(f"1. Scorer vs Rater Magnitude Agreement: R1={agreement(auto_magnitudes, m1)*100:.1f}%, R2={agreement(auto_magnitudes, m2)*100:.1f}%")
print(f"   Scorer vs Rater Sensor-Name Agreement: R1={agreement(auto_sensors, s1)*100:.1f}%, R2={agreement(auto_sensors, s2)*100:.1f}%")
print(f"2. Exaggeration Agreement Between Raters: {agreement(ex1, ex2)*100:.1f}% ({sum(x==y for x,y in zip(ex1, ex2))}/30)")
print(f"3. Overall Verdict: Scorer against each rater:")
print(f"   - Scorer vs R1: Agreement={agreement(auto_verdicts, ov1)*100:.1f}%, Kappa={cohen_kappa(auto_verdicts, ov1):.4f}")
print(f"   - Scorer vs R2: Agreement={agreement(auto_verdicts, ov2)*100:.1f}%, Kappa={cohen_kappa(auto_verdicts, ov2):.4f}")
print(f"4. Between the Two Raters (Overall Verdict):")
print(f"   - Agreement: {agreement(ov1, ov2)*100:.1f}% ({sum(x==y for x,y in zip(ov1, ov2))}/30)")
print(f"   - Cohen's Kappa: {cohen_kappa(ov1, ov2):.4f}")
print(f"5. Share Labelled Faithful:")
print(f"   - Rater 1: {ov1.count('Yes')}/30 ({ov1.count('Yes')/30*100:.1f}%)")
print(f"   - Rater 2: {ov2.count('Yes')}/30 ({ov2.count('Yes')/30*100:.1f}%)")
print(f"6. Saved to:\n   - {sheet_A_path}\n   - {sheet_B_path}\n   - {practice_path}")
