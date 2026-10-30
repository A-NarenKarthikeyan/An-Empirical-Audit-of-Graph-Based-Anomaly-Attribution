"""Master runner for Stages S2 through S8 after Stage S1 completes."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from scratch.option_b.s2_baselines import run_all_baselines
from scratch.option_b.s3_attribution import run_s3_attribution
from scratch.option_b.s4_conformal import run_s4_conformal
from scratch.option_b.s5_abstention import run_s5_abstention_screening
from scratch.option_b.s6_faithfulness import run_s6_faithfulness_audit
from scratch.option_b.s7_human_audit import run_s7_human_audit
from scratch.option_b.s8_summary_manifest import build_paper_summary_and_manifest

def run_all_remaining():
    print("\n" + "=" * 80)
    print("STARTING EXECUTION OF STAGES S2 THROUGH S8")
    print("=" * 80 + "\n")
    
    t_start = time.time()
    
    # Stage S2: Same-split baselines
    print("\n>>> EXECUTING STAGE S2: BASELINES")
    run_all_baselines()
    
    # Stage S3: Attribution on 327 segments
    print("\n>>> EXECUTING STAGE S3: ATTRIBUTION")
    run_s3_attribution()
    
    # Stage S4: Conformal Calibration
    print("\n>>> EXECUTING STAGE S4: CONFORMAL CALIBRATION")
    run_s4_conformal()
    
    # Stage S5: Abstention Screening
    print("\n>>> EXECUTING STAGE S5: ABSTENTION SCREENING")
    run_s5_abstention_screening()
    
    # Stage S6: Faithfulness Audit (and GATE G6 check)
    print("\n>>> EXECUTING STAGE S6: FAITHFULNESS AUDIT / GATE G6")
    run_s6_faithfulness_audit()
    
    # Stage S7: Blind Human Audit
    print("\n>>> EXECUTING STAGE S7: BLIND HUMAN AUDIT SHEET")
    run_s7_human_audit()
    
    # Stage S8: Summary for Paper & Manifest
    print("\n>>> EXECUTING STAGE S8: SUMMARY & MANIFEST")
    build_paper_summary_and_manifest()
    
    # Packaging for User: Word DOCX and ZIP archive in Downloads
    print("\n>>> PACKAGING RESULTS AS ZIP AND DOC IN DOWNLOADS")
    from scratch.option_b.package_downloads import package_all_results_to_downloads
    package_all_results_to_downloads()
    
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"ALL STAGES S2 THROUGH S8 COMPLETED IN {total_time:.2f}s ({total_time / 60.0:.2f} min)!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_all_remaining()
