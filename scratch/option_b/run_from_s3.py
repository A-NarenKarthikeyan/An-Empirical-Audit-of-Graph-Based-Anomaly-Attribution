"""Resume pipeline from Stage S3 (S1 and S2 already completed)."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from scratch.option_b.s3_attribution import run_s3_attribution
from scratch.option_b.s4_conformal import run_s4_conformal
from scratch.option_b.s5_abstention import run_s5_abstention_screening
from scratch.option_b.s6_faithfulness import run_s6_faithfulness_audit
from scratch.option_b.s7_human_audit import run_s7_human_audit
from scratch.option_b.s8_summary_manifest import build_paper_summary_and_manifest
from scratch.option_b.package_downloads import package_all_results_to_downloads

def main():
    print("\n" + "=" * 80)
    print("RESUMING FROM STAGE S3 (S1 and S2 already complete)")
    print("=" * 80 + "\n")

    t_start = time.time()

    print("\n>>> STAGE S3: ATTRIBUTION (327 segments, 28 machines)")
    run_s3_attribution()

    print("\n>>> STAGE S4: CONFORMAL CALIBRATION")
    run_s4_conformal()

    print("\n>>> STAGE S5: ABSTENTION SCREENING")
    run_s5_abstention_screening()

    print("\n>>> STAGE S6: FAITHFULNESS AUDIT / GATE G6")
    run_s6_faithfulness_audit()

    print("\n>>> STAGE S7: BLIND HUMAN AUDIT SHEET")
    run_s7_human_audit()

    print("\n>>> STAGE S8: SUMMARY & MANIFEST")
    build_paper_summary_and_manifest()

    print("\n>>> PACKAGING RESULTS AS ZIP AND DOCX IN DOWNLOADS")
    package_all_results_to_downloads()

    total = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"STAGES S3-S8 + PACKAGING COMPLETE in {total:.1f}s ({total/60:.1f} min)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
