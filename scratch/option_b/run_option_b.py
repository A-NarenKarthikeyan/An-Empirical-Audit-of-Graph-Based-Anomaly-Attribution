"""Master end-to-end runner for Option B: S1 through S8."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from scratch.option_b.s1_orchestrator import main as s1_main
from scratch.option_b.run_remaining_stages import run_all_remaining

def main():
    print("\n" + "=" * 80)
    print("STARTING FULL OPTION B END-TO-END EXECUTION (S1 -> S8)")
    print("=" * 80 + "\n")
    
    t_start = time.time()
    
    # 1. Run Stage S1 Orchestrator
    print("\n>>> EXECUTING STAGE S1: DETECTORS ORCHESTRATOR")
    s1_main()
    
    # 2. Run Stages S2 through S8
    print("\n>>> EXECUTING STAGES S2 THROUGH S8")
    run_all_remaining()
    
    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"FULL STUDY OPTION B COMPLETE IN {total_elapsed:.2f}s ({total_elapsed / 3600.0:.2f} hours)!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
