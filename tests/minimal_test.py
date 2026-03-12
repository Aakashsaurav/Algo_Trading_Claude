#!/usr/bin/env python3
"""Minimal test for engine_v3.py"""

import sys
sys.path.insert(0, '/workspaces/Algo_Trading_Claude')

print("Testing engine_v3.py import...")

try:
    from backtester.engine_v3 import BacktestEngineV3, BacktestConfigV3
    print("✓ SUCCESS: Engine V3 imported successfully")
except ImportError as e:
    print(f"✗ FAILED: Import error - {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ FAILED: Unexpected error - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✓ All basic tests passed!")