#!/usr/bin/env python3

try:
    from flight_implementation_code import FlightDataCollector, DelayPredictor, GeneticScheduleOptimizer, CascadingImpactAnalyzer
    print("✅ All imports successful!")
    print("- FlightDataCollector: OK")
    print("- DelayPredictor: OK") 
    print("- GeneticScheduleOptimizer: OK")
    print("- CascadingImpactAnalyzer: OK")
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
