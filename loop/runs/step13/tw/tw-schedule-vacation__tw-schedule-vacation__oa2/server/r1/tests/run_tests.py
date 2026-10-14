import sys, traceback
sys.path.insert(0, "/tests")
import test_outputs as t
ok = True
for name in sorted(n for n in dir(t) if n.startswith("test_")):
    try:
        getattr(t, name)()
        print("PASSED", name)
    except Exception:
        ok = False
        print("FAILED", name)
        traceback.print_exc()
sys.exit(0 if ok else 1)
