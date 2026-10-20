"""Run local tests and record their actual result for the project report."""
from pathlib import Path
import sys,unittest,json,io,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'));result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
report=ROOT/'reports';report.mkdir(exist_ok=True)
(report/'tests.txt').write_text(stream.getvalue(),encoding='utf-8')
(report/'tests.json').write_text(json.dumps({'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
print(stream.getvalue());sys.exit(0 if result.wasSuccessful() else 1)
