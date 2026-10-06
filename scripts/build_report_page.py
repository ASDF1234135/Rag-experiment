"""Fill report/template.html with report/report_data.json.

python -m scripts.build_report_data && python -m scripts.build_report_page
Writes report/ragbench_report.html (the page to publish) and report/ragbench_report.preview.html
(same page inside a standard doctype/head, for opening locally).
"""
import json
import sys
from pathlib import Path

DATA = sys.argv[1] if len(sys.argv) > 1 else "report/report_data.json"
NAME = sys.argv[2] if len(sys.argv) > 2 else "ragbench_report"   # output basename in report/

tpl = Path("report/template.html").read_text(encoding="utf-8")
data = json.loads(Path(DATA).read_text(encoding="utf-8"))
for ds in data["spend"]["by_dataset"].values():
    ds.pop("_tokens", None)  # raw token counts are not needed on the page
page = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
Path(f"report/{NAME}.html").write_text(page, encoding="utf-8")
shell = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
         '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>'
         + page + "</body></html>")
Path(f"report/{NAME}.preview.html").write_text(shell, encoding="utf-8")
print(f"report/{NAME}.html", len(page) // 1024, "KB")
