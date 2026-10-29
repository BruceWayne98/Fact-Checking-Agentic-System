import io
import sys
from pathlib import Path

# Ensure UTF-8 console output for Windows cmd/powershell
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from graph.baseline_graph import build_baseline_graph

def run_pipeline(text: str):
    print("=" * 60)
    print("🚀 TRUST Agents: Collaborative Multi-Agent Fact Verification")
    print("=" * 60)
    print(f"INPUT STATEMENT / ARTICLE:\n\"{text}\"\n")

    app = build_baseline_graph()
    result = app.invoke({"article": text})

    print("\n" + "=" * 60)
    print("📄 VERIFICATION AUDIT REPORT")
    print("=" * 60)
    print(result.get("final_report", "No report generated."))
    return result

if __name__ == "__main__":
    sample_statement = (
        "The US unemployment rate fell to 3.5% in January 2024, the lowest since 1969. "
        "Meanwhile, health care reform legislation was passed to mandate free sex change surgeries."
    )
    run_pipeline(sample_statement)
