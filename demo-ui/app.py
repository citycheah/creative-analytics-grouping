"""Deprecated standalone Streamlit demo UI (retired in favor of federated Next.js Portal).

Use `brand-analytics-agent/frontend-portal/` backed by `brand-analytics-agent` FastAPI.
Static CSV artifacts in `demo-ui/data/` remain frozen for Malaysia Telco ablation regression tests.
"""

import sys


def main() -> None:
    print(
        "creative-analytics-grouping `demo-ui/` has been retired. "
        "Launch the unified 5-page Portal in `brand-analytics-agent/frontend-portal/`."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
