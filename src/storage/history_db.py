"""
VisionGuard AI - Inspection Audit History Database (SQLite)
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_DB_PATH = Path(os.getenv("INSPECTION_DB_PATH", "data/inspections.db"))


class InspectionHistoryDB:
    """Thread-safe SQLite logger for quality inspection records."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    category TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    raw_score REAL NOT NULL,
                    normalized_score REAL NOT NULL,
                    z_score REAL NOT NULL,
                    defect_area_pct REAL NOT NULL,
                    latency_ms REAL NOT NULL
                )
                """
            )

    def log_inspection(self, filename: str, result: Dict[str, Any]) -> int:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO inspections (
                    timestamp, filename, category, decision,
                    raw_score, normalized_score, z_score, defect_area_pct, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    filename,
                    result["category"],
                    result["decision"],
                    float(result["raw_score"]),
                    float(result["normalized_score"]),
                    float(result["z_score"]),
                    float(result["defect_area_pct"]),
                    float(result["latency_ms"]),
                ),
            )
            return int(cur.lastrowid)

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_summary_stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN decision = 'PASS' THEN 1 ELSE 0 END) AS pass_count,
                    SUM(CASE WHEN decision = 'REVIEW' THEN 1 ELSE 0 END) AS review_count,
                    SUM(CASE WHEN decision = 'FAIL' THEN 1 ELSE 0 END) AS fail_count,
                    AVG(latency_ms) AS avg_latency_ms
                FROM inspections
                """
            ).fetchone()

        total = int(row["total"] or 0)
        pass_count = int(row["pass_count"] or 0)
        review_count = int(row["review_count"] or 0)
        fail_count = int(row["fail_count"] or 0)
        avg_latency = round(float(row["avg_latency_ms"] or 0.0), 2)
        pass_rate = round((pass_count / total) * 100.0, 1) if total > 0 else 0.0

        return {
            "total_inspected": total,
            "pass_count": pass_count,
            "review_count": review_count,
            "fail_count": fail_count,
            "pass_rate_pct": pass_rate,
            "avg_latency_ms": avg_latency,
        }
