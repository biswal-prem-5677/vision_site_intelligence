import sqlite3
import os
from datetime import datetime
from app.config import DB_PATH

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            track_id INTEGER,
            class_name TEXT,
            confidence REAL,
            x1 REAL,
            y1 REAL,
            x2 REAL,
            y2 REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            track_id INTEGER,
            message TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS asset_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL UNIQUE,
            active_seconds REAL DEFAULT 0,
            idle_seconds REAL DEFAULT 0,
            utilisation_percent REAL DEFAULT 0,
            updated_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_detection(timestamp, track_id, class_name, confidence, x1, y1, x2, y2):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO detections (timestamp, track_id, class_name, confidence, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp, track_id, class_name, confidence, x1, y1, x2, y2),
    )
    conn.commit()
    conn.close()


def insert_event(timestamp, event_type, severity, track_id, message):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events (timestamp, event_type, severity, track_id, message) VALUES (?, ?, ?, ?, ?)",
        (timestamp, event_type, severity, track_id, message),
    )
    conn.commit()
    conn.close()


def get_recent_events(limit=50):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM events ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_asset_metrics(track_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM asset_metrics WHERE track_id = ?",
        (track_id,),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_asset_metrics(track_id, active_seconds, idle_seconds, utilisation_percent):
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute(
        """
        INSERT INTO asset_metrics (track_id, active_seconds, idle_seconds, utilisation_percent, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(track_id) DO UPDATE SET
            active_seconds = excluded.active_seconds,
            idle_seconds = excluded.idle_seconds,
            utilisation_percent = excluded.utilisation_percent,
            updated_at = excluded.updated_at
        """,
        (track_id, active_seconds, idle_seconds, utilisation_percent, now),
    )
    conn.commit()
    conn.close()


def get_all_asset_metrics():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM asset_metrics")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_all_data():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM detections")
    cur.execute("DELETE FROM events")
    cur.execute("DELETE FROM asset_metrics")
    conn.commit()
    conn.close()
