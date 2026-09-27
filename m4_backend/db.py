from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "netguard.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS devices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        device_id TEXT NOT NULL UNIQUE,
        name TEXT,
        last_seen REAL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id TEXT,
        severity TEXT NOT NULL,
        message TEXT NOT NULL,
        source TEXT NOT NULL,
        resolved INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        device_id TEXT,
        status TEXT NOT NULL,
        forecast_json TEXT NOT NULL,
        created_at REAL NOT NULL
    );
    """)
    conn.commit()
    conn.close()

def create_user(email, password_hash):
    conn = get_conn()
    cur = conn.execute("INSERT INTO users(email,password_hash,created_at) VALUES(?,?,?)",
                       (email.lower().strip(), password_hash, time.time()))
    conn.commit()
    value = cur.lastrowid
    conn.close()
    return int(value)

def get_user(email):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE email=?", (email.lower().strip(),)).fetchone()
    conn.close()
    return dict(row) if row else None

def upsert_device(device_id, user_id=None, name=None):
    conn = get_conn()
    conn.execute("""
    INSERT INTO devices(user_id,device_id,name,last_seen) VALUES(?,?,?,?)
    ON CONFLICT(device_id) DO UPDATE SET
      user_id=COALESCE(excluded.user_id,devices.user_id),
      name=COALESCE(excluded.name,devices.name),
      last_seen=excluded.last_seen
    """, (user_id, device_id, name, time.time()))
    conn.commit()
    conn.close()

def insert_alert(device_id, severity, message, source):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO alerts(device_id,severity,message,source,created_at) VALUES(?,?,?,?,?)",
        (device_id, severity, message, source, time.time()))
    conn.commit()
    value = cur.lastrowid
    conn.close()
    return int(value)

def list_alerts(limit=50):
    conn = get_conn()
    rows = conn.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def create_incident(device_id, forecast, status="open"):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO incidents(device_id,status,forecast_json,created_at) VALUES(?,?,?,?)",
        (device_id, status, json.dumps(forecast), time.time()))
    conn.commit()
    value = cur.lastrowid
    conn.close()
    return int(value)

def get_incident(incident_id):
    conn = get_conn()
    row = conn.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
    conn.close()
    return dict(row) if row else None
