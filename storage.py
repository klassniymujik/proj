"""
Модуль хранения данных.

Использует SQLite (через aiosqlite) для простоты развёртывания.
Для production-сред легко переключается на PostgreSQL
через замену строки подключения (DATABASE_URL).

Таблицы:
  scenes      — конфигурация сцены (размеры зала)
  cameras     — параметры камер
  homographies — точки калибровки между камерами
  snapshots   — агрегированные аналитические снимки (каждые N секунд)
  tracks      — траектории объектов
"""

import json
import time
import sqlite3
import os

DB_PATH = os.environ.get("DATABASE_URL", "analytics.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS scenes (
            id INTEGER PRIMARY KEY,
            width REAL NOT NULL DEFAULT 20,
            height REAL NOT NULL DEFAULT 20,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scene_id INTEGER NOT NULL DEFAULT 1,
            x REAL, y REAL, height REAL,
            yaw REAL, pitch REAL, fov REAL,
            img_width INTEGER, img_height INTEGER,
            label TEXT,
            address TEXT DEFAULT '',
            FOREIGN KEY(scene_id) REFERENCES scenes(id)
        );

        CREATE TABLE IF NOT EXISTS homographies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cam_a INTEGER NOT NULL,
            cam_b INTEGER NOT NULL,
            points_a TEXT NOT NULL,
            points_b TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            visitor_count INTEGER NOT NULL DEFAULT 0,
            active_tracks INTEGER NOT NULL DEFAULT 0,
            heatmap TEXT
        );

        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            ts REAL NOT NULL,
            world_x REAL,
            world_y REAL,
            cam_idx INTEGER DEFAULT 0
        );
    """)

    cur.execute("INSERT OR IGNORE INTO scenes(id, width, height, updated_at) VALUES(1, 20, 20, ?)",
                (time.time(),))

    existing = [row[1] for row in cur.execute("PRAGMA table_info(cameras)").fetchall()]
    if "address" not in existing:
        cur.execute("ALTER TABLE cameras ADD COLUMN address TEXT DEFAULT ''")

    conn.commit()
    conn.close()


# ---- Scene ----

def save_scene(width: float, height: float):
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO scenes(id, width, height, updated_at) VALUES(1, ?, ?, ?)",
        (width, height, time.time())
    )
    conn.commit()
    conn.close()


def load_scene():
    conn = get_connection()
    row = conn.execute("SELECT * FROM scenes WHERE id=1").fetchone()
    conn.close()
    return dict(row) if row else {"id": 1, "width": 20, "height": 20}


# ---- Cameras ----

def save_cameras(cameras: list):
    conn = get_connection()
    conn.execute("DELETE FROM cameras WHERE scene_id=1")
    for cam in cameras:
        conn.execute("""
            INSERT INTO cameras(scene_id, x, y, height, yaw, pitch, fov, img_width, img_height, label, address)
            VALUES(1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cam.get("x", 0), cam.get("y", 0), cam.get("height", 3.0),
            cam.get("yaw", 0), cam.get("pitch", -60), cam.get("fov", 90),
            cam.get("img_width", 1280), cam.get("img_height", 720),
            cam.get("label", ""), cam.get("address", "")
        ))
    conn.commit()
    conn.close()


def load_cameras():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM cameras WHERE scene_id=1").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- Homography ----

def save_homography(cam_a: int, cam_b: int, points_a: list, points_b: list):
    conn = get_connection()
    conn.execute("DELETE FROM homographies WHERE cam_a=? AND cam_b=?", (cam_a, cam_b))
    conn.execute(
        "INSERT INTO homographies(cam_a, cam_b, points_a, points_b, created_at) VALUES(?, ?, ?, ?, ?)",
        (cam_a, cam_b, json.dumps(points_a), json.dumps(points_b), time.time())
    )
    conn.commit()
    conn.close()


def load_homographies():
    conn = get_connection()
    rows = conn.execute("SELECT cam_a, cam_b, points_a, points_b FROM homographies").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- Snapshots ----

def save_snapshot(visitor_count: int, active_tracks: int, heatmap: list):
    conn = get_connection()
    conn.execute(
        "INSERT INTO snapshots(ts, visitor_count, active_tracks, heatmap) VALUES(?, ?, ?, ?)",
        (time.time(), visitor_count, active_tracks, json.dumps(heatmap))
    )
    conn.commit()
    conn.close()


def load_snapshots(limit=100):
    conn = get_connection()
    rows = conn.execute(
        "SELECT ts, visitor_count, active_tracks FROM snapshots ORDER BY ts DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- Track points ----

def save_track_points(points: list, cam_idx: int = 0):
    if not points:
        return
    conn = get_connection()
    ts = time.time()
    conn.executemany(
        "INSERT INTO tracks(track_id, ts, world_x, world_y, cam_idx) VALUES(?, ?, ?, ?, ?)",
        [(p["id"], ts, p.get("x"), p.get("y"), cam_idx) for p in points]
    )
    conn.commit()
    conn.close()
