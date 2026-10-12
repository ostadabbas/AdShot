import sqlite3
import time
from pathlib import Path

class PersistentLeakyLimiter:
    """
    Enforces a smooth average rate: one start every `period_sec`.
    Persists `next_allowed` in SQLite so it survives restarts and coordinates across processes.
    """
    def __init__(self, db_path: Path, period_sec: float):
        self.db_path = str(db_path)
        self.period = float(period_sec)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        conn = self._connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS state (
                k TEXT PRIMARY KEY,
                v REAL NOT NULL
            );
        """)
        # initialize next_allowed if missing
        now = time.time()
        conn.execute(
            "INSERT OR IGNORE INTO state(k, v) VALUES('next_allowed', ?);",
            (now,)
        )
        conn.commit()
        conn.close()

    def acquire(self):
        while True:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                now = time.time()
                row = conn.execute(
                    "SELECT v FROM state WHERE k='next_allowed';"
                ).fetchone()
                next_allowed = float(row[0]) if row else now

                if now < next_allowed:
                    wait = next_allowed - now
                    conn.commit()
                else:
                    # schedule the next allowed time
                    new_next = now + self.period
                    conn.execute(
                        "UPDATE state SET v=? WHERE k='next_allowed';",
                        (new_next,)
                    )
                    conn.commit()
                    return
            finally:
                conn.close()

            time.sleep(max(0.02, wait))


if __name__ == "__main__":
    def fmt(x): 
        return f"{x:.3f}s"

    import random
    db = Path("_leaky_test.sqlite")

    # Fresh run so old state doesn't affect this test
    if db.exists():
        db.unlink()

    period = 1.8
    rl = PersistentLeakyLimiter(db, period_sec=period)

    print("=== PersistentLeakyLimiter sanity check ===")
    print("DB:", db.resolve())
    print("Target period:", period, "sec\n")

    t_prev = time.perf_counter()
    for i in range(1, 6):
        rl.acquire()
        t_now = time.perf_counter()
        dt = t_now - t_prev
        time.sleep(random.uniform(0, 4))
        print(f"[same instance] acquire #{i}: dt={fmt(dt)}")
        t_prev = t_now

    print("\n=== New instance, same DB (persistence check) ===")
    rl2 = PersistentLeakyLimiter(db, period_sec=period)

    t0 = time.perf_counter()
    rl2.acquire()
    dt = time.perf_counter() - t0
    print(f"[new instance] acquire: dt={fmt(dt)}")

    print("\nExpected behavior:")
    print("- First acquire is ~0s")
    print(f"- Next acquires are ~{period}s (some jitter is normal)")

