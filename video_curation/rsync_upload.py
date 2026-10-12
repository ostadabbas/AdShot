"""
Upload the video folders listed in a batch's videos CSV to the cluster, then verify with --checksum.

    python rsync_upload.py batch4      # run from WSL; reads LOGS_DIR/batch4_videos.csv
"""
import subprocess, time, datetime, sys
from pathlib import Path
import pandas as pd

from paths import DATA_ROOT as LOCAL_ROOT, LOGS_DIR, remote

REMOTE, DEST = remote()

MAX_RETRIES = 0  # 0 = infinite
BACKOFF = 10
MAX_BACKOFF = 300

log_path = f"{time.strftime('%Y%m%d_%H%M%S')}_rsync_selected.log"

SSH_OPTS = [
    "-o", "ServerAliveInterval=10",
    "-o", "ServerAliveCountMax=30",
    "-o", "TCPKeepAlive=yes",
]
RSYNC_SSH = " ".join(["ssh", *SSH_OPTS, "-o", "IPQoS=throughput"])

EXCLUDES = [
    "--exclude=*.part",
    "--exclude=*.tmp",
    "--exclude=*.ytdl",
    "--exclude=*.download",
]

def log(msg: str):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def run(cmd):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        return p.wait()

def read_targets(video_csv: Path):
    """Sorted unique parent folders of the 'relative_path' column, e.g. 'videomedia/Brand'."""
    if not video_csv.exists():
        raise FileNotFoundError(f"Missing csv file: {video_csv}")

    df = pd.read_csv(video_csv)
    if "relative_path" not in df.columns:
        raise ValueError(f"'relative_path' column not found in {video_csv}")

    folders = (
        df["relative_path"]
        .dropna()
        .map(lambda rp: str(Path(rp).parent).replace("\\", "/"))
        .loc[lambda x: (x != "") & (x != ".")]
        .unique()
    )
    if not len(folders):
        raise ValueError(f"No valid relative_path folders found in {video_csv}")
    return sorted(folders)

def rsync_one(rel_folder: str, verify_checksum: bool = False):
    # rel_folder like: videomedia/SN®  ->  remote DEST/videomedia/SN®/
    src = (LOCAL_ROOT / rel_folder).resolve()
    if not src.exists():
        print(f"❌ Local folder not found: {src}")
        return 2

    rel_parent = str(Path(rel_folder).parent).replace("\\", "/")
    remote_parent = f"{DEST}/{rel_parent}" if rel_parent != "." else DEST
    rc = run(["ssh", *SSH_OPTS, REMOTE, f"mkdir -p '{remote_parent}'"])
    if rc != 0:
        return rc

    cmd = ["rsync", "-aH", "--partial", "--inplace", "--info=progress2", *EXCLUDES, "-e", RSYNC_SSH]
    if verify_checksum:
        cmd += ["--checksum"]
    # trailing slash: copy folder contents into same folder name at dest
    cmd += [str(src) + "/", f"{REMOTE}:{DEST}/{rel_folder}/"]
    return run(cmd)

def sync_all(targets, label: str, verify_checksum: bool):
    """rsync every target, retrying each with exponential backoff until it succeeds."""
    for rel in targets:
        attempt, back = 0, BACKOFF
        while True:
            attempt += 1
            log(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {label} {rel} | Attempt #{attempt}")
            rc = rsync_one(rel, verify_checksum=verify_checksum)
            if rc == 0:
                print(f"✅ {label} OK: {rel}")
                break
            print(f"❌ {label} failed ({rc}) for {rel}. Sleeping {back}s...")
            time.sleep(back)
            back = min(back * 2, MAX_BACKOFF)
            if MAX_RETRIES and attempt >= MAX_RETRIES:
                sys.exit(rc)

def main(batch: str):
    run(["ssh", *SSH_OPTS, REMOTE, f"mkdir -p '{DEST}'"])
    targets = read_targets(LOGS_DIR / f"{batch}_videos.csv")
    print("Targets:")
    for t in targets:
        print(" -", t)

    sync_all(targets, "Upload", verify_checksum=False)       # fast pass
    print("\n=== Verification pass (--checksum) ===")
    sync_all(targets, "Verify", verify_checksum=True)        # confirms identical content

    print("\n✅ All selected folders uploaded + verified.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python rsync_upload.py <batch>   e.g. batch4")
    main(sys.argv[1])
