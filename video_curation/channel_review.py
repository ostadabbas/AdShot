import os, json, time, tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Channel Review", layout="wide")

# ----------------------------
# Atomic write (Windows-safe)
# ----------------------------
def atomic_write_json(path: Path, obj: dict, retries: int = 15, delay: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + ".", suffix=".tmp.json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        for _ in range(retries):
            try:
                os.replace(tmp_name, str(path))
                return
            except PermissionError:
                time.sleep(delay)

        # fallback (rare)
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def load_flags(flag_path: Path) -> dict:
    if not flag_path.exists():
        return {"version": 1, "channels": {}}
    d = load_json(flag_path)
    d.setdefault("version", 1)
    d.setdefault("channels", {})
    return d

def get_channel_id(ch: dict) -> str:
    return ch.get("channel_id") or ch.get("channelId") or ""

def get_handle(ch: dict) -> str:
    return ch.get("handle") or ch.get("customUrl") or ""

def build_df(raw: dict, flags: dict) -> pd.DataFrame:
    """
    One row per channel_id.
    If channel appears in both featured+search => source='both'
    """
    fmap = flags.get("channels", {})

    tmp = {}  # channel_id -> record

    def add(list_key: str, src: str):
        for rank, ch in enumerate(raw.get(list_key, []) or [], start=1):
            cid = get_channel_id(ch)
            if not cid:
                continue
            handle = get_handle(ch)
            title = ch.get("title") or ""
            desc = ch.get("description") or ""

            if cid not in tmp:
                tmp[cid] = {
                    "channel_id": cid,
                    "title": title,
                    "handle": handle,
                    "description": desc,
                    "featured_rank": None,
                    "search_rank": None,
                    "source": src,
                }
            else:
                # merge source
                prev = tmp[cid]["source"]
                if prev != src:
                    tmp[cid]["source"] = "both"
                # keep the richer title/desc if needed
                if len((tmp[cid]["description"] or "")) < len(desc or ""):
                    tmp[cid]["description"] = desc
                if len((tmp[cid]["title"] or "")) < len(title or ""):
                    tmp[cid]["title"] = title
                if not tmp[cid]["handle"] and handle:
                    tmp[cid]["handle"] = handle

            if src == "featured" and tmp[cid]["featured_rank"] is None:
                tmp[cid]["featured_rank"] = rank
            if src == "search" and tmp[cid]["search_rank"] is None:
                tmp[cid]["search_rank"] = rank

    add("featured_channels", "featured")
    add("searched_channels", "search")

    rows = []
    for cid, r in tmp.items():
        include = fmap.get(cid, {}).get("include", True)  # default ✅
        source = r["source"]
        badge = "🟩 featured" if source == "featured" else ("🟨 search" if source == "search" else "🟩🟨 both")

        rows.append({
            "include": bool(include),
            "badge": badge,
            "source": source,
            "featured_rank": r["featured_rank"],
            "search_rank": r["search_rank"],
            "title": r["title"],
            "handle": r["handle"],
            "channel_url": f"https://www.youtube.com/channel/{cid}",
            "handle_url": f"https://www.youtube.com/{r['handle']}" if r["handle"] else "",
            "description": r["description"],
            "channel_id": cid,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # stable ordering: featured first, then both, then search; and ranks inside
    src_order = {"featured": 0, "both": 1, "search": 2}
    df["__src_order"] = df["source"].map(src_order).fillna(9).astype(int)
    df["__rank"] = df["featured_rank"].fillna(df["search_rank"]).fillna(10**9).astype(float)
    df = df.sort_values(["__src_order", "__rank", "title"], kind="mergesort").drop(columns=["__src_order", "__rank"])
    df = df.reset_index(drop=True)
    return df

def compute_updates(edited: pd.DataFrame, flags: dict) -> dict:
    existing = flags.get("channels", {})
    now = datetime.now().isoformat(timespec="seconds")
    updates = {}

    for _, row in edited.iterrows():
        cid = str(row["channel_id"])
        if not cid:
            continue
        new_inc = bool(row["include"])
        old_inc = existing.get(cid, {}).get("include", True)
        if new_inc != old_inc:
            updates[cid] = {"include": new_inc, "updated_at": now}

    return updates

# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("Files")
    raw_path = Path(st.text_input("Raw channels JSON (read-only)", "data/channels/channels_raw.json"))
    flag_path = Path(st.text_input("Flags JSON (writable)", "data/channels/flag.json"))
    export_path = Path(st.text_input("Export pruned JSON", "out/channels_pruned.json"))

st.title("Channel Review")

if not raw_path.exists():
    st.error(f"Raw file not found: {raw_path}")
    st.stop()

raw = load_json(raw_path)
flags = load_flags(flag_path)
df = build_df(raw, flags)

if df.empty:
    st.warning("No channels found in raw JSON.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", len(df))
c2.metric("Included", int(df["include"].sum()))
c3.metric("Excluded", int((~df["include"]).sum()))
c4.metric("Flag file entries", len(flags.get("channels", {})))

# Filters (single place)
fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
q = fcol1.text_input("Search", "")
src_filter = fcol2.multiselect("Source", ["featured", "both", "search"], default=["featured", "both", "search"])
only_excluded = fcol3.checkbox("Only excluded", value=False)

mask = df["source"].isin(src_filter)
if q.strip():
    qq = q.lower()
    mask &= (
        df["title"].fillna("").str.lower().str.contains(qq) |
        df["handle"].fillna("").str.lower().str.contains(qq) |
        df["description"].fillna("").str.lower().str.contains(qq)
    )
if only_excluded:
    mask &= (df["include"] == False)

view = df[mask].copy()

st.caption("🟩 featured is safer; 🟨 search is noisier. Checkbox decides inclusion. Default is included ✅.")

edited = st.data_editor(
    view,
    use_container_width=True,
    num_rows="fixed",
    key="one_editor",
    column_config={
        "include": st.column_config.CheckboxColumn("include"),
        "channel_url": st.column_config.LinkColumn("channel_url"),
        "handle_url": st.column_config.LinkColumn("handle_url"),
        "description": st.column_config.TextColumn("description", width="large"),
        "badge": st.column_config.TextColumn(""),
        "channel_id": st.column_config.TextColumn("channel_id", disabled=True),
        "featured_rank": st.column_config.NumberColumn("featured_rank", disabled=True),
        "search_rank": st.column_config.NumberColumn("search_rank", disabled=True),
        "source": st.column_config.TextColumn("source", disabled=True),
    },
    disabled=["badge", "source", "featured_rank", "search_rank", "channel_id"],
)

colA, colB = st.columns([1, 1])

if colA.button("Save flags"):
    updates = compute_updates(edited, flags)

    # Only touch changed ids; everything else in flag.json stays as-is.
    if updates:
        flags["channels"].update(updates)
        atomic_write_json(flag_path, flags)
        st.success(f"Saved {len(updates)} changed ids to {flag_path}")
    else:
        st.info("No changes detected.")

if colB.button("Export pruned JSON"):
    flags_now = load_flags(flag_path)
    df_now = build_df(raw, flags_now)
    kept = df_now[df_now["include"] == True].copy()

    out = kept[["source", "featured_rank", "search_rank", "title", "handle", "channel_id", "description"]].to_dict(orient="records")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    st.success(f"Wrote {export_path} ({len(out)} channels)")
