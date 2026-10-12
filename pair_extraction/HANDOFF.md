CONTEXT HANDOFF — pair_extraction/s1bc_fused.py work
node d4055, conda env `shot`, cwd /projects/adsum-mllm-om/mllm/pair_extraction

WHAT THIS IS
Fused stage s1b (keyframe extract) + s1c (Swin3D embed) in one CPU->GPU pipeline.
Must produce embeddings byte-identical to the reference run that built data/fullmatch823.csv.

REFERENCE (byte-identity target):
- embeddings: /projects/adsum-mllm-om/data/0.1/embeddings/<brand_id>_<brand>_embeddings.h5
- clusters:   /projects/adsum-mllm-om/data/clusters/
- keyframes:  /projects/adsum-mllm-om/data/0.1/keyframes/<brand_id>/
- pairs:      /projects/adsum-mllm-om/data/fullmatch823.csv
- original byte-identical decode: pair_extraction/s1_keyframes.py::get_frames (whole-video decode + index)

DONE THIS SESSION (already in code):
1. PERF: thread oversubscription was making the run >24h (load 442 on 32 cores).
   Fixed in s1bc_fused.py: OMP/MKL/OPENBLAS_NUM_THREADS=1 at top of file;
   cv2.setNumThreads(0)+torch.set_num_threads(1) in _decode_worker. Run -> ~1h, ~2.3 it/s.
2. BYTE-IDENTITY BUG (root cause found + fixed): _decode_indices in s1_keyframes.py
   had used `select`+`vsync=0`+`vframes`. That remaps frame indices on VFR videos vs the
   default-vsync whole-video decode the shotsplits were built on -> wrong frames
   (keyframe maxpixdiff up to 195, embedding cosine ~0.78, fullmatch.csv missing 10/14
   pairs for the 7 test brands). REVERTED _decode_indices to decode whole video via
   get_frames() + index. Added -nostdin to ffmpeg (pixel-neutral; stops ffmpeg disabling
   terminal echo, which was freezing the shell). VERIFIED: 200/200 keyframes now
   byte-identical to data/0.1/keyframes/125/.
   Divergence map: shot detection MATCHES; break was at keyframe decode; embeddings/
   clusters/pairs all inherited it. Clustering was NOT the cause.

EDITED FILES (working tree only, not committed):
- pair_extraction/s1_keyframes.py  (_decode_indices reverted to whole-video decode; -nostdin on get_frames)
- pair_extraction/s1bc_fused.py     (thread caps at top + in _decode_worker)

TODO (in order):
1. (optional, recommended) End-to-end identity check on ONE ad: decode -> jpeg q95 ss0 ->
   transform_clip -> Swin3D forward, compare to data/0.1/.../*.h5 (expect cosine >= 0.9999).
2. Regenerate outputs made with the BROKEN decode (they're contaminated):
   - test set: /projects/adsum-mllm-om/data2/0.1_test/{embeddings,clusters,fullmatch.csv}
   - the batch1 fused run.
3. Re-verify fullmatch.csv vs data/fullmatch823.csv for the 7 test brands -> expect 0 extra, 0 missing.

NOT done (further perf levers if needed, only after byte-identity confirmed):
- ffmpeg children still spawn 84 threads each; cap with ffmpeg threads= .
- stage videos to node-local scratch (NFS IO waits in decode).
- job is decode-bound, so more CPU helps only sublinearly.

KNOWN-OK edge case: ~7 videos hit "select returned N-1" / frame-count mismatch (e.g.
Capital One 98f7DmVi0dI, EKU62BAfNFc, Uuupwns8XWE; Lovesac AP7YmUNHi8w, 1KG7Jj2gM3A,
Nf8YYfYAzvc, uQi1WvJDJRE). The reference drops these same videos too — expected, not a bug.
The reverted decode now drops only out-of-range shots instead of failing the whole video.

RE-RUN COMMAND (fused stage):
cd /projects/adsum-mllm-om/mllm/pair_extraction
export TORCH_HOME=/projects/adsum-mllm-om/.hf_cache/torch
python s1bc_fused.py --data_csv /projects/adsum-mllm-om/data2/batch1_videos_fix.csv \
  --num_decode_workers 24 --batch_size 256 --queue_size 64
(launch under tmux or with: nohup ... > logs/s1bc_fused.log 2>&1 &  so a bad tty on exit can't trap the shell)

Memory file with same context: ~/.claude/projects/-projects-adsum-mllm-om-mllm/memory/s1bc-fused-byte-identity.md
