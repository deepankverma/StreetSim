# attach_sounds_street_randomized_v5.py
# Focus: attach ONE speaker per tree by targeting only "Tree_*_Branch_0(.###)"
# Extras:
#  - --tree_mode branch0|auto (default branch0). 'auto' groups by base id and prefers Branch_0 if present.
#  - --tree_clean true|false (default true) removes speakers parented to other tree parts (e.g., Branch_3.*).

import bpy, sys, os, random, re, math
from mathutils import Vector, Matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(path):
    if not path:
        return path
    path = os.path.expanduser(path)
    if path.startswith("//"):
        return bpy.path.abspath(path)
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)

# ---------------- CLI helpers ----------------
def arg(flag, default=None, as_float=False, as_int=False):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i+1 < len(sys.argv):
            v = sys.argv[i+1]
            if as_float:
                try: return float(v)
                except Exception: return float(default) if default is not None else None
            if as_int:
                try: return int(v)
                except Exception: return int(default) if default is not None else None
            return v
    return default

def arg_bool(flag, default=False):
    v = arg(flag, None)
    if v is None: return default
    s = str(v).strip().lower()
    return s in ("1","true","t","yes","y","on")

# ---------------- Config from CLI ----------------
DIR_CAR   = resolve_path(arg("--dir_car", os.path.join(SCRIPT_DIR, "sounds", "car")))
DIR_WALK  = resolve_path(arg("--dir_walk", os.path.join(SCRIPT_DIR, "sounds", "walk")))
DIR_RUN   = resolve_path(arg("--dir_run", os.path.join(SCRIPT_DIR, "sounds", "run")))
DIR_BIRD  = resolve_path(arg("--dir_bird", os.path.join(SCRIPT_DIR, "sounds", "bird")))
DIR_WIND  = resolve_path(arg("--dir_wind", os.path.join(SCRIPT_DIR, "sounds", "wind")))
DIR_AMB   = resolve_path(arg("--dir_amb", os.path.join(SCRIPT_DIR, "sounds", "amb")))

SEED      = arg("--seed", None, as_int=True)
_AMB_COUNT_ARG = arg("--amb_count", 1, as_int=True)
AMB_COUNT = max(0, 1 if _AMB_COUNT_ARG is None else int(_AMB_COUNT_ARG))
TREE_PROB = max(0.0, min(1.0, arg("--tree_prob", 0.30, as_float=True) or 0.30))
OUT_PATH  = arg("--outblend", None)

CAR_MAX   = arg("--car_max", 120.0, as_float=True) or 120.0
PED_MAX   = arg("--ped_max", 60.0, as_float=True) or 60.0
TREE_MAX  = arg("--tree_max", 80.0, as_float=True) or 80.0
AMB_MAX   = arg("--amb_max", 200.0, as_float=True) or 200.0

CAR_ATT   = arg("--car_att", 0.7, as_float=True) or 0.7
PED_ATT   = arg("--ped_att", 1.0, as_float=True) or 1.0
TREE_ATT  = arg("--tree_att", 0.9, as_float=True) or 0.9
AMB_ATT   = arg("--amb_att", 0.5, as_float=True) or 0.5

TREE_MODE  = (arg("--tree_mode", "branch0") or "branch0").lower()  # branch0 | auto
TREE_CLEAN = arg_bool("--tree_clean", True)

# Phase randomization (seconds). Set to 0 to disable random start phase.
PHASE_MAX_S = arg("--phase_max_s", 12.0, as_float=True)
PHASE_MAX_S = 0.0 if PHASE_MAX_S is None else float(PHASE_MAX_S)

AUDIO_EXTS = {".wav", ".ogg", ".mp3", ".flac", ".aif", ".aiff"}
BRANCH0_RE = re.compile(r'^Tree_.*_Branch_0(?:\.\d+)?$')

# ---------------- Small utils ----------------
def _safe_set(obj, attr, val):
    try: setattr(obj, attr, val)
    except Exception: pass

def list_audio(dirpath):
    if not dirpath: return []
    dp = bpy.path.abspath(dirpath)
    if not os.path.isdir(dp):
        print(f"[Audio] Directory not found: {dp}")
        return []
    return sorted([os.path.join(dp, fn) for fn in os.listdir(dp)
                   if os.path.splitext(fn)[1].lower() in AUDIO_EXTS])

def ensure_loop(sound, speaker_data):
    if sound:
        for a in ("use_loop", "loop"):
            if hasattr(sound, a):
                _safe_set(sound, a, True)
    if speaker_data:
        for a in ("use_loop", "loop"):
            if hasattr(speaker_data, a):
                _safe_set(speaker_data, a, True)

def load_sound(path):
    abspath = bpy.path.abspath(path)
    name = os.path.basename(abspath)
    snd = bpy.data.sounds.get(name)
    if snd is None:
        try:
            snd = bpy.data.sounds.load(abspath, check_existing=True)
        except Exception as e:
            print(f"[Audio] Could not load sound: {abspath} ({e})")
            return None
    for a, v in (("use_memory_cache", True), ("use_mono", True)):
        if hasattr(snd, a): _safe_set(snd, a, v)
    ensure_loop(snd, None)
    return snd

# ---------------- Timeline helpers ----------------
def _scene_start():
    scn = bpy.context.scene
    return int(getattr(scn, "frame_start", 0))

def _scene_end():
    scn = bpy.context.scene
    return int(getattr(scn, "frame_end", 250))

def _fps_real():
    scn = bpy.context.scene
    fps  = float(getattr(scn.render, "fps", 24))
    base = float(getattr(scn.render, "fps_base", 1.0) or 1.0)
    return fps / base

def set_playhead_to_scene_start():
    scn = bpy.context.scene
    s = _scene_start()
    try: scn.use_preview_range = False
    except: pass
    try:
        scn.frame_current = s
        scn.frame_set(s)  # creation ops use current frame
    except Exception:
        pass

def retime_all_speaker_soundclips(start_frame=None, loop_to_scene=True):
    """
    Move every Speaker's NLA SOUND strip to 'start_frame' (defaults to scene start)
    and optionally repeat it to cover the full timeline.
    """
    s = _scene_start() if start_frame is None else int(start_frame)
    e = _scene_end()
    for spk in [o for o in bpy.data.objects if o.type == 'SPEAKER']:
        ad = getattr(spk.data, "animation_data", None)
        if not ad: 
            continue
        for tr in getattr(ad, "nla_tracks", []):
            for st in getattr(tr, "strips", []):
                if getattr(st, "type", None) == 'SOUND':
                    try:
                        # keep strip length, just reposition
                        length = max(1.0, (st.frame_end - st.frame_start))
                        st.frame_start = float(s)
                        st.mute = False
                        if loop_to_scene:
                            total  = max(0.0, (e - s))
                            st.repeat = max(1.0, (total / length) + 1.0)
                    except Exception:
                        pass

def randomize_speaker_start_offsets(max_offset_s=3.0, seed=None):
    """
    Randomize each SPEAKER's phase by moving its NLA SOUND strip to start BEFORE scene start
    so at frame 0 you're already mid-loop. Uses 'aud' to cap by real length when available.
    """
    if not max_offset_s or max_offset_s <= 0.0:
        return
    try:
        import aud
    except Exception:
        aud = None

    s = _scene_start()
    e = _scene_end()
    fr = _fps_real()
    rng = random.Random(seed) if seed is not None else random
    changed = 0
    ambience_offsets = []

    for spk in [o for o in bpy.data.objects if o.type == 'SPEAKER']:
        sd = spk.data
        snd = getattr(sd, "sound", None)
        if snd is None:
            continue

        # Estimate duration (seconds)
        dur_s = max_offset_s
        if aud is not None and hasattr(snd, "filepath"):
            try:
                sndobj = aud.Sound(bpy.path.abspath(snd.filepath))
                if math.isfinite(sndobj.length) and sndobj.length > 0.0:
                    dur_s = float(sndobj.length)
            except Exception:
                pass

        off_s = rng.uniform(0.0, min(max_offset_s, dur_s))
        off_f = off_s * fr
        clip_len_f = max(1.0, dur_s * fr)

        ad = getattr(sd, "animation_data", None)
        if not ad:
            continue
        for tr in getattr(ad, "nla_tracks", []):
            for st in getattr(tr, "strips", []):
                if getattr(st, "type", None) == 'SOUND':
                    try:
                        # start early by 'off_f' so at frame s we are already offset into the clip
                        st.frame_start = float(s) - off_f
                        st.mute = False
                        total_needed = (e - (s - off_f))
                        st.repeat = max(1.0, (total_needed / clip_len_f) + 1.0)
                        changed += 1
                        if spk.name.startswith("Ambience_"):
                            ambience_offsets.append(
                                f"{spk.name}:{os.path.basename(getattr(snd, 'filepath', '') or snd.name)}@{off_s:.2f}s"
                            )
                    except Exception:
                        pass
    print(f"[Audio] Randomized start phase on {changed} speaker strip(s), max={max_offset_s:.2f}s.")
    if ambience_offsets:
        print("[Audio] Ambience phases: " + ", ".join(ambience_offsets))

def normalize_timeline_audio(frame_start=0):
    scn = bpy.context.scene
    # ensure we truly start at requested frame and aren't constrained by preview range
    try: scn.use_preview_range = False
    except: pass
    if hasattr(scn, "frame_start"):
        scn.frame_start = frame_start
    if hasattr(scn, "frame_current"):
        scn.frame_current = frame_start
    try: scn.sync_mode = 'AUDIO_SYNC'
    except: pass
    if hasattr(scn, "render"):
        _safe_set(scn.render, "use_audio", True)
        _safe_set(scn.render, "use_audio_scrub", True)
    # purge any VSE sound strips that could mask 3D speaker audio
    se = getattr(scn, "sequence_editor", None)
    if se and hasattr(se, "sequences_all"):
        for strip in list(se.sequences_all):
            if getattr(strip, "type", "") == "SOUND":
                try: se.sequences.remove(strip)
                except: pass
    # enforce loop flags and zero known offsets on speakers + sound datablocks
    for spk in [o for o in bpy.data.objects if o.type == 'SPEAKER']:
        sd = spk.data
        _safe_set(sd, "muted", False)
        for a in ("use_loop", "loop"):
            if hasattr(sd, a): _safe_set(sd, a, True)
        for a in ("delay", "time", "offset_time", "time_offset"):
            if hasattr(sd, a): _safe_set(sd, a, 0.0)
        snd = getattr(sd, "sound", None)
        if snd:
            for a in ("use_loop", "loop"):
                if hasattr(snd, a): _safe_set(snd, a, True)
            if hasattr(snd, "use_memory_cache"): _safe_set(snd, "use_memory_cache", True)
            if hasattr(snd, "use_mono"):         _safe_set(snd, "use_mono", True)
    try: scn.frame_set(frame_start)
    except: pass
    print("[Audio] Timeline normalized: preview off, VSE cleared, loops on, offsets zeroed, frame set.")

# ---------------- Speaker creation ----------------
def make_speaker(name, location=(0,0,0), parent=None, sound=None,
                 volume=1.0, pitch=1.0, distance_reference=1.5,
                 distance_max=40.0, attenuation=1.0, angle_outer=360.0):
    # Ensure the SOUND strip is created at scene start
    set_playhead_to_scene_start()

    bpy.ops.object.speaker_add(location=location)
    spk = bpy.context.object
    spk.name = name
    if parent:
        spk.parent = parent
        # Anchor to parent's origin (no offset drift)
        spk.matrix_parent_inverse = Matrix.Identity(4)
        spk.location = (0.0, 0.0, 0.0)

    if sound: spk.data.sound = sound
    _safe_set(spk.data, "volume", float(volume))
    _safe_set(spk.data, "pitch", float(pitch))
    _safe_set(spk.data, "muted", False)
    _safe_set(spk.data, "distance_reference", float(distance_reference))
    _safe_set(spk.data, "distance_max", float(distance_max))
    _safe_set(spk.data, "attenuation", float(attenuation))
    _safe_set(spk.data, "cone_angle_outer", float(angle_outer))
    _safe_set(spk.data, "cone_angle_inner", 360.0)
    _safe_set(spk.data, "cone_volume_outer", 0.0)
    ensure_loop(sound, spk.data)

    # Ensure just-created SOUND strips start at scene start & will repeat
    try:
        retime_all_speaker_soundclips(start_frame=_scene_start(), loop_to_scene=True)
    except Exception:
        pass

    return spk

# ---------------- Scene audio defaults ----------------
def configure_scene_audio():
    scn = bpy.context.scene
    try: scn.sync_mode = 'AUDIO_SYNC'
    except Exception: pass
    a = getattr(scn, "audio", None)
    if a is not None:
        for (attr, val) in (
            ("distance_model", 'INVERSE_CLAMPED'),
            ("use_doppler", True),
            ("doppler_factor", 1.0),
            ("speed_of_sound", 343.3),
        ):
            _safe_set(a, attr, val)
    for (attr, val) in (
        ("audio_distance_model", 'INVERSE_CLAMPED'),
        ("audio_doppler_factor", 1.0),
        ("audio_doppler_speed", 343.3),
    ):
        if hasattr(scn, attr): _safe_set(scn, attr, val)
    if hasattr(scn, "render"):
        _safe_set(scn.render, "use_audio", True)
        _safe_set(scn.render, "use_audio_scrub", True)

# ---------------- Movers discovery ----------------
def find_movers_and_kinds():
    cars, peds = [], []
    for o in bpy.data.objects:
        if not o.name.endswith("_MOVE"): continue
        if o.name.startswith("Car_"):
            cars.append(o)
        elif o.name.startswith("Human_"):
            kind = "walk"
            arm = next((c for c in o.children if c.type == 'ARMATURE'), None)
            if arm and "kind" in arm:
                kind = str(arm["kind"]).lower()
            else:
                low = o.name.lower()
                if any(k in low for k in ("run","jog","sprint")):
                    kind = "run"
            peds.append((o, "run" if kind == "run" else "walk"))
    return dict(cars=cars, peds=peds)

# --------- Tree root detection ----------
def base_id_from_name(name: str):
    if "_Branch_" in name:
        return name.split("_Branch_")[0]
    return None

def find_tree_roots(mode="branch0"):
    if mode == "branch0":
        return [o for o in bpy.data.objects if BRANCH0_RE.match(o.name)]
    buckets = {}
    for o in bpy.data.objects:
        if not o.name.startswith("Tree_"):
            continue
        base = base_id_from_name(o.name)
        if not base:
            continue
        buckets.setdefault(base, []).append(o)
    roots = []
    for base, parts in buckets.items():
        b0 = [p for p in parts if BRANCH0_RE.match(p.name)]
        if b0:
            roots.append(b0[0]); continue
        parts_sorted = sorted(parts, key=lambda p: (0 if p.parent is None else 1, len(p.name)))
        roots.append(parts_sorted[0])
    return roots

def cleanup_tree_speakers(valid_roots):
    valid_ids = {id(o) for o in valid_roots}
    removed = 0
    for o in list(bpy.data.objects):
        if o.type != 'SPEAKER': continue
        par = o.parent
        if not par or not par.name.startswith("Tree_"): continue
        if id(par) not in valid_ids:
            try:
                bpy.data.objects.remove(o, do_unlink=True); removed += 1
            except Exception:
                pass
    if removed:
        print(f"[Clean] Removed {removed} tree-attached speaker(s) not on root parts].")

# --------- Attachments ----------
def attach_vehicle_speakers(car_movers, car_files):
    if not car_movers or not car_files:
        print("[Attach] Cars skipped (no movers or no car sounds).")
        return 0
    col = bpy.data.collections.get("Audio_Cars") or bpy.data.collections.new("Audio_Cars")
    if col.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(col)
    n=0
    for mv in car_movers:
        snd = load_sound(random.choice(car_files))
        if not snd: continue
        spk = make_speaker(mv.name.replace("_MOVE","_SPK"), parent=mv, sound=snd,
                           volume=random.uniform(0.7, 1.0),
                           pitch=random.uniform(0.97, 1.05),
                           distance_reference=1.6, distance_max=CAR_MAX, attenuation=CAR_ATT)
        try:
            for c in list(spk.users_collection): c.objects.unlink(spk)
            col.objects.link(spk)
        except Exception: pass
        n+=1
    print(f"[Attach] Vehicle speakers: {n}")
    return n

def attach_ped_speakers(ped_pairs, walk_files, run_files):
    if not ped_pairs or (not walk_files and not run_files):
        print("[Attach] Peds skipped (no movers or no ped sounds).")
        return 0
    col = bpy.data.collections.get("Audio_Peds") or bpy.data.collections.new("Audio_Peds")
    if col.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(col)
    n=0
    for mv, kind in ped_pairs:
        pool = run_files if (kind == "run" and run_files) else (walk_files if walk_files else run_files)
        if not pool: continue
        snd = load_sound(random.choice(pool))
        if not snd: continue
        if kind == "run":
            vol, pitch = random.uniform(0.7, 1.1), random.uniform(1.02, 1.10)
        else:
            vol, pitch = random.uniform(0.5, 0.9), random.uniform(0.96, 1.05)
        spk = make_speaker(mv.name.replace("_MOVE","_SPK"), parent=mv, sound=snd,
                           volume=vol, pitch=pitch,
                           distance_reference=1.2, distance_max=PED_MAX, attenuation=PED_ATT)
        try:
            for c in list(spk.users_collection): c.objects.unlink(spk)
            col.objects.link(spk)
        except Exception: pass
        n+=1
    print(f"[Attach] Pedestrian speakers: {n}")
    return n

def attach_tree_speakers(bird_files, wind_files, prob=0.30):
    """
    ONE looping speaker per Tree_*_Branch_0(.###), at the tree object's ORIGIN.
    Probability 'prob' controls how many trees get a speaker. Safe to re-run.
    """
    if not ((bird_files and len(bird_files)) or (wind_files and len(wind_files))):
        print("[Attach] Trees skipped (no bird/wind sounds).")
        return 0

    # Ensure / get collection
    col = bpy.data.collections.get("Audio_Trees")
    if not col:
        col = bpy.data.collections.new("Audio_Trees")
        bpy.context.scene.collection.children.link(col)

    roots = find_tree_roots(TREE_MODE)
    if TREE_CLEAN:
        cleanup_tree_speakers(roots)

    def pick_tree_clip():
        if bird_files and wind_files:
            return random.choice(bird_files) if random.random() < 0.6 else random.choice(wind_files)
        return random.choice(bird_files) if bird_files else random.choice(wind_files)

    placed = 0
    for t in roots:
        if random.random() > prob:
            continue
        if any(ch.type == 'SPEAKER' for ch in t.children):
            continue

        snd_path = pick_tree_clip()
        snd = load_sound(snd_path) if snd_path else None
        if not snd:
            continue

        # Create, parent, zero local (land exactly at tree origin)
        bpy.ops.object.speaker_add()
        spk = bpy.context.object
        spk.name = t.name + "_SPK"
        spk.data.sound = snd
        ensure_loop(snd, spk.data)

        spk.parent = t
        spk.matrix_parent_inverse = Matrix.Identity(4)
        spk.location = (0.0, 0.0, 0.0)

        # Outdoor falloff
        _safe_set(spk.data, "volume", random.uniform(0.4, 0.9))
        _safe_set(spk.data, "pitch",  random.uniform(0.96, 1.05))
        for (attr, val) in (
            ("distance_reference", 2.0),
            ("distance_max",       float(TREE_MAX)),
            ("attenuation",        float(TREE_ATT)),
            ("cone_angle_inner",   360.0),
            ("cone_angle_outer",   360.0),
            ("cone_volume_outer",  0.0),
        ):
            if hasattr(spk.data, attr): _safe_set(spk.data, attr, val)

        # Move to collection
        try:
            for c in list(spk.users_collection): c.objects.unlink(spk)
            col.objects.link(spk)
        except: pass

        placed += 1

    print(f"[Attach] Tree speakers placed on {placed} root(s) (mode={TREE_MODE}, p={prob}).")
    return placed

def attach_ambience_speakers(amb_files, count=2):
    if not amb_files or count <= 0:
        print("[Attach] Ambience skipped (no files or count=0).")
        return 0
    col = bpy.data.collections.get("Audio_Ambience") or bpy.data.collections.new("Audio_Ambience")
    if col.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(col)
    def world_bbox(o):
        M = o.matrix_world
        return [M @ Vector(c) for c in o.bound_box]
    def y_span(o):
        pts = world_bbox(o); ys = [p.y for p in pts]
        return (min(ys), max(ys))
    candidates = [o for o in bpy.data.objects if o.type == "MESH" and any(k in o.name.lower() for k in ("driveway","lane","road","street","footpath","sidewalk"))]
    y0,y1 = -20.0, 20.0
    if candidates:
        y0 = min(y_span(o)[0] for o in candidates)
        y1 = max(y_span(o)[1] for o in candidates)
    length = max(40.0, y1 - y0)
    xs = [0.0] * count
    ys = [y0 + (i+0.5)*(length/count) for i in range(count)]
    zs = [4.0] * count
    amb_pool = list(amb_files)
    random.shuffle(amb_pool)
    selected = []
    n=0
    for i in range(count):
        if not amb_pool:
            amb_pool = list(amb_files)
            random.shuffle(amb_pool)
        snd_path = amb_pool.pop()
        selected.append(os.path.basename(snd_path))
        snd = load_sound(snd_path)
        if not snd: continue
        spk = make_speaker(f"Ambience_{i+1:02d}", location=(xs[i], ys[i], zs[i]), parent=None, sound=snd,
                           volume=random.uniform(0.25,0.55), pitch=random.uniform(0.98,1.02),
                           distance_reference=5.0, distance_max=AMB_MAX, attenuation=AMB_ATT)
        try:
            for c in list(spk.users_collection): c.objects.unlink(spk)
            col.objects.link(spk)
        except Exception: pass
        n+=1
    print(f"[Attach] Ambience bed speakers: {n}")
    if selected:
        print("[Attach] Ambience clips: " + ", ".join(selected))
    return n

# ---------------- Main ----------------
def main():
    if SEED is not None: random.seed(SEED)
    configure_scene_audio()

    # Make sure any speaker created from now on gets its SOUND strip at the scene start
    set_playhead_to_scene_start()

    car_files  = list_audio(DIR_CAR)
    walk_files = list_audio(DIR_WALK)
    run_files  = list_audio(DIR_RUN)
    bird_files = list_audio(DIR_BIRD)
    wind_files = list_audio(DIR_WIND)
    amb_files  = list_audio(DIR_AMB)

    print("[Audio] Found:",
          f"{len(car_files)} car, {len(walk_files)} walk, {len(run_files)} run,",
          f"{len(bird_files)} bird, {len(wind_files)} wind, {len(amb_files)} ambience clips.")

    movers = find_movers_and_kinds()
    cars = movers["cars"]; peds = movers["peds"]

    attach_vehicle_speakers(cars, car_files)
    attach_ped_speakers(peds, walk_files, run_files)
    attach_tree_speakers(bird_files, wind_files, prob=TREE_PROB)
    attach_ambience_speakers(amb_files, count=AMB_COUNT)

    # Normalize timeline (disables preview range, clears VSE SOUND strips, enforces loop flags)
    normalize_timeline_audio(frame_start=0)

    # Ensure all existing speakers (cars/peds/trees/amb) cover the full timeline.
    retime_all_speaker_soundclips(start_frame=_scene_start(), loop_to_scene=True)

    # Phase randomization must happen last; otherwise final retiming resets all clips to 0.00.
    if PHASE_MAX_S > 0.0:
        randomize_speaker_start_offsets(max_offset_s=PHASE_MAX_S, seed=SEED)

    if OUT_PATH:
        out = bpy.path.abspath(OUT_PATH)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        bpy.ops.wm.save_mainfile(filepath=out)
        print(f"[IO] Saved with audio to: {out}")
    else:
        print("[IO] No --out provided; leaving file open.")

if __name__ == "__main__":
    main()
