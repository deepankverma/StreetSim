# animate_mixamo_peds_and_cars_v2.py — Mixamo pedestrians (in-place safe) + cars
# -------------------------------------------------------------------------------
# Usage (PowerShell):
# blender -b "D:\downloads\test_export_driveways_new.blend" `
#   -P "D:\downloads\animate_mixamo_peds_and_cars_v2.py" -- `
#   --mixamo_dir "D:\downloads\mixamo_fbx" `
#   --fps 24 --duration 20 `
#   --peds_move_ratio 1.0 `
#   --walk_fallback_mps 1.5 --run_fallback_mps 3.5 `
#   --speed_scale_min 0.95 --speed_scale_max 1.05 `
#   --cars_move_ratio 1.0 `
#   --speed_min_car 6.0 --speed_max_car 12.0 `
#   --dir_left_ped + --dir_right_ped - `
#   --dir_left_car + --dir_right_car - `
#   --out "D:\downloads\test_export_driveways_new_anim.blend"
# -------------------------------------------------------------------------------



# animate_mixamo_peds_and_cars_height_orient.py
# Blender 4.4+ — Mixamo pedestrians (exact 1.70 m height + proper facing + robust loop) & cars

import bpy, sys, os, math, random
from mathutils import Vector, Euler
from math import ceil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(path):
    if not path:
        return path
    path = os.path.expanduser(path)
    if path.startswith("//"):
        return bpy.path.abspath(path)
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)

# ---------------- CLI / Args ----------------
def arg(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return sys.argv[i+1] if i+1 < len(sys.argv) else default
    return default

# Timing / IO
FPS              = int(arg("--fps", "24"))
DURATION_S       = float(arg("--duration", "20"))
SAVE_OUT_PATH    = arg("--out", None)

# Pedestrians (Mixamo)
PEDS_MOVE_RATIO  = float(arg("--peds_move_ratio", "1.0"))
MIXAMO_DIR       = resolve_path(arg("--mixamo_dir", os.path.join(SCRIPT_DIR, "mixamo_fbx")))
SPEED_SCALE_MIN  = float(arg("--speed_scale_min", "0.95"))
SPEED_SCALE_MAX  = float(arg("--speed_scale_max", "1.05"))
DIR_LEFT_PED     = 1 if arg("--dir_left_ped", "-") in ("-","1","pos","plus") else -1
DIR_RIGHT_PED    = -1 if arg("--dir_right_ped", "+") in ("+","-1","neg","minus") else 1
WALK_FALLBACK    = float(arg("--walk_fallback_mps", "1.5"))
RUN_FALLBACK     = float(arg("--run_fallback_mps", "3.5"))

# --- Multi-track controls ---
PED_TRACKS          = int(arg("--ped_tracks", "2"))     # 1, 2, or 3
CAR_TRACKS          = int(arg("--car_tracks", "2"))     # 1, 2, or 3
TRACK_MARGIN_X      = float(arg("--track_margin_x", "2"))  # m kept from slab edges
TRACK_ASSIGN        = (arg("--track_assign", "random") or "random").lower()  # random|roundrobin
TRACK_SPEED_MODE    = (arg("--track_speed_mode", "bands") or "bands").lower() # bands|random|equal
TRACK_SPEED_MIN_MULT= float(arg("--track_speed_min_mult", "0.9"))
TRACK_SPEED_MAX_MULT= float(arg("--track_speed_max_mult", "1.1"))
INTRA_TRACK_PHASE   = (arg("--intra_track_phase", "even").lower() in ("even","true","1","yes"))


# NEW: Facing & Height options
RIG_FORWARD      = arg("--rig_forward", "-X")     # how your Mixamo rigs face on import: +Y, -Y, +X, -X
PED_H_MIN        = float(arg("--ped_height_min", "2.70")) ## script does not recognise height, so tune this with trial and error
PED_H_MAX        = float(arg("--ped_height_max", "2.70"))

# Cars
CARS_MOVE_RATIO  = float(arg("--cars_move_ratio", "1.0"))
SPEED_MIN_CAR    = float(arg("--speed_min_car", "6.0"))
SPEED_MAX_CAR    = float(arg("--speed_max_car", "7.0"))
DIR_LEFT_CAR     = 1 if arg("--dir_left_car", "+") in ("+","1","pos","plus") else -1
DIR_RIGHT_CAR    = -1 if arg("--dir_right_car", "-") in ("-","-1","neg","minus") else 1

END_MARGIN = 0.5  # keep actors off slab ends

# ___________________ for model rotations __________#
def normalize_angle(rad):
    return (rad + math.pi) % (2.0 * math.pi) - math.pi

def yaw_to_face_plusY(forward_axis="+Y"):
    m = {"+Y": 0.0, "-Y": math.pi, "+X": math.pi/2.0, "-X": -math.pi/2.0}
    return m.get(forward_axis.upper(), 0.0)

def set_facing_for_lane(arm_obj, lane_sign, rig_forward_axis):
    """
    Force absolute world yaw so the rig faces exactly along the lane.
    lane_sign >= 0 → +Y (0°), lane_sign < 0 → -Y (180°).
    rig_forward_axis tells us how the FBX faces when its Z=0 (e.g. +X).
    """
    # rotation that makes a zero-yaw rig face +Y
    forward_offset = yaw_to_face_plusY(rig_forward_axis)   # e.g. +X → +90°
    lane_yaw = 0.0 if lane_sign >= 0 else math.pi          # +Y or -Y
    world_yaw = normalize_angle(forward_offset + lane_yaw)

    # zero out any inherited pitch/roll and set absolute yaw
    arm_obj.rotation_euler = Euler((0.0, 0.0, world_yaw), 'XYZ')

# ---------------- Geometry helpers ----------------
# def bbox_world(obj):
#     M = obj.matrix_world
#     return [M @ Vector(c) for c in obj.bound_box]

def bbox_world(obj):
    # world-space bounding box corners
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

def _x_bounds(obj):
    xs = [p.x for p in bbox_world(obj)]
    return (min(xs), max(xs))

def _compute_tracks_for_lane(lane, n_tracks, margin_x):
    """Return list of X positions for n_tracks inside lane [x0, x1], keeping margin at both sides."""
    x0 = lane["x0"] + margin_x
    x1 = lane["x1"] - margin_x
    if x1 <= x0 or n_tracks <= 1:
        return [lane["cx"]]
    step = (x1 - x0) / (n_tracks - 1)
    return [x0 + i*step for i in range(n_tracks)]

def slab_info(slab):
    pts = bbox_world(slab)
    xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
    cx = 0.5*(min(xs)+max(xs)); y0,y1 = min(ys), max(ys); topz = max(zs)
    return cx, y0, y1, topz

def find_slab(names_exact, keywords_any=None, side=None):
    # try exact names first
    for nm in names_exact:
        ob = bpy.data.objects.get(nm)
        if ob and ob.type == "MESH": return ob
    # fallback by keywords
    cands = []
    if keywords_any:
        for o in bpy.data.objects:
            if o.type != "MESH": continue
            low = o.name.lower()
            if any(kw in low for kw in keywords_any): cands.append(o)
    if not cands: return None
    def center_x(o):
        xs = [p.x for p in bbox_world(o)]
        return 0.5*(min(xs)+max(xs))
    if side == "left":  return min(cands, key=center_x)
    if side == "right": return max(cands, key=center_x)
    return cands[0]

def _bbox_world(o):
    M = o.matrix_world
    return [M @ Vector(c) for c in o.bound_box]

def _geom_bbox_world(root):
    pts = []
    def add(obj):
        if obj.type == 'MESH':
            pts.extend(_bbox_world(obj))
    add(root)
    for ch in root.children_recursive:
        add(ch)
    return pts or _bbox_world(root)

def _anchor_bottom_center_world(root):
    pts = _geom_bbox_world(root)
    xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
    return Vector(((min(xs)+max(xs))*0.5, (min(ys)+max(ys))*0.5, min(zs)))

# ---------------- Movers & Drivers ----------------
def make_mover_and_snap_to_centerline(child_obj, center_x, top_z, face_sign=+1, offset_x=0.0):
    scn = bpy.context.scene
    mover = bpy.data.objects.new(child_obj.name + "_MOVE", None)
    mover.empty_display_type = 'PLAIN_AXES'
    mover.location = Vector((center_x + float(offset_x), child_obj.matrix_world.translation.y, top_z))
    mover.rotation_euler = Euler((0.0, 0.0, 0.0 if face_sign >= 0 else math.pi), 'XYZ')
    scn.collection.objects.link(mover)

    try:
        child_obj.delta_location = (0,0,0)
        child_obj.delta_rotation_euler = (0,0,0)
        child_obj.delta_scale = (1,1,1)
    except: pass

    anchor_world = _anchor_bottom_center_world(child_obj)  # your existing helper
    M = child_obj.matrix_world.copy()
    child_obj.parent = mover
    child_obj.matrix_parent_inverse = mover.matrix_world.inverted()
    child_obj.matrix_world = M
    anchor_in_mover = mover.matrix_world.inverted() @ anchor_world
    child_obj.location -= anchor_in_mover
    return mover



def add_linear_wrap_driver_Y(mover, y_start, span, speed_mps, fps, frame0, phase_m, direction_sign):
    """ Y(t) = y_start + sgn * ((speed*(frame-frame0)/fps + phase_m) % span) """
    drv = mover.driver_add('location', 1).driver  # channel Y
    v = drv.variables.new(); v.name = "f"; v.type = 'SINGLE_PROP'
    v.targets[0].id_type = 'SCENE'; v.targets[0].id = bpy.context.scene
    v.targets[0].data_path = 'frame_current'
    sgn = 1 if direction_sign >= 0 else -1
    drv.expression = f"{y_start} + ({sgn})*(({speed_mps}*(f-{frame0})/{fps} + {phase_m}) % {span})"

# ---------------- Lanes ----------------
def build_foot_lanes():
    left_fp  = find_slab(["left_footpath","footpath_left"],  ["footpath","sidewalk"], "left")
    right_fp = find_slab(["right_footpath","footpath_right"],["footpath","sidewalk"], "right")
    lanes = {}
    if left_fp:
        cx,y0,y1,topz = slab_info(left_fp)
        x0,x1 = _x_bounds(left_fp); w = max(0.0, x1 - x0)
        lanes["left"] = dict(cx=cx, topz=topz, y0=y0, y1=y1,
                             span=max((y1 - y0) - 2*END_MARGIN, 0.001),
                             sgn=DIR_LEFT_PED, x0=x0, x1=x1, w=w)
    if right_fp:
        cx,y0,y1,topz = slab_info(right_fp)
        x0,x1 = _x_bounds(right_fp); w = max(0.0, x1 - x0)
        lanes["right"] = dict(cx=cx, topz=topz, y0=y0, y1=y1,
                              span=max((y1 - y0) - 2*END_MARGIN, 0.001),
                              sgn=DIR_RIGHT_PED, x0=x0, x1=x1, w=w)
    return lanes

def build_drive_lanes():
    left_rd  = find_slab(["left_driveway","driveway_left","left_lane"],   ["driveway","carriageway","lane","road","street"], "left")
    right_rd = find_slab(["right_driveway","driveway_right","right_lane"],["driveway","carriageway","lane","road","street"], "right")
    lanes = {}
    if left_rd:
        cx,y0,y1,topz = slab_info(left_rd)
        x0,x1 = _x_bounds(left_rd); w = max(0.0, x1 - x0)
        lanes["left"] = dict(cx=cx, topz=topz, y0=y0, y1=y1,
                             span=max((y1 - y0) - 2*END_MARGIN, 0.001),
                             sgn=DIR_LEFT_CAR, x0=x0, x1=x1, w=w)
    if right_rd:
        cx,y0,y1,topz = slab_info(right_rd)
        x0,x1 = _x_bounds(right_rd); w = max(0.0, x1 - x0)
        lanes["right"] = dict(cx=cx, topz=topz, y0=y0, y1=y1,
                              span=max((y1 - y0) - 2*END_MARGIN, 0.001),
                              sgn=DIR_RIGHT_CAR, x0=x0, x1=x1, w=w)
    return lanes



def lane_start_for_sign(lane, sign):
    # sign >= 0 → start at min-Y+margin; sign < 0 → start at max-Y−margin
    return (lane["y0"] + END_MARGIN) if sign >= 0 else (lane["y1"] - END_MARGIN)


# ---------------- Mixamo helpers ----------------
def guess_kind(name):
    n = name.lower()
    return "run" if any(k in n for k in ("run","jog","sprint")) else "walk"

def find_hips_name(arm):
    names = [b.name for b in arm.data.bones]
    for cand in ("Hips","mixamorig:Hips","mixamorig:Hips01","hips"):
        if cand in names: return cand
    for n in names:
        if "hips" in n.lower(): return n
    return names[0] if names else "Hips"

def measure_forward_speed_mps(arm, hips_name, fps):
    """Measures ΔY of Hips over the action; In-Place clips will be ~0."""
    ad = arm.animation_data
    if not ad or not ad.action: return 0.0
    act = ad.action
    f0, f1 = act.frame_range
    span = max(1.0, (f1 - f0))
    dp = f'pose.bones["{hips_name}"].location'
    fcy = next((fc for fc in act.fcurves if fc.data_path == dp and fc.array_index == 1), None)
    if not fcy: return 0.0
    dy = abs(float(fcy.evaluate(f1) - fcy.evaluate(f0)))
    scale = (arm.scale.x + arm.scale.y + arm.scale.z) / 3.0
    return (dy * scale) * (fps / span)

def make_action_in_place(arm, hips_name):
    """Zero Hips X/Y so clip is in-place (harmless if already In-Place)."""
    ad = arm.animation_data
    if not ad or not ad.action: return
    act = ad.action
    dp = f'pose.bones["{hips_name}"].location'
    for fc in list(act.fcurves):
        if fc.data_path == dp and fc.array_index in (0,1):
            for kp in fc.keyframe_points:
                kp.co[1] = 0.0
                kp.handle_left[1] = 0.0
                kp.handle_right[1] = 0.0
            fc.update()

# ---------------- Looping strip (no Cycles modifiers) ----------------
def clear_all_nla(obj):
    if obj.animation_data and obj.animation_data.nla_tracks:
        for tr in list(obj.animation_data.nla_tracks):
            obj.animation_data.nla_tracks.remove(tr)

def build_looping_strip_covering_scene(obj, action, scene_start, scene_end, phase_frames=0):
    """
    Robust loop WITHOUT Cycles modifiers:
      - Start strip BEFORE scene by phase so frame 1 is already mid-walk (no rest pose).
      - use_sync_length=True and compute repeat to exceed scene_end.
      - Trim action_frame_end = action_end - 1 to avoid duplicated last-frame seam.
      - Solo the track and clear action slot so only NLA drives the rig.
    """
    if not obj.animation_data: obj.animation_data_create()
    clear_all_nla(obj)

    a0f, a1f = action.frame_range
    a0 = int(round(a0f)); a1 = int(round(a1f))
    loop_end = a1 - 1 if (a1 - a0) >= 2 else a1
    length = max(1, loop_end - a0)
    phase = int(max(0, min(length - 1, phase_frames)))

    track = obj.animation_data.nla_tracks.new()
    track.name = "Loop"
    track.is_solo = True

    strip_start = int(scene_start - phase)
    strip = track.strips.new(action.name, strip_start, action)
    strip.action_frame_start = float(a0)
    strip.action_frame_end   = float(loop_end)
    strip.use_sync_length    = True
    strip.extrapolation      = 'NOTHING'
    strip.blend_type         = 'REPLACE'
    strip.influence          = 1.0

    needed = (scene_end - strip.frame_start) + length + 2
    strip.repeat = max(1, int(ceil(needed / float(length))))

    # NLA-only evaluation
    obj.animation_data.action = None
    return strip

# ---------------- Prototypes: import Mixamo FBX ----------------
def load_mixamo_prototypes(folder):
    """
    Import all FBX into hidden collection 'ASSETS_PEDS'.
    Keep Action in slot; store: kind, hips_name, native_speed_mps, action_name.
    """
    col = bpy.data.collections.get("ASSETS_PEDS")
    if not col:
        col = bpy.data.collections.new("ASSETS_PEDS")
        bpy.context.scene.collection.children.link(col)
        col.hide_viewport = True; col.hide_render = True

    protos = [o for o in col.objects if o.type == 'ARMATURE']
    if protos: return protos

    if not folder or not os.path.isdir(folder):
        print("[Mixamo] Invalid --mixamo_dir."); return []

    fbx_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".fbx")]
    if not fbx_files:
        print("[Mixamo] No FBX found."); return []

    for path in fbx_files:
        before = set(o.name for o in bpy.data.objects)
        try:
            bpy.ops.import_scene.fbx(filepath=path, automatic_bone_orientation=True)
        except Exception as e:
            print(f"[Mixamo] Import failed: {path} ({e})"); continue
        newobs = [o for o in bpy.data.objects if o.name not in before]
        arm = next((o for o in newobs if o.type == 'ARMATURE'), None)
        if not arm: continue
        for o in newobs:
            for c in list(o.users_collection): c.objects.unlink(o)
            col.objects.link(o)

        hips = find_hips_name(arm)
        speed = measure_forward_speed_mps(arm, hips, FPS)  # 0.0 for In-Place (expected)
        make_action_in_place(arm, hips)                    # harmless if already In-Place
        act_name = arm.animation_data.action.name if (arm.animation_data and arm.animation_data.action) else ""

        arm["kind"] = guess_kind(os.path.basename(path))
        arm["hips_name"] = hips
        arm["native_speed_mps"] = float(speed)
        arm["action_name"] = act_name
        print(f"[Mixamo] Proto '{arm.name}': kind={arm['kind']}, act='{act_name}', measured_mps={speed:.3f}")

    return [o for o in col.objects if o.type == 'ARMATURE']

def dup_rig_with_action(proto):
    """Duplicate the prototype armature + meshes, relinking Armature modifiers; keep Action in slot."""
    arm = proto.copy(); arm.data = proto.data.copy()
    bpy.context.scene.collection.objects.link(arm)
    mapping = {proto: arm}
    for ch in proto.children_recursive:
        cp = ch.copy()
        if getattr(ch, "data", None): cp.data = ch.data
        bpy.context.scene.collection.objects.link(cp)
        cp.parent = mapping.get(ch.parent, arm)
        cp.matrix_parent_inverse = ch.matrix_parent_inverse.copy()
        mapping[ch] = cp
        if cp.type == 'MESH':
            for m in cp.modifiers:
                if m.type == 'ARMATURE': m.object = arm
    if not arm.animation_data: arm.animation_data_create()
    act = None
    if proto.animation_data and proto.animation_data.action:
        act = proto.animation_data.action
    elif "action_name" in proto:
        act = bpy.data.actions.get(proto["action_name"])
    if act: arm.animation_data.action = act
    return arm

# ---------------- Orientation & Height helpers ----------------
def yaw_to_face_plusY(forward_axis="+Y"):
    m = {"+Y": 0.0, "-Y": math.pi, "+X": math.pi/2.0, "-X": -math.pi/2.0}
    return m.get(forward_axis.upper(), 0.0)

def _measure_mover_height_world(mover, use_depsgraph=True):
    """Measure world-space Z span of all MESH children evaluated under mover."""
    deps = bpy.context.evaluated_depsgraph_get() if use_depsgraph else None
    zs = []
    for ch in mover.children_recursive:
        if ch.type != 'MESH': continue
        o = ch.evaluated_get(deps) if deps else ch
        M = o.matrix_world
        for c in o.bound_box:
            zs.append((M @ Vector(c)).z)
    return (max(zs) - min(zs)) if zs else 0.0

def normalize_mover_height(mover, target_m=1.70, tol=0.005, max_iters=3):
    """
    Scale mover uniformly so the assembled character's world height equals target_m.
    Returns total scale factor applied (use this to scale translation speed).
    """
    total = 1.0
    for _ in range(max_iters):
        h = _measure_mover_height_world(mover)
        if h <= 1e-6: break
        f = target_m / h
        if abs(1.0 - f) <= tol / max(h, 1e-6): break
        s = mover.scale
        mover.scale = (s.x * f, s.y * f, s.z * f)
        total *= f
        bpy.context.view_layer.update()
    return total

# ---------------- Pedestrians ----------------
def add_ped_from_proto(h_placeholder, lane, proto, scene_start, scene_end,
                       forced_speed=None, phase_override=None, offset_x=0.0,
                       speed_scale_multiplier=1.0):
    arm = dup_rig_with_action(proto)
    arm.name = f"{h_placeholder.name}_Rig"
    arm.location = h_placeholder.location.copy()
    arm.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
    set_facing_for_lane(arm, lane["sgn"], RIG_FORWARD)

    if arm.animation_data and arm.animation_data.action:
        act = arm.animation_data.action
        a0f, a1f = act.frame_range
        cyc_len = max(1, int(round(a1f - a0f)))
        phase_frames = random.randint(0, cyc_len - 1) if cyc_len > 1 else 0
        build_looping_strip_covering_scene(arm, act, scene_start, scene_end, phase_frames)

    mover = make_mover_and_snap_to_centerline(arm, lane["cx"], lane["topz"], face_sign=lane["sgn"], offset_x=offset_x)
    span = lane["span"]
    if span <= 0.001: return False

    target_h = random.uniform(PED_H_MIN, PED_H_MAX)
    height_scale = normalize_mover_height(mover, target_h)

    measured = float(proto.get("native_speed_mps", 0.0))
    kind     = proto.get("kind", "walk")
    base     = measured if measured >= 0.3 else (RUN_FALLBACK if kind == "run" else WALK_FALLBACK)
    speed    = (forced_speed if forced_speed is not None else (base * random.uniform(SPEED_SCALE_MIN, SPEED_SCALE_MAX))) * height_scale
    speed   *= float(speed_scale_multiplier)

    eff_sign = -lane["sgn"]
    y_start  = lane_start_for_sign(lane, eff_sign)
    phase_m  = phase_override if phase_override is not None else random.uniform(0.0, span)
    frame0   = bpy.context.scene.frame_start

    add_linear_wrap_driver_Y(mover, y_start, span, speed, FPS, frame0, phase_m, eff_sign)
    h_placeholder.hide_set(True); h_placeholder.hide_render = True
    return True

def animate_pedestrians():
    scn = bpy.context.scene
    lanes = build_foot_lanes()
    if not lanes:
        print("[Peds] No footpath slabs. Skipping."); return 0

    protos = load_mixamo_prototypes(MIXAMO_DIR)
    if not protos:
        print("[Peds] No Mixamo prototypes. Skipping."); return 0

    walkers = [p for p in protos if p.get("kind","walk") == "walk"]
    runners = [p for p in protos if p.get("kind","walk") == "run"]
    if not walkers and not runners: walkers = protos

    humans_all = [o for o in bpy.data.objects if o.name.startswith("Human_")]
    if not humans_all:
        print("[Peds] No Human_* placeholders."); return 0
    humans = humans_all[:int(len(humans_all) * max(0.0, min(1.0, PEDS_MOVE_RATIO)))]

    def side_near(x):
        if "left" in lanes and "right" in lanes:
            return "left" if abs(x - lanes["left"]["cx"]) <= abs(x - lanes["right"]["cx"]) else "right"
        return "left" if "left" in lanes else "right"

    # bucket by side
    buckets = {"left": [], "right": []}
    for h in humans:
        s = side_near(h.location.x)
        if s in lanes: buckets[s].append(h)

    placed = 0
    for side, hs in buckets.items():
        if not hs: continue
        lane = lanes[side]
        tracks = _compute_tracks_for_lane(lane, max(1, PED_TRACKS), TRACK_MARGIN_X)
        nT = len(tracks)

        # build track buckets
        track_buckets = [[] for _ in range(nT)]
        if TRACK_ASSIGN == "roundrobin":
            for i, h in enumerate(hs):
                track_buckets[i % nT].append(h)
        else:  # random
            for h in hs:
                track_buckets[random.randrange(nT)].append(h)

        # per-track speed multipliers
        if TRACK_SPEED_MODE == "bands" and nT > 1:
            # linearly spaced multipliers (slowest outer → fastest inner by default)
            m0, m1 = TRACK_SPEED_MIN_MULT, TRACK_SPEED_MAX_MULT
            track_mults = [m0 + (m1 - m0)*(i/(nT-1)) for i in range(nT)]
        elif TRACK_SPEED_MODE == "equal":
            track_mults = [1.0]*nT
        else:  # random per track in range
            track_mults = [random.uniform(TRACK_SPEED_MIN_MULT, TRACK_SPEED_MAX_MULT) for _ in range(nT)]

        for t_idx, group in enumerate(track_buckets):
            if not group: continue
            span = lane["span"]; n = len(group)
            # phases inside this track
            phases = ([(i + 0.5) * (span / n) for i in range(n)] if INTRA_TRACK_PHASE
                      else [random.uniform(0.0, span) for _ in range(n)])

            # little per-agent jitter around track multiplier
            jitter = lambda: random.uniform(0.98, 1.02)
            for i, h in enumerate(group):
                use_run = any(k in h.name.lower() for k in ("run","jog","sprint"))
                pool = runners if (use_run and runners) else (walkers if walkers else protos)
                proto = random.choice(pool)

                if add_ped_from_proto(h, lane, proto,
                                      scn.frame_start, scn.frame_end,
                                      forced_speed=None,
                                      phase_override=phases[i],
                                      offset_x=(tracks[t_idx] - lane["cx"]),
                                      speed_scale_multiplier=track_mults[t_idx] * jitter()):
                    placed += 1

    print(f"[Peds] Animated {placed}/{len(humans)} pedestrians across {PED_TRACKS} tracks/side.")
    return placed


# ---------------- Cars ----------------
def animate_cars():
    lanes = build_drive_lanes()
    if not lanes:
        print("[Cars] No driveway/road slabs. Skipping."); return 0

    cars_all = [o for o in bpy.data.objects if o.name.startswith("Car_")]
    if not cars_all:
        print("[Cars] No Car_* objects."); return 0
    cars = cars_all[:int(len(cars_all) * max(0.0, min(1.0, CARS_MOVE_RATIO)))]

    def side_near(x):
        if "left" in lanes and "right" in lanes:
            return "left" if abs(x - lanes["left"]["cx"]) <= abs(x - lanes["right"]["cx"]) else "right"
        return "left" if "left" in lanes else "right"

    # bucket by side
    side_buckets = {"left": [], "right": []}
    for c in cars:
        s = side_near(c.location.x)
        if s in lanes: side_buckets[s].append(c)

    moved = 0
    scn = bpy.context.scene
    for side, cs in side_buckets.items():
        if not cs: continue
        lane = lanes[side]
        tracks = _compute_tracks_for_lane(lane, max(1, CAR_TRACKS), TRACK_MARGIN_X)
        nT = len(tracks)

        # track buckets
        track_buckets = [[] for _ in range(nT)]
        if TRACK_ASSIGN == "roundrobin":
            for i, c in enumerate(cs):
                track_buckets[i % nT].append(c)
        else:
            for c in cs:
                track_buckets[random.randrange(nT)].append(c)

        # per-track speed multipliers
        if TRACK_SPEED_MODE == "bands" and nT > 1:
            m0, m1 = TRACK_SPEED_MIN_MULT, TRACK_SPEED_MAX_MULT
            track_mults = [m0 + (m1 - m0)*(i/(nT-1)) for i in range(nT)]
        elif TRACK_SPEED_MODE == "equal":
            track_mults = [1.0]*nT
        else:
            track_mults = [random.uniform(TRACK_SPEED_MIN_MULT, TRACK_SPEED_MAX_MULT) for _ in range(nT)]

        for t_idx, group in enumerate(track_buckets):
            if not group: continue
            span = lane["span"]; n = len(group)
            phases = ([(i + 0.5) * (span / n) for i in range(n)] if INTRA_TRACK_PHASE
                      else [random.uniform(0.0, span) for _ in range(n)])

            for i, car in enumerate(group):
                mover  = make_mover_and_snap_to_centerline(car, lane["cx"], lane["topz"],
                                                           face_sign=lane["sgn"],
                                                           offset_x=(tracks[t_idx] - lane["cx"]))
                # base speed in m/s, scaled per track and small jitter
                base = random.uniform(min(SPEED_MIN_CAR, SPEED_MAX_CAR), max(SPEED_MIN_CAR, SPEED_MAX_CAR))
                speed = base * track_mults[t_idx] * random.uniform(0.98, 1.02)

                frame0 = scn.frame_start
                eff_sign = -lane["sgn"]
                y_start  = lane_start_for_sign(lane, eff_sign)

                add_linear_wrap_driver_Y(mover, y_start, lane["span"], speed, FPS, frame0, phases[i], eff_sign)
                moved += 1

    print(f"[Cars] Animated {moved}/{len(cars)} cars across {CAR_TRACKS} tracks/side.")
    return moved


# ---------------- Save & Entry ----------------
def save_after_anim():
    out = bpy.path.abspath(SAVE_OUT_PATH) if SAVE_OUT_PATH else (
        os.path.splitext(bpy.data.filepath or "street_scene.blend")[0] + "_anim.blend"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_mainfile(filepath=out)
    print(f"[IO] Saved .blend to: {out}")

def main():
    scn = bpy.context.scene
    scn.render.fps  = FPS
    scn.frame_start = 1
    scn.frame_end   = int(FPS * DURATION_S)
    scn.frame_set(scn.frame_start)  # force depsgraph update

    p = animate_pedestrians()
    c = animate_cars()
    if p or c:
        save_after_anim()
    else:
        print("[Info] Nothing animated; saving handoff blend unchanged.")
        save_after_anim()

if __name__ == "__main__":
    main()

