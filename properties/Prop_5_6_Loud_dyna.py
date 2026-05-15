# compute_street_metrics_patched_FIXED.py
# Outputs JSON:
#  enclosure_HW_incl_zeros, enclosure_HW_valid_only,
#  prospect_area_norm, prospect_compactness, svf, shade_fraction,
#  loudness_idx, sound_IR, dynamism_index
#
# Defaults (as requested earlier):
#  - probe_mode = ped
#  - probes_per_side = 1 (so typically two probes: left & right sidewalks)
#  - isovist_maxdist (R) = 100 m
#
# Prospect (primary):
#  area_norm = area(isovist polygon in XY) / (π * R^2)
#  Printed per probe and averaged across probes.
#
# NOTE: Isovist ray-casts here SKIP any previously drawn viz geometry
#       (e.g., IsovistRays*/IsovistPoly*) so results match your viz tool.

import bpy, json, math, sys, os, datetime as _dt
from mathutils import Vector
from collections import Counter

# ---------------- CLI ----------------
def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--")+1:]
    else:
        argv = []
    args = {
        "out": None,
        "probe_mode": "ped",
        "probes_n": 1,               # used only if probes_per_side == 0
        "probes_per_side": 1,        # DEFAULT: 1 probe per lateral track/side
        "svf_rays": 400,
        "isovist_rays": 360,
        "isovist_maxdist": 100.0,    # DEFAULT RADIUS = 100 m
        # solar defaults (Berlin, 21 June, 12–13 local, +Y is North)
        "lat": 52.52,
        "lon": 13.405,
        "date": "2025-06-21",
        "tstart": 12.0,
        "tend": 13.0,
        "tz": 2.0,
        "north": 0.0,
    }
    i=0
    while i < len(argv):
        a = argv[i]
        if a=="--out" and i+1<len(argv): args["out"]=argv[i+1]; i+=2; continue
        if a=="--probe_mode" and i+1<len(argv): args["probe_mode"]=argv[i+1]; i+=2; continue
        if a=="--probes" and i+1<len(argv): args["probes_n"]=max(1,int(argv[i+1])); i+=2; continue
        if a=="--probes_per_side" and i+1<len(argv): args["probes_per_side"]=max(0,int(argv[i+1])); i+=2; continue
        if a=="--svf_rays" and i+1<len(argv): args["svf_rays"]=max(40,int(argv[i+1])); i+=2; continue
        if a=="--isovist_rays" and i+1<len(argv): args["isovist_rays"]=max(60,int(argv[i+1])); i+=2; continue
        if a=="--isovist_maxdist" and i+1<len(argv): args["isovist_maxdist"]=float(argv[i+1]); i+=2; continue
        if a=="--lat" and i+1<len(argv): args["lat"]=float(argv[i+1]); i+=2; continue
        if a=="--lon" and i+1<len(argv): args["lon"]=float(argv[i+1]); i+=2; continue
        if a=="--date" and i+1<len(argv): args["date"]=argv[i+1]; i+=2; continue
        if a=="--tstart" and i+1<len(argv): args["tstart"]=float(argv[i+1]); i+=2; continue
        if a=="--tend" and i+1<len(argv): args["tend"]=float(argv[i+1]); i+=2; continue
        if a=="--tz" and i+1<len(argv): args["tz"]=float(argv[i+1]); i+=2; continue
        if a=="--north" and i+1<len(argv): args["north"]=float(argv[i+1]); i+=2; continue
        i+=1
    return args

# ---------------- Basics & compatibility helpers ----------------
def dg(): return bpy.context.evaluated_depsgraph_get()

def eval_obj(o):
    try:
        return o.evaluated_get(dg())
    except Exception:
        return o

def ray_cast(origin: Vector, direction: Vector, dist=1e6):
    """Version-robust scene.ray_cast → (hit, loc, obj, distance)."""
    hit, loc, norm, face_idx, obj, _ = bpy.context.scene.ray_cast(dg(), origin, direction, distance=dist)
    if not hit:
        return False, None, None, None
    return True, loc, obj, (loc-origin).length

def ray_cast_skip_viz(origin: Vector, direction: Vector, dist=1e6, max_hops=8, eps=1e-4):
    """
    Ray cast but skip any previously drawn isovist/metrics viz geometry
    (e.g., objects tagged 'is_isovist_viz' or named like 'IsovistRays*'/'IsovistPoly*' or containing 'isovist').
    """
    deps = dg()
    start = origin.copy()
    remain = dist
    for _ in range(max_hops):
        hit, loc, norm, face_idx, obj, _ = bpy.context.scene.ray_cast(deps, start, direction, distance=remain)
        if not hit:
            return False, None, None, None
        nm = obj.name.lower() if obj else ""
        if obj and (obj.get("is_isovist_viz") or nm.startswith("isovistrays") or nm.startswith("isovistpoly") or "isovist" in nm):
            step = (loc - start).length + eps
            start = start + direction * step
            remain = max(0.0, dist - (start - origin).length)
            continue
        return True, loc, obj, (loc - origin).length
    return False, None, None, None

def world_bbox(obj):
    eo = eval_obj(obj)
    M  = eo.matrix_world
    try:
        corners = [M @ Vector(c) for c in eo.bound_box]
    except Exception:
        corners = [M @ Vector(v.co) for v in eo.data.vertices]
    mn = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    mx = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return mn, mx

def scene_bounds(objs=None):
    if objs is None:
        objs = [o for o in bpy.data.objects
                if o.type in {'MESH','CURVE','SURFACE','META'} and (not getattr(o,"hide_render",False))]
    ok=False
    mins=Vector(( 1e9, 1e9, 1e9))
    maxs=Vector((-1e9,-1e9,-1e9))
    for o in objs:
        try:
            mn,mx = world_bbox(o)
        except Exception:
            continue
        mins.x=min(mins.x,mn.x); mins.y=min(mins.y,mn.y); mins.z=min(mins.z,mn.z)
        maxs.x=max(maxs.x,mx.x); maxs.y=max(maxs.y,mx.y); maxs.z=max(maxs.z,mx.z)
        ok=True
    if not ok:
        mins=Vector((-5,-5,0)); maxs=Vector((5,5,3))
    return mins, maxs

def obj_midpoint_xy(o):
    mn, mx = world_bbox(o)
    return Vector(((mn.x+mx.x)*0.5,(mn.y+mx.y)*0.5,(mn.z+mx.z)*0.5))

# ---------------- object-type helpers ----------------
def name_has_sidewalk(nm: str) -> bool:
    nm = nm.lower()
    return any(k in nm for k in ["footpath","sidewalk","fp","pavement","walkway"])

def name_has_road(nm: str) -> bool:
    nm = nm.lower()
    return any(k in nm for k in ["driveway","carriage","carriageway","driving","road","lane","asphalt","tarmac","street"])

def is_building_part(o):
    if o.type != 'MESH' or getattr(o,"hide_render",False): return False
    nm = o.name.lower()
    return ("building" in nm) or ("roof" in nm)

def is_ground_like(obj):
    if obj is None: return False
    nm = obj.name.lower()
    return name_has_sidewalk(nm) or name_has_road(nm) or any(k in nm for k in ["ground","plaza","parking","curb","kerb"])

# ---------------- left_/right_ finders ----------------
def find_sidewalks_lr():
    L, R = [], []
    for o in bpy.data.objects:
        if o.type!='MESH' or getattr(o,"hide_render",False): continue
        nm=o.name.lower()
        if nm.startswith("left_") and name_has_sidewalk(nm): L.append(o)
        elif nm.startswith("right_") and name_has_sidewalk(nm): R.append(o)
    return L, R, (L+R)

def find_roads_lr():
    L, R = [], []
    for o in bpy.data.objects:
        if o.type!='MESH' or getattr(o,"hide_render",False): continue
        nm=o.name.lower()
        if nm.startswith("left_") and name_has_road(nm): L.append(o)
        elif nm.startswith("right_") and name_has_road(nm): R.append(o)
    return L, R, (L+R)

# ---------------- Probe placement + forward azimuth ----------------
def dominant_az_from_objs(objs):
    mn, mx = scene_bounds(objs)
    dx = mx.x - mn.x; dy = mx.y - mn.y
    # 0 rad = +Y forward; 90° = +X
    return 0.0 if dy >= dx else math.radians(90.0)

def ground_z_at(x, y, z_top=200.0, eye_h=1.6, max_iters=12, eps=0.01):
    """Robustly snap to ground-like meshes (roads/sidewalks/plaza), ignoring canopies/overhangs."""
    # top-down
    origin = Vector((x,y,z_top))
    z = None; last_z = z_top
    for _ in range(max_iters):
        hit, loc, obj, d = ray_cast(origin, Vector((0,0,-1)), dist=abs(last_z)+z_top)
        if not hit: break
        if is_ground_like(obj): z = loc.z; break
        last_z = loc.z - eps; origin = Vector((x,y,last_z))
    # bottom-up if needed
    if z is None:
        origin = Vector((x,y,-z_top)); last_z = -z_top
        for _ in range(max_iters):
            hit, loc, obj, d = ray_cast(origin, Vector((0,0,1)), dist=2*z_top)
            if not hit: break
            if is_ground_like(obj): z = loc.z; break
            last_z = loc.z + eps; origin = Vector((x,y,last_z))
    if z is None: return eye_h
    return z + eye_h

def pick_probes(probe_mode='ped', probes_n=1, probes_per_side=1):
    # If user empties exist and per-side is disabled, use up to probes_n of them.
    user = [o for o in bpy.data.objects if o.type=='EMPTY' and o.name.lower().startswith('probe')]
    if user and (probes_per_side or 0) <= 0:
        user_sorted = sorted(user, key=lambda o:o.name.lower())
        if probes_n is not None and probes_n > 0:
            user_sorted = user_sorted[:probes_n]
        pts = []
        mn, mx = scene_bounds()
        for o in user_sorted:
            P = o.matrix_world.translation.copy()
            P.z = ground_z_at(P.x, P.y, z_top=max(50.0, (mx.z - mn.z) + 10.0))
            pts.append(P)
        fwd = dominant_az_from_objs([o for o in bpy.data.objects if o.type=='MESH' and not getattr(o,"hide_render",False)])
        return pts, fwd

    # Otherwise derive tracks from sidewalks/roads
    Lsw,Rsw,AllSw = find_sidewalks_lr()
    Lrd,Rrd,AllRd = find_roads_lr()
    base = AllSw if AllSw else (AllRd if AllRd else [o for o in bpy.data.objects if o.type=='MESH' and not getattr(o,"hide_render",False)])
    fwd_az = dominant_az_from_objs(base)
    mn, mx = scene_bounds(base)

    x_tracks = []
    if (Lsw or Rsw) and (probe_mode in ('ped','both')):
        xL = (sum(obj_midpoint_xy(o).x for o in Lsw)/max(1,len(Lsw))) if Lsw else None
        xR = (sum(obj_midpoint_xy(o).x for o in Rsw)/max(1,len(Rsw))) if Rsw else None
        if xL is not None: x_tracks.append(xL)
        if xR is not None: x_tracks.append(xR)
    elif (AllRd) and (probe_mode in ('center','both')):
        rmn, rmx = scene_bounds(AllRd)
        x_tracks.append(0.5*(rmn.x + rmx.x))
    elif (Lrd or Rrd) and (probe_mode in ('ped','both')):
        rmn, rmx = scene_bounds(AllRd)
        margin = 1.25
        x_tracks.append(rmn.x - margin); x_tracks.append(rmx.x + margin)
    else:
        W = mx.x - mn.x
        x_tracks.append(mn.x + 0.25*W); x_tracks.append(mx.x - 0.25*W)
    if not x_tracks:
        x_tracks = [0.5*(mn.x + mx.x)]

    # rows along Y
    y_min, y_max = scene_bounds(base)
    y_min, y_max = y_min.y, y_max.y
    rows = max(1, int(probes_per_side)) if probes_per_side else 1
    if rows == 1:
        ys = [0.5*(y_min + y_max)]
    else:
        seg = (y_max - y_min) / rows
        ys = [y_min + (i+0.5)*seg for i in range(rows)]

    probes=[]
    for x in x_tracks:
        for y in ys:
            z = ground_z_at(x, y, z_top=max(50.0,(mx.z-mn.z)+10.0))
            probes.append(Vector((x,y,z)))
    return probes, fwd_az

# ---------------- Spatial metrics ----------------
def polygon_area_perimeter(pts_xy):
    n = len(pts_xy)
    if n < 3: return 0.0, 0.0
    A = 0.0; P = 0.0
    for i in range(n):
        x1,y1 = pts_xy[i]
        x2,y2 = pts_xy[(i+1)%n]
        A += x1*y2 - x2*y1
        dx = x2-x1; dy = y2-y1
        P += math.hypot(dx, dy)
    A = abs(A)*0.5
    return A, P

def isovist_area_norm_at(probe, n_rays=360, maxdist=100.0):
    """360° XY rays to first hit → normalized isovist area A / (π R²)."""
    R = max(0.01, float(maxdist))
    pts_xy = []
    for i in range(n_rays):
        ang = 2*math.pi * i / n_rays
        dir_xy = Vector((math.cos(ang), math.sin(ang), 0.0))
        hit, loc, obj, d = ray_cast_skip_viz(probe, dir_xy, R)
        end = (loc if hit else (probe + dir_xy*R))
        pts_xy.append((end.x, end.y))
    A, _P = polygon_area_perimeter(pts_xy)
    return 0.0 if R <= 0 else (A / (math.pi * R * R))

def isovist_compactness_at(probe, n_rays=360, maxdist=300.0):
    """Legacy: convex-hull compactness mapped to [0..1]."""
    pts=[]
    for i in range(n_rays):
        ang = 2*math.pi * i / n_rays
        dir_xy = Vector((math.cos(ang), math.sin(ang), 0.0))
        hit, loc, obj, d = ray_cast_skip_viz(probe, dir_xy, maxdist)
        end = (loc if hit else (probe + dir_xy*maxdist))
        pts.append((end.x, end.y))
    P = sorted(set(pts))
    if len(P) < 3: return 0.0
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]; upper=[]
    for p in P:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p) <= 0: lower.pop()
        lower.append(p)
    for p in reversed(P):
        while len(upper)>=2 and cross(upper[-2], upper[-1], p) <= 0: upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3: return 0.0
    A, Pm = polygon_area_perimeter(hull)
    if A <= 1e-6: return 0.0
    c = (Pm*Pm)/(4.0*math.pi*A)  # 1 for circle
    if c <= 1.0: return 1.0
    if c >= 5.0: return 0.0
    return 1.0 - (c-1.0)/4.0

def svf_at(probe, rays=400, maxdist=200.0):
    # Fibonacci-like hemisphere sampling; count non-hits as sky
    sky = 0; tot = 0
    phi = (1 + 5 ** 0.5) / 2
    for i in range(rays):
        v = (i / rays)
        e = math.acos(1 - v) * 0.5
        a = 2*math.pi*((i/phi) % 1.0)
        dir3 = Vector((math.cos(a)*math.cos(e), math.sin(a)*math.cos(e), math.sin(e)))
        hit, loc, obj, d = ray_cast(probe, dir3, maxdist)
        tot += 1
        if not hit: sky += 1
    return sky / max(1, tot)

# ---------------- Enclosure (H/W) with zeros included ----------------
def _building_cross_sections(step=5.0):
    import math
    bld = [o for o in bpy.data.objects if is_building_part(o)]
    if not bld:
        return []
    entries = []
    all_x = []
    for b in bld:
        mn, mx = world_bbox(b)
        h = (mx.z - mn.z)
        midx = 0.5 * (mn.x + mx.x)
        entries.append((b, mn, mx, h, midx))
        all_x.extend([mn.x, mx.x])
    if not all_x:
        return []
    centerline_x = 0.5 * (min(all_x) + max(all_x))
    ymins = [mn.y for (_b, mn, mx, h, midx) in entries]
    ymaxs = [mx.y for (_b, mn, mx, h, midx) in entries]
    y_min = min(ymins); y_max = max(ymaxs)
    length = y_max - y_min
    if length <= 0.1:
        return []
    step = max(0.5, float(step))
    sample_ys = []
    y = y_min + 0.5 * step
    while y < y_max:
        sample_ys.append(y); y += step
    if not sample_ys:
        sample_ys = [0.5 * (y_min + y_max)]
    sections = []
    for y in sample_ys:
        left_heights  = []
        right_heights = []
        left_xs  = []
        right_xs = []
        for (_b, mn, mx, h, midx) in entries:
            if mn.y <= y <= mx.y:
                if midx < centerline_x:
                    left_heights.append(h);  left_xs.extend([mn.x, mx.x])
                else:
                    right_heights.append(h); right_xs.extend([mn.x, mx.x])
        if not left_heights or not right_heights or not left_xs or not right_xs:
            continue
        H_left  = sum(left_heights)  / len(left_heights)
        H_right = sum(right_heights) / len(right_heights)
        H = 0.5 * (H_left + H_right)
        x_L = max(left_xs)
        x_R = min(right_xs)
        W = x_R - x_L
        if W <= 0.1:
            continue
        sections.append((y, x_L, x_R, H))
    return sections

def enclosure_HW(step=5.0):
    sections = _building_cross_sections(step=step)
    ratios = []
    for (_y, x_L, x_R, H) in sections:
        W = x_R - x_L
        if W > 0.1:
            ratios.append(H / W)
    if ratios:
        return sum(ratios) / len(ratios)
    bld = [o for o in bpy.data.objects if is_building_part(o)]
    if not bld:
        mn, mx = scene_bounds()
        W = (mx.x - mn.x)
        return 8.0 / max(W, 0.1)
    heights = []; xs = []
    for b in bld:
        mn, mx = world_bbox(b)
        heights.append(mx.z - mn.z)
        xs.extend([mn.x, mx.x])
    H = sum(heights) / len(heights) if heights else 8.0
    W = (max(xs) - min(xs)) if xs else (scene_bounds()[1].x - scene_bounds()[0].x)
    return H / max(W, 0.1)

def enclosure_HW_with_zeros(step=5.0, slab_width=None, slab_height=None, thickness=0.06, keep_debug=False, return_detail=False):
    bparts = [o for o in bpy.data.objects if is_building_part(o)]
    if not bparts:
        return (0.0, 0.0) if not return_detail else \
               {"samples":0,"valid":0,"mean_including_zeros":0.0,"mean_valid_only":0.0,"per_section":[]}
    min_x=min_y=min_z=  1e9; max_x=max_y=max_z= -1e9
    for o in bparts:
        mn, mx = world_bbox(o)
        min_x=min(min_x,mn.x); max_x=max(max_x,mx.x)
        min_y=min(min_y,mn.y); max_y=max(max_y,mx.y)
        min_z=min(min_z,mn.z); max_z=max(max_z,mx.z)
    if (max_y-min_y) <= 1e-3 or (max_x-min_x) <= 1e-3:
        return (0.0, 0.0) if not return_detail else \
               {"samples":0,"valid":0,"mean_including_zeros":0.0,"mean_valid_only":0.0,"per_section":[]}
    centerline_x = 0.5*(min_x+max_x); base_z = min_z
    W_span = (max_x - min_x); H_span = (max_z - min_z)
    width  = slab_width  if slab_width  is not None else (W_span * 1.2)
    height = slab_height if slab_height is not None else (H_span + 2.0)
    thick  = max(0.02, float(thickness))
    st = max(0.5, float(step))
    ys = []; y = min_y + 0.5*st
    while y < max_y:
        ys.append(y); y += st
    if not ys: ys = [0.5*(min_y+max_y)]
    coll = bpy.data.collections.get("EnclosureTmp")
    if coll is None:
        coll = bpy.data.collections.new("EnclosureTmp")
        bpy.context.scene.collection.children.link(coll)
    else:
        for obj in list(coll.objects):
            try: bpy.data.objects.remove(obj, do_unlink=True)
            except Exception: pass
    def _make_box(name, cx, yy, bz, w, h, t):
        w2 = w*0.5; t2 = t*0.5
        xL, xR = cx - w2, cx + w2
        yF, yB = yy - t2, yy + t2
        z0, z1 = bz, bz + h
        me = bpy.data.meshes.new(name)
        me.from_pydata(
            [(xL,yF,z0),(xR,yF,z0),(xR,yB,z0),(xL,yB,z0),
             (xL,yF,z1),(xR,yF,z1),(xR,yB,z1),(xL,yB,z1)],
            [],
            [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
        )
        me.update()
        ob = bpy.data.objects.new(name, me); coll.objects.link(ob); return ob
    def _boolean_apply(obj, target, op='INTERSECT'):
        bpy.context.view_layer.objects.active = obj
        for oo in bpy.context.selected_objects: oo.select_set(False)
        obj.select_set(True)
        mod = obj.modifiers.new(name="Bool", type='BOOLEAN')
        mod.operation = op; mod.solver = 'EXACT'; mod.object = target
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name); return True
        except Exception:
            return False
    def _join_objects(objs, name):
        if not objs: return None
        for oo in bpy.context.selected_objects: oo.select_set(False)
        for oo in objs: oo.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
        objs[0].name = name; return objs[0]
    per_section = []; valid = 0
    for idx, yv in enumerate(ys):
        tmpl = _make_box(f"Encl_Slab_{idx:02d}_T", centerline_x, yv, base_z, width, height, thick)
        slices = []
        for bp in bparts:
            slab = tmpl.copy(); slab.data = tmpl.data.copy(); coll.objects.link(slab)
            ok = _boolean_apply(slab, bp, op='INTERSECT')
            if not ok or len(slab.data.polygons) == 0:
                try: bpy.data.objects.remove(slab, do_unlink=True)
                except Exception: pass
                continue
            slices.append(slab)
        try: bpy.data.objects.remove(tmpl, do_unlink=True)
        except Exception: pass
        if not slices:
            per_section.append(0.0); continue
        joined = _join_objects(slices, f"Encl_Envelope_{idx:02d}")
        coords = [(joined.matrix_world @ v.co) for v in joined.data.vertices]
        eps = 1e-6
        left  = [c for c in coords if c.x < centerline_x - eps]
        right = [c for c in coords if c.x > centerline_x + eps]
        if not left or not right:
            per_section.append(0.0)
            try:
                bpy.data.objects.remove(joined, do_unlink=True)
            except Exception: pass
            continue
        H_L = max(c.z for c in left)  - base_z
        H_R = max(c.z for c in right) - base_z
        Havg = 0.5 * (H_L + H_R)
        x_L_inner = max(c.x for c in left)
        x_R_inner = min(c.x for c in right)
        W_canyon  = x_R_inner - x_L_inner
        if W_canyon <= 1e-6 or Havg <= 0.0:
            per_section.append(0.0)
        else:
            per_section.append(Havg / W_canyon); valid += 1
        try:
            bpy.data.objects.remove(joined, do_unlink=True)
        except Exception: pass
    for obj in list(coll.objects):
        try: bpy.data.objects.remove(obj, do_unlink=True)
        except Exception: pass
    samples_n = len(per_section)
    mean_all = (sum(per_section) / samples_n) if samples_n else 0.0
    valid_list = [v for v in per_section if v > 0.0]
    mean_valid = (sum(valid_list) / len(valid_list)) if valid_list else 0.0
    return mean_all, mean_valid

# ---------------- Audio + visibility bits used by dynamism ----------------
def classify_speaker(spk_obj):
    nm = (spk_obj.parent.name.lower()+" "+spk_obj.name.lower()) if spk_obj.parent else spk_obj.name.lower()
    if any(k in nm for k in ['car','vehicle','bus','truck','van']): return 'veh'
    if any(k in nm for k in ['human','ped','person','crowd']): return 'ped'
    return 'amb'

def speakers_by_class():
    ped, veh, amb = [], [], []
    for o in bpy.data.objects:
        if o.type == 'SPEAKER' and (not getattr(o,"hide_render",False)):
            cls = classify_speaker(o)
            if cls=='ped': ped.append(o)
            elif cls=='veh': veh.append(o)
            else: amb.append(o)
    return ped, veh, amb

def inverse_clamped_gain(distance, dref, rolloff, dmax):
    if distance <= dref: g = 1.0
    else: g = dref / max(dref + rolloff * (distance - dref), 1e-6)
    if dmax > 0 and distance > dmax: g = 0.0
    return g

def analytic_loudness_at(probe, ped_speakers, veh_speakers, amb_speakers):
    def sum_class(objs):
        s = 0.0
        for spk in objs:
            so = spk.data
            loc = spk.matrix_world.translation
            d = (loc - probe).length
            dref = getattr(so, "distance_reference", 1.0) or 1.0
            rolloff = getattr(so, "attenuation", 1.0) or 1.0
            dmax = getattr(so, "distance_max", 0.0) or 0.0
            vol = getattr(so, "volume", 1.0) or 1.0
            s += vol * inverse_clamped_gain(d, dref, rolloff, dmax)
        return s
    Lp = sum_class(ped_speakers)
    Lv = sum_class(veh_speakers)
    La = sum_class(amb_speakers)
    return Lp, Lv, La

def visible_agents_and_ped_speed(probe, fov_deg=90.0, maxdist=30.0, frame_step=2, fwd_az=0.0):
    scene = bpy.context.scene
    ped_objs = [o for o in bpy.data.objects if (not getattr(o,"hide_render",False)) and any(k in o.name.lower() for k in ['human','ped','person'])]
    veh_objs = [o for o in bpy.data.objects if (not getattr(o,"hide_render",False)) and any(k in o.name.lower() for k in ['car','vehicle','bus','truck','van'])]
    def in_fov_visible(o):
        tgt = o.matrix_world.translation.copy()
        dir3 = (tgt - probe)
        dist = dir3.length
        if dist < 1e-3 or dist > maxdist: return False
        dir3.normalize()
        yaw = math.atan2(dir3.x, dir3.y)  # 0 along +Y
        yaw_rel = ((yaw - fwd_az + math.pi) % (2*math.pi)) - math.pi
        if abs(yaw_rel) > math.radians(fov_deg/2.0): return False
        hit, loc, obj, d = ray_cast(probe, dir3, dist+1e-3)
        return hit and (obj == o or (o.parent and obj == o.parent))
    cur = scene.frame_current
    try:
        Np = sum(1 for o in ped_objs if in_fov_visible(o))
        Nv = sum(1 for o in veh_objs if in_fov_visible(o))
        fs, fe = scene.frame_current, min(scene.frame_current + frame_step, scene.frame_end)
        sum_sp = 0.0
        for o in ped_objs:
            p0 = o.matrix_world.translation.copy()
            scene.frame_set(fe); p1 = o.matrix_world.translation.copy()
            scene.frame_set(fs)
            v = (p1 - p0).length / ( (fe-fs)/max(scene.render.fps,1) or 1e-6 )
            if in_fov_visible(o): sum_sp += v
    finally:
        scene.frame_set(cur)
    return Np, Nv, sum_sp

# ---------------- Dynamism (entropy × intensity) ----------------
def _bin3(x): return 0 if x < 1/3 else (1 if x < 2/3 else 2)

def _norm_entropy(hist_counts, k):
    n = sum(hist_counts)
    if n == 0: return 0.0
    H = 0.0
    for c in hist_counts:
        if c <= 0: continue
        p = c / n
        H -= p * math.log(p + 1e-12)
    return H / math.log(max(k, 2))

def dynamism_index_entropy(probe, frames_sampled=30, fwd_az=0.0):
    scene = bpy.context.scene
    fs, fe = scene.frame_start, scene.frame_end
    step = max(1, (fe - fs)//max(1, frames_sampled))
    ped_speakers, veh_speakers, amb_speakers = speakers_by_class()
    M_vals = []; A_vals = []
    cur = scene.frame_current
    try:
        for f in range(fs, fe+1, step):
            scene.frame_set(f)
            Np, Nv, _sum_sp = visible_agents_and_ped_speed(probe, fwd_az=fwd_az)
            M = max(0.0, min(1.0, (Np + 0.5*Nv)/12.0))
            M_vals.append(M)
            Lp, Lv, _La = analytic_loudness_at(probe, ped_speakers, veh_speakers, amb_speakers)
            A_vals.append(0.7*Lp + 0.3*Lv)
    finally:
        scene.frame_set(cur)
    if not M_vals or not A_vals:
        return 0.0
    As = sorted(A_vals)
    A90 = As[int(0.9*(len(As)-1))]; A90 = max(A90, 1e-6)
    S_vals = [max(0.0, min(1.0, a / A90)) for a in A_vals]
    counts = Counter()
    for M, S in zip(M_vals, S_vals):
        counts[(_bin3(M), _bin3(S))] += 1
    hist = [counts.get((i,j), 0) for i in range(3) for j in range(3)]
    H = _norm_entropy(hist, 9)
    I = 0.5 * (sum(M_vals)/len(M_vals) + sum(S_vals)/len(S_vals))
    return 100.0 * H * I

def dynamism_metric_across_probes(probes, fwd_az):
    vals = [dynamism_index_entropy(p, fwd_az=fwd_az) for p in probes]
    return sum(vals)/max(1, len(vals))

# ---------------- Loudness & impulsivity (kept) ----------------
def loudness_index_and_ir(probe, frames_sampled=50):
    scene = bpy.context.scene
    fs, fe = scene.frame_start, scene.frame_end
    step = max(1, (fe - fs)//max(1, frames_sampled))
    ped_speakers, veh_speakers, amb_speakers = speakers_by_class()
    A_vals=[]
    cur = scene.frame_current
    try:
        for f in range(fs, fe+1, step):
            scene.frame_set(f)
            Lp, Lv, La = analytic_loudness_at(probe, ped_speakers, veh_speakers, amb_speakers)
            A_vals.append(0.6*Lp + 0.3*Lv + 0.1*La)
    finally:
        scene.frame_set(cur)
    if not A_vals: return 0.0, 0.0
    s = sorted(A_vals)
    Aref = max(1e-6, s[int(0.9*(len(s)-1))])
    loud_idx = max(0.0, min(1.0, sum(a/Aref for a in A_vals)/len(A_vals)))
    meanA = sum(A_vals)/len(A_vals)
    thr = 1.5 * meanA
    ir = sum(1 for a in A_vals if a >= thr) / len(A_vals)
    return loud_idx, ir

# ---------------- Shade fraction with solar model ----------------
def _day_of_year(dt_local):
    return dt_local.timetuple().tm_yday

def solar_pos_az_el(lat_deg, lon_deg, dt_local, tz_offset_hours=0.0):
    # NOAA-like analytic; azimuth CW from North (0=N, 90=E), elevation deg
    h = dt_local.hour + dt_local.minute/60.0 + dt_local.second/3600.0
    n = _day_of_year(dt_local)
    gamma = 2.0*math.pi/365.0 * (n - 1 + (h - 12.0)/24.0)
    decl = (0.006918
            - 0.399912*math.cos(gamma)
            + 0.070257*math.sin(gamma)
            - 0.006758*math.cos(2*gamma)
            + 0.000907*math.sin(2*gamma)
            - 0.002697*math.cos(3*gamma)
            + 0.00148 *math.sin(3*gamma))
    eq_time = 229.18*(0.000075
                      + 0.001868*math.cos(gamma)
                      - 0.032077*math.sin(gamma)
                      - 0.014615*math.cos(2*gamma)
                      - 0.040849*math.sin(2*gamma))
    tst = h*60.0 + eq_time + 4.0*lon_deg - 60.0*tz_offset_hours
    ha = (tst/4.0) - 180.0
    lat = math.radians(lat_deg); ha = math.radians(ha)
    cos_zen = (math.sin(lat)*math.sin(decl) + math.cos(lat)*math.cos(decl)*math.cos(ha))
    cos_zen = max(-1.0, min(1.0, cos_zen))
    zen = math.acos(cos_zen)
    el  = math.degrees(math.pi/2 - zen)
    sin_az = math.sin(ha) * math.cos(decl) / max(math.sin(zen), 1e-9)
    cos_az = (math.sin(decl) - math.sin(lat)*math.cos(zen)) / max(math.cos(lat)*math.sin(zen), 1e-9)
    az = math.degrees(math.atan2(sin_az, cos_az))
    if az < 0: az += 360.0
    return az, el

def sun_dir_from_az_el(az_deg_from_north_cw, el_deg, north_deg=0.0):
    az = math.radians((az_deg_from_north_cw - north_deg) % 360.0)
    el = math.radians(el_deg)
    x = math.sin(az) * math.cos(el)
    y = math.cos(az) * math.cos(el)
    z = math.sin(el)
    return Vector((x, y, z)).normalized()

def shade_fraction_on_sidewalks_with_dir(dir_to_sun, samples_per_side=200):
    """Robust shade: origins above sidewalk, ignore self/roads/env-domes, count only real blockers."""
    dir_to_sun = dir_to_sun.normalized()
    sc_mn, sc_mx = scene_bounds()
    def is_env_dome(obj):
        nm = obj.name.lower()
        if any(k in nm for k in ["skydome","sky","hdr","hdri","env","environment","dome","background","world"]):
            return True
        mn, mx = world_bbox(obj)
        return (mn.x <= sc_mn.x-0.1 and mx.x >= sc_mx.x+0.1 and
                mn.y <= sc_mn.y-0.1 and mx.y >= sc_mx.y+0.1 and
                mn.z <= sc_mn.z-0.1 and mx.z >= sc_mx.z+0.1)
    def is_blocker(obj):
        if obj is None: return False
        nm = obj.name.lower()
        if ("building" in nm) or ("facade" in nm) or ("façade" in nm): return True
        if ("tree" in nm) or ("branch" in nm) or ("trunk" in nm):     return True
        if ("car" in nm) or ("parkedcar" in nm) or ("vehicle" in nm) or ("bus" in nm) or ("truck" in nm): return True
        if ("roof" in nm) or ("awning" in nm) or ("canopy" in nm):    return True
        if ("wall" in nm) or ("fence" in nm):                         return True
        return False
    Lsw, Rsw, all_sw = find_sidewalks_lr()
    targets = all_sw if all_sw else [o for o in bpy.data.objects if o.type == 'MESH' and name_has_sidewalk(o.name)]
    if not targets:
        return 0.0
    hits = 0; total = 0
    for sw in targets:
        mn, mx = world_bbox(sw)
        count = max(1, int(round(samples_per_side / max(1, len(targets)))))
        for i in range(count):
            u = (i * 1.3247) % 1.0
            v = (i * 0.6180) % 1.0
            x = mn.x + u * (mx.x - mn.x)
            y = mn.y + v * (mx.y - mn.y)
            origin = Vector((x, y, mx.z + 0.05)) + dir_to_sun * 0.02
            shaded = False; start  = origin
            for _ in range(3):
                hit, loc, obj, d = ray_cast(start, dir_to_sun, dist=1000.0)
                if not hit:
                    shaded = False; break
                nm = obj.name.lower()
                if obj == sw or name_has_sidewalk(nm) or name_has_road(nm) or is_env_dome(obj):
                    start = loc + dir_to_sun*0.05; continue
                shaded = is_blocker(obj); break
            total += 1
            if shaded: hits += 1
    return hits / max(1, total)

def shade_fraction_timewindow(lat, lon, date_ymd="2025-06-21",
                              hour_start=15.0, hour_end=16.0,
                              tz_offset_hours=0.0, north_deg=0.0,
                              step_minutes=10, samples_per_side=200):
    """Average shade fraction over a local time window using integer-minute steps."""
    y, m, d = [int(t) for t in date_ymd.split("-")]
    start_min = max(0, int(round(hour_start * 60.0)))
    end_min   = max(start_min, int(round(hour_end   * 60.0)))
    step      = max(1, int(step_minutes))
    vals = []
    for minute in range(start_min, end_min + 1, step):
        hh, mm = divmod(minute, 60)
        if not (0 <= hh <= 23): continue
        dt_local = _dt.datetime(y, m, d, hh, mm, 0)
        az, el = solar_pos_az_el(lat, lon, dt_local, tz_offset_hours=tz_offset_hours)
        if el <= 0.0: continue
        sdir = sun_dir_from_az_el(az, el, north_deg=north_deg)
        vals.append(shade_fraction_on_sidewalks_with_dir(sdir, samples_per_side=samples_per_side))
    return (sum(vals) / len(vals)) if vals else 0.0

# ---------------- Metric wrappers ----------------
def prospect_area_norm_metric(probes, n_rays=360, R=100.0, verbose=True):
    vals = []
    for i, p in enumerate(probes):
        v = isovist_area_norm_at(p, n_rays=n_rays, maxdist=R)
        vals.append(v)
        if verbose:
            A_norm = v
            print(f"[ProspectArea] Probe {i}: area_norm={A_norm:.4f}  R={{R:.1f}}  rays={n_rays}  at ({p.x:.2f},{p.y:.2f},{p.z:.2f})")
    mean_v = sum(vals)/max(1,len(vals))
    if verbose:
        print(f"[ProspectArea] Probes={len(vals)}  Prospect (normalized isovist area) MEAN = {mean_v:.4f}")
    return mean_v

def prospect_compactness(probes, n_rays=360, maxdist=300.0):
    vals = [isovist_compactness_at(p, n_rays=n_rays, maxdist=maxdist) for p in probes]
    return sum(vals)/max(1,len(vals))

def svf_metric(probes, rays=400):
    vals = [svf_at(p, rays=rays) for p in probes]
    return sum(vals)/max(1,len(vals))

def loudness_and_ir_metric(probes):
    Ls, IRs = [], []
    for p in probes:
        L, IR = loudness_index_and_ir(p)
        Ls.append(L); IRs.append(IR)
    return (sum(Ls)/max(1,len(Ls))), (sum(IRs)/max(1,len(IRs)))

# ---------------- Main ----------------
def main():
    args = parse_args()
    out_path = args["out"] or os.path.join(os.path.dirname(bpy.data.filepath) or ".", "street_metrics.json")

    # Probes & forward azimuth
    probes, fwd_az = pick_probes(probe_mode=args["probe_mode"],
                                 probes_n=args["probes_n"],
                                 probes_per_side=args["probes_per_side"])

    metrics = {}

    # Enclosure (both means)
    enc_all, enc_valid = enclosure_HW_with_zeros(step=5.0)
    metrics['enclosure_HW_incl_zeros'] = round(enc_all, 4)
    metrics['enclosure_HW_valid_only'] = round(enc_valid, 4)

    # Prospect (primary): area-normalized isovist
    metrics["prospect_area_norm"] = round(
        prospect_area_norm_metric(probes, n_rays=args["isovist_rays"], R=args["isovist_maxdist"], verbose=True), 4
    )

    # (Optional legacy compactness in JSON)
    metrics["prospect_compactness"] = round(
        prospect_compactness(probes, n_rays=args["isovist_rays"], maxdist=args["isovist_maxdist"]), 4
    )

    # SVF
    metrics["svf"] = round(svf_metric(probes, rays=args["svf_rays"]), 4)

    # Loudness + impulsivity
    L_idx, IR = loudness_and_ir_metric(probes)
    metrics["loudness_idx"] = round(L_idx, 4)
    metrics["sound_IR"]     = round(IR, 4)

    # Dynamism
    metrics["dynamism_index"] = round(dynamism_metric_across_probes(probes, fwd_az), 2)

    # Shade
    metrics["shade_fraction"] = round(shade_fraction_timewindow(
                                         lat=args["lat"], lon=args["lon"],
                                         date_ymd=args["date"],
                                         hour_start=args["tstart"], hour_end=args["tend"],
                                         tz_offset_hours=args["tz"], north_deg=args["north"],
                                         step_minutes=10, samples_per_side=200
                                      ), 4)

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[StreetMetrics] Saved metrics to: {out_path}")
        print(json.dumps(metrics, indent=2))
    except Exception as e:
        print(f"[StreetMetrics] ERROR writing metrics: {e}")

if __name__ == "__main__":
    main()
