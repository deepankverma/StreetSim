# Isovist/prospect viz + metric at eye height, with per-side probe placement and
# normalized isovist area output (area(ray polygon) / (pi * R^2)).
#
# Defaults (as requested):
#   --probe_mode ped
#   --probes_per_side 1
#   --radius 100
#
# CLI flags:
#   --probe_mode ped|center|both
#   --probes_per_side N         (rows along street per lateral track; default 1)
#   --probes N                  (total probes, only used if probes_per_side=0)
#   --radius R                  (default 100)
#   --rays_iso K                (default 360)
#   --thick T                   (default 0.02; curve bevel)
#   --eye_h H                   (default 1.6)
#   --respect_user_z true|false (default false)
#   --ignore_user_probes true|false (default false)
#   --flatten true|false        (default true; project hits to eye plane)
#   --clear true|false          (default true)
#   --save_as path.blend        (optional)
#
# Example:
# blender your_scene.blend --background --python metrics_viz_in_scene_fixed.py -- \
#   --probe_mode ped --probes_per_side 1 --radius 100 --rays_iso 360 --flatten true

import bpy, sys, math
from mathutils import Vector

# ---------- Args ----------
def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--")+1:]
    else:
        argv = []
    args = {
        "probe_mode": "ped",      # default = ped
        "probes_n": 1,            # used only if probes_per_side == 0
        "probes_per_side": 0,     # default = 1 per side (so usually 2 total in ped mode)
        "radius": 50.0,          # default = 100 m visibility cap
        "rays_iso": 360,
        "thick": 0.02,
        "clear": True,
        "save_as": None,
        "eye_h": 2.2,
        "respect_user_z": False,
        "flatten": True,
        "ignore_user_probes": False,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--probe_mode" and i+1 < len(argv):
            args["probe_mode"] = argv[i+1]; i += 2; continue
        if a == "--probes" and i+1 < len(argv):
            args["probes_n"] = max(1, int(argv[i+1])); i += 2; continue
        if a == "--probes_per_side" and i+1 < len(argv):
            args["probes_per_side"] = max(0, int(argv[i+1])); i += 2; continue
        if a == "--radius" and i+1 < len(argv):
            args["radius"] = float(argv[i+1]); i += 2; continue
        if a == "--rays_iso" and i+1 < len(argv):
            args["rays_iso"] = max(8, int(argv[i+1])); i += 2; continue
        if a == "--thick" and i+1 < len(argv):
            args["thick"] = float(argv[i+1]); i += 2; continue
        if a == "--clear" and i+1 < len(argv):
            args["clear"] = argv[i+1].lower() in ("1","true","yes","y"); i += 2; continue
        if a == "--save_as" and i+1 < len(argv):
            args["save_as"] = argv[i+1]; i += 2; continue
        if a == "--eye_h" and i+1 < len(argv):
            args["eye_h"] = float(argv[i+1]); i += 2; continue
        if a == "--respect_user_z" and i+1 < len(argv):
            args["respect_user_z"] = argv[i+1].lower() in ("1","true","yes","y"); i += 2; continue
        if a == "--ignore_user_probes" and i+1 < len(argv):
            args["ignore_user_probes"] = argv[i+1].lower() in ("1","true","yes","y"); i += 2; continue
        if a == "--flatten" and i+1 < len(argv):
            args["flatten"] = argv[i+1].lower() in ("1","true","yes","y"); i += 2; continue
        i += 1
    return args

# ---------- Utils ----------
def dg():
    return bpy.context.evaluated_depsgraph_get()

def ensure_emissive(name, rgba, strength=3.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emi = nt.nodes.new("ShaderNodeEmission")
    emi.inputs["Color"].default_value = rgba
    emi.inputs["Strength"].default_value = strength
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])
    return mat

def ray_cast(origin, direction, dist=1e6):
    hit, loc, norm, face_idx, obj, _ = bpy.context.scene.ray_cast(dg(), origin, direction, distance=dist)
    return hit, loc, obj

# Add this helper anywhere above ray_cast_skip_viz:
# --- helpers: classify objects we should/shouldn't skip -----------------------

def looks_like_structure(obj):
    """True for occluding built stuff we must NOT skip."""
    if obj is None:
        return False
    nm = obj.name.lower()
    return any(t in nm for t in [
        "building", "facade", "façade", "roof", "wall", "fence",
        "parapet", "balcony", "bridge", "storefront"
    ])
def is_viz_obj(obj):
    """Any isovist viz we created (so rays ignore it)."""
    if obj is None:
        return False
    if obj.get("is_isovist_viz"):
        return True
    nm = obj.name
    return nm.startswith(("IsovistRays", "IsovistPoly"))

def is_structure_or_vehicle(obj):
    """Things that MUST occlude (never skip these)."""
    if obj is None:
        return False
    nm = obj.name.lower()
    return any(t in nm for t in [
        # built mass
        "building","facade","façade","roof","wall","fence","parapet","balcony","bridge","storefront",
        # vehicles
        "vehicle","car","bus","truck","van","tram","train","bike","bicycle","motorcycle","scooter",
        # vegetation
        "tree","trunk","branch","leaf","plant"
    ])

def is_ped_strict(obj):
    """
    True only for actual pedestrians (meshes/rigs), not for 'ped' in lane/collection names.
    """
    if obj is None:
        return False
    # Guard: if it clearly looks like structure/vehicle/veg, NOT a pedestrian.
    if is_structure_or_vehicle(obj):
        return False

    nm = obj.name.lower()

    # Strong/explicit human patterns (do NOT use raw 'ped')
    if nm.startswith(("human_", "human.")) or nm.startswith("human"):
        return True
    if any(w in (" " + nm + " ").replace("_"," ") for w in (" person ", " pedestrian ")):
        return True

    # Allow an ARMATURE parent called Human_* etc.
    par = obj.parent
    if par and par.type == 'ARMATURE':
        pn = par.name.lower()
        if pn.startswith(("human_", "human.")) or pn.startswith("human"):
            return True
        if any(w in (" " + pn + " ").replace("_"," ") for w in (" person ", " pedestrian ")):
            return True

    return False

def ray_cast_skip_viz_and_peds(origin, direction, dist=1e6, max_hops=12, eps=1e-4):
    """
    Ray cast that ignores:
      • our own viz geometry (IsovistRays*/IsovistPoly* or tagged is_isovist_viz)
      • actual pedestrians (Human_* / Person / Pedestrian),
    but NEVER skips buildings/vehicles/trees.
    """
    deps = bpy.context.evaluated_depsgraph_get()
    start = origin.copy()
    remain = dist

    for _ in range(max_hops):
        hit, loc, norm, face_idx, obj, _ = bpy.context.scene.ray_cast(deps, start, direction, distance=remain)
        if not hit:
            return False, None, None

        # Always ignore our viz
        if is_viz_obj(obj):
            step = (loc - start).length + eps
            start = start + direction * step
            remain = max(0.0, dist - (start - origin).length)
            continue

        # Ignore pedestrians ONLY (and only if it's not a structure/vehicle/veg)
        if is_ped_strict(obj):
            step = (loc - start).length + eps
            start = start + direction * step
            remain = max(0.0, dist - (start - origin).length)
            continue

        # Real occluder hit
        return True, loc, obj

    # No hit within distance after hops
    return False, None, None

def is_viz_obj(obj):
    """Our own viz geometry (rays/polys) that must be ignored in ray-casts."""
    if obj is None:
        return False
    if obj.get("is_isovist_viz"):
        return True
    nm = obj.name
    return nm.startswith(("IsovistRays", "IsovistPoly"))

def is_ped(obj):
    """
    Return True only for *actual* pedestrian meshes/rigs.
    Avoid generic 'ped' substring — it causes false positives.
    """
    if obj is None:
        return False

    # If it obviously looks like any structure/vehicle, do NOT treat as ped
    if looks_like_structure(obj):
        return False

    nm = obj.name.lower()

    # explicit tags win
    if obj.get("is_ped") is True:
        return True

    # strong human patterns (safer than raw "ped")
    human_tokens = ("human_", "human.", "human", "person", "pedestrian")
    if nm.startswith(("human_", "human.")) or any(f"_{t}_" in f"_{nm}_" for t in human_tokens):
        return True

    # also allow parent if it's an ARMATURE with human-ish name
    par = obj.parent
    if par and par.type == 'ARMATURE':
        pn = par.name.lower()
        if pn.startswith(("human_", "human.")) or any(f"_{t}_" in f"_{pn}_" for t in human_tokens):
            return True

    return False

# --- ray cast that skips only viz + real pedestrians --------------------------

def ray_cast_skip_viz(origin, direction, dist=1e6, max_hops=12, eps=1e-4):
    """
    Ray cast that ignores:
      • our viz geometry (IsovistRays*/IsovistPoly* or tagged is_isovist_viz)
      • pedestrians (Human_*, Person, Pedestrian) — but NEVER buildings/vehicles/etc.
    """
    deps = dg()
    start = origin.copy()
    remain = dist

    for _ in range(max_hops):
        hit, loc, norm, face_idx, obj, _ = bpy.context.scene.ray_cast(deps, start, direction, distance=remain)
        if not hit:
            return False, None, None

        # Skip viz geometry unconditionally
        if is_viz_obj(obj):
            step = (loc - start).length + eps
            start = start + direction * step
            remain = max(0.0, dist - (start - origin).length)
            continue

        # Skip pedestrians only (never skip if it looks like a structure)
        if is_ped(obj):
            step = (loc - start).length + eps
            start = start + direction * step
            remain = max(0.0, dist - (start - origin).length)
            continue

        # Real hit (building, vehicle, tree, etc.)
        return True, loc, obj

    # Gave up hopping; treat as no hit
    return False, None, None


# --- drop in (reuse from enclosure.py) ---
def ensure_emission_alpha(name, rgb, alpha=0.25, strength=1.0):
    """
    Emission mixed with Transparent BSDF using 'alpha' as opacity.
    Works in Eevee/Cycles; guards shadow settings across Blender versions.
    """
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out   = nt.nodes.new("ShaderNodeOutputMaterial")
    emis  = nt.nodes.new("ShaderNodeEmission")
    trans = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix   = nt.nodes.new("ShaderNodeMixShader")

    emis.inputs["Color"].default_value    = (rgb[0], rgb[1], rgb[2], 1.0)
    emis.inputs["Strength"].default_value = strength
    # Mix factor = 1 - alpha → output = alpha*Emission + (1-alpha)*Transparent
    mix.inputs["Fac"].default_value = max(0.0, min(1.0, 1.0 - float(alpha)))

    nt.links.new(emis.outputs["Emission"], mix.inputs[1])
    nt.links.new(trans.outputs["BSDF"],    mix.inputs[2])
    nt.links.new(mix.outputs["Shader"],    out.inputs["Surface"])

    mat.blend_method = 'BLEND'
    if hasattr(mat, 'shadow_method'):   # 2.9–3.6
        mat.shadow_method = 'NONE'
    elif hasattr(mat, 'shadow_mode'):   # 4.x fallback
        mat.shadow_mode = 'NONE'
    mat.use_backface_culling = False
    return mat
# -----------------------------------------





def obj_midpoint_xy(o):
    bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
    xs = [p.x for p in bb]; ys = [p.y for p in bb]; zs = [p.z for p in bb]
    return Vector(((min(xs)+max(xs))*0.5, (min(ys)+max(ys))*0.5, (min(zs)+max(zs))*0.5))

def scene_bounds(objs=None):
    if objs is None:
        objs = [o for o in bpy.data.objects if o.type in {'MESH','CURVE','SURFACE','META'} and (not getattr(o, "hide_render", False))]
    mins = Vector((1e9,1e9,1e9)); maxs = Vector((-1e9,-1e9,-1e9)); ok=False
    for o in objs:
        try:
            bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
        except Exception:
            continue
        xs=[p.x for p in bb]; ys=[p.y for p in bb]; zs=[p.z for p in bb]
        mn=Vector((min(xs),min(ys),min(zs))); mx=Vector((max(xs),max(ys),max(zs)))
        mins.x=min(mins.x,mn.x); mins.y=min(mins.y,mn.y); mins.z=min(mins.z,mn.z)
        maxs.x=max(maxs.x,mx.x); maxs.y=max(maxs.y,mx.y); maxs.z=max(maxs.z,mx.z)
        ok=True
    if not ok:
        mins=Vector((-5,-5,0)); maxs=Vector((5,5,3))
    return mins, maxs

def name_has_sidewalk(nm: str) -> bool:
    nm = nm.lower(); return any(k in nm for k in ["footpath","sidewalk","fp","pavement","walkway"])

def name_has_road(nm: str) -> bool:
    nm = nm.lower(); return any(k in nm for k in ["driveway","carriage","carriageway","driving","road","lane","asphalt","tarmac","street"])

def is_ground_like(obj):
    if obj is None: return False
    nm = obj.name.lower()
    return name_has_sidewalk(nm) or name_has_road(nm) or any(k in nm for k in ["ground","plaza","parking","curb","kerb"])

def find_sidewalks_lr():
    L, R = [], []
    for o in bpy.data.objects:
        if o.type != 'MESH' or getattr(o, "hide_render", False): continue
        nm = o.name.lower()
        if nm.startswith("left_") and name_has_sidewalk(nm): L.append(o)
        elif nm.startswith("right_") and name_has_sidewalk(nm): R.append(o)
    return L, R, (L+R)

def find_roads_left_right():
    L, R = [], []
    for o in bpy.data.objects:
        if o.type != 'MESH' or getattr(o, "hide_render", False): continue
        nm = o.name.lower()
        if nm.startswith("left_") and name_has_road(nm): L.append(o)
        elif nm.startswith("right_") and name_has_road(nm): R.append(o)
    return L, R, (L+R)

def ground_z_at(x, y, z_top=200.0, eye_h=1.6, max_iters=12, eps=0.01):
    """Find local ground z (sidewalk/road/plaza) beneath (x,y); return z + eye_h."""
    origin = Vector((x,y,z_top)); z=None; last_z=z_top
    for _ in range(max_iters):
        hit, loc, obj = ray_cast(origin, Vector((0,0,-1)), dist=abs(last_z)+z_top)
        if not hit: break
        if is_ground_like(obj): z=loc.z; break
        last_z = loc.z - eps; origin = Vector((x,y,last_z))
    if z is None:
        origin = Vector((x,y,-z_top)); last_z = -z_top
        for _ in range(max_iters):
            hit, loc, obj = ray_cast(origin, Vector((0,0,1)), dist=2*z_top)
            if not hit: break
            if is_ground_like(obj): z=loc.z; break
            last_z = loc.z + eps; origin = Vector((x,y,last_z))
    if z is None: return eye_h
    return z + eye_h

def pick_probes(probe_mode='ped', probes_n=1, eye_h=1.6, respect_user_z=False, probes_per_side=1, ignore_user_probes=False):
    # If user-placed empties exist and we are NOT in per-side mode, use them (up to probes_n).
    user = [o for o in bpy.data.objects if o.type=='EMPTY' and o.name.lower().startswith('probe')]
    if user and not ignore_user_probes and (probes_per_side or 0) <= 0:
        user_sorted = sorted(user, key=lambda o: o.name.lower())
        if probes_n is not None and probes_n > 0:
            user_sorted = user_sorted[:probes_n]
        out = []
        for o in user_sorted:
            P = o.matrix_world.translation.copy()
            if not respect_user_z:
                mn, mx = scene_bounds()
                P.z = ground_z_at(P.x, P.y, z_top=max(50.0, (mx.z-mn.z)+10.0), eye_h=eye_h)
            out.append(P)
        print(f"[Isovist] Using {len(out)} user PROBE_* empties.")
        return out

    # Otherwise auto-place per side/track
    Lsw, Rsw, AllSw = find_sidewalks_lr()
    Lrd, Rrd, AllRd = find_roads_left_right()
    fallback_objs = AllSw if AllSw else (AllRd if AllRd else [o for o in bpy.data.objects if o.type=='MESH' and (not getattr(o, 'hide_render', False))])
    mn, mx = scene_bounds(fallback_objs)
    y_min, y_max = mn.y, mx.y

    x_tracks = []
    if probe_mode in ('ped','both') and (Lsw or Rsw):
        xL = (sum(obj_midpoint_xy(o).x for o in Lsw)/max(1,len(Lsw))) if Lsw else None
        xR = (sum(obj_midpoint_xy(o).x for o in Rsw)/max(1,len(Rsw))) if Rsw else None
        if xL is not None: x_tracks.append(xL)
        if xR is not None: x_tracks.append(xR)
    elif probe_mode in ('center','both') and (AllRd or AllSw):
        src = AllRd if AllRd else AllSw
        rmn, rmx = scene_bounds(src)
        x_tracks.append(0.5*(rmn.x + rmx.x))
    elif (Lrd or Rrd):
        rmn, rmx = scene_bounds(AllRd)
        margin = 1.25
        x_tracks.append(rmn.x - margin); x_tracks.append(rmx.x + margin)
    else:
        W = mx.x - mn.x
        x_tracks.append(mn.x + 0.25*W); x_tracks.append(mx.x - 0.25*W)

    if not x_tracks:
        x_tracks = [0.5*(mn.x + mx.x)]

    # Place exactly probes_per_side rows on each track
    rows = max(1, int(probes_per_side)) if probes_per_side else 1
    if rows == 1:
        ys = [0.5*(y_min + y_max)]
    else:
        seg = (y_max - y_min) / rows
        ys = [y_min + (i+0.5)*seg for i in range(rows)]

    probes = []
    for x in x_tracks:
        for y in ys:
            z = ground_z_at(x, y, z_top=max(50.0, (mx.z-mn.z)+10.0), eye_h=eye_h)
            probes.append(Vector((x,y,z)))
    print(f"[Isovist] Placed {len(probes)} probes = {rows} per side × {len(x_tracks)} side(s).")
    return probes

# ---------- Materials & Collections ----------
def ensure_collection(name, parent=None, clear=False):
    coll = bpy.data.collections.get(name)
    if coll and clear:
        for scn in bpy.data.scenes:
            try: scn.collection.children.unlink(coll)
            except Exception: pass
        bpy.data.collections.remove(coll); coll = None
    if coll is None:
        coll = bpy.data.collections.new(name)
        if parent: parent.children.link(coll)
        else: bpy.context.scene.collection.children.link(coll)
    return coll

def ensure_viz_materials():
    return {
        # Make the FILL semi-transparent (tweak alpha 0..1; lower = more transparent)
        "isovist_fill": ensure_emission_alpha("viz_isovist_fill",
                                              (1.00, 0.85, 0.25),  # warm fill color
                                              alpha=0.25,          # opacity ~25%
                                              strength=1.0),

        # Keep rays opaque (or set them however you like)
        "rays":  ensure_emissive("viz_isovist_rays", (0.0, 0.0, 0.0, 1.0), strength=1.0),

        "probe": ensure_emissive("viz_probe", (0.00, 0.00, 0.00, 1.0), strength=1.0),
    }



# ---------- Builders ----------
def add_curve_object(name, splines_points_lists, thickness=0.02, material=None, collection=None):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.fill_mode = 'FULL'
    cu.bevel_depth = thickness
    for pts in splines_points_lists:
        sp = cu.splines.new('POLY')
        sp.points.add(len(pts)-1)
        for i,p in enumerate(pts):
            sp.points[i].co = (p[0], p[1], p[2], 1.0)
    ob = bpy.data.objects.new(name, cu)
    ob["is_isovist_viz"] = True  # tag so rays ignore this
    if material: ob.data.materials.append(material)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob

def add_mesh_polygon(name, verts_xy, z=0.02, material=None, collection=None):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    vs = [bm.verts.new((x,y,z)) for (x,y) in verts_xy]
    if len(vs) >= 3:
        bm.faces.new(vs)
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    ob["is_isovist_viz"] = True  # tag so rays ignore this
    if material: ob.data.materials.append(material)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob

# ---------- Isovist construction + metric ----------
def build_isovist_and_metric(probe, R, rays, mats, coll, thickness, flatten=True):
    """
    Returns dict with:
      distances: [d_i], endpoints_xy: [(x,y)], area, area_norm, mean_d_norm, horizon_frac
    """
    distances = []
    endpoints_xy = []
    ray_splines = []

    for i in range(rays):
        ang = 2*math.pi*i/rays
        dir3 = Vector((math.cos(ang), math.sin(ang), 0.0))
        hit, loc, obj = ray_cast_skip_viz_and_peds(probe, dir3, R)
        if hit:
            end = Vector((loc.x, loc.y, probe.z if flatten else loc.z))
            d = (Vector((end.x, end.y, probe.z)) - Vector((probe.x, probe.y, probe.z))).length
        else:
            end = Vector((probe.x, probe.y, probe.z)) + dir3*R
            d = R
        distances.append(d)
        endpoints_xy.append((end.x, end.y))
        ray_splines.append([(probe.x, probe.y, probe.z), (end.x, end.y, end.z)])

    # Shoelace area in XY
    A = 0.0
    n = len(endpoints_xy)
    if n >= 3:
        for i in range(n):
            x1, y1 = endpoints_xy[i]
            x2, y2 = endpoints_xy[(i+1) % n]
            A += x1*y2 - x2*y1
        A = abs(0.5 * A)
    A_disc = math.pi * (R**2)
    area_norm = (A / A_disc) if A_disc > 0 else 0.0
    mean_d_norm = sum(distances) / (n * R) if (n and R > 0) else 0.0
    horizon_frac = sum(1 for d in distances if abs(d - R) < 1e-6) / n if n else 0.0

    # Draw viz (one fill + one rays object)
    add_mesh_polygon("IsovistPoly", endpoints_xy, z=probe.z+0.02, material=mats["isovist_fill"], collection=coll)
    add_curve_object("IsovistRays", ray_splines, thickness=thickness, material=mats["rays"], collection=coll)

    return {
        "distances": distances,
        "endpoints_xy": endpoints_xy,
        "area": A,
        "area_norm": area_norm,
        "mean_d_norm": mean_d_norm,
        "horizon_frac": horizon_frac,
    }

# ---------- Main ----------
def main():
    args = parse_args()

    root = ensure_collection("MetricsViz", clear=args["clear"])
    mats = ensure_viz_materials()

    probes = pick_probes(
        probe_mode=args["probe_mode"],
        probes_n=args["probes_n"],
        eye_h=args.get("eye_h", 2.2),
        respect_user_z=args.get("respect_user_z", False),
        probes_per_side=args["probes_per_side"],
        ignore_user_probes=args.get("ignore_user_probes", False),
    )


        # --- keep only one probe (Probe 0) ---
    if isinstance(probes, tuple):  # some versions return (probes, fwd_az)
        probes, _fwd = probes

    if len(probes) > 1:
        print(f"[VizInScene] Keeping only Probe_00; dropping {len(probes)-1} extra.")
        probes = [probes[0]]

    if not probes:
        print("[Isovist] No probes found."); return

    results = []
    for idx, P in enumerate(probes):
        sub = ensure_collection(f"Probe_{idx:02d}", parent=root, clear=False)
        empty = bpy.data.objects.new(f"Probe_{idx:02d}_Empty", None)
        empty.empty_display_type = 'PLAIN_AXES'
        empty.empty_display_size = 0.3
        empty.location = P
        sub.objects.link(empty)

        res = build_isovist_and_metric(P, args["radius"], args["rays_iso"], mats, sub, args["thick"], flatten=args["flatten"])
        results.append(res)

        print(f"[Isovist] Probe {idx}: area={res['area']:.3f}  area_norm={res['area_norm']:.4f}  "
              f"mean_d_norm={res['mean_d_norm']:.4f}  horizon_frac={res['horizon_frac']:.3f}  "
              f"R={args['radius']}  rays={args['rays_iso']}  at {tuple(round(c,3) for c in P)}")

    # Average normalized area across all probes (with ped+1 per side, this is your 2-probe mean)
    if results:
        mean_area_norm = sum(r["area_norm"] for r in results) / len(results)
        print(f"[Isovist] Probes={len(results)}  Prospect (normalized isovist area) MEAN = {mean_area_norm:.4f}")

    if args["save_as"]:
        outp = bpy.path.abspath(args["save_as"])
        bpy.ops.wm.save_as_mainfile(filepath=outp, compress=False, copy=True)
        print(f"[Isovist] Saved as {outp}")

if __name__ == "__main__":
    main()
