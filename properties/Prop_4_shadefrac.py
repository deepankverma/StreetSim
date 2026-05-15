# shade_viz_in_scene.py
# Visualize sidewalk shade with Sun positioned from lat/lon/date/time.
# Produces a "ShadeViz" collection with sub-collections per timestamp:
#  - ShadePoints_SUN (green discs) and ShadePoints_SHADE (red discs)
#  - (optional) ShadeRays_SUN / ShadeRays_SHADE as curves
#
# Example:
# blender scene.blend --background --python shade_viz_in_scene.py -- \
#   --lat 52.52 --lon 13.405 --date 2025-06-21 --tstart 15 --tend 16 --step 10 \
#   --tz 2 --north 0 --samples 400 --draw_rays true --save_as out.blend
#
import bpy, sys, math, datetime as _dt
from mathutils import Vector

# ---------- CLI ----------
def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--")+1:]
    else:
        argv = []
    args = {
        "lat": 52.52,
        "lon": 13.405,
        "date": "2025-06-21",
        "tstart": 15.0,         # local hour
        "tend": 15.0,           # local hour
        "step": 10,             # minutes between samples for viz collections
        "tz": 2.0,              # hours offset from UTC (Berlin in June ~ CEST = 2)
        "north": 0.0,           # deg CW from +Y being North (0:+Y, 90:+X)
        "samples": 300,         # total samples across all sidewalks per timestamp
        "draw_rays": True,      # draw hit/miss rays toward sun
        "disc_radius": 0.8,    # meters
        "ray_maxdist": 600.0,
        "clear": True,
        "save_as": None,
    }
    i=0
    while i < len(argv):
        a = argv[i]
        if a=="--lat" and i+1<len(argv): args["lat"]=float(argv[i+1]); i+=2; continue
        if a=="--lon" and i+1<len(argv): args["lon"]=float(argv[i+1]); i+=2; continue
        if a=="--date" and i+1<len(argv): args["date"]=argv[i+1]; i+=2; continue
        if a=="--tstart" and i+1<len(argv): args["tstart"]=float(argv[i+1]); i+=2; continue
        if a=="--tend" and i+1<len(argv): args["tend"]=float(argv[i+1]); i+=2; continue
        if a=="--step" and i+1<len(argv): args["step"]=int(argv[i+1]); i+=2; continue
        if a=="--tz" and i+1<len(argv): args["tz"]=float(argv[i+1]); i+=2; continue
        if a=="--north" and i+1<len(argv): args["north"]=float(argv[i+1]); i+=2; continue
        if a=="--samples" and i+1<len(argv): args["samples"]=int(argv[i+1]); i+=2; continue
        if a=="--draw_rays" and i+1<len(argv): args["draw_rays"]=argv[i+1].lower() in ("1","true","yes","y"); i+=2; continue
        if a=="--disc_radius" and i+1<len(argv): args["disc_radius"]=float(argv[i+1]); i+=2; continue
        if a=="--ray_maxdist" and i+1<len(argv): args["ray_maxdist"]=float(argv[i+1]); i+=2; continue
        if a=="--clear" and i+1<len(argv): args["clear"]=argv[i+1].lower() in ("1","true","yes","y"); i+=2; continue
        if a=="--save_as" and i+1<len(argv): args["save_as"]=argv[i+1]; i+=2; continue
        i+=1
    return args

# ---------- Helpers ----------
def dg(): return bpy.context.evaluated_depsgraph_get()

def scene_ray(origin, direction, dist=1e6):
    hit, loc, norm, face_idx, obj, _ = bpy.context.scene.ray_cast(dg(), origin, direction, distance=dist)
    return hit, loc, obj

def world_bbox(obj):
    eo = obj.evaluated_get(dg())
    M  = eo.matrix_world
    try:
        corners = [M @ Vector(c) for c in eo.bound_box]
    except Exception:
        corners = [M @ v.co for v in eo.data.vertices]
    mn = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    mx = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return mn, mx

def scene_bounds(objs=None):
    if objs is None:
        objs = [o for o in bpy.data.objects if o.type in {'MESH','CURVE','SURFACE','META'} and not getattr(o,"hide_render",False)]
    mn = Vector(( 1e9, 1e9, 1e9)); mx = Vector((-1e9,-1e9,-1e9)); ok=False
    for o in objs:
        try:
            a,b = world_bbox(o)
        except Exception:
            continue
        mn.x=min(mn.x,a.x); mn.y=min(mn.y,a.y); mn.z=min(mn.z,a.z)
        mx.x=max(mx.x,b.x); mx.y=max(mx.y,b.y); mx.z=max(mx.z,b.z)
        ok=True
    if not ok:
        mn = Vector((-5,-5,0)); mx = Vector((5,5,3))
    return mn,mx

def ensure_coll(name, parent=None, clear=False):
    c = bpy.data.collections.get(name)
    if c and clear:
        # unlink from all scenes
        for scn in bpy.data.scenes:
            try: scn.collection.children.unlink(c)
            except Exception: pass
        bpy.data.collections.remove(c)
        c=None
    if not c:
        c = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(c)
    return c

def ensure_emissive(name, rgba, strength=3.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emi = nt.nodes.new("ShaderNodeEmission")
    emi.inputs["Color"].default_value = rgba
    emi.inputs["Strength"].default_value = strength
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])
    return m

def mats():
    return {
        "sun":   ensure_emissive("viz_sunlit", (0.10,0.85,0.20,1.0), strength=4.0),   # green
        "shade": ensure_emissive("viz_shaded", (0.95,0.20,0.20,1.0), strength=4.0),   # red
        "rayS":  ensure_emissive("viz_ray_sun", (0.10,0.85,0.20,1.0), strength=2.5),
        "rayH":  ensure_emissive("viz_ray_hit", (0.95,0.20,0.20,1.0), strength=2.5),
    }

def name_has_sidewalk(nm: str) -> bool:
    nm = nm.lower()
    return any(k in nm for k in ["footpath","sidewalk","fp"])

def name_has_road(nm: str) -> bool:
    nm = nm.lower()
    return any(k in nm for k in ["driveway","carriage","carriageway","driving","road","lane","asphalt"])

def find_sidewalks_lr():
    L, R = [], []
    for o in bpy.data.objects:
        if o.type!='MESH' or getattr(o,"hide_render",False): continue
        nm = o.name.lower()
        if nm.startswith("left_") and name_has_sidewalk(nm): L.append(o)
        elif nm.startswith("right_") and name_has_sidewalk(nm): R.append(o)
    return L, R, (L+R)

# ---------- Solar position ----------
def _doy(dt): return dt.timetuple().tm_yday

def solar_pos_az_el(lat_deg, lon_deg, dt_local, tz_offset_hours=0.0):
    # NOAA-like analytic; az CW from North (0=N, 90=E), el deg
    h = dt_local.hour + dt_local.minute/60.0 + dt_local.second/360.0
    n = _doy(dt_local)
    g = 2.0*math.pi/365.0 * (n - 1 + (h - 12.0)/24.0)
    decl = (0.006918
            - 0.399912*math.cos(g)
            + 0.070257*math.sin(g)
            - 0.006758*math.cos(2*g)
            + 0.000907*math.sin(2*g)
            - 0.002697*math.cos(3*g)
            + 0.00148 *math.sin(3*g))
    eqt = 229.18*(0.000075
                  + 0.001868*math.cos(g)
                  - 0.032077*math.sin(g)
                  - 0.014615*math.cos(2*g)
                  - 0.040849*math.sin(2*g))
    tst = h*60.0 + eqt + 4.0*lon_deg - 60.0*tz_offset_hours
    ha = (tst/4.0) - 180.0
    lat = math.radians(lat_deg); ha = math.radians(ha)
    cos_zen = (math.sin(lat)*math.sin(decl) + math.cos(lat)*math.cos(decl)*math.cos(ha))
    cos_zen = max(-1.0, min(1.0, cos_zen))
    zen = math.acos(cos_zen)
    el  = math.degrees(math.pi/2 - zen)
    sin_az = math.sin(ha)*math.cos(decl) / max(math.sin(zen), 1e-9)
    cos_az = (math.sin(decl) - math.sin(lat)*math.cos(zen)) / max(math.cos(lat)*math.sin(zen), 1e-9)
    az = math.degrees(math.atan2(sin_az, cos_az))
    if az < 0: az += 360.0
    return az, el

def sun_dir_from_az_el(az_deg_from_north_cw, el_deg, north_deg=0.0):
    az = math.radians((az_deg_from_north_cw - north_deg) % 360.0)
    el = math.radians(el_deg)
    x = math.sin(az)*math.cos(el)
    y = math.cos(az)*math.cos(el)
    z = math.sin(el)
    return Vector((x,y,z)).normalized()

def ensure_sun(name="Viz_Sun", dir_vec=None, strength=3.0, angle=0.5):
    sun = bpy.data.objects.get(name)
    if sun is None:
        data = bpy.data.lights.new(name=name, type='SUN')
        sun = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(sun)
    if dir_vec is not None:
        q = dir_vec.normalized().to_track_quat('-Z','Y')
        sun.rotation_euler = q.to_euler()
    try:
        sun.data.energy = strength
        sun.data.angle  = math.radians(angle)
    except Exception:
        pass
    return sun

# ---------- Build geometry ----------
def add_discs_mesh(name, centers, radius, mat, coll):
    # Build many little flat discs (8-gon) as a single mesh
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    seg = 8
    for c in centers:
        cx,cy,cz = c
        verts=[]
        for i in range(seg):
            ang = 2*math.pi*i/seg
            vx = cx + radius*math.cos(ang)
            vy = cy + radius*math.sin(ang)
            vz = cz
            verts.append(bm.verts.new((vx,vy,vz)))
        try:
            bm.faces.new(verts)
        except ValueError:
            # face may already exist if duplicates; ignore
            pass
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    if mat: 
        ob.data.materials.append(mat)
    coll.objects.link(ob)
    return ob

def add_segments_curve(name, segments, mat, coll, thickness=0.01):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.bevel_depth = thickness
    for s,e in segments:
        sp = cu.splines.new('POLY')
        sp.points.add(1)
        sp.points[0].co = (s[0], s[1], s[2], 1.0)
        sp.points[1].co = (e[0], e[1], e[2], 1.0)
    ob = bpy.data.objects.new(name, cu)
    if mat:
        ob.data.materials.append(mat)
    coll.objects.link(ob)
    return ob

def is_env_dome(obj, sc_mn=None, sc_mx=None):
    nm = obj.name.lower()
    if any(k in nm for k in ["skydome","sky","hdr","hdri","env","environment","dome","background","world"]):
        return True
    if sc_mn is None or sc_mx is None: 
        sc_mn, sc_mx = scene_bounds()
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

def make_timestamp_label(t_minutes):
    hh, mm = divmod(int(round(t_minutes)), 60)
    return f"{hh:02d}:{mm:02d}"

def build_shade_viz_for_time(coll_root, t_minutes, lat, lon, date_ymd, tz, north, total_samples, draw_rays, disc_radius, ray_maxdist):
    # Compute Sun direction
    y,m,d = [int(x) for x in date_ymd.split("-")]
    hh, mm = divmod(int(round(t_minutes)), 60)
    dt_local = _dt.datetime(y,m,d, hh, mm, 0)
    az, el = solar_pos_az_el(lat, lon, dt_local, tz_offset_hours=tz)
    if el <= 0:
        print(f"[ShadeViz] {make_timestamp_label(t_minutes)} sun below horizon (el={el:.2f}°). Skipping.")
        return None

    dir_sun = sun_dir_from_az_el(az, el, north_deg=north)
    ensure_sun(dir_vec=dir_sun, strength=3.0, angle=0.5)

    # Prepare collections/materials
    lab = make_timestamp_label(t_minutes).replace(":","h")
    coll = ensure_coll(f"Shade_{lab}", parent=coll_root, clear=False)
    m = mats()

    # Collect sidewalks
    Lsw, Rsw, all_sw = find_sidewalks_lr()
    sidewalks = all_sw if all_sw else [o for o in bpy.data.objects if o.type=='MESH' and name_has_sidewalk(o.name)]
    if not sidewalks:
        print("[ShadeViz] No sidewalk-like meshes found.")
        return None

    # Distribute samples across sidewalks
    per = max(1, total_samples // max(1,len(sidewalks)))

    sc_mn, sc_mx = scene_bounds()
    sun_pts   = []  # [(x,y,z)]
    shade_pts = []  # [(x,y,z)]
    rays_sun  = []  # [((x,y,z),(x2,y2,z2))]
    rays_hit  = []

    for sw in sidewalks:
        mn, mx = world_bbox(sw)
        for i in range(per):
            # low-discrepancy on bbox
            u = (i * 1.3247) % 1.0
            v = (i * 0.6180) % 1.0
            x = mn.x + u*(mx.x - mn.x)
            y = mn.y + v*(mx.y - mn.y)
            origin = Vector((x, y, mx.z + 0.05)) + dir_sun * 0.02

            shaded = False
            start  = origin
            end_for_ray = origin + dir_sun * ray_maxdist

            # up to 3 recasts to skip self/ground/env domes
            for _ in range(3):
                hit, loc, obj = scene_ray(start, dir_sun, dist=ray_maxdist)
                if not hit:
                    shaded = False
                    break
                # ignore self, roads/grounds, env domes
                nm = obj.name.lower()
                if obj == sw or name_has_sidewalk(nm) or name_has_road(nm) or is_env_dome(obj, sc_mn, sc_mx):
                    start = loc + dir_sun*0.05
                    end_for_ray = origin + dir_sun * ray_maxdist
                    continue
                shaded = is_blocker(obj)
                end_for_ray = loc
                break

            if shaded:
                shade_pts.append((origin.x, origin.y, origin.z))
                if draw_rays:
                    rays_hit.append(((origin.x,origin.y,origin.z),
                                     (end_for_ray.x,end_for_ray.y,end_for_ray.z)))
            else:
                sun_pts.append((origin.x, origin.y, origin.z))
                if draw_rays:
                    rays_sun.append(((origin.x,origin.y,origin.z),
                                     (end_for_ray.x,end_for_ray.y,end_for_ray.z)))

    # Build geometry
    if sun_pts:
        add_discs_mesh(f"ShadePoints_SUN_{lab}", sun_pts, disc_radius, m["sun"], coll)
    if shade_pts:
        add_discs_mesh(f"ShadePoints_SHADE_{lab}", shade_pts, disc_radius, m["shade"], coll)
    if draw_rays:
        if rays_sun:
            add_segments_curve(f"ShadeRays_SUN_{lab}", rays_sun, m["rayS"], coll, thickness=0.01)
        if rays_hit:
            add_segments_curve(f"ShadeRays_SHADE_{lab}", rays_hit, m["rayH"], coll, thickness=0.012)

    # Report numeric fraction for this timestamp
    total = len(sun_pts) + len(shade_pts)
    frac  = (len(shade_pts) / total) if total else 0.0
    print(f"[ShadeViz] {make_timestamp_label(t_minutes)}  shade_fraction={frac:.4f}  (shade={len(shade_pts)}, sun={len(sun_pts)}, total={total})")
    bpy.context.scene["viz_shade_fraction"] = frac
    bpy.context.scene["viz_shade_label"]    = make_timestamp_label(t_minutes)
    bpy.context.scene["viz_shade_az_el"]    = (float(az), float(el))

    return frac

# ---------- Main ----------
def main():
    args = parse_args()

    root = ensure_coll("ShadeViz", clear=args["clear"])
    # Per-timestamp sub-collections
    start_min = int(round(args["tstart"] * 60.0))
    end_min   = int(round(args["tend"]   * 60.0))
    step      = max(1, int(args["step"]))

    if start_min > end_min:
        start_min, end_min = end_min, start_min

    # Build viz for each timestamp
    fracs=[]
    for minute in range(start_min, end_min + 1, step):
        f = build_shade_viz_for_time(
            coll_root=root,
            t_minutes=minute,
            lat=args["lat"], lon=args["lon"], date_ymd=args["date"],
            tz=args["tz"], north=args["north"],
            total_samples=args["samples"],
            draw_rays=args["draw_rays"],
            disc_radius=args["disc_radius"],
            ray_maxdist=args["ray_maxdist"]
        )
        if f is not None:
            fracs.append(f)

    # Print average if multiple timestamps
    if fracs:
        avg = sum(fracs)/len(fracs)
        print(f"[ShadeViz] Average shade_fraction across {len(fracs)} timestamps = {avg:.4f}")
        bpy.context.scene["viz_shade_fraction_avg"] = float(avg)

    # Save
    if args["save_as"]:
        path = bpy.path.abspath(args["save_as"])
        bpy.ops.wm.save_as_mainfile(filepath=path, compress=False, copy=True)
        print(f"[ShadeViz] Saved to {path}")

if __name__ == "__main__":
    main()
