# metrics_viz_svf_in_scene.py
# Creates in-scene geometry to visualize SVF sampling:
# - A wire dome (lat/lon curves)
# - Colored tick marks on the hemisphere for each sampled direction (sky/hit classes)
# Usage:
#   blender /path/your_scene.blend --background \
#     --python metrics_viz_svf_in_scene.py -- \
#     --probe_mode ped --probes 1 --radius 40 --az 48 --el 12 --tick 0.5 --hits false --save_as /path/scene_svf.blend
#
# Flags:
#   --probe_mode ped|center|both   (default ped; uses left_/right_ naming)
#   --probes 1|3                   (# of Y positions; 1 = mid; 3 = quartiles)
#   --radius <m>                   (hemisphere radius, default 40)
#   --az <n>                       (# azimuth divisions, default 48)
#   --el <n>                       (# elevation rings, default 12, from horizon→zenith)
#   --tick <m>                     (tick length along the ray, default 0.5)
#   --hits true|false              (also draw full obstruction rays; default false)
#   --clear true|false             (delete existing SVFViz; default true)
#   --save_as <path>               (optional: write a new .blend)
#
import bpy, sys, os, math
from mathutils import Vector


try:
    bpy.context.view_layer.material_override = None
except Exception:
    pass

# ---------------- CLI ----------------
def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--")+1:]
    else:
        argv = []
    args = {
        "probe_mode": "ped",
        "probes_n": 1,
        "radius": 25.0,
        "az": 48,
        "el": 12,
        "tick": 0.5,
        "hits": False,
        "clear": True,
        "save_as": None,
        "fill": True,
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--probe_mode" and i+1 < len(argv): args["probe_mode"] = argv[i+1].lower(); i+=2; continue
        if a == "--probes" and i+1 < len(argv): args["probes_n"] = max(1,int(argv[i+1])); i+=2; continue
        if a == "--radius" and i+1 < len(argv): args["radius"] = float(argv[i+1]); i+=2; continue
        if a == "--az" and i+1 < len(argv): args["az"] = max(8,int(argv[i+1])); i+=2; continue
        if a == "--el" and i+1 < len(argv): args["el"] = max(2,int(argv[i+1])); i+=2; continue
        if a == "--tick" and i+1 < len(argv): args["tick"] = float(argv[i+1]); i+=2; continue
        if a == "--hits" and i+1 < len(argv): args["hits"] = argv[i+1].lower() in ("1","true","yes","y"); i+=2; continue
        if a == "--clear" and i+1 < len(argv): args["clear"] = argv[i+1].lower() in ("1","true","yes","y"); i+=2; continue
        if a == "--save_as" and i+1 < len(argv): args["save_as"] = argv[i+1]; i+=2; continue
        if a == "--fill" and i+1 < len(argv):args["fill"] = argv[i+1].lower() in ("1","true","yes","y"); i+=2; continue
        i += 1
    return args

# ---------------- Depsgraph & ray ----------------
def dg(): return bpy.context.evaluated_depsgraph_get()
def ray_cast(origin, direction, dist=1e6):
    hit, loc, norm, face_idx, obj, _ = bpy.context.scene.ray_cast(dg(), origin, direction, distance=dist)
    if not hit:
        return (False, None, None, None, None)
    return (True, loc, norm, obj, (loc-origin).length)

# ---------------- Bounds & naming helpers ----------------
def world_bbox(obj):
    eo = obj.evaluated_get(dg())
    mat = eo.matrix_world
    corners = [mat @ Vector(c) for c in eo.bound_box]
    mn = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    mx = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return mn, mx

def scene_bounds(objs=None):
    if objs is None:
        objs = [o for o in bpy.data.objects if o.type in {'MESH','CURVE','SURFACE','META'} and (not getattr(o,"hide_render",False))]
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

def obj_mid_xy(o):
    mn,mx = world_bbox(o)
    return Vector(((mn.x+mx.x)*0.5,(mn.y+mx.y)*0.5,(mn.z+mx.z)*0.5))

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

def ground_z_at(x,y,z_top=1000.0):
    hit, loc, norm, obj, d = ray_cast(Vector((x,y,z_top)), Vector((0,0,-1)), dist=z_top*2.0)
    if hit: return loc.z + 1.6
    hit2, loc2, norm2, obj2, d2 = ray_cast(Vector((x,y,-z_top)), Vector((0,0,1)), dist=z_top*2.0)
    return (loc2.z + 1.6) if hit2 else 1.6

# ---------------- Probes ----------------
def pick_probes(probe_mode='ped', probes_n=1):
    # Manual override: empties named PROBE_*
    user = [o for o in bpy.data.objects if o.type=='EMPTY' and o.name.lower().startswith('probe')]
    if user:
        return [o.matrix_world.translation.copy() for o in user]
    Lsw,Rsw,AllSw = find_sidewalks_lr()
    Lrd,Rrd,AllRd = find_roads_lr()
    base_objs = AllSw if AllSw else (AllRd if AllRd else [o for o in bpy.data.objects if o.type=='MESH' and (not getattr(o,"hide_render",False))])
    mn,mx = scene_bounds(base_objs)
    y0 = 0.5*(mn.y+mx.y)
    ys = [y0] if probes_n==1 else [0.25*mn.y+0.75*y0, y0, 0.75*mx.y+0.25*y0]
    probes=[]
    def addp(x,y):
        z = ground_z_at(x,y, z_top=max(50.0,(mx.z-mn.z)+10.0))
        probes.append(Vector((x,y,z)))
    if probe_mode in ('ped','both') and (Lsw or Rsw):
        xL = (sum(obj_mid_xy(o).x for o in Lsw)/max(1,len(Lsw))) if Lsw else None
        xR = (sum(obj_mid_xy(o).x for o in Rsw)/max(1,len(Rsw))) if Rsw else None
        for y in ys:
            if xL is not None: addp(xL,y)
            if xR is not None: addp(xR,y)
    elif probe_mode in ('center','both') and AllRd:
        rmn,rmx = scene_bounds(AllRd)
        xC = 0.5*(rmn.x+rmx.x)
        for y in ys: addp(xC,y)
    elif AllRd:
        rmn,rmx = scene_bounds(AllRd)
        margin=1.25
        for y in ys:
            addp(rmn.x-margin, y); addp(rmx.x+margin, y)
    else:
        W = mx.x-mn.x
        for y in ys:
            addp(mn.x+0.25*W, y); addp(mx.x-0.25*W, y)
    return probes

# ---------------- Materials & curves ----------------
def ensure_material(name, rgba=(0.29, 0.61, 0.83, 0.35), alpha=None, blend=True):
    import bpy
    r, g, b = rgba[0], rgba[1], rgba[2]
    a_from_rgba = rgba[3] if len(rgba) > 3 else 1.0
    a = float(alpha) if alpha is not None else float(a_from_rgba)

    def _safe_set(obj, attr, value):
        if hasattr(obj, attr):
            try: setattr(obj, attr, value)
            except Exception: pass

    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True

    # Rebuild nodes so alpha changes take effect every call
    nt = mat.node_tree
    nt.nodes.clear()
    out  = nt.nodes.new("ShaderNodeOutputMaterial");  out.location  = (400, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled");  bsdf.location = (0, 0)
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Alpha"].default_value      = a

    if blend:
        # Mix Transparent with Principled so Eevee shows real transparency
        tr  = nt.nodes.new("ShaderNodeBsdfTransparent"); tr.location = (-200, -100)
        mix = nt.nodes.new("ShaderNodeMixShader");       mix.location = (200, -50)
        inv = nt.nodes.new("ShaderNodeMath");            inv.operation = "SUBTRACT"
        inv.inputs[0].default_value = 1.0
        inv.inputs[1].default_value = a
        inv.location = (0, -200)

        nt.links.new(inv.outputs[0],      mix.inputs["Fac"])
        nt.links.new(tr.outputs["BSDF"],  mix.inputs[1])
        nt.links.new(bsdf.outputs["BSDF"],mix.inputs[2])
        nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

        _safe_set(mat, "blend_method", 'BLEND')   # Eevee/Cycles (if available)
        _safe_set(mat, "shadow_method", 'NONE')   # Eevee only; guard it
        _safe_set(mat, "use_backface_culling", False)
        mat.diffuse_color = (r, g, b, a)
    else:
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        _safe_set(mat, "blend_method", 'OPAQUE')
        _safe_set(mat, "shadow_method", 'OPAQUE')  # may not exist → safely ignored
        mat.diffuse_color = (r, g, b, 1.0)

    mat["__SVF_HELPER__"] = True
    return mat



PAL = {
    "sky":      (0.29,0.61,0.83,1.0),
    "veg":      (0.19,0.62,0.26,1.0),
    "building": (0.42,0.42,0.42,1.0),
    "vehicle":  (0.85,0.56,0.02,1.0),
    "ped":      (0.83,0.30,0.54,1.0),
    "other":    (0.6,0.6,0.6,1.0),
    "dome":     (0.8,0.8,0.8,1.0),
}

def mats():
    return {k: ensure_material(f"svf_{k}", PAL[k], alpha=1.0, blend=False) for k in PAL}

def add_curve(name, splines_points_lists, thickness=0.01, mat=None, coll=None):
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
    if mat: ob.data.materials.append(mat)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob

def ensure_collection(name, parent=None, clear=False):
    coll = bpy.data.collections.get(name)
    if coll and clear:
        # unlink & delete
        for scn in bpy.data.scenes:
            try: scn.collection.children.unlink(coll)
            except: pass
        bpy.data.collections.remove(coll)
        coll=None
    if coll is None:
        coll = bpy.data.collections.new(name)
        (parent or bpy.context.scene.collection).children.link(coll)
    return coll

# ---------- Mesh helpers (3D quads) ----------
def add_mesh_quads(name, quads_pts, mat=None, coll=None):
    # quads_pts: list of [(x,y,z),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4)]
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    for quad in quads_pts:
        vs = [bm.verts.new(p) for p in quad]
        try:
            bm.faces.new(vs)
        except ValueError:
            # face may already exist if sampling grids overlap — ignore
            pass
    bm.to_mesh(me); bm.free()
    ob = bpy.data.objects.new(name, me)
    if mat: ob.data.materials.append(mat)
    (coll or bpy.context.scene.collection).objects.link(ob)
    return ob

def sph_point_on_hemisphere(probe, R, az, el):
    # el in [0, pi/2], az in [0, 2pi)
    x = probe.x + R*math.cos(el)*math.cos(az)
    y = probe.y + R*math.cos(el)*math.sin(az)
    z = probe.z + R*math.sin(el)
    return (x, y, z)

def build_svf_sky_patches(probe, R, n_az=48, n_el=12, inset=0.02, mat=None, coll=None):
    """Create translucent quads on cells whose center ray sees SKY."""
    # Slightly shrink radius so patches don't z-fight the wire:
    Rin = max(0.0, R - inset)
    quads = []
    for j in range(n_el):
        el0 = (j    / n_el) * (math.pi/2.0)
        el1 = ((j+1)/ n_el) * (math.pi/2.0)
        elc = ((j+0.5)/n_el) * (math.pi/2.0)
        cosc, sinc = math.cos(elc), math.sin(elc)

        for i in range(n_az):
            az0 = (i    / n_az) * (2*math.pi)
            az1 = ((i+1)/ n_az) * (2*math.pi)
            azc = ((i+0.5)/n_az) * (2*math.pi)

            # Cast from cell center
            dir3 = Vector((math.cos(azc)*cosc, math.sin(azc)*cosc, sinc)).normalized()
            hit, loc, norm, obj, _d = ray_cast(probe, dir3, dist=R)

            # Only fill if SKY
            if hit:
                continue

            p00 = sph_point_on_hemisphere(probe, Rin, az0, el0)
            p10 = sph_point_on_hemisphere(probe, Rin, az1, el0)
            p11 = sph_point_on_hemisphere(probe, Rin, az1, el1)
            p01 = sph_point_on_hemisphere(probe, Rin, az0, el1)
            quads.append([p00, p10, p11, p01])

    if quads:
        add_mesh_quads("SVF_Sky_Patches", quads, mat=mat, coll=coll)


# ---------------- Categorization ----------------
def hit_category(obj):
    if obj is None: return 'sky'
    nm = obj.name.lower()
    if any(k in nm for k in ['tree','leaf','plant','veg']): return 'veg'
    if any(k in nm for k in ['building','facade','façade']): return 'building'
    if any(k in nm for k in ['car','vehicle','truck','bus']): return 'vehicle'
    if any(k in nm for k in ['human','ped','person']): return 'ped'
    return 'other'

# ---------------- Dome & ticks ----------------
def build_dome_wire(probe, R, n_az=48, n_el=12, thickness=0.005, mat=None, coll=None):
    # Latitudes (exclude horizon at 0 to avoid occlusion with ground)
    lat_spl = []
    for j in range(1, n_el+1):
        el = (j/(n_el)) * (math.pi/2.0)  # 0..pi/2
        r = R * math.cos(el)
        z = probe.z + R * math.sin(el)
        ring=[]
        for i in range(n_az+1):
            az = (i/n_az) * 2*math.pi
            x = probe.x + r*math.cos(az)
            y = probe.y + r*math.sin(az)
            ring.append((x,y,z))
        lat_spl.append(ring)
    add_curve("SVF_Dome_Lats", lat_spl, thickness=thickness, mat=mat, coll=coll)
    # Longitudes (semi-circles)
    lon_spl = []
    for i in range(n_az):
        az = (i/n_az) * 2*math.pi
        arc=[]
        for j in range(0, n_el+1):
            el = (j/(n_el)) * (math.pi/2.0)
            x = probe.x + R*math.cos(el)*math.cos(az)
            y = probe.y + R*math.cos(el)*math.sin(az)
            z = probe.z + R*math.sin(el)
            arc.append((x,y,z))
        lon_spl.append(arc)
    add_curve("SVF_Dome_Lons", lon_spl, thickness=thickness, mat=mat, coll=coll)

def build_svf_ticks(probe, R, n_az=48, n_el=12, tick_len=0.4, include_hit_rays=False, coll=None, mat_map=None):
    buckets = {k: [] for k in ['sky','veg','building','vehicle','ped','other']}
    hit_rays = []  # optional full obstruction rays
    # Grid over hemisphere: az in [0,2pi), el in (0, pi/2]
    for j in range(0, n_el):
        el0 = (j+0.5)/n_el * (math.pi/2.0)  # center of ring
        cosel = math.cos(el0); sinel = math.sin(el0)
        for i in range(0, n_az):
            az = (i+0.5)/n_az * 2*math.pi
            dir3 = Vector((math.cos(az)*cosel, math.sin(az)*cosel, sinel)).normalized()
            hit, loc, norm, obj, d = ray_cast(probe, dir3, dist=R)
            cat = hit_category(obj if hit else None)
            # place a short radial tick near dome surface
            p1 = probe + dir3 * (R - tick_len)
            p2 = probe + dir3 * R
            buckets[cat].append([ (p1.x,p1.y,p1.z), (p2.x,p2.y,p2.z) ])
            if include_hit_rays and hit and d < R:
                hit_rays.append([ (probe.x,probe.y,probe.z), (loc.x,loc.y,loc.z) ])
    # Build curve objects
    for cat, spl in buckets.items():
        if not spl: continue
        add_curve(f"SVF_Ticks_{cat}", spl, thickness=0.01, mat=mat_map.get(cat), coll=coll)
    if include_hit_rays and hit_rays:
        add_curve("SVF_Rays_Hits", hit_rays, thickness=0.007, mat=mat_map.get("other"), coll=coll)

    total_samples = int(n_az) * int(n_el)
    sky_samples   = len(buckets.get('sky', []))
    svf = (sky_samples / total_samples) if total_samples else 0.0
    return svf, sky_samples, total_samples

def center_probe():
    """Return one probe at the center of the street (or whole scene if no road meshes)."""
    Lrd, Rrd, AllRd = find_roads_lr()
    if AllRd:
        rmn, rmx = scene_bounds(AllRd)
        cx = 0.5 * (rmn.x + rmx.x)
        cy = 0.5 * (rmn.y + rmx.y)
        z  = ground_z_at(cx, cy, z_top=max(50.0, (rmx.z - rmn.z) + 10.0))
    else:
        mn, mx = scene_bounds()
        cx = 0.5 * (mn.x + mx.x)
        cy = 0.5 * (mn.y + mx.y)
        z  = ground_z_at(cx, cy, z_top=max(50.0, (mx.z - mn.z) + 10.0))
    P = Vector((cx, cy, z))
    print(f"[SVF] Single center probe @ ({cx:.3f}, {cy:.3f}, {z:.3f})")
    return P

# ---------------- Main ----------------
def main():
    args = parse_args()
    root = ensure_collection("SVFViz", clear=args["clear"])
    mats_map = mats()

 # Always use one auto-centered probe
    probes = [center_probe()]

    if not probes:
        print("[SVF] No probes found."); return

    fill_on = bool(args.get("fill", True))
    if fill_on:
        skyfill = ensure_material("svf_skyfill", (0.29, 0.61, 0.83, 0.35), alpha=0.8, blend=False)

    svfs = []
    for idx, P in enumerate(probes):
        sub = ensure_collection(f"SVF_Probe_{idx:02d}", parent=root, clear=False)

        # locator + dome
        empty = bpy.data.objects.new(f"SVF_Probe_{idx:02d}_Empty", None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.15
        empty.location = P
        sub.objects.link(empty)

        build_dome_wire(
            P, args["radius"],
            n_az=args["az"], n_el=args["el"],
            thickness=0.005, mat=mats_map["dome"], coll=sub
        )

        # cast rays & count sky BEFORE adding any filled quads
        svf, sky_n, total_n = build_svf_ticks(
            P, args["radius"], n_az=args["az"], n_el=args["el"],
            tick_len=args["tick"], include_hit_rays=args["hits"], coll=sub, mat_map=mats_map
        )
        svfs.append(svf)
        print(f"[SVF] Probe {idx} @ {tuple(round(c,3) for c in P)}  SVF={svf:.4f}  (sky={sky_n}, total={total_n})")
        bpy.context.scene[f"viz_svf_probe_{idx:02d}"] = float(svf)

        # Now add translucent sky patches for visualization
        if fill_on:
            build_svf_sky_patches(
                P, args["radius"],
                n_az=args["az"], n_el=args["el"],
                inset=0.02, mat=skyfill, coll=sub
            )

    if svfs:
        avg = sum(svfs) / len(svfs)
        print(f"[SVF] Average SVF across {len(svfs)} probe(s) = {avg:.4f}")
        bpy.context.scene["viz_svf_avg"] = float(avg)

    if args["save_as"]:
        outp = bpy.path.abspath(args["save_as"])
        bpy.ops.wm.save_as_mainfile(filepath=outp, compress=False, copy=True)
        print(f"[SVF] Saved as {outp}")




if __name__ == "__main__":
    main()
