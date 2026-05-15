
# enclosure_hw_viz.py
# Compute & visualize street enclosure as classic H/W per cross-section,
# using boolean slices to find inner facades and heights.
#
# Now with semi-transparent slabs (backdrop) and optional envelope transparency.
#
# Enclosure per section:
#   - H_L  = max roof z on LEFT  side - base_z
#   - H_R  = max roof z on RIGHT side - base_z
#   - Havg = 0.5 * (H_L + H_R)
#   - W    = x_right_inner - x_left_inner  (inner facade span)
#   - E    = Havg / W
#
# Street enclosure = mean(E) across samples (zeros where a side is missing).
#
# It also draws:
#   - EnclosureEnvelope_## : magenta boolean intersection slice (for inspection)
#   - EnclosureBackdrop_## : pale slab per section; in "holes" mode it subtracts the envelope
#
# Usage examples:
# blender your_scene.blend --background --python enclosure_hw_viz.py -- \
#   --width 32.0 --height 12.0 --samples 25 --thickness 0.06 \
#   --backdrop true --backdrop_mode holes --backdrop_alpha 0.35 \
#   --clear true --save_as enclosure_hw.blend
#
# blender your_scene.blend --background --python enclosure_hw_viz.py -- \
#   --width 32.0 --height 12.0 --step 5.0 --backdrop_alpha 0.35 --clear true
#
import bpy, sys
from mathutils import Vector

# ---------- CLI ----------
def parse_args():
    argv = sys.argv
    if "--" in argv: argv = argv[argv.index("--")+1:]
    else: argv = []
    args = {
        "width": None,          # slab width (m); if None, defaults to 0.9 * building-span
        "height": 12.0,         # slab height (m)
        "thickness": 0.06,      # slab thickness along Y (m) for robust boolean
        "samples": None,        # number of evenly spaced sections; overrides --step
        "step": 5.0,            # spacing in meters if --samples not given
        "clear": True,          # clear EnclosureViz collection first
        "save_as": None,        # optional output path
        "backdrop": True,       # draw reference slab per section
        "backdrop_mode": "holes",   # "holes" (subtract envelope) or "solid"
        "backdrop_alpha": 0.1,     # transparency for the slab (0..1); lower = more transparent
        "envelope_alpha": 1.0,      # transparency for the envelope; default opaque
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--width" and i+1 < len(argv):
            try: args["width"] = float(argv[i+1])
            except: pass
            i += 2; continue
        if a == "--height" and i+1 < len(argv):
            try: args["height"] = float(argv[i+1])
            except: pass
            i += 2; continue
        if a == "--thickness" and i+1 < len(argv):
            try: args["thickness"] = max(0.01, float(argv[i+1]))
            except: pass
            i += 2; continue
        if a == "--samples" and i+1 < len(argv):
            try: args["samples"] = max(1, int(argv[i+1]))
            except: pass
            i += 2; continue
        if a == "--step" and i+1 < len(argv):
            try: args["step"] = max(0.5, float(argv[i+1]))
            except: pass
            i += 2; continue
        if a == "--clear" and i+1 < len(argv):
            args["clear"] = argv[i+1].lower() in ("1","true","yes","y")
            i += 2; continue
        if a == "--save_as" and i+1 < len(argv):
            args["save_as"] = argv[i+1]
            i += 2; continue
        if a == "--backdrop" and i+1 < len(argv):
            args["backdrop"] = argv[i+1].lower() in ("1","true","yes","y")
            i += 2; continue
        if a == "--backdrop_mode" and i+1 < len(argv):
            args["backdrop_mode"] = argv[i+1].lower()
            i += 2; continue
        if a == "--backdrop_alpha" and i+1 < len(argv):
            try: args["backdrop_alpha"] = max(0.0, min(1.0, float(argv[i+1])))
            except: pass
            i += 2; continue
        if a == "--envelope_alpha" and i+1 < len(argv):
            try: args["envelope_alpha"] = max(0.0, min(1.0, float(argv[i+1])))
            except: pass
            i += 2; continue
        i += 1
    return args

# ---------- Helpers ----------
def is_building_part(o):
    if o.type != 'MESH': return False
    if getattr(o, "hide_render", False): return False
    nm = o.name.lower()
    return ("building" in nm) or ("roof" in nm)

def scene_bounds_of_buildings():
    objs = [o for o in bpy.data.objects if is_building_part(o)]
    if not objs: return None
    min_x=min_y=min_z=  1e9
    max_x=max_y=max_z= -1e9
    for o in objs:
        bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
        xs=[p.x for p in bb]; ys=[p.y for p in bb]; zs=[p.z for p in bb]
        min_x=min(min_x, min(xs)); max_x=max(max_x, max(xs))
        min_y=min(min_y, min(ys)); max_y=max(max_y, max(ys))
        min_z=min(min_z, min(zs)); max_z=max(max_z, max(zs))
    return (min_x, max_x, min_y, max_y, min_z, max_z)

def ensure_collection(name, clear=False):
    coll = bpy.data.collections.get(name)
    if coll and clear:
        # unlink from all scenes and remove
        for scn in bpy.data.scenes:
            try: scn.collection.children.unlink(coll)
            except Exception: pass
        bpy.data.collections.remove(coll)
        coll = None
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll

def ensure_emission_alpha(name, rgb, alpha=1.0, strength=3.0):
    """
    Create an emission material mixed with Transparent BSDF using 'alpha'.
    Sets Eevee/Cycles blend to BLEND and disables shadow casting to avoid darkening.
    """
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree

    # clear nodes
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emis = nt.nodes.new("ShaderNodeEmission")
    emis.inputs["Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    emis.inputs["Strength"].default_value = strength

    trans = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = max(0.0, min(1.0, 1.0 - float(alpha)))  # 0->opaque, 1->fully transparent

    nt.links.new(emis.outputs["Emission"], mix.inputs[1])
    nt.links.new(trans.outputs["BSDF"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    # enable transparency in render/viewport
    mat.blend_method = 'BLEND'
    # Blender 4.5: 'shadow_method' was removed. Guard it for older versions.
    if hasattr(mat, 'shadow_method'):
        mat.shadow_method = 'NONE'  # 2.9x–3.6
    elif hasattr(mat, 'shadow_mode'):
        mat.shadow_mode = 'NONE'    # 4.x fallback (if present)
    mat.use_backface_culling = False
    return mat

def make_box(name, center_x, y, base_z, width, height, thickness, collection):
    # axis-aligned rectangular box: width along X, thickness along Y, height along Z
    w = width * 0.5
    t = thickness * 0.5
    xL, xR = center_x - w, center_x + w
    yF, yB = y - t, y + t
    z0, z1 = base_z, base_z + height
    me = bpy.data.meshes.new(name)
    me.from_pydata(
        [(xL,yF,z0),(xR,yF,z0),(xR,yB,z0),(xL,yB,z0),
         (xL,yF,z1),(xR,yF,z1),(xR,yB,z1),(xL,yB,z1)],
        [],
        [(0,1,2,3),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    )
    me.update()
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    return ob

def boolean_apply(obj, target, op='INTERSECT'):
    bpy.context.view_layer.objects.active = obj
    for o in bpy.context.selected_objects: o.select_set(False)
    obj.select_set(True)
    mod = obj.modifiers.new(name="Bool", type='BOOLEAN')
    mod.operation = op
    mod.solver = 'EXACT'
    mod.object = target
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
        return True
    except Exception:
        return False

def join_objects(objs, name):
    if not objs: return None
    for o in bpy.context.selected_objects: o.select_set(False)
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    objs[0].name = name
    return objs[0]

# ---------- Main ----------
def main():
    args = parse_args()
    coll = ensure_collection("EnclosureViz", clear=args["clear"])

    # materials for visualization
    mat_envelope = ensure_emission_alpha("Enclosure_Envelope",
                                         (0.95, 0.35, 0.8),
                                         alpha=args["envelope_alpha"],
                                         strength=3.0)  # magenta-ish, optionally transparent
    mat_backdrop = ensure_emission_alpha("Enclosure_Backdrop",
                                         (0.85, 0.88, 0.92),
                                         alpha=args["backdrop_alpha"],
                                         strength=1.0) # pale, semi-transparent

    # gather building parts
    bparts = [o for o in bpy.data.objects if is_building_part(o)]
    if not bparts:
        print("[EnclosureHW] No Building_/Roof_ meshes found."); return

    # extents & defaults
    ext = scene_bounds_of_buildings()
    if not ext:
        print("[EnclosureHW] Could not compute building extents."); return
    min_x, max_x, min_y, max_y, min_z, max_z = ext
    centerline_x = 0.5 * (min_x + max_x)

    width  = args["width"] if args["width"] is not None else (max_x - min_x) * 0.9
    height = args["height"]
    thick  = args["thickness"]

    # sample y positions
    if args["samples"]:
        N = args["samples"]
        if N == 1: ys = [0.5*(min_y+max_y)]
        else:
            seg = (max_y - min_y) / N
            ys = [min_y + (i+0.5)*seg for i in range(N)]
    else:
        ys = []
        step = args["step"]
        y = min_y + 0.5*step
        while y < max_y:
            ys.append(y); y += step
        if not ys: ys = [0.5*(min_y+max_y)]

    # simple ground base: use min_z across building parts
    base_z = min_z

    per_section = []
    valid = 0

    for idx, y in enumerate(ys):
        # build a thin slab at this y
        tmpl = make_box(f"Slab_{idx:02d}_Tmpl", centerline_x, y, base_z, width, height, thick, coll)

        # intersect slab with each building/roof, keep pieces
        slices = []
        for bp in bparts:
            slab = tmpl.copy(); slab.data = tmpl.data.copy(); coll.objects.link(slab)
            ok = boolean_apply(slab, bp, op='INTERSECT')
            if not ok or len(slab.data.polygons) == 0:
                try: bpy.data.objects.remove(slab, do_unlink=True)
                except Exception: pass
                continue
            # assign transparent/opaque envelope material
            if mat_envelope:
                slab.data.materials.clear(); slab.data.materials.append(mat_envelope)
            slices.append(slab)

        # remove template
        try: bpy.data.objects.remove(tmpl, do_unlink=True)
        except Exception: pass

        if not slices:
            per_section.append(0.0)
            print(f"[EnclosureHW] Sec {idx:02d} y={y:.2f}: no intersection -> H/W=0.000")
            continue

        joined = join_objects(slices, f"EnclosureEnvelope_{idx:02d}")

        # world-space coords of envelope verts
        coords = [(joined.matrix_world @ v.co) for v in joined.data.vertices]

        # split into left/right of centerline
        eps = 1e-6
        left  = [c for c in coords if c.x < centerline_x - eps]
        right = [c for c in coords if c.x > centerline_x + eps]

        if not left or not right:
            per_section.append(0.0)
            print(f"[EnclosureHW] Sec {idx:02d} y={y:.2f}: one side missing -> H/W=0.000")
            continue

        # heights above base
        H_L = max(c.z for c in left)  - base_z
        H_R = max(c.z for c in right) - base_z
        Havg = 0.5 * (H_L + H_R)

        # inner facade span
        x_L_inner = max(c.x for c in left)    # rightmost on the left
        x_R_inner = min(c.x for c in right)   # leftmost on the right
        W_canyon  = x_R_inner - x_L_inner

        if W_canyon <= 1e-6 or Havg <= 0.0:
            per_section.append(0.0)
            print(f"[EnclosureHW] Sec {idx:02d} y={y:.2f}: degenerate span/height -> H/W=0.000")
            continue

        E = Havg / W_canyon
        per_section.append(E)
        valid += 1

        print(f"[EnclosureHW] Sec {idx:02d} y={y:.2f}: H_L={H_L:.2f} H_R={H_R:.2f} "
              f"H_avg={Havg:.2f} W={W_canyon:.2f}  H/W={E:.3f}")

        # optional figure backdrop (semi-transparent)
        if args["backdrop"]:
            slab_bg = make_box(f"EnclosureBackdrop_{idx:02d}", centerline_x, y, base_z, width, height, thick, coll)
            if mat_backdrop:
                slab_bg.data.materials.clear(); slab_bg.data.materials.append(mat_backdrop)
            if args["backdrop_mode"] == "holes":
                ok = boolean_apply(slab_bg, joined, op='DIFFERENCE')
                if not ok:
                    print(f"[EnclosureHW] Backdrop boolean failed at section {idx:02d}")

    if per_section:
        mean_all  = sum(per_section) / len(per_section)
        nonzero   = [v for v in per_section if v > 0.0]
        mean_valid = (sum(nonzero) / len(nonzero)) if nonzero else 0.0
        print(f"[EnclosureHW] Samples={len(per_section)}, Valid={valid}, "
              f"Mean H/W (incl. zeros)={mean_all:.3f}, Mean H/W (valid-only)={mean_valid:.3f}")
    else:
        print("[EnclosureHW] No samples.")

    # save if requested
    if args["save_as"]:
        outp = bpy.path.abspath(args["save_as"])
        bpy.ops.wm.save_as_mainfile(filepath=outp, compress=False, copy=True)
        print(f"[EnclosureHW] Saved as {outp}")

if __name__ == "__main__":
    main()
