# apply_textures_street_plus.py
# Example:
# blender -b "C:/Users/Bhupender/Downloads/blender/3d_build.blend" --python "C:/Users/Bhupender/Downloads/blender/apply_textures_street.py" -- 
#   --texdir "C:/Users/Bhupender/Downloads/blender/textures" 
#   --outblend "C:/Users/Bhupender/Downloads/blender/textured.blend" 
#   --pack true --relative true --mapping box --seed 123 --maxdepth 3
#
# Expects subfolders under --texdir:
#   branches/  bikepath/  parking/  median/  footpath/  driveway/  wall/  roof/
# Each may contain textures directly OR inside 1–2 nested subfolders (e.g. */*.blend/textures/*.png).
# If multiple candidates exist, one is chosen automatically (deterministic if --seed is set).

import bpy, os, random, sys, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(path):
    if not path:
        return path
    path = os.path.expanduser(path)
    if path.startswith("//"):
        return bpy.path.abspath(path)
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)

# ---------------- CLI helpers ----------------
def arg(flag, default=None, as_float3=False, as_int=False):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            val = sys.argv[i+1]
            if as_int:
                try: return int(val)
                except: return default
            if as_float3:
                try:
                    parts = [float(x) for x in val.split(",")]
                    if len(parts) == 3: return tuple(parts)
                except: pass
                return default
            return val
    return default

TEX_ROOT = arg("--texdir", os.path.join(SCRIPT_DIR, "textures"))

OVERRIDE = {
    "branches":  arg("--tex_branches",  None),
    "bikepath":  arg("--tex_bikepath",  None),
    "parking":   arg("--tex_parking",   None),
    "median":    arg("--tex_median",    None),
    "footpath":  arg("--tex_footpath",  None),
    "driveway":  arg("--tex_driveway",  None),
    "wall":      arg("--tex_wall",      None),
    "roof":      arg("--tex_roof",      None),
}

MAPPING_MODE = (arg("--mapping", "box") or "box").lower()
SEED = arg("--seed", None, as_int=True)
if SEED is not None:
    random.seed(SEED)

TEXTURE_CATEGORIES = ("branches", "bikepath", "parking", "median", "footpath", "driveway", "wall", "roof")
OBJECT_CATEGORY_PRIORITY = ("roof", "wall", "median", "driveway", "parking", "bikepath", "footpath", "branches")
PER_OBJECT_VARIANT_CATEGORIES = {"branches", "wall", "roof"}

# Per-category texture size in meters per repeat.
MP = {
    "branches": float(arg("--mp_branches", "3.0") or 3.0),
    "bikepath": float(arg("--mp_bikepath", "4.0") or 4.0),
    "parking":  float(arg("--mp_parking",  "4.0") or 4.0),
    "median":   float(arg("--mp_median",   "6.0") or 6.0),
    "footpath": float(arg("--mp_footpath", "4.0") or 4.0),
    "driveway": float(arg("--mp_driveway", "8.0") or 8.0),
    "wall":     float(arg("--mp_wall",     "2.0") or 2.0),
    "roof":     float(arg("--mp_roof",     "3.0") or 3.0),
}


# Randomize UV offset for large repetitive surfaces to reduce tiling.
# Keep street-level surfaces unrotated so paving/asphalt patterns align with the street.
RAND_UV_OFFSET = {
    "bikepath": True,
    "parking": True,
    "median": True,
    "footpath": True,
    "driveway": True,
    "wall": True,
    "roof": True,
    "branches": True,
}
RAND_UV_ROTATION = {
    "wall": True,
    "roof": True,
    "branches": True,
}

# Save options
OUT_BLEND = arg("--outblend", None)
PACK_DATA = (arg("--pack", "true") or "true").lower() in ("1","true","yes")
RELATIVE  = (arg("--relative", "true") or "true").lower() in ("1","true","yes")

# Recursive depth for scanning textures (e.g., ...\branches\asset.blend\textures\*.png)
MAX_DEPTH = arg("--maxdepth", 3, as_int=True) or 3

# ---------------- file helpers ----------------
IMG_EXT = {".png",".jpg",".jpeg",".tga",".tif",".tiff",".bmp",".exr",".hdr",".dds",".webp"}

def _folder_for(category):
    if OVERRIDE.get(category):
        return resolve_path(OVERRIDE[category])
    if not TEX_ROOT: return None
    cand = os.path.join(resolve_path(TEX_ROOT), category)
    return cand if os.path.isdir(cand) else None

def list_images_recursive(folder, max_depth=3):
    """Return all image paths under 'folder' up to max_depth levels deep."""
    if not folder or not os.path.isdir(folder): return []
    folder = os.path.abspath(folder)
    root_depth = folder.rstrip(os.sep).count(os.sep)
    out = []
    for root, dirs, files in os.walk(folder):
        depth = root.rstrip(os.sep).count(os.sep) - root_depth
        if depth >= max_depth:
            # prevent walking deeper
            dirs[:] = []
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXT:
                out.append(os.path.join(root, fn))
    return out

def list_images_shallow(folder):
    """Return image paths directly inside folder (non-recursive)."""
    if not folder or not os.path.isdir(folder): return []
    out = []
    for fn in sorted(os.listdir(folder)):
        path = os.path.join(folder, fn)
        if os.path.isfile(path) and os.path.splitext(fn)[1].lower() in IMG_EXT:
            out.append(path)
    return out

def _discover_texture_sets(folder):
    """
    Discover candidate texture-set folders for a category.
    The category folder itself counts as one set if it contains images directly.
    Each immediate child folder containing images recursively also counts as one set.
    """
    if not folder or not os.path.isdir(folder):
        return []

    sets = []
    if list_images_shallow(folder):
        sets.append(folder)

    try:
        child_names = sorted(os.listdir(folder))
    except Exception:
        child_names = []

    for name in child_names:
        child = os.path.join(folder, name)
        if os.path.isdir(child) and list_images_recursive(child, max_depth=MAX_DEPTH):
            sets.append(child)

    deduped = []
    seen = set()
    for path in sets:
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(path)
    return deduped

def _next_texture_set(category, variant_state):
    """Cycle through shuffled texture sets so variation is spread across objects."""
    sets = variant_state.get(category, {}).get("sets", [])
    if not sets:
        return None

    bag = variant_state[category].get("bag", [])
    if not bag:
        bag = list(sets)
        random.shuffle(bag)
    choice = bag.pop()
    variant_state[category]["bag"] = bag
    return choice

def _safe_material_name(category, obj_name, texture_set_folder):
    suffix = os.path.basename(os.path.normpath(texture_set_folder)) or category
    raw = f"MAT_{category}_{obj_name}_{suffix}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", raw)

# Heuristic PBR map finder across nested files
def find_pbr_images(folder):
    files = list_images_recursive(folder, max_depth=MAX_DEPTH)
    if not files: return {}
    low = {f: os.path.basename(f).lower() for f in files}

    def pick(*tags):
        # return first match by tag order
        for f in files:
            name = low[f]
            if any(t in name for t in tags):
                return f
        return None

    base   = pick("basecolor","base_color","albedo","diffuse","_diff","color","col","_bc","_d")
    rough  = pick("roughness","rough","_rough","_r")
    normal = pick("nor_gl","normal_gl","_gl") or pick("nor_dx","normal_dx","_dx","normal","_nor","nrm","norm")
    height = pick("disp","displacement","height","_h","bump")
    ao     = pick("ao","ambientocclusion","occlusion","ambient_occlusion")
    metal  = pick("metallic","metal","mtl")

    if not base:
        # safer fallback: pick a jpg/png that isn't clearly a data map
        jpgpng = [f for f in files if os.path.splitext(f)[1].lower() in (".jpg",".jpeg",".png")]
        bad = ("rough","normal","nrm","nor","metal","metallic","mtl","height","disp","bump","ao","ambient")
        jpgpng = [f for f in jpgpng if not any(b in low[f] for b in bad)]
        base = jpgpng[0] if jpgpng else files[0]

    out = {"basecolor": base}
    if rough:  out["roughness"] = rough
    if normal: out["normal"]    = normal
    if height: out["height"]    = height
    if ao:     out["ao"]        = ao
    if metal:  out["metallic"]  = metal
    return out



# --------------- material builder ---------------
def make_pbr_material(name, folder, scale_xyz=(1,1,1), mapping_mode="box", randomize_offset=False, randomize_rotation=False):
    maps = find_pbr_images(folder)
    if not maps:
        print(f"[Tex] No images found in '{folder}'. Skipping material {name}.")
        return None

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (1200, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (900, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # Mapping chain
    texcoord = nt.nodes.new("ShaderNodeTexCoord"); texcoord.location = (0, 0)
    mapping  = nt.nodes.new("ShaderNodeMapping"); mapping.location = (250, 0)
    # mapping.inputs["Scale"].default_value = (scale_xyz[0], scale_xyz[1], scale_xyz[2])

    # meters_per_repeat comes in via scale_xyz or category; compute 1 / meters
    def _meters_to_scale(x):
        try:
            x = float(x)
        except:
            x = 1.0
        return 1.0 / max(1e-6, x)

    if isinstance(scale_xyz, (tuple, list)) and len(scale_xyz) == 3:
        sx, sy, sz = (_meters_to_scale(scale_xyz[0]),
                    _meters_to_scale(scale_xyz[1]),
                    _meters_to_scale(scale_xyz[2]))
    else:
        s = _meters_to_scale(scale_xyz)
        sx = sy = sz = s

    mapping.inputs["Scale"].default_value = (sx, sy, sz)


    if randomize_rotation:
        mapping.inputs["Rotation"].default_value[2] = random.random()*6.28318
    if randomize_offset:
        mapping.inputs["Location"].default_value[0] = random.uniform(-10, 10)
        mapping.inputs["Location"].default_value[1] = random.uniform(-10, 10)

    if mapping_mode == "uv":
        nt.links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
    else:
        nt.links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])

    def _img_node(path, colorspace="sRGB", y=0):
        img = bpy.data.images.load(path, check_existing=True)
        node = nt.nodes.new("ShaderNodeTexImage"); node.image = img
        node.location = (500, y)
        if colorspace.lower() != "srgb":
            node.image.colorspace_settings.name = "Non-Color"
        if mapping_mode != "uv":
            try:
                node.projection = 'BOX'
                node.projection_blend = 0.2
            except Exception: pass
        nt.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    n_col = None
    if maps.get("basecolor"):
        n_col = _img_node(maps["basecolor"], "sRGB", y=0)
        nt.links.new(n_col.outputs["Color"], bsdf.inputs["Base Color"])

    if maps.get("roughness"):
        n_r = _img_node(maps["roughness"], "Non-Color", y=-220)
        nt.links.new(n_r.outputs["Color"], bsdf.inputs["Roughness"])

    if maps.get("metallic"):
        n_m = _img_node(maps["metallic"], "Non-Color", y=-440)
        nt.links.new(n_m.outputs["Color"], bsdf.inputs["Metallic"])

    # Specular (optional)
    if maps.get("specular"):
        n_s = _img_node(maps["specular"], "Non-Color", y=-660)

        # Principled BSDF v1 (<= Blender 3.x) had "Specular"
        # Principled BSDF v2 (Blender 4.x) uses "Specular IOR Level"
        spec_input = (bsdf.inputs.get("Specular")
                      or bsdf.inputs.get("Specular IOR Level")
                      or bsdf.inputs.get("Specular Level"))  # some builds label it this way

        if spec_input:
            nt.links.new(n_s.outputs["Color"], spec_input)
        else:
            print("[Tex] Specular map found but Principled has no compatible specular input; skipping.")


    # Normal (handles GL vs DX)
    if maps.get("normal"):
        n_n = _img_node(maps["normal"], "Non-Color", y=-880)

        filename = os.path.basename(maps["normal"]).lower()
        is_dx = any(tag in filename for tag in ("nor_dx", "normal_dx", "_dx"))

        # Build the Normal Map node
        n_nm = nt.nodes.new("ShaderNodeNormalMap"); n_nm.location = (950, -880)
        n_nm.space = 'TANGENT'

        if is_dx:
            # Invert the green channel (DX → GL)
            sep = nt.nodes.new("ShaderNodeSeparateRGB"); sep.location = (700, -930)
            inv = nt.nodes.new("ShaderNodeInvert");      inv.location = (820, -920)
            comb = nt.nodes.new("ShaderNodeCombineRGB"); comb.location = (820, -1030)

            nt.links.new(n_n.outputs["Color"], sep.inputs["Image"])
            nt.links.new(sep.outputs["G"],     inv.inputs["Color"])
            nt.links.new(sep.outputs["R"],     comb.inputs["R"])
            nt.links.new(inv.outputs["Color"], comb.inputs["G"])
            nt.links.new(sep.outputs["B"],     comb.inputs["B"])
            nt.links.new(comb.outputs["Image"], n_nm.inputs["Color"])
        else:
            # GL normal → feed directly
            nt.links.new(n_n.outputs["Color"], n_nm.inputs["Color"])

        nt.links.new(n_nm.outputs["Normal"], bsdf.inputs["Normal"])


    # Height / Displacement
    if maps.get("height"):
        n_h = _img_node(maps["height"], "Non-Color", y=-1100)

        use_true_disp = (bpy.context.scene.render.engine == 'CYCLES') and \
                        ((arg("--true_displacement", "false") or "false").lower() in ("1","true","yes"))

        if use_true_disp:
            # Material output displacement
            n_hc = nt.nodes.new("ShaderNodeDisplacement"); n_hc.location = (900, -1100)
            n_hc.inputs["Scale"].default_value = float(arg("--disp_scale", "0.03") or 0.03)
            nt.links.new(n_h.outputs["Color"], n_hc.inputs["Height"])
            nt.links.new(n_hc.outputs["Displacement"], out.inputs["Displacement"])
            # Ensure material is allowed to displace
            try:
                mat.cycles.displacement_method = 'BOTH'
            except Exception:
                pass
        else:
            # Bump fallback (works in Eevee & Cycles)
            n_b = nt.nodes.new("ShaderNodeBump"); n_b.location = (900, -1100)
            n_b.inputs["Strength"].default_value = float(arg("--bump_strength", "1.0") or 1.0)
            nt.links.new(n_h.outputs["Color"], n_b.inputs["Height"])
            # Chain with normal map if present
            if 'n_nm' in locals():
                nt.links.new(n_nm.outputs["Normal"], n_b.inputs["Normal"])
            nt.links.new(n_b.outputs["Normal"], bsdf.inputs["Normal"])

    if maps.get("ao") and n_col is not None:
        n_ao = _img_node(maps["ao"], "Non-Color", y=220)
        n_mix = nt.nodes.new("ShaderNodeMixRGB"); n_mix.location = (900, 220)
        n_mix.blend_type = 'MULTIPLY'; n_mix.inputs["Fac"].default_value = 1.0
        nt.links.new(n_col.outputs["Color"], n_mix.inputs["Color1"])
        nt.links.new(n_ao.outputs["Color"], n_mix.inputs["Color2"])
        nt.links.new(n_mix.outputs["Color"], bsdf.inputs["Base Color"])

    return mat

# --------------- UV helpers ---------------
def ensure_uv_map(obj):
    if obj.type != 'MESH': return
    me = obj.data
    if me.uv_layers.active: return
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
    except Exception as e:
        print(f"[UV] Smart Project failed on {obj.name}: {e}")
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
        obj.select_set(False)
        bpy.context.view_layer.objects.active = None

def assign_material(obj, mat, mapping_mode):
    if not mat or obj.type != 'MESH': return
    if mapping_mode == 'uv':
        ensure_uv_map(obj)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

# --------------- name classifiers ---------------
def is_driveway(n):
    nl = n.lower()
    return ("driveway" in nl) or n in ("left_driveway","right_driveway")

def is_bikepath(n):
    nl = n.lower()
    return ("bikepath" in nl) or n in ("left_bikepath","right_bikepath")

def is_parking(n):
    nl = n.lower()
    return ("street_parking" in nl) or ("parking" in nl and "street" in nl)

def is_median(n):
    return n.lower() == "median"

def is_footpath(n):
    nl = n.lower()
    return ("footpath" in nl) or ("sidewalk" in nl) or n in ("left_footpath","right_footpath")

def is_roof(n):
    nl = n.lower()
    return n.startswith("Roof_") or nl.startswith("roof")

def is_wall(n):
    nl = n.lower()
    return (n.startswith("Building_") or "wall" in nl or "facade" in nl) and not is_roof(n)

def is_branch(n):
    nl = n.lower()
    # Tree bark meshes from 01_model.py are named like Tree_left_0_Wood,
    # with optional Blender duplicate suffixes such as .001.
    base = nl.split(".", 1)[0]
    return "_branch_" in base or (base.startswith("tree_") and (base.endswith("_wood") or "branch" in base))

def classify_texture_category(name):
    """Return the first matching texture category for an object name."""
    checks = {
        "roof": is_roof,
        "wall": is_wall,
        "median": is_median,
        "driveway": is_driveway,
        "parking": is_parking,
        "bikepath": is_bikepath,
        "footpath": is_footpath,
        "branches": is_branch,
    }
    for category in OBJECT_CATEGORY_PRIORITY:
        if checks[category](name):
            return category
    return None

# --------------- legacy shared-material path (not used by main) ---------------
def _legacy_build_category_materials():
    """Deprecated shared-material implementation retained for reference only."""
    mats = {}
    for cat in TEXTURE_CATEGORIES:
        folder = _folder_for(cat)
        if folder:
            print(f"[Tex] Category '{cat}' → {folder} (maxdepth={MAX_DEPTH})")
            mats[cat] = make_pbr_material(
                f"MAT_{cat.capitalize()}",
                folder,
                scale_xyz=(MP[cat], MP[cat], 1.0),   # meters → converted inside to 1/meters
                mapping_mode=MAPPING_MODE,
                randomize_offset=bool(RAND_UV_OFFSET.get(cat, False)),
                randomize_rotation=bool(RAND_UV_ROTATION.get(cat, False)),
            )

        else:
            print(f"[Tex] Category '{cat}' missing (folder not found).")
            mats[cat] = None
    return mats

# --------------- material selection ---------------
def build_category_materials():
    """Build street-level materials and collect per-object pools where useful."""
    mats = {}
    variant_state = {}
    for cat in TEXTURE_CATEGORIES:
        folder = _folder_for(cat)
        if folder:
            texture_sets = _discover_texture_sets(folder)
            if texture_sets:
                if cat in PER_OBJECT_VARIANT_CATEGORIES:
                    print(f"[Tex] Category '{cat}' -> {folder} ({len(texture_sets)} texture set(s), per-object random)")
                    mats[cat] = None
                    variant_state[cat] = {"sets": texture_sets, "bag": []}
                else:
                    texture_set = random.choice(texture_sets)
                    texture_name = os.path.basename(os.path.normpath(texture_set)) or cat
                    print(f"[Tex] Category '{cat}' -> {folder} ({len(texture_sets)} texture set(s), selected '{texture_name}')")
                    mats[cat] = make_pbr_material(
                        f"MAT_{cat.capitalize()}_{texture_name}",
                        texture_set,
                        scale_xyz=(MP[cat], MP[cat], 1.0),
                        mapping_mode=MAPPING_MODE,
                        randomize_offset=bool(RAND_UV_OFFSET.get(cat, False)),
                        randomize_rotation=bool(RAND_UV_ROTATION.get(cat, False)),
                    )
            else:
                print(f"[Tex] Category '{cat}' missing usable texture sets in {folder}.")
                mats[cat] = None
        else:
            print(f"[Tex] Category '{cat}' missing (folder not found).")
            mats[cat] = None
    return mats, variant_state

def assign_material_for_category(obj, category, mats, variant_state):
    """Assign street-level material or per-object material for organic/building categories."""
    if category in PER_OBJECT_VARIANT_CATEGORIES:
        texture_set = _next_texture_set(category, variant_state)
        if not texture_set:
            return False
        mat = make_pbr_material(
            _safe_material_name(category, obj.name, texture_set),
            texture_set,
            scale_xyz=(MP[category], MP[category], 1.0),
            mapping_mode=MAPPING_MODE,
            randomize_offset=bool(RAND_UV_OFFSET.get(category, False)),
            randomize_rotation=bool(RAND_UV_ROTATION.get(category, False)),
        )
        assign_material(obj, mat, MAPPING_MODE)
        return mat is not None

    mat = mats.get(category)
    if not mat:
        return False
    assign_material(obj, mat, MAPPING_MODE)
    return True

def save_blend(filepath, pack=PACK_DATA, make_relative=RELATIVE):
    path = bpy.path.abspath(filepath)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    if make_relative:
        try:
            bpy.ops.file.make_paths_relative()
        except Exception as e:
            print(f"[Save] make_paths_relative failed: {e}")
        bpy.ops.wm.save_mainfile()
    if pack:
        try:
            bpy.ops.file.pack_all()
        except Exception as e:
            print(f"[Save] Pack failed (continuing): {e}")
        bpy.ops.wm.save_mainfile()
    print(f"[Save] Wrote {path} (relative={make_relative}, packed={pack})")

# --------------- main ---------------
def main():
    if not TEX_ROOT and not any(OVERRIDE.values()):
        print("[Tex] Provide --texdir OR per-category --tex_* folders.")
        return

    mats, variant_state = build_category_materials()
    counts = {k:0 for k in TEXTURE_CATEGORIES}

    for o in bpy.data.objects:
        category = classify_texture_category(o.name)
        if category and assign_material_for_category(o, category, mats, variant_state):
            counts[category] += 1

    print("[Tex] Assigned materials:")
    for k,v in counts.items():
        print(f"  {k:9s}: {v}")

    if OUT_BLEND:
        save_blend(OUT_BLEND)

if __name__ == "__main__":
    main()
