# Blender Python Script: Procedural Street Generator with Thickness

import bpy
import colorsys
import math
import random
from mathutils import Vector, Matrix
import bmesh # Required for roof modifications
import sys, json, tempfile,  copy
import os, time

# --- for model paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _argv_get_early(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else default
    return default

ASSET_ROOT = os.path.abspath(os.path.expanduser(
    _argv_get_early(
        "--asset-root",
        _argv_get_early("--assets-dir", os.environ.get("PROCEDURAL_ASSET_ROOT", os.path.dirname(SCRIPT_DIR))),
    )
))

def script_path(*parts):
    return os.path.join(ASSET_ROOT, *parts)

def resolve_asset_path(path):
    if not path:
        return path
    path = os.path.expanduser(path)
    if path.startswith("//"):
        return bpy.path.abspath(path)
    if os.path.isabs(path):
        return path
    for base in (ASSET_ROOT, SCRIPT_DIR):
        candidate = os.path.join(base, path)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(ASSET_ROOT, path)

# --- Script Configuration ---
CLEAR_SCENE = True

# --- Constants for Thicknesses ---
DEFAULT_GROUND_THICKNESS = 0.1
DEFAULT_ROAD_THICKNESS = 0.1
DEFAULT_CURB_THICKNESS = 0.25

# --- START: OBJECT-SPECIFIC CONSTANTS ---
CAR_Z_OFFSET = 0.8
MIN_CAR_SPACING = 7.0
# CAR_ROTATION_FIX_RAD = math.pi / 2
# PROBLEM_CAR_FILENAME = None

LAMP_Z_OFFSET = 2.0
LAMP_DEFAULT_SPACING = 25.0
LAMP_ROTATION_FIX_RAD = 0
LAMP_X_OFFSET_FROM_BOUNDARY = 0.6

# --- NEW: Human Constants ---
HUMAN_Z_OFFSET = 1.5 # Small lift for humans
# --- END: OBJECT-SPECIFIC CONSTANTS ---

# --- Utility Functions ---
# (clear_scene, get_material, create_slab - unchanged)
def clear_scene():
    """Deletes all objects in the current scene."""
    if bpy.ops.object.mode_set.poll(): bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0: bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0: bpy.data.materials.remove(block)
    for block in bpy.data.textures:
        if block.users == 0: bpy.data.textures.remove(block)
    for block in bpy.data.images:
        if block.users == 0: bpy.data.images.remove(block)
    for block in bpy.data.collections:
        if block.users == 0: bpy.data.collections.remove(block)
    print("Scene Cleared.")

def get_material(name, color):
    """Gets or creates a material with a specified base color."""
    if len(color) == 3: color = (*color, 1.0)
    mat = bpy.data.materials.get(name)
    if mat is None: mat = bpy.data.materials.new(name=name); mat.use_nodes = True
    else:
        if not mat.use_nodes: mat.use_nodes = True
    if mat.use_nodes and mat.node_tree:
        principled_bsdf = None
        if 'Principled BSDF' in mat.node_tree.nodes: principled_bsdf = mat.node_tree.nodes['Principled BSDF']
        else:
            output_node = next((n for n in mat.node_tree.nodes if isinstance(n, bpy.types.ShaderNodeOutputMaterial)), None)
            if output_node:
                 if output_node.inputs['Surface'].links:
                     input_shader = output_node.inputs['Surface'].links[0].from_node
                     if input_shader.bl_idname == 'ShaderNodeBsdfPrincipled': principled_bsdf = input_shader
                 if not principled_bsdf: # Create if not found or not linked
                    principled_bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
                    while output_node.inputs['Surface'].links: mat.node_tree.links.remove(output_node.inputs['Surface'].links[0])
                    mat.node_tree.links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])
        if principled_bsdf:
            if 'Base Color' in principled_bsdf.inputs: principled_bsdf.inputs['Base Color'].default_value = color
            if 'Roughness' in principled_bsdf.inputs: principled_bsdf.inputs['Roughness'].default_value = 0.8
            if 'Specular' in principled_bsdf.inputs: principled_bsdf.inputs['Specular'].default_value = 0.25
    else: mat.diffuse_color = color[:3]
    return mat

def vary_color_rgba(color, hue_jitter=0.02, sat_jitter=0.08, val_jitter=0.10):
    """Apply a subtle HSV variation while keeping the color in the same family."""
    if len(color) == 3:
        color = (*color, 1.0)

    r, g, b, a = [float(c) for c in color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    h = (h + random.uniform(-hue_jitter, hue_jitter)) % 1.0
    s = max(0.0, min(1.0, s * random.uniform(1.0 - sat_jitter, 1.0 + sat_jitter)))
    v = max(0.0, min(1.0, v * random.uniform(1.0 - val_jitter, 1.0 + val_jitter)))

    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (r2, g2, b2, a)

def create_slab(width, length, thickness, center_location, name="Slab", material_color=(0.8, 0.8, 0.8, 1)):
    """Creates a solid slab (cuboid)."""
    if width <= 0 or length <= 0 or thickness <= 0: return None
    try:
        bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=center_location)
        slab = bpy.context.object; slab.name = name
        slab.scale = (width, length, thickness); bpy.ops.object.transform_apply(scale=True)
        mat = get_material(f"{name}_Mat", material_color)
        if slab.data.materials: slab.data.materials[0] = mat
        else: slab.data.materials.append(mat)
        print(f"Created Slab: {name} at {center_location} size (W:{width:.2f}, L:{length:.2f}, T:{thickness:.2f})")
        return slab
    except Exception as e:
        print(f"Error creating slab '{name}': {e}")
        if 'slab' in locals() and slab and slab.name in bpy.context.scene.objects: bpy.data.objects.remove(slab, do_unlink=True)
        return None

# --- Tree Generation (FAST BMESH VERSION) ---

def _bm_add_branch_segment(bm, start, end, radius, segments=6):
    vec = end - start
    length = vec.length
    if length <= 1e-6:
        return

    direction = vec.normalized()
    mid = (start + end) * 0.5

    ret = bmesh.ops.create_cone(
        bm,
        cap_ends=False,
        cap_tris=False,
        segments=max(3, int(segments)),
        radius1=radius,
        radius2=max(radius * 0.72, 0.002),
        depth=length,
    )
    verts = ret["verts"]

    rot = Vector((0, 0, 1)).rotation_difference(direction).to_matrix().to_4x4()
    mat = Matrix.Translation(mid) @ rot
    bmesh.ops.transform(bm, verts=verts, matrix=mat)


def _bm_add_leaf_blob(bm, center, radius, subdivisions=0):
    r = max(0.04, float(radius))

    # low-poly rounded blob: much nicer than cube, much cheaper than dense icospheres
    ret = bmesh.ops.create_uvsphere(
        bm,
        u_segments=5,
        v_segments=4,
        radius=r,
    )
    verts = ret["verts"]

    # slight random non-uniform scale so blobs do not all look identical
    sx = random.uniform(0.9, 1.15)
    sy = random.uniform(0.9, 1.15)
    sz = random.uniform(0.85, 1.20)

    mat = (
        Matrix.Translation(center) @
        Matrix.Diagonal((sx, sy, sz, 1.0))
    )
    bmesh.ops.transform(bm, verts=verts, matrix=mat)

def _grow_tree_bmesh(start, direction, length, radius, depth, max_depth, tree_data, bm_wood, bm_leaf):
    if length <= 0.01 or radius <= 0.005:
        return

    max_foliage_depth = tree_data.get('max_foliage_depth', max_depth)

    # terminal foliage
    if depth > max_foliage_depth or length < tree_data.get('min_foliage_length', 0.2):
        clusters = tree_data.get('leaf_clusters_per_tip', 1)
        leaf_base_size = tree_data.get('leaf_size_factor', 0.5)
        leaf_subdiv = tree_data.get('leaf_subdivisions', 0)

        for _ in range(clusters):
            offset_factor = 0.3
            offset = Vector((
                random.uniform(-offset_factor, offset_factor) * length,
                random.uniform(-offset_factor, offset_factor) * length,
                random.uniform(-offset_factor, offset_factor) * length,
            ))
            leaf_radius = max(0.02, length * leaf_base_size * random.uniform(0.8, 1.2))
            _bm_add_leaf_blob(bm_leaf, start + offset, leaf_radius, leaf_subdiv)
        return

    dir_normalized = direction.normalized()
    if dir_normalized.length < 1e-6:
        dir_normalized = Vector((0, 0, 1))

    end = start + dir_normalized * length

    _bm_add_branch_segment(
        bm_wood,
        start,
        end,
        radius,
        segments=tree_data.get('branch_vertices', 6)
    )

    num_children = tree_data.get('branch_count', 3)
    if tree_data.get('vary_branch_count', True):
        num_children = max(1, num_children + random.randint(-1, 1))

    branch_angle_rad = math.radians(tree_data.get('branch_angle_deg', 30))

    up = Vector((0, 0, 1))
    axis = up.cross(dir_normalized)
    if axis.length < 1e-6:
        axis = Vector((1, 0, 0)).cross(dir_normalized)
    if axis.length < 1e-6:
        axis = Vector((0, 1, 0))
    axis.normalize()

    for _ in range(num_children):
        rot_axis = axis.copy()
        tilt_variance = tree_data.get('branch_angle_variance', 0.3)
        tilt = branch_angle_rad * random.uniform(1.0 - tilt_variance, 1.0 + tilt_variance)
        twist = random.uniform(0, 2 * math.pi)

        try:
            mat_twist = Matrix.Rotation(twist, 4, dir_normalized)
            mat_tilt = Matrix.Rotation(tilt, 4, rot_axis)
            rot_mat = mat_twist @ mat_tilt
        except ValueError:
            rot_mat = Matrix.Identity(4)

        new_dir_4d = rot_mat @ dir_normalized.to_4d()
        new_dir = new_dir_4d.to_3d().normalized()
        if new_dir.length < 0.99 or not all(math.isfinite(c) for c in new_dir):
            new_dir = dir_normalized

        length_factor = tree_data.get('branch_length_decay', 0.7)
        length_decay_variance = tree_data.get('branch_length_decay_variance', 0.1)
        new_length = length * length_factor * random.uniform(
            1.0 - length_decay_variance,
            1.0 + length_decay_variance
        )

        radius_factor = tree_data.get('branch_radius_decay', 0.65)
        radius_decay_variance = tree_data.get('branch_radius_decay_variance', 0.1)
        new_radius = radius * radius_factor * random.uniform(
            1.0 - radius_decay_variance,
            1.0 + radius_decay_variance
        )

        _grow_tree_bmesh(
            end,
            new_dir,
            new_length,
            new_radius,
            depth + 1,
            max_depth,
            tree_data,
            bm_wood,
            bm_leaf
        )


def create_tree(location, tree_config, side, index):
    """Creates a complete tree as 2 mesh objects (wood + leaves) parented to one root."""
    tree_name = f"Tree_{side}_{index}"
    print(f"Creating Tree: {tree_name} at {location}")

    trunk_height = random.uniform(tree_config.get('min_height', 3.0), tree_config.get('max_height', 5.0))
    trunk_radius = tree_config.get('trunk_radius', 0.3) * random.uniform(0.9, 1.1)

    start_direction = Vector(tree_config.get('trunk_direction', (0, 0, 1))).normalized()
    if start_direction.length < 0.9:
        start_direction = Vector((0, 0, 1))

    bm_wood = bmesh.new()
    bm_leaf = bmesh.new()

    _grow_tree_bmesh(
        Vector((0, 0, 0)),
        start_direction,
        trunk_height,
        trunk_radius,
        0,
        tree_config.get('branch_levels', 3),
        tree_config,
        bm_wood,
        bm_leaf
    )

    wood_mesh = bpy.data.meshes.new(f"{tree_name}_WoodMesh")
    leaf_mesh = bpy.data.meshes.new(f"{tree_name}_LeafMesh")

    bm_wood.to_mesh(wood_mesh)
    bm_leaf.to_mesh(leaf_mesh)
    bm_wood.free()
    bm_leaf.free()

    wood_mesh.update()
    leaf_mesh.update()

    root = bpy.data.objects.new(tree_name, None)
    root.location = Vector(location)
    bpy.context.scene.collection.objects.link(root)

    wood_obj = bpy.data.objects.new(f"{tree_name}_Wood", wood_mesh)
    leaf_obj = bpy.data.objects.new(f"{tree_name}_Leaves", leaf_mesh)

    bpy.context.scene.collection.objects.link(wood_obj)
    bpy.context.scene.collection.objects.link(leaf_obj)

    wood_obj.parent = root
    leaf_obj.parent = root

    wood_obj.location = (0, 0, 0)
    leaf_obj.location = (0, 0, 0)

    bark_mat = get_material("BarkMat", tree_config.get('bark_material_color', (0.3, 0.15, 0.1, 1)))
    leaf_color = vary_color_rgba(
        tree_config.get('leaf_material_color', (0.1, 0.5, 0.1, 1)),
        hue_jitter=float(tree_config.get('leaf_color_hue_jitter', 0.02)),
        sat_jitter=float(tree_config.get('leaf_color_sat_jitter', 0.08)),
        val_jitter=float(tree_config.get('leaf_color_val_jitter', 0.10)),
    )
    leaf_mat = get_material(f"{tree_name}_LeafMat", leaf_color)

    if wood_obj.data.materials:
        wood_obj.data.materials[0] = bark_mat
    else:
        wood_obj.data.materials.append(bark_mat)

    if leaf_obj.data.materials:
        leaf_obj.data.materials[0] = leaf_mat
    else:
        leaf_obj.data.materials.append(leaf_mat)

    for p in wood_obj.data.polygons:
        p.use_smooth = True
    for p in leaf_obj.data.polygons:
        p.use_smooth = True

    return root

# --- Building Generation ---
# (create_roof - unchanged)
def create_roof(building_obj, roof_config, side, index):
    """Creates a roof for the given building object based on config."""
    if not building_obj: print(f"Error: No building obj for roof {side}_{index}."); return None
    try:
        bpy.context.view_layer.update()
        b_dims_local = [Vector(co) for co in building_obj.bound_box] # Local coords if scale applied
        b_world_loc = building_obj.matrix_world.translation
        min_x=min(v.x for v in b_dims_local); max_x=max(v.x for v in b_dims_local)
        min_y=min(v.y for v in b_dims_local); max_y=max(v.y for v in b_dims_local)
        min_z=min(v.z for v in b_dims_local); max_z=max(v.z for v in b_dims_local)
        dim_x = max_x-min_x; dim_y = max_y-min_y; dim_z_local = max_z-min_z
        building_top_z = b_world_loc.z + dim_z_local / 2 # World Z top based on center origin
    except Exception as e: print(f"Error getting building dims/location {building_obj.name}: {e}"); return None

    def _resolve_roof_type(cfg):
        roof_type = str(cfg.get('roof_type', 'random')).strip().lower()
        if roof_type in ('random', 'mixed', 'random_gabled_flat'):
            choices = cfg.get('roof_type_choices', ('gabled', 'flat'))
            if not isinstance(choices, (list, tuple)):
                choices = ('gabled', 'flat')
            choices = [str(choice).strip().lower() for choice in choices if str(choice).strip()]
            valid_choices = [choice for choice in choices if choice in ('gabled', 'flat', 'pitched')]
            if not valid_choices:
                valid_choices = ['gabled', 'flat']
            return random.choice(valid_choices)
        return roof_type

    roof_height_fallback = float(roof_config.get('roof_height', 1.0) or 1.0)
    min_roof_height = float(roof_config.get('min_roof_height', roof_height_fallback) or roof_height_fallback)
    max_roof_height = float(roof_config.get('max_roof_height', roof_height_fallback) or roof_height_fallback)
    if max_roof_height < min_roof_height:
        min_roof_height, max_roof_height = max_roof_height, min_roof_height
    roof_height = random.uniform(min_roof_height, max_roof_height)
    flat_roof_height = float(roof_config.get('flat_roof_height', 0.2) or 0.2)
    roof_type = _resolve_roof_type(roof_config)
    roof_mat_color = roof_config.get('roof_material', (0.5, 0.1, 0.1, 1)); roof_mat_name = f"Roof_{side}_Mat"; roof_name = f"Roof_{side}_{index}"
    print(f"Creating Roof: {roof_name} (Type: {roof_type})")
    try: bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,0)); roof = bpy.context.object; roof.name = roof_name
    except Exception as e: print(f"Error creating base cube for roof {roof_name}: {e}"); return None
    roof_mat = get_material(roof_mat_name, roof_mat_color)
    if roof.data.materials: roof.data.materials[0] = roof_mat
    else: roof.data.materials.append(roof_mat)

    if roof_type == 'flat':
        roof.scale = (dim_x, dim_y, flat_roof_height); bpy.ops.object.transform_apply(scale=True)
        roof.location = (b_world_loc.x, b_world_loc.y, building_top_z + flat_roof_height / 2)
    elif roof_type in ['gabled', 'pitched']:
        roof.scale = (dim_x, dim_y, roof_height); bpy.ops.object.transform_apply(scale=True)
        roof.location = (b_world_loc.x, b_world_loc.y, building_top_z + roof_height / 2)
        try:
            bpy.context.view_layer.objects.active = roof; roof.select_set(True)
            bpy.ops.object.mode_set(mode='EDIT')
            bm = bmesh.from_edit_mesh(roof.data); bm.verts.ensure_lookup_table()
            local_top_z_threshold = (roof.dimensions.z / 2.0) * 0.9
            top_verts = [v for v in bm.verts if v.co.z > local_top_z_threshold]
            if len(top_verts) >= 4:
                if roof_type == 'gabled':
                    mid_y_local = (min(v.co.y for v in top_verts) + max(v.co.y for v in top_verts)) / 2
                    for v in top_verts: v.co.y = mid_y_local
                elif roof_type == 'pitched':
                    lower_positive_x_edge = (side == 'left'); local_base_z = -roof.dimensions.z / 2.0
                    for v in top_verts:
                        if (lower_positive_x_edge and v.co.x > 0.0) or (not lower_positive_x_edge and v.co.x < 0.0):
                            v.co.z = local_base_z
            else: print(f"Warning: Expected >=4 top verts for {roof_type} roof {roof_name}, found {len(top_verts)}. Reverting."); raise ValueError("Incorrect vertex count")
            if bpy.context.object.mode == 'EDIT': bmesh.update_edit_mesh(roof.data); bpy.ops.object.mode_set(mode='OBJECT')
        except Exception as e:
            print(f"Error during BMesh mod for {roof_name}: {e}. Creating flat roof instead.")
            if bpy.context.object and bpy.context.object.mode == 'EDIT': bpy.ops.object.mode_set(mode='OBJECT')
            roof.scale = (dim_x, dim_y, flat_roof_height); bpy.ops.object.transform_apply(scale=True)
            roof.location = (b_world_loc.x, b_world_loc.y, building_top_z + flat_roof_height / 2)
        finally: bpy.ops.object.select_all(action='DESELECT')
    else: # Unknown roof type
        print(f"Warning: Unknown roof_type '{roof_type}' for {roof_name}. Creating flat roof.")
        roof.scale = (dim_x, dim_y, flat_roof_height); bpy.ops.object.transform_apply(scale=True)
        roof.location = (b_world_loc.x, b_world_loc.y, building_top_z + flat_roof_height / 2)
    if roof and roof.select_get(): roof.select_set(False)
    if bpy.context.view_layer.objects.active == roof: bpy.context.view_layer.objects.active = None
    return roof

# --- Street Layout Generation (with random tree Y) ---
def build_street_side(start_offset, side_data, side, length):
    """Builds one side of the street including all elements with thickness."""
    print(f"\n--- Building Street Side: {side.upper()} ---")
    current_offset = start_offset; direction = -1 if side == 'left' else 1
    # --- Calculate boundaries first ---
    driveway_start_x = current_offset
    driveway_width = side_data.get('driveway',{}).get('width',0) if side_data.get('driveway',{}).get('present') else 0
    driveway_end_x = driveway_start_x + driveway_width * direction
    bikepath_start_x = driveway_end_x
    bikepath_width = side_data.get('bikepath',{}).get('width',0) if side_data.get('bikepath',{}).get('present') else 0
    bikepath_end_x = bikepath_start_x + bikepath_width * direction
    footpath_start_x = bikepath_end_x
    footpath_width = side_data.get('footpath',{}).get('width',0) if side_data.get('footpath',{}).get('present') else 0
    footpath_end_x = footpath_start_x + footpath_width * direction
    # --- Define target top surface Z levels ---
    road_top_z = DEFAULT_ROAD_THICKNESS * 0.5
    curb_top_z = road_top_z + (DEFAULT_CURB_THICKNESS - DEFAULT_ROAD_THICKNESS) + 0.01
    # --- Create Transportation Elements ---
    base_road_thickness = DEFAULT_ROAD_THICKNESS; last_transport_element_top_z = 0.0
    
    ################
    # helpers
    def _use_parking_on_this_side(cfg, side_name, driveway_w):
        if not cfg or not cfg.get('present', False):
            return False
        sides = (cfg.get('sides') or 'both').lower()
        if not (sides == 'both' or sides == side_name.lower()):
            return False
        return driveway_w >= float(cfg.get('min_driveway_width', 3.5))

    def _parking_width(cfg, side_name):
        import random
        side = side_name.lower()

        # Stable per-side seeding (keeps your current determinism)
        if cfg.get('seed', 0) is not None:
            random.seed(int(cfg.get('seed', 0)) + (0 if side == 'left' else 1))

        # 1) Absolute per-side width wins
        per_side = cfg.get('width_m_per_side') or {}
        if isinstance(per_side, dict) and per_side.get(side) is not None:
            return float(per_side[side])

        # Convenience aliases (also per-side)
        key = f'width_m_{side}'
        if cfg.get(key) is not None:
            return float(cfg[key])

        # 2) Single absolute width (legacy) applies to both sides
        if cfg.get('width_m') is not None:
            return float(cfg['width_m'])

        # 3) Per-side random range (if provided)
        rng_map = cfg.get('width_range_per_side') or {}
        if isinstance(rng_map, dict) and rng_map.get(side):
            w0, w1 = rng_map[side]
            return max(0.0, float(random.uniform(w0, w1)))

        # 4) Fallback to legacy common range (current behavior)
        w0, w1 = cfg.get('width_range', (3.5, 5.0))
        return max(0.0, float(random.uniform(w0, w1)))


    parking_cfg = street_data.get('street_parking', {}) or {}

    # --- DRIVEWAY ---
    if driveway_width > 0:
        config = side_data['driveway']
        elem_thickness = config.get('thickness', base_road_thickness)
        center_z = road_top_z - (elem_thickness / 2.0)  # top-anchored to road
        element_center_x = driveway_start_x + (driveway_width / 2 * direction)

        slab_obj = create_slab(
            driveway_width, length, elem_thickness,
            (element_center_x, 0, center_z),
            f"{side}_driveway",
            config.get('material', (0.5, 0.5, 0.5, 1))
        )
        if slab_obj:
            last_transport_element_top_z = road_top_z
        
             # --- Simple gutter strip (visual only) ---
        gutter_width = 0.22
        gutter_thick = 0.010
        gutter_color = (0.07, 0.07, 0.07, 1)
        top_z = road_top_z + 0.006     # slightly sunken for a shadow line
        gz = top_z - gutter_thick * 0.5
        curb_x = driveway_start_x + driveway_width * direction
        gutter_cx = curb_x - direction * (gutter_width * 0.5)
        create_slab(gutter_width, length, gutter_thick, (gutter_cx, 0.0, gz), f"{side}_gutter", gutter_color)
        
    # --- STREET PARKING (between driveway and bikepath) ---
    parking_width = 0.0
    if _use_parking_on_this_side(parking_cfg, side, driveway_width):
        parking_width = _parking_width(parking_cfg, side)
        if parking_width > 0.0:
            p_thick = float(parking_cfg.get('thickness') or base_road_thickness)
            p_color = parking_cfg.get('color', (0.22, 0.22, 0.22, 1))
            p_center_z = road_top_z - (p_thick * 0.5)

            parking_start_x = driveway_start_x + (driveway_width * direction)
            p_center_x = parking_start_x + (parking_width / 2.0 * direction)

            slab_obj = create_slab(
                parking_width, length, p_thick,
                (p_center_x, 0.0, p_center_z),
                f"{side}_street_parking",
                p_color
            )
            if slab_obj:
                last_transport_element_top_z = road_top_z
            street_data.setdefault('_runtime', {}).setdefault(side, {})['parking_width'] = float(parking_width)
                
    # --- push downstream elements outward so they don't overlap parking ---
    if parking_width > 0.0:
        # move bikepath start by the parking width
        bikepath_start_x += (parking_width * direction)

        # if footpath_start_x (or other outward starts) were computed earlier, bump them too
        try:
            footpath_start_x += (parking_width * direction)
        except NameError:
            pass
        
        # NEW: recompute ends so everyone downstream uses the updated edges
        bikepath_end_x  = bikepath_start_x  + (bikepath_width  * direction)
        footpath_end_x  = footpath_start_x  + (footpath_width  * direction)

        # If you have any other precomputed outward refs, nudge them as well:
        # e.g., tree_line_x, building_front_start_x, etc.
        for var_name in ("tree_line_x", "building_start_x", "setback_start_x"):
            if var_name in locals():
                locals()[var_name] += (parking_width * direction)

        print(f"[Parking] Shifted downstream starts by {parking_width:.2f} m on {side}.")

                
                ##########################################
        
    if bikepath_width > 0:
        config = side_data['bikepath']; elem_thickness = config.get('thickness', base_road_thickness)
        center_z = road_top_z + (elem_thickness / 2.0); element_center_x = bikepath_start_x + (bikepath_width / 2 * direction)
        slab_obj = create_slab(bikepath_width, length, elem_thickness, (element_center_x, 0, center_z), f"{side}_bikepath", config.get('material', (0.0, 0.4, 0.1, 1)))
        if slab_obj: last_transport_element_top_z = road_top_z
        
    # --- Tree Placement (Random Y) ---
    tree_boundary_x = bikepath_end_x; tree_base_z = last_transport_element_top_z
    if side_data.get('trees') and side_data['trees'].get('present', True):
        tree_config = side_data['trees']; num_trees = tree_config.get('count', 0)
        if num_trees > 0:
            print(f"  Placing {num_trees} Trees randomly near X = {tree_boundary_x:.2f}, Base Z = {tree_base_z:.3f}")
            for i in range(num_trees):
                y_pos = random.uniform(-length * 0.48, length * 0.48) # Random Y
                x_jitter = random.uniform(-0.15, 0.15); y_jitter = random.uniform(-0.5, 0.5)
                tree_loc_z = tree_base_z + tree_config.get('base_z_offset', 0.01)
                tree_loc = (tree_boundary_x + x_jitter, y_pos + y_jitter, tree_loc_z)
                create_tree(tree_loc, tree_config, side, i)
        else: print(f"  Skipping Trees on side '{side}': count is zero.")
    else: print(f"  Skipping Trees on side '{side}': not present or no config.")
    
    # --- Create Footpath ---
    last_element_top_z = curb_top_z
    if footpath_width > 0:
        config = side_data['footpath']; elem_thickness = config.get('thickness', DEFAULT_CURB_THICKNESS)
        center_z = curb_top_z - (elem_thickness / 2.0); element_center_x = footpath_start_x + (footpath_width / 2 * direction)
        print(f"  Placing Footpath starting at X = {footpath_start_x:.2f}, Top Z = {curb_top_z:.3f}")
        slab_obj = create_slab(footpath_width, length, elem_thickness, (element_center_x, 0, center_z), f"{side}_footpath", config.get('material', (0.6, 0.6, 0.6, 1)))
    else: print(f"  Skipping Footpath on side '{side}': width is zero."); last_element_top_z = tree_base_z # Fallback Z for buildings
    # --- Building Generation ---
    building_start_x = footpath_end_x; building_base_z = last_element_top_z
    if side_data.get('buildings') and side_data['buildings'].get('present', True):
        b_config = side_data['buildings']; building_count = b_config.get('count', 1)
        if building_count > 0:
            building_gap=b_config.get('gap',1.0); building_depth=b_config.get('depth',10.0); building_width=b_config.get('width',10.0)
            height_fallback = float(b_config.get('height', 8.0) or 8.0)
            min_height = float(b_config.get('min_height', height_fallback) or height_fallback)
            max_height = float(b_config.get('max_height', height_fallback) or height_fallback)
            if max_height < min_height:
                min_height, max_height = max_height, min_height
            setback=b_config.get('setback',0.0)
            bldg_mat_color=b_config.get('material',(0.7,0.7,0.7,1)); bldg_mat_name = f"Building_{side}_Mat"
            print(f"  Placing {building_count} Buildings starting setback from X = {building_start_x:.2f}, Base Z = {building_base_z:.3f}")
            total_building_space_y = building_count * building_width + max(0, building_count - 1) * building_gap
            start_y = -length / 2 + (length - total_building_space_y) / 2
            building_center_x = building_start_x + (setback + building_depth / 2) * direction
            for i in range(building_count):
                building_height = random.uniform(min_height, max_height)
                building_center_y = start_y + i * (building_width + building_gap) + building_width / 2
                body_center_z = building_base_z + building_height / 2.0
                try:
                    bpy.ops.mesh.primitive_cube_add(size=1, location=(building_center_x, building_center_y, body_center_z))
                    building = bpy.context.object; building.name = f"Building_{side}_{i}"
                    building.scale = (building_depth, building_width, building_height)
                    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                    body_mat = get_material(bldg_mat_name, bldg_mat_color)
                    if building.data.materials: building.data.materials[0] = body_mat
                    else: building.data.materials.append(body_mat)
                    create_roof(building, b_config, side, i)
                except Exception as e: print(f"Error creating building {side}_{i}: {e}")
        else: print(f"  Skipping Buildings on side '{side}': count is zero.")
    else: print(f"  Skipping Buildings on side '{side}': not present or no config.")
    print(f"--- Finished Street Side: {side.upper()} ---")

# --- Car Importing Function ---
# (import_and_place_car - unchanged)
def import_and_place_car(name, model_path, location, rotation_euler, uniform_scale):
    """Imports a car model, scales it, places it, and handles basic parenting."""
    abs_model_path = resolve_asset_path(model_path)
    if not os.path.exists(abs_model_path): print(f"Error: Car model file not found: {abs_model_path}"); return None
    objects_before = set(bpy.context.scene.objects); main_car_object = None
    try:
        print(f"Importing car: {name} from {os.path.basename(abs_model_path)}")
        file_ext = os.path.splitext(abs_model_path)[1].lower()
        if bpy.ops.object.select_all.poll(): bpy.ops.object.select_all(action='DESELECT')
        if file_ext in ['.glb', '.gltf']: bpy.ops.import_scene.gltf(filepath=abs_model_path)
        elif file_ext == '.fbx': bpy.ops.import_scene.fbx(filepath=abs_model_path, use_manual_orientation=False, global_scale=1.0, axis_forward='-Z', axis_up='Y')
        elif file_ext == '.obj': bpy.ops.import_scene.obj(filepath=abs_model_path, axis_forward='-Z', axis_up='Y')
        else: print(f"Error: Unsupported file format '{file_ext}' for {abs_model_path}"); return None
        objects_after = set(bpy.context.scene.objects); new_objects = list(objects_after - objects_before)
        if not new_objects:
            if bpy.context.selected_objects: new_objects = list(bpy.context.selected_objects)
            else: print(f"Error: No objects detected after importing {abs_model_path}"); return None
        potential_parents = [obj for obj in new_objects if obj.type == 'EMPTY']; mesh_objects = [obj for obj in new_objects if obj.type == 'MESH']
        if potential_parents: main_car_object = potential_parents[0]
        elif mesh_objects: main_car_object = mesh_objects[0]
        elif new_objects: main_car_object = new_objects[0]
        else: print("Error: Could not identify main object after import."); return None
        print(f"  Identified main object: '{main_car_object.name}' (Type: {main_car_object.type})")
        if bpy.ops.object.select_all.poll(): bpy.ops.object.select_all(action='DESELECT')
        if len(new_objects) > 1:
            print(f"  Parenting {len(new_objects)-1} parts to '{main_car_object.name}'...")
            for obj in new_objects:
                if obj != main_car_object: obj.select_set(True)
            bpy.context.view_layer.objects.active = main_car_object; main_car_object.select_set(True)
            try: bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
            except Exception as e: print(f"  Warning: Could not parent objects: {e}")
            if bpy.ops.object.select_all.poll(): bpy.ops.object.select_all(action='DESELECT')
        main_car_object.name = name; main_car_object.scale = (uniform_scale, uniform_scale, uniform_scale)
        bpy.context.view_layer.objects.active = main_car_object; main_car_object.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        main_car_object.select_set(False)
        main_car_object.location = location; main_car_object.rotation_euler = rotation_euler
        print(f"Placed Car: {name} at {tuple(round(c, 2) for c in location)}")
        return main_car_object
    except Exception as e:
        print(f"Error during import or placement of '{name}': {e}")
        objects_after_error = set(bpy.context.scene.objects); failed_objects = list(objects_after_error - objects_before)
        if failed_objects:
             print(f"  Cleaning up {len(failed_objects)} objects from failed import.")
             if bpy.ops.object.select_all.poll(): bpy.ops.object.select_all(action='DESELECT')
             for obj in failed_objects:
                  if obj.name in bpy.data.objects: obj.select_set(True)
             if bpy.context.selected_objects: bpy.ops.object.delete(use_global=False)
        return None


def place_parked_cars(street_data, median_offset):
    cfg = street_data.get('parked_cars', {}) or {}
    if not cfg or not cfg.get('present', False):
        print("[ParkedCars] Skipping (not present).")
        return

    import math, random, os
    # Deterministic randomness if desired
    seed = cfg.get('seed', 1)
    if seed is not None:
        random.seed(seed)

    street_length = float(street_data.get('length', 0.0) or 0.0)
    if street_length <= 0.0:
        print("[ParkedCars] Skipping (no street length).")
        return

    # Models & scale: fall back to driving-cars config if not provided
    car_cfg = street_data.get('cars', {}) or {}
    model_paths = (cfg.get('model_paths') or car_cfg.get('model_paths') or [])
    car_scale   = float(cfg.get('scale') if cfg.get('scale') is not None else car_cfg.get('scale', 1.0))

    # Validate models
    valid_model_paths = []
    for p in model_paths:
        ap = bpy.path.abspath(p)
        if os.path.exists(ap):
            valid_model_paths.append(p)
    if not valid_model_paths:
        print("[ParkedCars] No valid model paths; skipping.")
        return

    # Params
    count_per_side = int(cfg.get('count_per_side', 4))
    wide_threshold = float(cfg.get('wide_threshold_m', 2.5))
    angle_deg      = float(cfg.get('angle_deg_if_wide', 45.0))
    min_spacing    = float(cfg.get('min_spacing_m', 5.0))
    end_margin     = float(cfg.get('end_margin_m', 4.0))
    x_inset        = float(cfg.get('x_inset_m', 0.30))
    y_lo = -street_length * 0.5 + end_margin
    y_hi =  street_length * 0.5 - end_margin

    # Helper: build y positions with spacing (like your moving cars)
    def spaced_y_positions(n):
        if n <= 0 or y_hi <= y_lo:
            return []
        seeds = sorted(random.uniform(y_lo, y_hi) for _ in range(n))
        adjusted = []
        last_y = -float("inf")
        for y in seeds:
            y_req = last_y + min_spacing
            y_adj = max(y, y_req)
            y_adj = max(y_lo, min(y_adj, y_hi))
            if adjusted and y_adj < adjusted[-1] + min_spacing:
                y_adj = min(adjusted[-1] + min_spacing, y_hi)
            adjusted.append(y_adj)
            last_y = y_adj
        return adjusted

    def _side_loop(side_name):
        # Side geometry
        side = street_data['sides'].get(side_name) or {}
        direction = -1 if side_name == 'left' else 1
        driveway_w = float(side.get('driveway', {}).get('width', 0.0) or 0.0)

        # Street-parking width saved at build time (don’t reseed now)
        parking_w = float(street_data.get('_runtime', {}).get(side_name, {}).get('parking_width', 0.0) or 0.0)
        if parking_w <= 0.0 or driveway_w <= 0.0:
            return 0

        # Reconstruct the same X chain used during build
        driveway_start_x = (-median_offset if side_name == 'left' else median_offset)
        parking_start_x  = driveway_start_x + (driveway_w * direction)     # curb edge between driveway and parking
        curb_edge_x      = parking_start_x
        parking_center_x = parking_start_x + (parking_w * 0.5 * direction)

        # ---------- ORIENTATION (UPDATED) ----------
        # Reuse driving-cars' per-side yaw map + optional global correction.
        # Falls back to sensible defaults if nothing is configured.
        traffic = (car_cfg.get('traffic') or 'RHT').upper()
        rot_map_deg = car_cfg.get('rotation_by_side_deg')
        if not isinstance(rot_map_deg, dict):
            rot_map_deg = {'left': 180.0, 'right': 0.0} if traffic == 'RHT' else {'left': 0.0, 'right': 180.0}
        base_rot = math.radians(float(rot_map_deg[side_name]))
        is_wide  = parking_w >= wide_threshold
        rot = (base_rot + ((+1 if side_name == 'left' else -1) * math.radians(angle_deg))) if is_wide else base_rot
        # ------------------------------------------

        # X placement (slightly toward curb for angled so tails don’t clip bikepath)
        px = (curb_edge_x + direction * min(parking_w * 0.45, x_inset + parking_w * 0.35)) if is_wide else parking_center_x

        ys = spaced_y_positions(count_per_side)
        placed = 0
        road_surface_z = DEFAULT_ROAD_THICKNESS * 0.5

        for i, y in enumerate(ys):
            car_name = f"ParkedCar_{side_name[0].upper()}_{i+1}"
            model_path = random.choice(valid_model_paths)
            obj = import_and_place_car(car_name, model_path, (0, 0, 0), (0, 0, 0), car_scale)
            if not obj:
                continue
            # Land it cleanly on the surface (same trick as moving cars)
            bpy.context.view_layer.update()
            local_min_z = min(corner[2] for corner in obj.bound_box)
            z_offset_needed = -local_min_z
            final_z = road_surface_z + z_offset_needed + CAR_Z_OFFSET

            obj.location = (px, y, final_z)
            obj.rotation_euler = (0.0, 0.0, rot)
            obj.select_set(False)
            if bpy.context.view_layer.objects.active == obj:
                bpy.context.view_layer.objects.active = None
            placed += 1
        return placed

    sides_setting = (cfg.get('sides') or 'both').lower()
    total = 0
    if sides_setting in ('left', 'both'):
        total += _side_loop('left')
    if sides_setting in ('right', 'both'):
        total += _side_loop('right')

    print(f"[ParkedCars] Placed {total} parked car(s).")

def _side_yaw_rad(street_data, side_name: str) -> float:
    """
    Resolve per-side yaw in radians, combining:
      - rotation_by_side_deg (if provided) OR traffic-based defaults
      - a single global axis correction (degrees)
    """
    import math
    car_cfg = street_data.get('cars', {}) or {}
    traffic = (car_cfg.get('traffic') or 'RHT').upper()

    # If you set rotation_by_side_deg in config, it wins; otherwise use traffic defaults
    rot_map = (car_cfg.get('rotation_by_side_deg')
               or ({'left': 180.0, 'right': 0.0} if traffic == 'RHT'
                   else {'left': 0.0, 'right': 180.0}))

    global_fix_deg = float(car_cfg.get('global_rotation_fix_deg', 0.0))
    return math.radians(float(rot_map[side_name.lower()])) + math.radians(global_fix_deg)



# --- Lamp Importing Function ---
# (import_and_place_lamp - unchanged)
def import_and_place_lamp(name, model_path, location, rotation_euler, uniform_scale):
    """Imports a lamp model, scales it, places it, and handles basic parenting."""
    abs_model_path = resolve_asset_path(model_path)
    if not os.path.exists(abs_model_path): print(f"Error: Lamp model file not found: {abs_model_path}"); return None
    objects_before = set(bpy.context.scene.objects); main_lamp_object = None
    try:
        print(f"Importing lamp: {name} from {os.path.basename(abs_model_path)}")
        file_ext = os.path.splitext(abs_model_path)[1].lower()
        if bpy.ops.object.select_all.poll(): bpy.ops.object.select_all(action='DESELECT')
        if file_ext in ['.glb', '.gltf']: bpy.ops.import_scene.gltf(filepath=abs_model_path)
        elif file_ext == '.fbx': bpy.ops.import_scene.fbx(filepath=abs_model_path, use_manual_orientation=False, global_scale=1.0, axis_forward='-Z', axis_up='Y')
        elif file_ext == '.obj': bpy.ops.import_scene.obj(filepath=abs_model_path, axis_forward='-Z', axis_up='Y')
        else: print(f"Error: Unsupported file format '{file_ext}' for {abs_model_path}"); return None
        objects_after = set(bpy.context.scene.objects); new_objects = list(objects_after - objects_before)
        if not new_objects:
            if bpy.context.selected_objects: new_objects = list(bpy.context.selected_objects)
            else: print(f"Error: No objects detected after importing {abs_model_path}"); return None
        potential_parents=[obj for obj in new_objects if obj.type=='EMPTY']; mesh_objects=[obj for obj in new_objects if obj.type=='MESH']
        if potential_parents: main_lamp_object = potential_parents[0]
        elif mesh_objects: main_lamp_object = mesh_objects[0]
        elif new_objects: main_lamp_object = new_objects[0]
        else: print("Error: Could not identify main object after import."); return None
        print(f"  Identified main object: '{main_lamp_object.name}' (Type: {main_lamp_object.type})")
        if bpy.ops.object.select_all.poll(): bpy.ops.object.select_all(action='DESELECT')
        if len(new_objects) > 1:
            print(f"  Parenting {len(new_objects)-1} parts to '{main_lamp_object.name}'...")
            for obj in new_objects:
                if obj != main_lamp_object: obj.select_set(True)
            bpy.context.view_layer.objects.active = main_lamp_object; main_lamp_object.select_set(True)
            try: bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
            except Exception as e: print(f"  Warning: Could not parent lamp objects: {e}")
            if bpy.ops.object.select_all.poll(): bpy.ops.object.select_all(action='DESELECT')
        main_lamp_object.name = name; main_lamp_object.scale = (uniform_scale, uniform_scale, uniform_scale)
        bpy.context.view_layer.objects.active = main_lamp_object; main_lamp_object.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        main_lamp_object.select_set(False)
        main_lamp_object.location = location; main_lamp_object.rotation_euler = rotation_euler
        print(f"Placed Lamp: {name} at {tuple(round(c, 2) for c in location)}")
        return main_lamp_object
    except Exception as e:
        print(f"Error during import or placement of '{name}': {e}")
        objects_after_error = set(bpy.context.scene.objects); failed_objects = list(objects_after_error - objects_before)
        if failed_objects:
             print(f"  Cleaning up {len(failed_objects)} objects from failed lamp import.")
             if bpy.ops.object.select_all.poll(): bpy.ops.object.select_all(action='DESELECT')
             for obj in failed_objects:
                  if obj.name in bpy.data.objects: obj.select_set(True)
             if bpy.context.selected_objects: bpy.ops.object.delete(use_global=False)
        return None

# --- START: ADDED HUMAN IMPORT FUNCTION ---
def import_and_place_human(name, model_path, location, rotation_euler, uniform_scale):
    """Imports a human model, scales it, places it, and handles basic parenting."""
    abs_model_path = resolve_asset_path(model_path)
    if not os.path.exists(abs_model_path):
        print(f"Error: Human model file not found: {abs_model_path}")
        return None
    objects_before = set(bpy.context.scene.objects); main_human_object = None
    try:
        print(f"Importing human: {name} from {os.path.basename(abs_model_path)}")
        file_ext = os.path.splitext(abs_model_path)[1].lower()
        if bpy.ops.object.select_all.poll(): bpy.ops.object.select_all(action='DESELECT')
        # --- Importer calls ---
        if file_ext in ['.glb', '.gltf']: bpy.ops.import_scene.gltf(filepath=abs_model_path)
        elif file_ext == '.fbx': bpy.ops.import_scene.fbx(filepath=abs_model_path, use_manual_orientation=False, global_scale=1.0, axis_forward='-Z', axis_up='Y')
        elif file_ext == '.obj': bpy.ops.import_scene.obj(filepath=abs_model_path, axis_forward='-Z', axis_up='Y')
        else: print(f"Error: Unsupported file format '{file_ext}' for {abs_model_path}"); return None
        # --- Identify imported ---
        objects_after = set(bpy.context.scene.objects); new_objects = list(objects_after - objects_before)
        if not new_objects:
            if bpy.context.selected_objects: new_objects = list(bpy.context.selected_objects)
            else: print(f"Error: No objects detected after importing {abs_model_path}"); return None
        # --- Find main object ---
        potential_parents=[obj for obj in new_objects if obj.type=='EMPTY']; mesh_objects=[obj for obj in new_objects if obj.type=='MESH']
        if potential_parents: main_human_object = potential_parents[0]
        elif mesh_objects: main_human_object = mesh_objects[0] # Often characters are single meshes
        elif new_objects: main_human_object = new_objects[0]
        else: print("Error: Could not identify main human object after import."); return None
        print(f"  Identified main object: '{main_human_object.name}' (Type: {main_human_object.type})")
        # --- Parent if needed ---
        if bpy.ops.object.select_all.poll(): bpy.ops.object.select_all(action='DESELECT')
        if len(new_objects) > 1:
            print(f"  Parenting {len(new_objects)-1} parts to '{main_human_object.name}'...")
            for obj in new_objects:
                if obj != main_human_object: obj.select_set(True)
            bpy.context.view_layer.objects.active = main_human_object; main_human_object.select_set(True)
            try: bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
            except Exception as e: print(f"  Warning: Could not parent human objects: {e}")
            if bpy.ops.object.select_all.poll(): bpy.ops.object.select_all(action='DESELECT')
        # --- Apply scale and transform ---
        main_human_object.name = name; main_human_object.scale = (uniform_scale, uniform_scale, uniform_scale)
        bpy.context.view_layer.objects.active = main_human_object; main_human_object.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        main_human_object.select_set(False)
        main_human_object.location = location; main_human_object.rotation_euler = rotation_euler
        print(f"Placed Human: {name} at {tuple(round(c, 2) for c in location)}")
        return main_human_object
    except Exception as e:
        print(f"Error during import or placement of '{name}': {e}")
        objects_after_error = set(bpy.context.scene.objects); failed_objects = list(objects_after_error - objects_before)
        if failed_objects:
             print(f"  Cleaning up {len(failed_objects)} objects from failed human import.")
             if bpy.ops.object.select_all.poll(): bpy.ops.object.select_all(action='DESELECT')
             for obj in failed_objects:
                  if obj.name in bpy.data.objects: obj.select_set(True)
             if bpy.context.selected_objects: bpy.ops.object.delete(use_global=False)
        return None
# --- END: ADDED HUMAN IMPORT FUNCTION ---

# --- Vegetation Helpers ------------------------------------------------------

def ensure_collection(name: str):
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll

def create_grass_tuft(name: str, base_location, tuft_cfg):
    """
    Create a low-poly grass tuft as a single mesh using bmesh (no add-operators).
    Geometry is built in LOCAL space around the origin, then the object is moved
    to base_location in WORLD space. This avoids alignment/parenting quirks.
    """
    import bpy, bmesh, random, math
    from mathutils import Matrix, Vector

    try:
        # --- config ---
        min_blades = int(tuft_cfg.get('min_blades', 5))
        max_blades = int(tuft_cfg.get('max_blades', 9))
        n = max(1, random.randint(min_blades, max_blades))

        h0, h1 = tuft_cfg.get('blade_height_range', (0.08, 0.18))
        r0, r1 = tuft_cfg.get('blade_radius_range', (0.002, 0.004))
        s0, s1 = tuft_cfg.get('tuft_scale_range', (0.7, 1.3))
        tuft_scale = random.uniform(s0, s1)

        grass_color = tuft_cfg.get('grass_color', (0.08, 0.45, 0.08, 1))
        mat = get_material("GrassMat", grass_color)

        # --- build mesh in local space (around origin) ---
        bm = bmesh.new()

        for _ in range(n):
            height = random.uniform(h0, h1)
            base_r = random.uniform(r0, r1)

            # slight local jitter so blades don't coincide
            jx = random.uniform(-base_r * 2.0, base_r * 2.0)
            jy = random.uniform(-base_r * 2.0, base_r * 2.0)

            # base triangle (around jitter pivot), tip is a small triangle near the top
            eps = 0.08  # how narrow the top triangle is (0..1)
            angs = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
            base = [Vector((math.cos(a) * base_r + jx, math.sin(a) * base_r + jy, 0.0)) for a in angs]
            top  = [Vector((math.cos(a) * base_r * eps + jx, math.sin(a) * base_r * eps + jy, height)) for a in angs]

            v0 = bm.verts.new(base[0]); v1 = bm.verts.new(base[1]); v2 = bm.verts.new(base[2])
            v3 = bm.verts.new(top[0]);  v4 = bm.verts.new(top[1]);  v5 = bm.verts.new(top[2])

            # sides (as quads) + caps
            bm.faces.new([v0, v1, v4, v3])
            bm.faces.new([v1, v2, v5, v4])
            bm.faces.new([v2, v0, v3, v5])
            bm.faces.new([v0, v2, v1])       # bottom cap
            bm.faces.new([v3, v4, v5])       # near-tip cap

            # random tilt & twist around the local jitter pivot
            tilt_x = random.uniform(-0.5, 0.5)
            tilt_y = random.uniform(-0.5, 0.5)
            twist_z = random.uniform(0.0, 2.0 * math.pi)

            pivot = Vector((jx, jy, 0.0))
            M = (Matrix.Translation(pivot) @
                 Matrix.Rotation(twist_z, 4, 'Z') @
                 Matrix.Rotation(tilt_x, 4, 'X') @
                 Matrix.Rotation(tilt_y, 4, 'Y') @
                 Matrix.Translation(-pivot))

            for v in (v0, v1, v2, v3, v4, v5):
                v.co = M @ v.co

        # uniform scale of the whole tuft geometry (still local)
        if abs(tuft_scale - 1.0) > 1e-6:
            S = Matrix.Scale(tuft_scale, 4)
            for v in bm.verts:
                v.co = S @ v.co

        # finalize mesh datablock
        mesh = bpy.data.meshes.new(name)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        # create object, link, material, shading
        obj = bpy.data.objects.new(name, mesh)
        # link into the active scene collection for now; caller can re-link to a custom collection
        if bpy.context.scene.collection not in obj.users_collection:
            bpy.context.scene.collection.objects.link(obj)

        if mat:
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)

        # smooth shading
        for p in obj.data.polygons:
            p.use_smooth = True

        # place object in WORLD at the requested spot
        obj.location = Vector(base_location)

        return obj

    except Exception as e:
        print(f"[Vegetation] Grass tuft error: {e}")
        return None



def scatter_vegetation_on_median(median_obj, median_width: float, street_length: float, surface_z: float, cfg: dict):
    """Scatter grass and shrubs on the median top surface."""
    if not cfg or not cfg.get('present', False): return
    if median_obj is None or median_width <= 0 or street_length <= 0: return

    # Deterministic randomness if seed provided
    seed = cfg.get('seed', 0)
    if seed is not None: random.seed(seed)

    density = max(0.0, cfg.get('density_per_m2', 1.0))
    area = median_width * street_length
    target = min(int(area * density), int(cfg.get('max_instances', 800)))
    if target <= 0:
        print("[Vegetation] Target=0; nothing to place.")
        return

    margin = cfg.get('edge_margin', 0.05)
    xmin = -median_width * 0.5 + margin
    xmax =  median_width * 0.5 - margin
    ymin = -street_length * 0.5 + margin
    ymax =  street_length * 0.5 - margin

    min_d2 = cfg.get('min_distance', 0.1) ** 2
    placed_xy = []

    grass_cfg = cfg.get('grass', {})
    shrub_cfg = cfg.get('shrubs', {})
    grass_on = grass_cfg.get('enabled', True)
    shrub_on = shrub_cfg.get('enabled', True)
    shrub_p = shrub_cfg.get('shrub_probability', 0.25)

    coll = ensure_collection("Vegetation")
    parent = bpy.data.objects.get("MedianVegetation")
    if parent is None:
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, surface_z))
        parent = bpy.context.object
        parent.name = "MedianVegetation"

    count = 0
    tries = 0
    max_tries = target * 20

    while count < target and tries < max_tries:
        tries += 1
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)

        # Simple rejection sampling for spacing
        if any((x - px) ** 2 + (y - py) ** 2 < min_d2 for (px, py) in placed_xy):
            continue

        placed_xy.append((x, y))
        z_nudge = float(grass_cfg.get('z_offset_m', -0.2))  # ~2 cm sink by default
        loc = (x, y, surface_z + z_nudge)

        place_shrub = shrub_on and (not grass_on or random.random() < shrub_p)
        obj = None

        if place_shrub:
            try:
                rad0, rad1 = shrub_cfg.get('radius_range', (0.15, 0.35))
                subs = int(shrub_cfg.get('subdivisions', 2))
                bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subs, radius=random.uniform(rad0, rad1), location=loc)
                obj = bpy.context.object
                obj.name = f"Shrub_{count+1}"
                shrub_color = vary_color_rgba(
                    shrub_cfg.get('color', (0.06, 0.35, 0.06, 1)),
                    hue_jitter=float(shrub_cfg.get('color_hue_jitter', 0.025)),
                    sat_jitter=float(shrub_cfg.get('color_sat_jitter', 0.10)),
                    val_jitter=float(shrub_cfg.get('color_val_jitter', 0.12)),
                )
                mat = get_material(f"{obj.name}_Mat", shrub_color)
                if obj.data.materials: obj.data.materials[0] = mat
                else: obj.data.materials.append(mat)
            except Exception as e:
                print(f"[Vegetation] Shrub error: {e}")
        else:
            obj = create_grass_tuft("GrassTuft", loc, grass_cfg)

        if obj:
            obj.rotation_euler[2] = random.uniform(0.0, 6.28318)
            # Move to Vegetation collection and parent for tidy outliner
            try:
                if obj.name not in coll.objects:
                    try:
                        bpy.context.scene.collection.objects.unlink(obj)
                    except Exception:
                        pass
                    coll.objects.link(obj)
            except Exception:
                pass
            obj.parent = parent
            count += 1

    print(f"[Vegetation] Placed {count} items on median (target {target}).")

# --- Drainage helpers --------------------------------------------------------

def ensure_collection(name: str):
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll

def _road_top_z():
    # Slightly above road to avoid z-fighting; adjust if your road spec differs
    return (DEFAULT_ROAD_THICKNESS * 0.5) + 0.0005

def _side_params(street_data):
    """Return geometry per side: median offset, widths, and the curb-edge X for gutter."""
    L = street_data['length']
    mw = float(street_data.get('median', {}).get('width', 0.0) or 0.0)
    med_off = mw * 0.5

    left = street_data['sides']['left']
    right = street_data['sides']['right']
    lw = float(left.get('driveway', {}).get('width', 0.0) or 0.0)
    rw = float(right.get('driveway', {}).get('width', 0.0) or 0.0)

    # Curb edge = boundary between driveway and bikepath/footpath
    curb_x_L = -med_off - lw
    curb_x_R =  med_off + rw
    return L, med_off, mw, lw, rw, curb_x_L, curb_x_R

def create_grate_mesh(name, width_x, length_y, thickness, bar_thickness, bar_gap, color_rgba):
    """Operator-free curb grate made from repeated bar boxes (bmesh)."""
    import bpy, bmesh

    mat = get_material("GrateMat", color_rgba)

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    x0 = -width_x / 2.0
    x1 =  width_x / 2.0
    y0 = -length_y / 2.0
    y1 =  length_y / 2.0
    z0 = -thickness / 2.0
    z1 =  thickness / 2.0

    x = x0
    # Build bars centered at origin, each as a thin rectangular prism.
    while x < x1 - 1e-9:
        w = min(bar_thickness, x1 - x)

        # Create this bar's 8 verts and remember them locally
        verts = [
            bm.verts.new((x,   y0, z0)),  # 0
            bm.verts.new((x+w, y0, z0)),  # 1
            bm.verts.new((x+w, y1, z0)),  # 2
            bm.verts.new((x,   y1, z0)),  # 3
            bm.verts.new((x,   y0, z1)),  # 4
            bm.verts.new((x+w, y0, z1)),  # 5
            bm.verts.new((x+w, y1, z1)),  # 6
            bm.verts.new((x,   y1, z1)),  # 7
        ]

        # Faces using the local verts list (avoids global index lookups)
        faces = [
            (0, 1, 2, 3),  # bottom
            (4, 5, 6, 7),  # top
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
        for f in faces:
            bm.faces.new([verts[i] for i in f])

        x += (bar_thickness + bar_gap)

    # Finalize mesh
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    # Material
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # Link to scene; the caller can also link it to a "Drainage" collection
    bpy.context.scene.collection.objects.link(obj)

    # Smooth shading (optional)
    for p in obj.data.polygons:
        p.use_smooth = True

    return obj


def add_curb_inlets(street_data):
    cfg = street_data.get('drainage', {})
    if not cfg or not cfg.get('present', False):
        return

    L, med_off, mw, lw, rw, curb_x_L, curb_x_R = _side_params(street_data)
    if lw <= 0 and rw <= 0:
        print("[Drainage] No driveways; skipping.")
        return

    sides_setting = (cfg.get('sides') or 'both').lower()
    do_left  = sides_setting in ('left', 'both') and lw > 0
    do_right = sides_setting in ('right', 'both') and rw > 0

    coll = ensure_collection("Drainage")

    # --- 3.a Gutter strip (visual lip) ---
    gut = cfg.get('gutter_strip', {})
    if gut.get('enabled', True):
        gW = float(gut.get('width', 0.28))
        gT = float(gut.get('thickness', 0.012))
        gLower = float(gut.get('lowering', 0.008))
        gCol = gut.get('color', (0.07,0.07,0.07,1))

        road_top = _road_top_z()
        top = road_top - gLower
        cz = top - gT/2.0

        if do_left:
            cx = curb_x_L + (gW * 0.5)   # into driveway (toward center)
            obj = create_slab(gW, L, gT, (cx, 0.0, cz), "GutterLeft", gCol)
            coll.objects.link(obj)
        if do_right:
            cx = curb_x_R - (gW * 0.5)
            obj = create_slab(gW, L, gT, (cx, 0.0, cz), "GutterRight", gCol)
            coll.objects.link(obj)

    # --- 3.b Inlets at intervals ---
    inlet = cfg.get('inlet', {})
    if not inlet.get('enabled', True):
        return

    gl = float(inlet.get('grate_length', 1.0))
    gw = float(inlet.get('grate_width', 0.30))
    gt = float(inlet.get('grate_thickness', 0.02))
    bt = float(inlet.get('bar_thickness', 0.02))
    bg = float(inlet.get('bar_gap', 0.03))
    rdepth = float(inlet.get('recess_depth', 0.35))
    rheight = float(inlet.get('recess_height', 0.25))
    col_grate = inlet.get('color_grate', (0.10,0.10,0.10,1))
    col_recess = inlet.get('color_recess', (0.02,0.02,0.02,1))
    zoff = float(inlet.get('z_offset', 0.003))

    spacing = max(4.0, float(cfg.get('spacing', 24.0)))
    y_margin = max(0.0, float(cfg.get('offset_from_ends', 6.0)))

    road_top = _road_top_z()

    # iterate Y positions
    y = -L * 0.5 + y_margin
    idx = 0
    while y <= (L * 0.5 - y_margin):
        # left side
        if do_left:
            # recess “hole” tucked under sidewalk
            rcx = curb_x_L + 0.12                      # slightly under the curb face
            rcz = road_top - (rheight / 2.0) - 0.02    # a bit below road surface
            recess = create_slab(rdepth, gl, rheight, (rcx, y, rcz), f"InletRecessL_{idx}", col_recess)
            coll.objects.link(recess)

            # grate centered in gutter, oriented along Y
            gcx = curb_x_L + (gw * 0.5)
            gcz = road_top - gt/2.0 + zoff
            grate = create_grate_mesh(f"CurbGrateL_{idx}", gw, gl, gt, bt, bg, col_grate)
            grate.location = (gcx, y, gcz)
            grate.rotation_euler = (0.0, 0.0, 0.0)
            if grate.name not in coll.objects:
                try: bpy.context.scene.collection.objects.unlink(grate)
                except Exception: pass
                coll.objects.link(grate)

        # right side
        if do_right:
            rcx = curb_x_R - 0.12
            rcz = road_top - (rheight / 2.0) - 0.02
            recess = create_slab(rdepth, gl, rheight, (rcx, y, rcz), f"InletRecessR_{idx}", col_recess)
            coll.objects.link(recess)

            gcx = curb_x_R - (gw * 0.5)
            gcz = road_top - gt/2.0 + zoff
            grate = create_grate_mesh(f"CurbGrateR_{idx}", gw, gl, gt, bt, bg, col_grate)
            grate.location = (gcx, y, gcz)
            grate.rotation_euler = (0.0, 0.0, 0.0)
            if grate.name not in coll.objects:
                try: bpy.context.scene.collection.objects.unlink(grate)
                except Exception: pass
                coll.objects.link(grate)

        idx += 1
        y += spacing

    print(f"[Drainage] Added {idx} inlet rows; sides L:{do_left} R:{do_right}.")

def ensure_collection(name: str):
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll

def _road_top_z():
    # top of road surface; adjust if your road placement differs
    return DEFAULT_ROAD_THICKNESS * 0.5

def _side_params(street_data):
    """Return handy geometry: L, median halfwidth, widths, and curb-edge X positions."""
    L = float(street_data['length'])
    mw = float(street_data.get('median', {}).get('width', 0.0) or 0.0)
    med_off = mw * 0.5

    left  = street_data['sides']['left']
    right = street_data['sides']['right']
    lw = float(left.get('driveway', {}).get('width', 0.0) or 0.0)
    rw = float(right.get('driveway', {}).get('width', 0.0) or 0.0)

    # curb-edge between driveway and bikepath (driveway outer edge)
    curb_x_L = -med_off - lw
    curb_x_R =  med_off + rw
    return L, med_off, mw, lw, rw, curb_x_L, curb_x_R

def add_slot_drain(street_data):
    cfg = street_data.get('slot_drain', {})
    if not cfg or not cfg.get('present', False):
        return

    L, med_off, mw, lw, rw, curb_x_L, curb_x_R = _side_params(street_data)
    if lw <= 0 and rw <= 0:
        print("[SlotDrain] No driveways; skipping.")
        return

    sides = (cfg.get('sides') or 'both').lower()
    do_left  = sides in ('left', 'both') and lw > 0
    do_right = sides in ('right', 'both') and rw > 0

    slit_w    = float(cfg.get('width', 0.14))
    open_t    = float(cfg.get('opening_thickness', 0.008))
    lowering  = float(cfg.get('lowering', 0.004))
    recess_h  = float(cfg.get('recess_height', 0.30))
    recess_in = float(cfg.get('recess_inset_under_bike', 0.06))  # extends under bikepath

    col_open  = cfg.get('color_opening', (0.05, 0.05, 0.05, 1))
    col_recess= cfg.get('color_recess',  (0.02, 0.02, 0.02, 1))

    road_top  = _road_top_z()

    coll = ensure_collection("Drainage")  # reuse your drainage collection

    # Each side: place a narrow opening at curb level, plus a deeper recess just below it.
    def _side_place(curb_x, side: str):
        # opening centered between driveway and bikepath
        # For LEFT: slit is outside (more negative) of curb edge; for RIGHT: more positive.
        sign = -1.0 if side == 'left' else 1.0
        open_cx = curb_x + sign * (slit_w * 0.5)

        # Z placement: opening is a thin strip slightly below road top (to catch highlights)
        open_top = road_top - lowering
        open_cz  = open_top - open_t * 0.5

        opening = create_slab(
            slit_w, L, open_t,
            (open_cx, 0.0, open_cz),
            f"SlotOpen_{side.capitalize()}",
            col_open
        )
        if opening and opening.name not in coll.objects:
            try: bpy.context.scene.collection.objects.unlink(opening)
            except Exception: pass
            coll.objects.link(opening)

        # recess (a hidden “void” under the opening, slightly wider and tucked under bikepath)
        recess_w = slit_w + recess_in
        recess_cx = curb_x + sign * (slit_w * 0.5 + recess_in * 0.5)
        recess_cz = (road_top - 0.02) - recess_h * 0.5   # a bit below road

        recess = create_slab(
            recess_w, L, recess_h,
            (recess_cx, 0.0, recess_cz),
            f"SlotRecess_{side.capitalize()}",
            col_recess
        )
        if recess and recess.name not in coll.objects:
            try: bpy.context.scene.collection.objects.unlink(recess)
            except Exception: pass
            coll.objects.link(recess)

    if do_left:  _side_place(curb_x_L, 'left')
    if do_right: _side_place(curb_x_R, 'right')

    print(f"[SlotDrain] Added slot drain: sides L:{do_left} R:{do_right}, width={slit_w:.3f}m.")

def _road_top_z():
    return DEFAULT_ROAD_THICKNESS * 0.5

def _paint_strip(width_x, length_y, center_xy, offset_z, color, name):
    paint_thickness = 0.002
    cx, cy = center_xy
    cz = _road_top_z() - paint_thickness * 0.5 + (offset_z or 0.0)
    return create_slab(width_x, length_y, paint_thickness, (cx, cy, cz), name, color)

def add_lane_markings_by_width(street_data):
    cfg = street_data.get('lane_markings', {})
    if not cfg or not cfg.get('present', False):
        return

    L = float(street_data['length'])
    mw = float(street_data.get('median', {}).get('width', 0.0) or 0.0)
    med_off = mw * 0.5
    lane_w = float(cfg.get('lane_width_m', 3.5))

    div = cfg.get('divider', {})
    div_on  = bool(div.get('enabled', True))
    dash    = float(div.get('dash_len', 6.0))
    gap     = float(div.get('gap_len', 6.0))
    line_w  = float(div.get('line_width', 0.12))
    oz      = float(div.get('offset_z', 0.012))
    color   = div.get('color', (1,1,1,1))

    edge = cfg.get('edge_lines', {})
    edge_on  = bool(edge.get('enabled', False))
    edge_w   = float(edge.get('width', 0.10))
    edge_oz  = float(edge.get('offset_z', 0.011))
    edge_col = edge.get('color', (1,1,1,1))

    def _dash_along_x(x_fixed, base_name):
        if not div_on or dash <= 0.0:
            return
        y = -L * 0.5
        i = 0
        while y < L * 0.5:
            seg = min(dash, L * 0.5 - y)
            _paint_strip(line_w, seg, (x_fixed, y + seg * 0.5), oz, color, f"{base_name}_{i}")
            y += dash + gap
            i += 1

    def _mark_side(width, inner_x, sign, tag):
        """inner_x: X at the median edge of this driveway. sign=-1 for left, +1 for right."""
        if width <= 0.0 or lane_w <= 0.0:
            return
        n_lanes = int(width // lane_w)  # whole lanes that fit
        if n_lanes < 2:
            return
        # draw a divider every lane_w from inner edge (don’t draw at the outer edge)
        for k in range(1, n_lanes):
            x = inner_x + sign * (k * lane_w)
            _dash_along_x(x, f"{tag}_Div{k}")

        if edge_on:
            inner_edge_x = inner_x
            outer_edge_x = inner_x + sign * width
            # nudge edge lines slightly into the driveway
            _paint_strip(edge_w, L, (inner_edge_x + sign * ( edge_w * 0.5), 0.0), edge_oz, edge_col, f"{tag}_EdgeInner")
            _paint_strip(edge_w, L, (outer_edge_x - sign * ( edge_w * 0.5), 0.0), edge_oz, edge_col, f"{tag}_EdgeOuter")

    # Left side (negative X)
    left_w = float(street_data['sides']['left'].get('driveway', {}).get('width', 0.0) or 0.0)
    _mark_side(left_w, -med_off, -1.0, "L")

    # Right side (positive X)
    right_w = float(street_data['sides']['right'].get('driveway', {}).get('width', 0.0) or 0.0)
    _mark_side(right_w,  med_off, +1.0, "R")

    print(f"[Markings] Lanes drawn where >= {lane_w*2:.2f} m; lane width={lane_w:.2f} m.")


def _road_top_z():
    return DEFAULT_ROAD_THICKNESS * 0.5

def _curb_top_z(elem_thickness, z_fudge=0.005):
    # Matches your curb/sidewalk anchoring: keeps top fixed
    return _road_top_z() + (elem_thickness - DEFAULT_ROAD_THICKNESS) + z_fudge

def _side_widths(street_data):
    L = float(street_data['length'])
    mw = float(street_data.get('median', {}).get('width', 0.0) or 0.0); med_off = mw * 0.5

    def side_dict(letter):
        Sd = street_data['sides']['left' if letter == 'L' else 'right']
        w_drive = float(Sd.get('driveway', {}).get('width', 0.0) or 0.0)
        w_bike  = float(Sd.get('bikepath', {}).get('width', 0.0) or 0.0)
        w_foot  = float(Sd.get('footpath', {}).get('width', 0.0) or 0.0)
        # NEW: runtime parking width
        w_park  = float(street_data.get('_runtime', {}).get('left' if letter == 'L' else 'right', {}).get('parking_width', 0.0) or 0.0)
        inner_x = (-med_off) if letter == 'L' else (+med_off)
        return {'drive': w_drive, 'park': w_park, 'bike': w_bike, 'foot': w_foot, 'inner_x': inner_x}

    return {'L': side_dict('L'), 'R': side_dict('R'), 'L_sign': -1.0, 'R_sign': +1.0, 'LENGTH': L}


def _footpath_top_for_side(street_data, side_letter):
    th = float(street_data['sides']['left' if side_letter=='L' else 'right'].get('footpath', {}).get('thickness', DEFAULT_CURB_THICKNESS))
    return _curb_top_z(th, z_fudge=0.005)

def _crosswalk_y_positions(street_data):
    cw = street_data.get('crosswalks', {})
    return [float(y) for y in (cw.get('y_positions') or [])] if cw and cw.get('present', False) else []

def create_utility_box(name, w, d, h, base_xy, top_z, body_color, plinth_h=0.05, plinth_color=(0.18,0.18,0.18,1)):
    """Main box + plinth (both slabs). top_z is the *top level* the box should reach."""
    # body
    cz_body = (top_z + h * 0.2)
    box = create_slab(w, d, h, (base_xy[0], base_xy[1], cz_body), name, body_color)

    # plinth (slightly below body base)
    if plinth_h > 0:
        base_z = cz_body + (h * 0.5)
        cz_plinth = base_z + plinth_h * 0.5
        plinth = create_slab(w * 1.02, d * 1.02, plinth_h, (base_xy[0], base_xy[1], cz_plinth), f"{name}_Base", plinth_color)
    return box

def add_utility_boxes(street_data):
    cfg = street_data.get('utility_boxes', {})
    if not cfg or not cfg.get('present', False):
        return

    # deterministic randomness if desired
    seed = cfg.get('seed', 0)
    if seed is not None:
        random.seed(seed)

    widths = _side_widths(street_data)
    L = widths['LENGTH']
    cw_y = _crosswalk_y_positions(street_data)
    y_clear = float(cfg.get('min_clear_y_from_crosswalk', 2.0))

    sides = (cfg.get('sides') or 'both').lower()
    use_left  = sides in ('left', 'both')
    use_right = sides in ('right', 'both')

    # Try to find actual lamp objects (collection named 'Lamps' or name contains 'lamp')
    lamp_objs = []
    lamps_coll = bpy.data.collections.get("Lamps")
    if lamps_coll:
        lamp_objs.extend([o for o in lamps_coll.objects])
    else:
        for o in bpy.data.objects:
            n = (o.name or "").lower()
            if 'lamp' in n or 'streetlamp' in n or 'lightpole' in n:
                lamp_objs.append(o)

    def _allowed_by_side(x):
        return (use_left and x < 0) or (use_right and x > 0)

    # Build a candidate list of anchor points (prefer real lamps)
    anchors = []
    if lamp_objs and (cfg.get('mode','near_lamps').lower() == 'near_lamps'):
        for o in lamp_objs:
            x, y, _ = o.matrix_world.translation
            if _allowed_by_side(x):
                anchors.append(('lamp', (x, y)))
    else:
        # Fallback: synthesize anchors from lamp spacing (if provided) or density
        spacing = float(street_data.get('lamps', {}).get('spacing', 25.0))
        y = -L * 0.5 + spacing * 0.5
        while y <= L * 0.5 - spacing * 0.25:
            if use_left:  anchors.append(('left',  (-0.1, y)))   # x is placeholder, we’ll compute later
            if use_right: anchors.append(('right', (+0.1, y)))
            y += spacing

    prob = float(cfg.get('probability_per_lamp', 0.6))
    w0, w1 = cfg.get('size', {}).get('w_range', (0.40, 0.55))
    d0, d1 = cfg.get('size', {}).get('d_range', (0.22, 0.35))
    h0, h1 = cfg.get('size', {}).get('h_range', (1.00, 1.40))
    plinth_h = float(cfg.get('size', {}).get('plinth_h', 0.05))

    off_x     = float(cfg.get('offset_from_lamp_x', 0.60))
    jit_ymax  = float(cfg.get('offset_from_lamp_y_jitter', 0.40))
    into_fp   = float(cfg.get('offset_into_footpath', 0.30))
    col_body  = cfg.get('color_body',   (0.55,0.58,0.55,1))
    col_plinth= cfg.get('color_plinth', (0.18,0.18,0.18,1))

    coll = ensure_collection("StreetUtilities")

    count = 0
    for kind, (ax, ay) in anchors:
        # skip some at random unless we’re in random mode with explicit density
        if kind == 'lamp' and random.random() > prob:
            continue

        # avoid crosswalk vicinity
        if any(abs(ay - cy) < y_clear for cy in cw_y):
            continue

        # determine side sign and surface Z (footpath top feels right for cabinets)
        side = 'L' if (kind == 'left' or ax < 0) else 'R'
        sign = widths[f"{side}_sign"]
        fp_top = _footpath_top_for_side(street_data, side)

        # compute X/Y placement
        if kind == 'lamp':
            # push outward from lamp toward footpath/building line
            x = ax + sign * off_x
            y = ay + random.uniform(-jit_ymax, jit_ymax)
        else:
            # build from section: inner_x + drive + bike + into_footpath
            sd = widths[side]
            x = sd['inner_x'] + sign * (sd['drive'] + sd.get('park', 0.0) + sd['bike'] + into_fp)
            y = ay + random.uniform(-jit_ymax, jit_ymax)

        # random size
        W = random.uniform(w0, w1)
        D = random.uniform(d0, d1)
        H = random.uniform(h0, h1)

        # top-anchored to the footpath top (box grows down); cabinet top ≈ curb level
        top_z = fp_top
        obj = create_utility_box(f"UtilityBox_{side}_{count}", W, D, H, (x, y), top_z, col_body, plinth_h, col_plinth)

        # move to collection (tidy)
        if obj and obj.name not in coll.objects:
            try: bpy.context.scene.collection.objects.unlink(obj)
            except Exception: pass
            coll.objects.link(obj)

        count += 1

    print(f"[Utilities] Placed {count} utility boxes (mode={cfg.get('mode','near_lamps')}, anchors={len(anchors)}).")

def add_median_trees(street_data, median_width, street_length, surface_z):
    """
    Minimal median tree placer (safe if median_width == 0).
    - Reuses street_data['trees'] look; optional street_data['median_trees']['overrides'] tweak it.
    - Single center row by default; 'rows':'double' gives two rows near edges.
    - No crosswalk logic; uses a local RNG; 'count' mode is random with min spacing.
    """
    import bpy, random

    # ---- robust numeric guards ----
    mw = float(median_width or 0.0)
    L  = float(street_length or 0.0)
    if mw <= 0.0 or L <= 0.0:
        # No median or no length → do nothing
        print("[MedianTrees] Skipping (no median).")
        return

    mt = dict(street_data.get('median_trees', {}) or {})
    if not mt.get('present', False):
        return

    # Local RNG so other seeds don’t affect this
    seed = mt.get('seed', 0)
    rng = random.Random(seed) if seed is not None else random.Random()

    # Base look + optional overrides
    base_tree_cfg = dict(street_data.get('trees', {}) or {})
    overrides = dict(mt.get('overrides', {}) or {})
    tree_cfg = {**base_tree_cfg, **overrides}

    # Jitter / spacing
    y_margin = float(mt.get('y_end_margin_m', 4.0))
    x_margin = float(mt.get('margin_x_m', 0.35))
    x_jitter = float(mt.get('x_jitter_m', 0.12))
    y_jitter = float(mt.get('y_jitter_m', 0.50))
    min_sp   = float(mt.get('min_spacing_m', 6.0))

    # Row layout
    rows = (mt.get('rows') or 'single').lower()
    xmin = -mw * 0.5 + x_margin
    xmax =  mw * 0.5 - x_margin
    if xmin > xmax:  # too narrow → center
        xmin = xmax = 0.0
    row_xs = [xmin, xmax] if rows == 'double' else [0.0]

    # Collection
    coll = bpy.data.collections.get("MedianTrees")
    if not coll:
        coll = bpy.data.collections.new("MedianTrees")
        bpy.context.scene.collection.children.link(coll)

    # Y domain
    ylo = -L * 0.5 + y_margin
    yhi =  L * 0.5 - y_margin

    # Y positions: 'count' = random with spacing, 'spacing' = stepped with jitter
    mode = (mt.get('mode') or 'count').lower()
    y_positions = []
    if mode == 'count':
        n = max(1, int(mt.get('count', max(1, base_tree_cfg.get('count', 4)))))
        tries = 0
        max_tries = max(100, n * 50)
        while len(y_positions) < n and tries < max_tries:
            tries += 1
            y = rng.uniform(ylo, yhi)
            if all(abs(y - py) >= min_sp for py in y_positions):
                y_positions.append(y)
        y_positions.sort()
    else:  # 'spacing'
        spacing = max(2.0, float(mt.get('spacing_m', 12.0)))
        y = ylo
        while y <= yhi + 1e-6:
            y_positions.append(y + rng.uniform(-0.3 * spacing, 0.3 * spacing))
            y += spacing

    # Place trees
    placed = 0
    for i, y in enumerate(y_positions):
        x_base = row_xs[i % len(row_xs)]
        x = x_base + (rng.uniform(-x_jitter, x_jitter) if x_jitter > 0 else 0.0)
        z = surface_z + float(tree_cfg.get('base_z_offset', 0.0))
        loc = (x, y, z)

        tree_obj = None
        try:
            tree_obj = create_tree(loc, tree_cfg, 'median', i)
        except TypeError:
            try:
                tree_obj = create_tree(loc)
            except Exception:
                tree_obj = None

        if tree_obj:
            try:
                if tree_obj.name not in coll.objects:
                    try:
                        bpy.context.scene.collection.objects.unlink(tree_obj)
                    except Exception:
                        pass
                    coll.objects.link(tree_obj)
            except Exception:
                pass
            placed += 1

    print(f"[MedianTrees] Placed {placed} tree(s) on median.")


# ============================================================================
# === MAIN EXECUTION =========================================================
# ============================================================================

# --- Street Configuration Data ---
street_data = {
    'length': 70.0,
    'median': {'width': 1,'present': True,'material': (0.4,0.4,0.4,1),'thickness': DEFAULT_CURB_THICKNESS},
    'sides': { # Side configurations... (unchanged)
        'left': {'driveway': {'width': 10.0,'present': True,'material': (0.25,0.25,0.25,1),'thickness': 0.15},
        'bikepath': {'width': 2.0,'present': True,'material': (0.0,0.4,0.1,1),'thickness': 0.20},
        'trees': {'present': True,'count': 1,'min_height': 2,'max_height': 6,'trunk_radius': 0.35,
        'branch_levels': 4,'branch_count': 3,'branch_angle_deg': 35,'branch_length_decay': 0.75,
        'branch_radius_decay': 0.60,'leaf_clusters_per_tip': 5,'leaf_size_factor': 0.60,
        'leaf_subdivisions': 2,'bark_material_color': (0.4,0.2,0.1,1),'leaf_material_color': (0.05,0.4,0.05,1),
        'leaf_color_hue_jitter': 0.02,'leaf_color_sat_jitter': 0.08,'leaf_color_val_jitter': 0.10,'base_z_offset': 0.02 },
        'footpath': {'width': 5,'present': True,'material': (0.7,0.7,0.65,1),'thickness': 0.20},
        'buildings': {'present': True,'count': 4,'height': 10.0,'min_height': 8.0,'max_height': 12.0,'depth': 11.0,'width': 10.0,'gap': 2.5,'setback': 0.0,'roof_type': 'random',
        'roof_type_choices': ('gabled', 'flat'),'roof_height': 3.5,'min_roof_height': 1.5,'max_roof_height': 3.5,'flat_roof_height': 0.2,'material': (0.8,0.75,0.7,1),'roof_material': (0.6,0.2,0.1,1) }},
        
        'right': {'driveway': {'width': 7,'present': True,'material': (0.25,0.25,0.25,1),'thickness': 0.15},
        'bikepath': {'width': 2.0,'present': True,'material': (0.0,0.4,0.1,1),'thickness': 0.20},
        'trees': {'present': True,'count': 1,'min_height': 2.0,'max_height': 6.0,'trunk_radius': 0.3,
        'branch_levels': 4,'branch_count': 3,'branch_angle_deg': 32,'branch_length_decay': 0.7,
        'branch_radius_decay': 0.65,'leaf_clusters_per_tip': 5,'leaf_size_factor': 0.7,
        'leaf_subdivisions': 2,'bark_material_color': (0.35,0.18,0.12,1),'leaf_material_color': (0.1,0.5,0.15,1),
        'leaf_color_hue_jitter': 0.02,'leaf_color_sat_jitter': 0.08,'leaf_color_val_jitter': 0.10,'base_z_offset': 0.01 },
        'footpath': {'width': 5,'present': True,'material': (0.7,0.7,0.65,1),'thickness': 0.20},
        'buildings': {'present': True,'count': 3,'height': 8,'min_height': 6.0,'max_height': 8.0,'depth': 8.0,'width': 8.0,'gap': 1.8,'setback': 0.0,'roof_type': 'random',
        'roof_type_choices': ('gabled', 'flat'),'roof_height': 2.5,'min_roof_height': 1.5,'max_roof_height': 2.5,'flat_roof_height': 0.2,'material': (0.7,0.75,0.85,1),'roof_material': (0.3,0.35,0.4,1) }}
    },
    'cars': {
        'present': True, 'count': 3, 'scale': 10.0,
        'model_paths': [
            script_path('cars', '1.glb'),
            script_path('cars', '2.glb'),
            script_path('cars', '3.glb'),
            script_path('cars', '4.glb'),
            script_path('cars', '5.glb'),
            script_path('cars', '6.glb'),
            script_path('cars', '7.glb'),
            script_path('cars', '8.glb'),
        ],
            # NEW:
        'rotation_by_side_deg': {'left': 0, 'right': 180.0},  # override if needed
        'traffic': 'RHT',            # or 'LHT' to swap sensible defaults
        'yaw_jitter_deg': 0.0,        # small random ±jitter if you want variation
        'global_rotation_fix_deg': 180.0,   # NEW: apply to all car models after side-yaw

    },


    'lamps': {
        'present': True, 'scale': 7.5, 'spacing': 25.0, 'z_offset': LAMP_Z_OFFSET,
        'model_paths': [
            script_path('lamp', '1.glb'),
            script_path('lamp', '2.glb'),
            script_path('lamp', '3.glb'),
            script_path('lamp', '4.glb'),
        ]
    },
    # --- START: ADDED HUMAN CONFIGURATION ---
    'humans': {
        'present': True,
        'count': 5,
        'scale': 5.0,
        'z_offset': HUMAN_Z_OFFSET,
        'model_paths': [
            script_path('people', '1.glb'),
            script_path('people', '2.glb'),
            script_path('people', '3.glb'),
            script_path('people', '4.glb'),
            script_path('people', '5.glb')
        ]
    },
    # --- END: ADDED HUMAN CONFIGURATION ---
    
        'vegetation': {
        'present': True,                 # turn ALL vegetation on/off
        'density_per_m2': 2,           # instances per square meter (increase for denser fill)
        'seed': 0,                       # random seed (set None for non-deterministic)
        'edge_margin': 0.06,             # keep a small border from median edges (meters)
        'min_distance': 0.10,            # simple anti-clump spacing (meters)
        'max_instances': 800,            # safety cap

        'grass': {                       # grass tufts (multiple skinny cones merged)
            'enabled': True,
            'min_blades': 15,
            'max_blades': 20,
            'blade_height_range': (0.08, 1),
            'blade_radius_range': (0.002, 0.004),
            'tuft_scale_range': (0.7, 1.3),
            'grass_color': (0.08, 0.45, 0.08, 1),
        },

        'shrubs': {                      # small round shrubs (icospheres)
            'enabled': True,
            'shrub_probability': 0.2,   # probability a placed item is a shrub (else grass)
            'radius_range': (0.15, 0.35),
            'subdivisions': 2,
            'color': (0.06, 0.35, 0.06, 1),
            'color_hue_jitter': 0.025,
            'color_sat_jitter': 0.10,
            'color_val_jitter': 0.12,
        },
    },
    
    
    'drainage': {
    'present': True,
    'sides': 'both',                 # 'left', 'right', or 'both'
    'spacing': 24.0,                 # meters between inlets
    'offset_from_ends': 6.0,         # keep away from the ends

    # Gutter strip (a narrow darker band at curb edge)
    'gutter_strip': {
        'enabled': True,
        'width': 0.22,               # ~28 cm wide
        'thickness': 0.012,          # ~1.2 cm
        'lowering': 0.015,           # top slightly below road to catch highlights
        'color': (0.07, 0.07, 0.07, 1),
    },

    # Curb inlet (grate + dark recess behind it)
    'inlet': {
        'enabled': True,
        'grate_length': 1.0,         # along Y
        'grate_width': 0.22,         # along X (into the lane)
        'grate_thickness': 0.02,
        'bar_thickness': 0.02,
        'bar_gap': 0.03,
        'recess_depth': 0.35,        # how far under the sidewalk it looks
        'recess_height': 0.25,       # vertical size of the “hole”
        'color_grate': (0.10, 0.10, 0.10, 1),
        'color_recess': (0.02, 0.02, 0.02, 1),
        'z_offset': 0.008            # tiny lift to avoid z-fighting with road
    },
},

'lane_markings': {
    'present': True,
    'lane_width_m': 3.5,      # lane width used to compute how many fit
    'divider': {
        'enabled': True,
        'dash_len': 3.0,
        'gap_len': 6.0,
        'line_width': 0.15,
        'offset_z': 0.012,
        'color': (1, 1, 1, 1),
    },
    'edge_lines': {           # optional
        'enabled': False,
        'width': 0.10,
        'offset_z': 0.011,
        'color': (1, 1, 1, 1),
    },
},
'utility_boxes': {
    'present': True,
    'mode': 'near_lamps',          # 'near_lamps' or 'random'
    'probability_per_lamp': 0.3,   # 60% chance to place a box next to a given lamp
    'random_density_per_100m': 1.0,# used only if mode='random'
    'sides': 'both',               # 'left', 'right', 'both'

    # geometry + lookl
    'size': {
        'w_range': (0.30, 0.40),   # X width (m)
        'd_range': (1.60, 1.80),   # Y depth (m)
        'h_range': (1.00, 2.00),   # Z height (m)
        'plinth_h': 0.05,          # small base under the box
    },
    'offset_from_lamp_x': 0.00,    # shove outward from lamp post (across the street)
    'offset_from_lamp_y_jitter': 2,  # forward/back jitter along the street
    'offset_into_footpath': 0.30,  # if we can’t find lamps, place box this far onto footpath side
    'min_clear_y_from_crosswalk': 2.0,  # meters; avoid crosswalk stripes

    'color_body':   (0.55, 0.58, 0.55, 1),  # dull green/gray
    'color_plinth': (0.18, 0.18, 0.18, 1),

    'seed': 2,                     # None for non-deterministic
},

'street_parking': {
    'present': True,
    'sides': 'both',
    'min_driveway_width': 3.5,

    # NEW (highest priority if present)
    'width_m_per_side': {'left': 2.0, 'right': 3.0},   # absolute widths per side
    # Optional per-side random ranges (used only if the above key or 'width_m_*' missing)
    'width_range_per_side': {'left': (2.0, 2.4), 'right': (2.8, 3.2)},

    # LEGACY (still works; applies to both sides if per-side values not provided)
    'width_m': None,
    'width_range': (3.5, 5.0),

    'seed': 0,
    'thickness': None,
    'color': (0.22, 0.22, 0.22, 1),
},


'median_trees': {
    'present': True,          # toggle on/off
    # placement: choose ONE mode
    'mode': 'count',          # 'count' or 'spacing'
    'count': 1,               # used if mode='count'
    'spacing_m': 12.0,        # used if mode='spacing'

    # layout
    'rows': 'auto',           # 'single' | 'double' | 'auto' (auto = single if narrow, else double)
    'seed': None,                # set None for non-deterministic
    'y_end_margin_m': 4.0,    # keep away from street ends
    'margin_x_m': 0.35,       # inset from median edges
    'x_jitter_m': 0.12,       # small sideways wobble
    'y_jitter_m': 0.50,       # small along-street wobble

    # (optional) style overrides ONLY for median trees; otherwise it uses your global 'trees' settings
    'overrides': {
        # Example tweaks (delete if you want the same look as roadside trees):
        'min_height': 2,
        'max_height': 6,
        # 'trunk_radius': 0.15,
        # 'leaf_size_factor': 0.60,
        # 'bark_material_color': (0.32,0.22,0.16,1),
        # 'leaf_material_color': (0.04,0.36,0.06,1),
        # 'base_z_offset': 0.00,
    },
},

'parked_cars': {
    'present': True,
    'sides': 'both',             # 'left' | 'right' | 'both'
    'count_per_side': 5,         # change this to vary how many get parked
    'model_paths': None,         # or provide a list; falls back to cars.model_paths if None
    'scale': None,               # or override; falls back to cars.scale
    'wide_threshold_m': 2.5,     # < threshold => parallel, >= threshold => angled
    'angle_deg_if_wide': 60.0,   # nose-in angle for wide bays
    'min_spacing_m': 6.0,        # along-street spacing after adjustment
    'end_margin_m': 1.0,         # keep away from segment ends
    'x_inset_m': 0.30,           # nudges angled cars toward the curb
    'seed': 0,                   # set None for non-deterministic
}



}


# --- Optional external config -------------------------------------------------
# Allow: blender -b base.blend -P test_export_driveways_new.py -- --config /path/street_data.json --out /path/scene.blend
# ---- External config merge + safe fallbacks ---------------------------------


def _argv_get(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return sys.argv[i+1] if i+1 < len(sys.argv) else default
    return default

# keep a deep copy of script defaults to restore from if needed
_DEFAULTS = copy.deepcopy(street_data)

def _deep_merge(base, override):
    # Dicts: recursive merge
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for k, v in override.items():
            out[k] = _deep_merge(base.get(k), v)
        return out
    # Lists: if override is an empty list, keep base (prevents wiping defaults)
    if isinstance(base, list) and isinstance(override, list):
        return base if len(override) == 0 else override
    # Scalars: use override when not None, else keep base
    return override if override is not None else base

def _migrate_schema(cfg):
    cfg = dict(cfg or {})
    # Map UI median.trees -> script's median_trees (if you use that field)
    mt = ((cfg.get('median') or {}).get('trees') or {})
    if mt and 'median_trees' not in cfg:
        cfg['median_trees'] = {
            'present': bool(mt.get('present', False)),
            'count': int(mt.get('count', 0)),
        }
    # Ensure per-side containers exist (defensive)
    cfg.setdefault('sides', {})
    for side in ('left', 'right'):
        cfg['sides'].setdefault(side, {})
    return cfg

# --- External config (BOM tolerant) ---
CFG = _argv_get("--config", None)
if CFG:
    # Use OS path resolution so it doesn't depend on a .blend's // base
    cfg_path = os.path.abspath(CFG)
    if not os.path.exists(cfg_path):
        print(f"[Config] WARNING: --config path not found: {cfg_path}")
    else:
        with open(cfg_path, "r", encoding="utf-8-sig") as f:  # accepts BOM
            incoming = json.load(f)
        incoming = _migrate_schema(incoming)
        street_data = _deep_merge(street_data, incoming)

        # Restore default model_paths when UI sends [] or omits it
        for key in ("cars", "humans"):
            node = street_data.get(key, {})
            if node.get("present"):
                mp = node.get("model_paths")
                if not mp:  # None or []
                    node["model_paths"] = _DEFAULTS.get(key, {}).get("model_paths", [])
                    if node["model_paths"]:
                        print(f"[Config] Using default {key}.model_paths ({len(node['model_paths'])} entries).")
                    else:
                        print(f"[Config] WARNING: No default {key}.model_paths; spawners may skip.")
        print(f"[Config] Loaded and merged: {cfg_path}")

# -----------------------------------------------------------------------------



# --- Scene Generation ---
if __name__ == "__main__":

    # if not bpy.data.filepath:
    #     def show_error_message(self, context): self.layout.label(text="SAVE the Blender file before running!"); self.layout.label(text="Relative paths need a saved file location.")
    #     bpy.context.window_manager.popup_menu(show_error_message, title="File Not Saved", icon='ERROR')
    #     raise Exception("Blender file must be saved first.")

    model_seed = street_data.get('seed', None)
    if model_seed is not None:
        try:
            random.seed(int(model_seed))
            print(f"[Seed] Model random seed: {int(model_seed)}")
        except Exception as exc:
            print(f"[Seed] Could not apply model seed {model_seed!r}: {exc}")

    if CLEAR_SCENE: clear_scene()

    # --- Create Ground Slab ---
    ground_buffer=100; est_max_width=street_data.get('median',{}).get('width',0)+2*(5+2+3+1+15)
    ground_size_x=est_max_width+ground_buffer; ground_size_y=street_data['length']+ground_buffer*2
    ground_thickness=DEFAULT_GROUND_THICKNESS; ground_center_z=-ground_thickness/2.0
    create_slab(ground_size_x, ground_size_y, ground_thickness, (0,0,ground_center_z), "Ground", (0.3,0.45,0.25,1))

    # --- Create Median Slab ---

    median_offset = 0
    median_obj = None
    if street_data.get('median') and street_data['median'].get('present', True):
        median_config = street_data['median']
        width = median_config.get('width', 0)
        if width > 0:
            elem_thickness = median_config.get('thickness', DEFAULT_CURB_THICKNESS)
            top_z = DEFAULT_ROAD_THICKNESS * 0.5 + (elem_thickness - DEFAULT_ROAD_THICKNESS) + 0.01
            center_z = top_z - elem_thickness / 2.0

            median_obj = create_slab(
                width,
                street_data['length'],
                elem_thickness,
                (0, 0, center_z),
                "Median",
                median_config.get('material', (0.8, 0.8, 0.8, 1))
            )
            median_offset = width / 2.0

            # --- NEW: Vegetation on median --------------------------------------
            if street_data.get('vegetation', {}).get('present', False):
                scatter_vegetation_on_median(
                    median_obj,
                    width,
                    street_data['length'],
                    top_z,
                    street_data['vegetation']
                )
        else:
            print("Skipping Median: width is zero.")
    else:
        print("Skipping Median: not present.")

########### median trees ###################
    # After you create the median slab and compute top_z
    median_cfg = street_data.get('median', {}) or {}
    median_width = float(median_cfg.get('width', 0.0) or 0.0)

    if median_cfg.get('present', True) and median_width > 0.0:
        # top_z should already be computed for the median slab; pass it in:
        add_median_trees(street_data, median_width, street_data['length'], top_z)

################# ######################


    # --- Build Left and Right Sides ---
    if 'left' in street_data['sides']: build_street_side(-median_offset, street_data['sides']['left'], 'left', street_data['length'])
    if 'right' in street_data['sides']: build_street_side(median_offset, street_data['sides']['right'], 'right', street_data['length'])

    ### drainage
    add_curb_inlets(street_data)

    ### road markings
    add_lane_markings_by_width(street_data)



    # --- Place Cars ---
    # (Car placement logic - unchanged)
    # ---------------- Place Cars (per-side facing; no per-model fixes) ----------------
    if street_data.get('cars') and street_data['cars'].get('present', True):
        car_config = street_data['cars']; num_cars_to_place = car_config.get('count', 0)
        model_paths = car_config.get('model_paths', []); car_scale = car_config.get('scale', 1.0)
        street_length = street_data['length']
        car_seed = car_config.get('seed', None)
        car_rng = random.Random(car_seed) if car_seed is not None else random.Random()

        valid_model_paths = []
        print("\nValidating car model paths...")
        for path in model_paths:
            abs_path = resolve_asset_path(path)
            if os.path.exists(abs_path):
                valid_model_paths.append(path); print(f"  [OK] Found: {abs_path}")
            else:
                print(f"  [NOT FOUND] Skipping: {abs_path} (from: {path})")

        if num_cars_to_place > 0 and valid_model_paths:
            # Per-side yaw + global correction (no per-model fixes)
            traffic = (car_config.get('traffic') or 'RHT').upper()
            rot_map_deg = car_config.get('rotation_by_side_deg')
            if not isinstance(rot_map_deg, dict):
                traffic = (car_config.get('traffic') or 'RHT').upper()
                # RHT → left faces -Y (180°), right faces +Y (0°). Swap for LHT.
                rot_map_deg = {
                    'left': 180.0 if traffic == 'RHT' else 0.0,
                    'right': 0.0 if traffic == 'RHT' else 180.0
                }
            yaw_jitter = math.radians(float(car_config.get('yaw_jitter_deg', 0.0)))
            global_fix = math.radians(float(car_config.get('global_rotation_fix_deg', 0.0)))

            print(f"\n--- Generating {num_cars_to_place} Car Placements...  "
                f"(yaw L={rot_map_deg['left']}°, R={rot_map_deg['right']}°, global={math.degrees(global_fix):.1f}°)")
            road_surface_z = DEFAULT_ROAD_THICKNESS * 0.5

            left_driveway_present  = street_data['sides']['left'].get('driveway', {}).get('present')
            right_driveway_present = street_data['sides']['right'].get('driveway', {}).get('present')

            left_driveway_width  = street_data['sides']['left'].get('driveway', {}).get('width', 0)  if left_driveway_present  else 0
            right_driveway_width = street_data['sides']['right'].get('driveway', {}).get('width', 0) if right_driveway_present else 0

            left_lane_center_x  = -median_offset - (left_driveway_width / 2.0)   if left_driveway_width  > 0 else None
            right_lane_center_x =  median_offset + (right_driveway_width / 2.0)  if right_driveway_width > 0 else None

            available_lanes = []
            if left_lane_center_x  is not None: available_lanes.append('left')
            if right_lane_center_x is not None: available_lanes.append('right')

            if not available_lanes:
                print("  Warning: No driveways for cars.")
            else:
                intended_placements = []
                for i in range(num_cars_to_place):
                    side = car_rng.choice(available_lanes)
                    model_path = car_rng.choice(valid_model_paths)

                    if side == 'left':
                        car_x = left_lane_center_x;  lane_width = left_driveway_width
                    else:
                        car_x = right_lane_center_x; lane_width = right_driveway_width

                    # lateral jitter inside lane
                    car_x += car_rng.uniform(-lane_width * 0.2, lane_width * 0.2)
                    # along-street seed
                    car_y = car_rng.uniform(-street_length * 0.48, street_length * 0.48)

                    base_yaw = _side_yaw_rad(street_data, side)  # unified per-side + global fix

                    intended_placements.append({
                        'id': i, 'side': side, 'model_path': model_path,
                        'x': car_x, 'y': car_y, 'base_yaw': base_yaw
                    })

                # spacing
                intended_placements.sort(key=lambda p: p['y'])
                print(f"Adjusting placements for minimum spacing ({MIN_CAR_SPACING:.1f} units)...")
                adjusted_placements = []; last_y = -float('inf')
                for i, placement in enumerate(intended_placements):
                    current_y = placement['y']; required_y = last_y + MIN_CAR_SPACING
                    adjusted_y = max(current_y, required_y)
                    adjusted_y = min(adjusted_y, street_length * 0.48); adjusted_y = max(adjusted_y, -street_length * 0.48)
                    placement['y'] = adjusted_y
                    if i > 0:
                        prev_y = adjusted_placements[i-1]['y']
                        if placement['y'] < prev_y + MIN_CAR_SPACING:
                            placement['y'] = min(prev_y + MIN_CAR_SPACING, street_length * 0.48)
                    adjusted_placements.append(placement); last_y = placement['y']

                print(f"--- Placing {len(adjusted_placements)} Adjusted Cars ---"); placed_car_count = 0
                for placement in adjusted_placements:
                    car_name = f"Car_{placement['id']+1}"
                    final_rot_z = placement['base_yaw'] + car_rng.uniform(-yaw_jitter, yaw_jitter)
                    final_rotation = (0, 0, final_rot_z)

                    imported_car_obj = import_and_place_car(
                        car_name, placement['model_path'], (0, 0, 0), (0, 0, 0), car_scale
                    )
                    if imported_car_obj:
                        bpy.context.view_layer.update()
                        local_min_z = min(corner[2] for corner in imported_car_obj.bound_box)
                        z_offset_needed = -local_min_z
                        car_final_z = road_surface_z + z_offset_needed + CAR_Z_OFFSET

                        imported_car_obj.location = (placement['x'], placement['y'], car_final_z)
                        side_by_x = 'left' if imported_car_obj.location.x < 0 else 'right'
                        imported_car_obj.rotation_euler[2] = _side_yaw_rad(street_data, side_by_x) + car_rng.uniform(-yaw_jitter, yaw_jitter)
                        imported_car_obj.select_set(False)
                        if bpy.context.view_layer.objects.active == imported_car_obj:
                            bpy.context.view_layer.objects.active = None
                        placed_car_count += 1
                    else:
                        print(f"  Skipping car {car_name} due to import error.")
                print(f"--- Successfully placed {placed_car_count} cars ---")
        elif num_cars_to_place > 0:
            print("\n--- Skipping Cars: No valid models found. ---")
        else:
            print("\n--- Skipping Cars (Count is zero) ---")
    else:
        print("\n--- Skipping Cars (Not present in config) ---")



    #####################
    # --- Place Parked Cars in On-Street Parking ---
    place_parked_cars(street_data, median_offset)

    # --- Place Lamps ---
    # (Lamp placement logic - unchanged)
        # --- Place Lamps ---
    # --- Place Lamps (pick ONE model for the whole scene) ---
    if street_data.get('lamps') and street_data['lamps'].get('present', True):
        lamp_config   = street_data['lamps']
        lamp_scale    = lamp_config.get('scale', 1.0)
        lamp_spacing  = lamp_config.get('spacing', LAMP_DEFAULT_SPACING)
        lamp_z_offset = lamp_config.get('z_offset', LAMP_Z_OFFSET)
        street_length = street_data['length']

        # collect candidate models
        model_paths = list(lamp_config.get('model_paths', []) or [])
        if not model_paths and lamp_config.get('model_path'):
            model_paths = [lamp_config['model_path']]

        # optional: scan a folder if provided and no explicit list
        models_dir = lamp_config.get('models_dir')
        if models_dir and not model_paths:
            abs_dir = resolve_asset_path(models_dir)
            if os.path.isdir(abs_dir):
                for f in os.listdir(abs_dir):
                    if os.path.splitext(f)[1].lower() in ('.glb', '.gltf', '.fbx', '.obj'):
                        model_paths.append(os.path.join(abs_dir, f))

        # validate paths
        valid_lamp_model_paths = []
        for p in model_paths:
            ap = resolve_asset_path(p)
            if os.path.exists(ap):
                valid_lamp_model_paths.append(p)

        if valid_lamp_model_paths:
            seed = lamp_config.get('seed', None)
            _rng = random.Random(seed) if seed is not None else random.Random()

            # >>> Choose exactly ONE model for the whole scene <<<
            selected_model_path = _rng.choice(valid_lamp_model_paths)
            print(f"\n--- Placing Street Lamps ({lamp_spacing:.1f}m spacing) ---")
            print(f"  Selected lamp model: {resolve_asset_path(selected_model_path)}")

            base_z = DEFAULT_ROAD_THICKNESS * 0.5
            lamp_count = 0

            for side in ['left', 'right']:
                side_data = street_data['sides'].get(side)
                direction = -1 if side == 'left' else 1
                if not side_data:
                    continue

                driveway_width = side_data.get('driveway', {}).get('width', 0) if side_data.get('driveway', {}).get('present') else 0
                bikepath_width = side_data.get('bikepath', {}).get('width', 0) if side_data.get('bikepath', {}).get('present') else 0
                park_w = float(street_data.get('_runtime', {}).get(side, {}).get('parking_width', 0.0) or 0.0)
                boundary_offset = driveway_width + park_w + bikepath_width

                lamp_boundary_x  = (median_offset * direction) + (boundary_offset * direction)
                lamp_placement_x = lamp_boundary_x + (LAMP_X_OFFSET_FROM_BOUNDARY * direction)
                print(f"  Placing lamps on '{side}' side at X = {lamp_placement_x:.2f} (Boundary was {lamp_boundary_x:.2f})")

                side_rotation_z = (math.pi / 2) * (-direction)  # face inward
                final_lamp_rot_z = side_rotation_z + LAMP_ROTATION_FIX_RAD
                final_lamp_rotation = (0, 0, final_lamp_rot_z)

                num_lamps_on_side = int(street_length // lamp_spacing) + 1
                start_y = -street_length / 2.0

                for i in range(num_lamps_on_side):
                    lamp_y = start_y + i * lamp_spacing
                    lamp_name = f"StreetLamp_{side}_{i+1}"

                    # Import and place THIS SAME model every time
                    imported_lamp_obj = import_and_place_lamp(
                        lamp_name,
                        selected_model_path,
                        (0, 0, 0),
                        (0, 0, 0),
                        lamp_scale
                    )
                    if imported_lamp_obj:
                        bpy.context.view_layer.update()
                        local_min_z = min(corner[2] for corner in imported_lamp_obj.bound_box)
                        z_offset_needed = -local_min_z
                        lamp_final_z = base_z + z_offset_needed + lamp_z_offset

                        imported_lamp_obj.location = (lamp_placement_x, lamp_y, lamp_final_z)
                        imported_lamp_obj.rotation_euler = final_lamp_rotation
                        imported_lamp_obj.select_set(False)
                        if bpy.context.view_layer.objects.active == imported_lamp_obj:
                            bpy.context.view_layer.objects.active = None
                        lamp_count += 1
                    else:
                        print(f"  Skipping lamp {lamp_name} due to import error.")

            print(f"--- Successfully placed {lamp_count} lamps (model fixed to one type) ---")
        else:
            single = lamp_config.get('model_path')
            if single:
                print(f"\n--- Skipping Lamps: Model not found at '{resolve_asset_path(single)}' ---")
            else:
                print("\n--- Skipping Lamps (Not present or no model paths) ---")

        
    ########################
    add_utility_boxes(street_data)


    # --- START: ADDED HUMAN PLACEMENT LOGIC ---
    # --- START: ADDED HUMAN PLACEMENT LOGIC ---
    if street_data.get('humans') and street_data['humans'].get('present', True):
        human_config = street_data['humans']
        num_humans_to_place = human_config.get('count', 0)
        model_paths = human_config.get('model_paths', [])
        human_scale = human_config.get('scale', 1.0)
        human_z_offset = human_config.get('z_offset', HUMAN_Z_OFFSET)
        street_length = street_data['length']
        # Use a local RNG so other features' seeds don't affect humans
        human_seed = human_config.get('seed', None)
        _rng = random.Random(human_seed) if human_seed is not None else random.Random()


        # Validate human model paths
        valid_human_model_paths = []
        print("\nValidating human model paths...")
        for path in model_paths:
             abs_path = resolve_asset_path(path)
             if os.path.exists(abs_path): valid_human_model_paths.append(path); print(f"  [OK] Found: {abs_path}")
             else: print(f"  [NOT FOUND] Skipping: {abs_path} (from: {path})")

        if num_humans_to_place > 0 and valid_human_model_paths:
            print(f"\n--- Placing {num_humans_to_place} Humans randomly on footpaths (using {len(valid_human_model_paths)} models) ---")

            placed_human_count = 0
            attempts = 0 # Limit attempts to prevent infinite loop if no footpaths exist
            max_attempts = num_humans_to_place * 5 # Try a few times per human requested

            # Pre-calculate the top Z of the footpath (curb)
            road_top_z = DEFAULT_ROAD_THICKNESS * 0.5
            footpath_top_z = road_top_z + (DEFAULT_CURB_THICKNESS - DEFAULT_ROAD_THICKNESS) + 0.01

            while placed_human_count < num_humans_to_place and attempts < max_attempts:
                attempts += 1
                side = _rng.choice(['left', 'right'])
                side_data = street_data['sides'].get(side)
                if not side_data or not side_data.get('footpath', {}).get('present'):
                    continue # Skip this side if no footpath configured

                footpath_config = side_data['footpath']
                footpath_width = footpath_config.get('width', 0)
                if footpath_width <= 0:
                    continue # Skip this side if footpath width is zero

                direction = -1 if side == 'left' else 1

                # Re-calculate footpath X boundaries for this side
                driveway_width = side_data.get('driveway',{}).get('width',0) if side_data.get('driveway',{}).get('present') else 0
                bikepath_width = side_data.get('bikepath',{}).get('width',0) if side_data.get('bikepath',{}).get('present') else 0
                footpath_width = side_data.get('footpath',{}).get('width',0) if side_data.get('footpath',{}).get('present') else 0
                
                # CHANGED: read parking width recorded at build time; DO NOT reseed here
                parking_w = float(street_data.get('_runtime', {}).get(side, {}).get('parking_width', 0.0) or 0.0)  # CHANGED

                # Now include parking_w in the offset chain
                footpath_start_x = (median_offset + driveway_width + parking_w + bikepath_width) * direction
                footpath_end_x   = footpath_start_x + (footpath_width * direction)

                footpath_min_x = min(footpath_start_x, footpath_end_x)
                footpath_max_x = max(footpath_start_x, footpath_end_x)
                

                # --- Generate Random Position on Footpath ---
                # Random Y along street
                human_y = _rng.uniform(-street_length * 0.48, street_length * 0.48)
                # Random X within the footpath width, slightly inset
                inset_factor = 0.1 # Percentage inset from edge
                human_x = _rng.uniform(footpath_min_x + footpath_width * inset_factor,
                       footpath_max_x - footpath_width * inset_factor)
                # Random Z rotation
                human_rot_z = _rng.uniform(0, 2 * math.pi)
                final_rotation = (0, 0, human_rot_z)

                # Choose a random model
                model_path = _rng.choice(valid_human_model_paths)
                human_name = f"Human_{placed_human_count + 1}" # Use placed count for name

                # Import human at origin
                imported_human_obj = import_and_place_human(human_name, model_path, (0,0,0), (0,0,0), human_scale)

                if imported_human_obj:
                    bpy.context.view_layer.update()
                    local_min_z = min(corner[2] for corner in imported_human_obj.bound_box)
                    z_offset_needed = -local_min_z
                    human_final_z = footpath_top_z + z_offset_needed + human_z_offset

                    # Set final location and rotation
                    imported_human_obj.location = (human_x, human_y, human_final_z)
                    imported_human_obj.rotation_euler = final_rotation

                    imported_human_obj.select_set(False)
                    if bpy.context.view_layer.objects.active == imported_human_obj: bpy.context.view_layer.objects.active = None
                    placed_human_count += 1
                else:
                    print(f"  Skipping human {human_name} due to import error.")

            print(f"--- Successfully placed {placed_human_count} humans ---")
            if attempts >= max_attempts and placed_human_count < num_humans_to_place:
                 print(f"  Warning: Reached max placement attempts ({max_attempts}). Check if footpaths exist and have width > 0.")

        elif num_humans_to_place > 0: print("\n--- Skipping Humans: No valid models found. ---")
        else: print("\n--- Skipping Humans (Count is zero) ---")
    else: print("\n--- Skipping Humans (Not present in config) ---")
    # --- END: ADDED HUMAN PLACEMENT LOGIC ---

    # --- END: ADDED HUMAN PLACEMENT LOGIC ---


    # --- Setup Camera ---
    # (Camera setup - unchanged)
    cam_dist_factor=1.2; cam_height_factor=0.7; cam_x=0; cam_y=-street_data['length']*cam_dist_factor*0.6; cam_z=street_data['length']*cam_height_factor
    cam_loc=(cam_x,cam_y,cam_z); scene_cam=bpy.context.scene.camera
    if scene_cam and scene_cam.name=="SceneCamera": camera=scene_cam; camera.location=cam_loc; print("Reusing SceneCamera.")
    else: bpy.ops.object.camera_add(location=cam_loc); camera=bpy.context.object; camera.name="SceneCamera"; bpy.context.scene.camera=camera; print(f"Added SceneCamera.")
    look_at_point=Vector((0, street_data['length']*0.1, 0)); direction_to_target = look_at_point - Vector(cam_loc)
    camera.rotation_euler = direction_to_target.to_track_quat('-Z','Y').to_euler()


    # --- Setup Lighting ---
    # (Lighting setup - unchanged)
    sun_loc=(150,-150,120)
    if "SunLight" in bpy.data.objects: sun=bpy.data.objects["SunLight"]; sun.location=sun_loc; print("Reusing SunLight.")
    else: bpy.ops.object.light_add(type='SUN', location=sun_loc); sun=bpy.context.object; sun.name="SunLight"; print(f"Added SunLight.")
    sun.data.energy=4.0; sun.data.angle=math.radians(1.5); sun_look_at=Vector((0,0,0)); sun_dir=sun_look_at-Vector(sun_loc)
    sun.rotation_euler = sun_dir.to_track_quat('-Z','Y').to_euler()
    fill_loc=(0,0,50)
    if "FillLight" in bpy.data.objects: fill_light=bpy.data.objects["FillLight"]; fill_light.location=fill_loc; print("Reusing FillLight.")
    else: bpy.ops.object.light_add(type='AREA', location=fill_loc); fill_light=bpy.context.object; fill_light.name="FillLight"; print("Added FillLight.")
    fill_light.data.energy=150; fill_light.scale=(50,50,50); fill_light.rotation_euler=(math.pi, 0, 0)


    bpy.ops.object.select_all(action='DESELECT')
    if bpy.context.view_layer.objects.active: bpy.context.view_layer.objects.active = None

################### to save a blend file ##########



# --- Final save / handoff ---
OUT_BLEND = _argv_get("--outblend", None)

# Build a simple "handoff" dictionary for downstream scripts (and humans)
handoff = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "blend_version": bpy.app.version_string,
    "cfg_used": os.path.abspath(CFG) if CFG else None,
    "notes": "Model stage completed",
}

if OUT_BLEND:
    out_path = os.path.abspath(OUT_BLEND)  # DO NOT use bpy.path.abspath here
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f"[Build] Saved .blend to: {out_path}")
    handoff["model_blend"] = out_path
else:
    # If no output path given, keep users informed where Blender defaulted to
    # (but try not to rely on this in batch pipelines)
    out_path = bpy.data.filepath
    print(f"[Build] WARNING: --outblend not provided. Current file: {out_path}")
    handoff["model_blend"] = out_path

# Store handoff into Scene custom props (simple dict with basic types)
try:
    bpy.context.scene["handoff"] = handoff
    print(f"[Handoff] {handoff}")
except Exception as e:
    print(f"[Handoff] Could not store in scene: {e}")

# Optional guard: ensure the file exists and is readable from the OS
if not out_path or not os.path.isfile(out_path):
    print("[Build] ERROR: Final save failed or path not found.")
    # Non-zero exit helps your PowerShell stop on this step
    sys.exit(2)

# ---------------------------------------------------------------
