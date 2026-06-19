# render_spin_with_audio_cycles.py  — modes: render1 (spin mp4), render2 (still), render3 (clay+outlines, fast EEVEE)
import bpy, sys, math, os

# ---------------- args & helpers ----------------
def _get_arg(flag, default=None, as_float=False, as_int=False, as_bool=False):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            raw = sys.argv[i+1]
            if as_bool:  return str(raw).strip().lower() in ("1","true","t","yes","y","on")
            if as_int:
                try: return int(raw)
                except: return int(default) if default is not None else None
            if as_float:
                try: return float(raw)
                except: return float(default) if default is not None else None
            return raw
    return default

def _parse_center(s):
    if not s: return (0.0, 0.0, 0.0)
    try:
        parts = [float(p) for p in s.split(",")]
        if len(parts) >= 3: return (parts[0], parts[1], parts[2])
    except: pass
    return (0.0, 0.0, 0.0)

# ---------------- defaults / CLI ----------------
MODE        = (_get_arg("--mode", "render1") or "render1").lower()  # render1|render2|render3
OUT_PATH    = _get_arg("--out", "spin.mp4")
ROTATIONS   = _get_arg("--rotations", None, as_float=True)
PAN_DEG     = _get_arg("--pan-deg", 25.0, as_float=True)
PAN_CENTER_DEG = _get_arg("--pan-center-deg", -90.0, as_float=True)
DURATION_S  = _get_arg("--duration_s", 4.0, as_float=True)
RADIUS      = _get_arg("--radius", 80.0, as_float=True) or 80.0
# was 100 for other properties render, 150 for svf
HEIGHT      = _get_arg("--height", 30.0, as_float=True) or 50.0
CENTER_STR  = _get_arg("--center", "0,0,0")
CENTER      = _parse_center(CENTER_STR)
FPS         = _get_arg("--fps", 24, as_int=True)
RESX        = _get_arg("--resx", None, as_int=True)   # if None, we set per-mode defaults
RESY        = _get_arg("--resy", None, as_int=True)
PAN_VIDEO_RESX = 640
PAN_VIDEO_RESY = 480
STILL_RESX = 2048
STILL_RESY = 1536
CAM_NAME    = _get_arg("--camera_name", None)
USE_CURCAM  = _get_arg("--use_current_camera", False, as_bool=True)

ENGINE      = (_get_arg("--engine", "cycles") or "cycles").strip().lower()
GPU         = _get_arg("--gpu", True, as_bool=True)
SAMPLES     = _get_arg("--samples", 128, as_int=True)
DENOISE     = (_get_arg("--denoise", "oidn") or "oidn").strip().lower()
MOTION_BLUR = _get_arg("--motion_blur", True, as_bool=True)
LIGHTING_MODE = (_get_arg("--lighting", "nishita") or "nishita").strip().lower()  # nishita|scene|both
RENDER_EXPOSURE = _get_arg("--exposure", -0.5, as_float=True)
NISHITA_SUN_SIZE = _get_arg("--sun-size", 0.545, as_float=True)
NISHITA_SUN_INTENSITY = _get_arg("--sun-intensity", 0.4, as_float=True)
NISHITA_SUN_ELEVATION_DEG = _get_arg("--sun-elevation-deg", 30.0, as_float=True)
NISHITA_SUN_ROTATION_DEG = _get_arg("--sun-rotation-deg", 15.0, as_float=True)
NISHITA_WORLD_STRENGTH = _get_arg("--world-strength", 0.4, as_float=True)

# still framing
STILL_YAW_DEG = 90.0

def scene(): return bpy.context.scene
def scene_start(): return int(getattr(scene(), "frame_start", 0))
def scene_end(): return int(getattr(scene(), "frame_end", 250))
def fps_real():
    scn = scene()
    fps  = float(getattr(scn.render, "fps", 24))
    base = float(getattr(scn.render, "fps_base", 1.0) or 1.0)
    return fps / base

def apply_render_fps():
    if FPS:
        scn = scene()
        scn.render.fps = int(FPS)
        scn.render.fps_base = 1.0

def link_object(obj):
    try:
        if obj.name not in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.link(obj)
    except Exception: pass

def ensure_empty(name, location=(0,0,0)):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        link_object(obj)
    obj.location = location
    return obj

def ensure_camera(name='SpinCamera'):
    if CAM_NAME and CAM_NAME in bpy.data.objects and bpy.data.objects[CAM_NAME].type == 'CAMERA':
        return bpy.data.objects[CAM_NAME]
    if USE_CURCAM and scene().camera is not None:
        return scene().camera
    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    link_object(cam)
    return cam

def clear_anim(obj):
    ad = getattr(obj, "animation_data", None)
    if ad and ad.action:
        fcurves = [fc for fc in ad.action.fcurves if fc.data_path == "rotation_euler"]
        for fc in fcurves:
            try: ad.action.fcurves.remove(fc)
            except: pass

def _safe_set(obj, attr, val):
    try: setattr(obj, attr, val)
    except Exception: pass

def set_render_visibility_for_lights(enabled, light_types=None):
    """
    Enable/disable light objects for rendering.
    If light_types is provided, only lights whose data.type matches are affected.
    """
    wanted = None if light_types is None else {str(t).upper() for t in light_types}
    for obj in bpy.data.objects:
        if obj.type != 'LIGHT':
            continue
        light_type = str(getattr(obj.data, "type", "")).upper()
        if wanted is not None and light_type not in wanted:
            continue
        obj.hide_render = not enabled

def set_shadow_casting_for_lights(enabled, light_types=None):
    """Enable/disable shadows from scene light objects without hiding their illumination."""
    wanted = None if light_types is None else {str(t).upper() for t in light_types}
    for obj in bpy.data.objects:
        if obj.type != 'LIGHT':
            continue
        light_type = str(getattr(obj.data, "type", "")).upper()
        if wanted is not None and light_type not in wanted:
            continue
        _safe_set(obj.data, "use_shadow", bool(enabled))
        cycles_visibility = getattr(obj, "cycles_visibility", None)
        if cycles_visibility:
            _safe_set(cycles_visibility, "shadow", bool(enabled))

def _pick_engine_id(scn, preferred=('BLENDER_EEVEE_NEXT','BLENDER_EEVEE','BLENDER_WORKBENCH','CYCLES')):
    items = tuple(scn.render.bl_rna.properties['engine'].enum_items.keys())
    for e in preferred:
        if e in items:
            return e
    return items[0] if items else 'CYCLES'


# ---------------- engines ----------------
def configure_engine_cycles():
    scn = scene()
    scn.render.engine = 'CYCLES'
    cycles = getattr(scn, "cycles", None)
    if cycles:
        _safe_set(cycles, "samples", int(SAMPLES))
        _safe_set(cycles, "preview_samples", max(1, min(32, int(SAMPLES//4) or 1)))
        _safe_set(cycles, "use_adaptive_sampling", True)
        _safe_set(cycles, "device", 'GPU' if GPU else 'CPU')
        _safe_set(cycles, "use_motion_blur", bool(MOTION_BLUR))
    for layer in scn.view_layers:
        cyc = getattr(layer, "cycles", None)
        if cyc:
            _safe_set(cyc, "use_denoising", DENOISE != "none")
    # try to enable any GPU
    try:
        prefs = bpy.context.preferences
        addon = prefs.addons.get("cycles")
        if addon:
            ap = addon.preferences
            for t in ("OPTIX","HIP","METAL","CUDA","OPENCL"):
                try:
                    ap.compute_device_type = t
                    break
                except Exception:
                    continue
            try:
                for dev in ap.get_devices()[0]:
                    dev.use = True
            except Exception:
                pass
    except Exception:
        pass

def configure_engine_eevee_for_outlines():
    scn = scene()
    # Pick EEVEE_NEXT if present, else EEVEE, else WORKBENCH
    scn.render.engine = _pick_engine_id(scn, ('BLENDER_EEVEE_NEXT','BLENDER_EEVEE','BLENDER_WORKBENCH'))

    # EEVEE settings object name stayed "eevee" in 4.x, but be defensive
    ee = getattr(scn, "eevee", None) or getattr(scn, "eevee_next", None)
    if ee:
        _safe_set(ee, "use_taa_reprojection", True)
        # AO flag changed name in some versions; set both safely
        _safe_set(ee, "use_gtao", True)
        _safe_set(ee, "use_ambient_occlusion", True)
        _safe_set(ee, "use_bloom", False)
        _safe_set(ee, "use_motion_blur", False)

    # Freestyle for outlines
    scn.render.use_freestyle = True


# ---------------- rig / anim ----------------
def build_spin_rig(center=(0,0,0), radius=30.0, height=5.0):
    tgt = ensure_empty("SpinTarget", center)
    rig = ensure_empty("SpinRig", center)
    cam = ensure_camera()
    cam.parent = rig
    cam.location = (radius, 0.0, height)
    con = None
    for c in cam.constraints:
        if c.type == 'TRACK_TO' and c.name == "SpinTrackTo":
            con = c; break
    if con is None:
        con = cam.constraints.new(type='TRACK_TO')
        con.name = "SpinTrackTo"
    con.target = tgt
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis    = 'UP_Y'
    scene().camera = cam
    return rig, tgt, cam

def animate_spin(rig, rotations=None, pan_degrees=25.0, pan_center_degrees=-90.0, start=None, end=None):
    if start is None: start = scene_start()
    if end   is None: end   = scene_end()
    rig.rotation_mode = 'XYZ'
    clear_anim(rig)
    center_angle = math.radians(float(pan_center_degrees if pan_center_degrees is not None else -90.0))
    if rotations is not None:
        start_angle = center_angle
        end_angle = center_angle + float(rotations) * (2.0 * math.pi)
    else:
        sweep = math.radians(float(pan_degrees if pan_degrees is not None else 25.0))
        start_angle = center_angle - 0.5 * sweep
        end_angle = center_angle + 0.5 * sweep
    rig.rotation_euler = (0.0, 0.0, start_angle)
    rig.keyframe_insert(data_path="rotation_euler", frame=start, index=2)
    rig.rotation_euler[2] = end_angle
    rig.keyframe_insert(data_path="rotation_euler", frame=end, index=2)
    ad = rig.animation_data
    act = ad.action if ad else None
    if act:
        for fc in act.fcurves:
            if fc.data_path == "rotation_euler" and fc.array_index == 2:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

def set_rig_yaw(rig, degrees):
    rig.rotation_mode = 'XYZ'
    clear_anim(rig)
    rig.rotation_euler = (0.0, 0.0, math.radians(degrees))

def set_render1_view_settings(exposure=0.0):
    scn = scene()
    try:
        scn.view_settings.exposure = float(exposure)
        scn.view_settings.gamma = 1.0
    except Exception:
        pass

def extend_nla_strips_to_scene_end():
    """Extend looped action strips so longer renders do not outlast walk/run cycles."""
    scn = scene()
    end = scene_end()
    changed = 0
    for obj in bpy.data.objects:
        ad = getattr(obj, "animation_data", None)
        if not ad:
            continue
        for track in getattr(ad, "nla_tracks", []):
            for strip in getattr(track, "strips", []):
                try:
                    length = max(1.0, float(strip.action_frame_end) - float(strip.action_frame_start))
                    needed = (float(end) - float(strip.frame_start)) + length + 2.0
                    repeat = max(1, math.ceil(needed / length))
                    if repeat > float(getattr(strip, "repeat", 1.0)):
                        strip.repeat = repeat
                        changed += 1
                except Exception:
                    pass
    if changed:
        print(f"[Anim] Extended {changed} NLA loop strips to frame {end}.")

# ---------------- audio ----------------
def configure_audio_and_retime_speakers():
    scn = scene()
    try: scn.sync_mode = 'AUDIO_SYNC'
    except: pass
    _set = lambda obj,a,v: setattr(obj,a,v) if hasattr(obj,a) else None
    if hasattr(scn, "render"):
        _set(scn.render, "use_audio", True)
        _set(scn.render, "use_audio_scrub", True)
    s = scene_start(); e = scene_end()
    for spk in [o for o in bpy.data.objects if o.type == 'SPEAKER']:
        ad = getattr(spk.data, "animation_data", None)
        if not ad: 
            continue
        for tr in getattr(ad, "nla_tracks", []):
            for st in getattr(tr, "strips", []):
                if getattr(st, "type", None) == 'SOUND':
                    try:
                        length = max(1.0, (st.frame_end - st.frame_start))
                        st.frame_start = float(s)
                        st.mute = False
                        total  = max(0.0, (e - s))
                        st.repeat = max(1.0, (total / length) + 1.0)
                    except Exception:
                        pass
    se = getattr(scn, "sequence_editor", None)
    if se and hasattr(se, "sequences_all"):
        for strip in list(se.sequences_all):
            if getattr(strip, "type", "") == "SOUND":
                try: se.sequences.remove(strip)
                except: pass

# ---------------- world: Nishita ----------------
def ensure_nishita_world():
    """Create/refresh World using Nishita Sky with fixed parameters."""
    sun_size = float(NISHITA_SUN_SIZE)
    sun_intensity = float(NISHITA_SUN_INTENSITY)
    sun_elevation_deg = float(NISHITA_SUN_ELEVATION_DEG)
    sun_rotation_deg = float(NISHITA_SUN_ROTATION_DEG)
    strength = float(NISHITA_WORLD_STRENGTH)

    world = bpy.context.scene.world or bpy.data.worlds.get("World")
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nt = world.node_tree

    # get/create nodes
    bg = next((n for n in nt.nodes if n.type == "BACKGROUND"), None) or nt.nodes.new("ShaderNodeBackground")
    out = next((n for n in nt.nodes if n.type == "OUTPUT_WORLD"), None) or nt.nodes.new("ShaderNodeOutputWorld")
    bg.location, out.location = (200,0), (400,0)

    # remove prior sky/env to avoid double-lighting
    for n in list(nt.nodes):
        if n.type in {"TEX_ENVIRONMENT", "TEX_SKY"}:
            nt.nodes.remove(n)

    sky = nt.nodes.new("ShaderNodeTexSky"); sky.location = (0, 0)
    sky.sky_type = 'NISHITA'
    if hasattr(sky, "sun_disc"): sky.sun_disc = True

    elev_rad = math.radians(sun_elevation_deg)
    rot_rad  = math.radians(sun_rotation_deg)

    if hasattr(sky, "sun_size"):        sky.sun_size = sun_size
    elif "Sun Size" in sky.inputs:      sky.inputs["Sun Size"].default_value = sun_size

    if hasattr(sky, "sun_intensity"):   sky.sun_intensity = sun_intensity
    elif "Sun Intensity" in sky.inputs: sky.inputs["Sun Intensity"].default_value = sun_intensity

    if hasattr(sky, "sun_elevation"):   sky.sun_elevation = elev_rad
    elif "Sun Elevation" in sky.inputs: sky.inputs["Sun Elevation"].default_value = elev_rad

    if hasattr(sky, "sun_rotation"):    sky.sun_rotation = rot_rad
    elif "Sun Rotation" in sky.inputs:  sky.inputs["Sun Rotation"].default_value = rot_rad

    # link Sky → Background → Output
    for l in list(nt.links):
        if l.to_node == bg and l.to_socket.name == "Color":
            nt.links.remove(l)
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    print(f"[World] Nishita sky set.")

def ensure_scene_lights_world(background_color=(1, 1, 1, 1)):
    """
    Keep a clean visible background for the camera while removing world lighting,
    so only scene light objects drive the render shadows.
    """
    world = bpy.context.scene.world or bpy.data.worlds.get("World")
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    light_path = nt.nodes.new("ShaderNodeLightPath")
    bg_unlit = nt.nodes.new("ShaderNodeBackground")
    bg_camera = nt.nodes.new("ShaderNodeBackground")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputWorld")

    light_path.location = (-500, 0)
    bg_unlit.location = (-250, -120)
    bg_camera.location = (-250, 120)
    mix.location = (0, 0)
    out.location = (220, 0)

    bg_unlit.inputs["Color"].default_value = (0, 0, 0, 1)
    bg_unlit.inputs["Strength"].default_value = 1.0
    bg_camera.inputs["Color"].default_value = background_color
    bg_camera.inputs["Strength"].default_value = 1.0

    nt.links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(bg_unlit.outputs["Background"], mix.inputs[1])
    nt.links.new(bg_camera.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
    print("[World] Scene-lights world set (camera background only, no world lighting).")

def configure_render1_lighting(mode):
    """
    render1 lighting modes:
      - nishita: Nishita sky only, scene lights disabled
      - scene: scene lights only, world lighting disabled
      - both: Nishita sky shadows plus non-shadow scene fill lights
    """
    raw_mode = (mode or "nishita").strip().lower()
    aliases = {
        "sky": "nishita",
        "world": "nishita",
        "lights": "scene",
        "scene_lights": "scene",
        "mixed": "both",
    }
    resolved = aliases.get(raw_mode, raw_mode)

    if resolved == "nishita":
        ensure_nishita_world()
        set_shadow_casting_for_lights(True)
        set_render_visibility_for_lights(False)
    elif resolved == "scene":
        ensure_scene_lights_world()
        set_render_visibility_for_lights(True)
        set_shadow_casting_for_lights(False)
        set_shadow_casting_for_lights(True, light_types=("SUN",))
    elif resolved == "both":
        ensure_nishita_world()
        set_shadow_casting_for_lights(False)
        set_render_visibility_for_lights(False, light_types=("SUN",))
        set_render_visibility_for_lights(True, light_types=("AREA", "POINT", "SPOT"))
    else:
        print(f"[Lighting] Unknown --lighting '{mode}', falling back to 'nishita'.")
        ensure_nishita_world()
        set_shadow_casting_for_lights(True)
        set_render_visibility_for_lights(False)
        resolved = "nishita"

    print(f"[Lighting] render1 mode = {resolved}")
    return resolved

# ---------------- clay + outlines ----------------
def setup_clay_and_outlines():
    """Force white material and enable black outline rendering with Freestyle."""
    # White clay material
    # White clay material (flat, unshaded)
    mat = bpy.data.materials.get("_ClayWhite") or bpy.data.materials.new("_ClayWhite")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)

    em = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em.inputs["Color"].default_value = (1, 1, 1, 1)   # pure white
    em.inputs["Strength"].default_value = 1.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])

    # ViewLayer material override
    vl = bpy.context.view_layer
    vl.material_override = mat

    # Freestyle setup (faster: silhouettes + border only)
    scn = scene()
    scn.render.use_freestyle = True
    fs = vl.freestyle_settings
    fs.use_culling = True  # ignore occluded strokes to speed up
    fs.crease_angle = math.radians(145.0)

    if not fs.linesets:
        fs.linesets.new("All")
    ls = fs.linesets[0]
    ls.select_silhouette = True
    ls.select_border = True
    ls.select_crease = False
    ls.select_edge_mark = False
    ls.select_contour = False

    line_style = ls.linestyle
    line_style.color = (0,0,0)
    line_style.alpha = 1.0
    line_style.thickness = 2.0  # px
    print("[Clay] Material override + Freestyle outlines (EEVEE) enabled.")

# ---------------- output ----------------
def configure_output_movie(out_path):
    scn = scene()
    try:
        abs_out = os.path.abspath(bpy.path.abspath(out_path))
        os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    except Exception: pass
    apply_render_fps()
    scn.render.resolution_x = int(RESX) if RESX is not None else PAN_VIDEO_RESX
    scn.render.resolution_y = int(RESY) if RESY is not None else PAN_VIDEO_RESY
    scn.render.image_settings.file_format = 'FFMPEG'
    scn.render.ffmpeg.format = 'MPEG4'
    scn.render.ffmpeg.codec = 'H264'
    scn.render.ffmpeg.use_max_b_frames = True
    scn.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    scn.render.ffmpeg.ffmpeg_preset = 'GOOD'
    scn.render.ffmpeg.audio_codec = 'AAC'
    scn.render.ffmpeg.audio_bitrate = 192
    scn.render.ffmpeg.audio_channels = 'STEREO'
    scn.render.ffmpeg.audio_mixrate = 48000
    scn.render.filepath = bpy.path.abspath(out_path)

def configure_output_still(out_path, fmt='PNG'):
    scn = scene()
    try:
        abs_out = os.path.abspath(bpy.path.abspath(out_path))
        os.makedirs(os.path.dirname(abs_out), exist_ok=True)
    except Exception: pass
    # Default still resolution if not supplied.
    scn.render.resolution_x = int(RESX) if RESX is not None else STILL_RESX
    scn.render.resolution_y = int(RESY) if RESY is not None else STILL_RESY
    scn.render.image_settings.file_format = fmt
    scn.render.filepath = bpy.path.abspath(out_path)

def ensure_duration_seconds(duration_s):
    if duration_s is None: return scene_start(), scene_end()
    scn = scene()
    fr = fps_real()
    s = scene_start()
    frame_count = max(1, int(round(float(duration_s) * fr)))
    e = int(s + frame_count - 1)
    scn.frame_end = e
    return s, e

def set_view_to_standard_paper_white():
    scn = bpy.context.scene
    scn.display_settings.display_device = 'sRGB'
    # Prefer Standard; if not available (older OCIO), fall back to Raw
    vt = getattr(scn.view_settings, "view_transform", "Standard")
    try:
        scn.view_settings.view_transform = 'Standard'
    except Exception:
        try: scn.view_settings.view_transform = 'Raw'
        except Exception: pass
    scn.view_settings.look = 'None'
    scn.view_settings.exposure = 0.0
    scn.view_settings.gamma = 1.0
    scn.render.film_transparent = False  # keep solid white

def setup_clay_world_white():
    world = bpy.context.scene.world or bpy.data.worlds.get("World")
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    bg  = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg.inputs["Color"].default_value = (1,1,1,1)  # pure white background
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

def force_ground_white():
    for o in bpy.data.objects:
        if o.type == "MESH" and "ground" in o.name.lower():
            if o.data.materials:
                o.data.materials[0] = bpy.data.materials["_ClayWhite"]
            else:
                o.data.materials.append(bpy.data.materials["_ClayWhite"])
    print("[Clay] Ground forced to white.")

def set_ground_material(mat):
    """Assign the same material to all ground meshes."""
    if not mat:
        return
    for o in ground_objects():
        if o.data.materials:
            o.data.materials[0] = mat
        else:
            o.data.materials.append(mat)
    print(f"[Ground] Applied material '{mat.name}' to ground meshes.")

def ground_objects(name_keys=("ground",)):
    keys = tuple(k.lower() for k in name_keys)
    return [o for o in bpy.data.objects
            if o.type == "MESH" and any(k in o.name.lower() for k in keys)]

def hide_ground_for_render3():
    """Non-destructive: exclude ground in render3 only."""
    for o in ground_objects():
        o.hide_render = True
        try: o.hide_viewport = True
        except Exception: pass
    print("[Render3] Ground hidden (not deleted).")

def ensure_white_ground_mat(emission=True, name="_GroundWhite2"):
    """White material for ground only (emission=True = paper-white)."""
    import bpy
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if emission:
        sh = nt.nodes.new("ShaderNodeEmission")
        sh.inputs["Color"].default_value = (1,1,1,1)
        sh.inputs["Strength"].default_value = 1.0
        nt.links.new(sh.outputs["Emission"], out.inputs["Surface"])
    else:
        sh = nt.nodes.new("ShaderNodeBsdfPrincipled")
        sh.inputs["Base Color"].default_value = (1,1,1,1)
        sh.inputs["Roughness"].default_value = 0.9
        nt.links.new(sh.outputs["BSDF"], out.inputs["Surface"])
    return m



# ---------------- main ----------------
def main():
    scn = scene()

    # timeline sane
    try:
        scn.frame_current = scene_start()
        scn.frame_set(scene_start())
    except: pass

    rig, tgt, cam = build_spin_rig(center=CENTER, radius=RADIUS, height=HEIGHT)

    # World lighting for all modes
    # ensure_nishita_world()

    if MODE == "render1":
        # Cycles animation → MP4
        configure_render1_lighting(LIGHTING_MODE)
        # Selectable from the CLI:
        #   --lighting nishita
        #   --lighting scene
        #   --lighting both
        set_ground_material(ensure_white_ground_mat(emission=False, name="_GroundWhiteRender1"))
        configure_engine_cycles()
        apply_render_fps()
        s, e = ensure_duration_seconds(DURATION_S)
        animate_spin(rig, rotations=ROTATIONS, pan_degrees=PAN_DEG, pan_center_degrees=PAN_CENTER_DEG, start=s, end=e)
        set_render1_view_settings(RENDER_EXPOSURE)
        extend_nla_strips_to_scene_end()
        configure_audio_and_retime_speakers()
        configure_output_movie(OUT_PATH)
        if ROTATIONS is not None:
            motion_desc = f"spin, {float(ROTATIONS) * 360.0:.1f} degrees"
        else:
            motion_desc = f"pan, {float(PAN_DEG):.1f} degrees around yaw {float(PAN_CENTER_DEG):.1f}"
        print(f"[Render] Mode=render1 ({motion_desc}) frames {s}->{e} @ {fps_real():.2f} fps")
        print(f"[Render] Output: {scene().render.filepath}")
        bpy.ops.render.render(animation=True, use_viewport=False)

    elif MODE == "render2":
        # Cycles still (textured)
        configure_engine_cycles()
        scene().render.use_freestyle = False  
        hide_ground_for_render3()
        setup_clay_world_white()
        set_view_to_standard_paper_white()
        set_rig_yaw(rig, STILL_YAW_DEG)
        scn.frame_current = scene_start()
        configure_output_still(OUT_PATH, fmt='PNG')
        print(f"[Render] Mode=render2 (still, textured)")
        print(f"[Render] Output: {scene().render.filepath}")
        bpy.ops.render.render(animation=False, write_still=True, use_viewport=False)

    elif MODE == "render3":
        # EEVEE still (clay + outlines) — fast, no denoise, no motion blur
        configure_engine_eevee_for_outlines()
        hide_ground_for_render3()  
        setup_clay_and_outlines()
        setup_clay_world_white()  
        set_view_to_standard_paper_white()
        force_ground_white()

        set_rig_yaw(rig, STILL_YAW_DEG)
        scn.frame_current = scene_start()
        configure_output_still(OUT_PATH, fmt='PNG')
        print(f"[Render] Mode=render3 (still, clay+outlines, EEVEE)")
        print(f"[Render] Output: {scene().render.filepath}")
        bpy.ops.render.render(animation=False, write_still=True, use_viewport=False)

    else:
        print(f"[Render] Unknown --mode '{MODE}'. Use render1|render2|render3.")
        sys.exit(2)

if __name__ == "__main__":
    main()
