import hashlib
import io
import json
import mimetypes
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import ExifTags, Image, ImageOps, TiffImagePlugin

try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None


# ============================================================
# App settings
# ============================================================
st.set_page_config(page_title="Mask Tool", layout="wide")

SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_SHARE_FILE_MB = 50
JPEG_QUALITY = 100
DEFAULT_THRESHOLD = 30


# ============================================================
# Secrets / gates
# ============================================================
def read_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read Streamlit secret safely.

    Local development fallback:
    - If .streamlit/secrets.toml does not exist, st.secrets can raise.
    - In that case, return the provided default instead of stopping the app.
    """
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None

    if value is None or str(value).strip() == "":
        return default
    return str(value)


def password_gate(
    secret_name: str,
    session_key: str,
    title: str,
    fallback_secret_name: Optional[str] = None,
    local_default: str = "dev",
) -> None:
    """Simple password gate.

    Public deployment:
    - Put APP_PASSWORD / SHARE_PAGE_PASSWORD in Streamlit secrets.
    Local development:
    - If secrets are missing, password defaults to "dev" so local testing works.
    """
    expected = read_secret(secret_name)
    using_local_default = False

    if expected is None and fallback_secret_name:
        expected = read_secret(fallback_secret_name)

    if expected is None:
        expected = local_default
        using_local_default = True

    if st.session_state.get(session_key) is True:
        return

    st.subheader(title)

    if using_local_default:
        st.info('Local dev mode: secrets not found. Use password "dev". Set .streamlit/secrets.toml before public deployment.')

    entered = st.text_input("Password", type="password", key=f"{session_key}_input")
    if st.button("Enter", key=f"{session_key}_button"):
        if secrets_equal(entered, expected):
            st.session_state[session_key] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()


def secrets_equal(a: str, b: str) -> bool:
    return hashlib.sha256(a.encode("utf-8")).digest() == hashlib.sha256(b.encode("utf-8")).digest()


# ============================================================
# Image helpers
# ============================================================
@st.cache_data(show_spinner=False)
def load_image_pair(file_bytes: bytes) -> Tuple[Image.Image, Image.Image]:
    raw = Image.open(io.BytesIO(file_bytes))
    raw.load()

    raw_copy = raw.copy()
    raw_copy.format = raw.format
    raw_copy.info = dict(raw.info)

    rgb = ImageOps.exif_transpose(raw).convert("RGB")
    return raw_copy, rgb


@st.cache_data(show_spinner=False)
def luminance_histogram(file_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    _, img = load_image_pair(file_bytes)
    arr = np.asarray(img, dtype=np.float32)
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    hist, bins = np.histogram(lum.ravel(), bins=256, range=(0, 255))
    return hist.astype(np.int64), bins.astype(np.float32)


@st.cache_data(show_spinner=False)
def build_mask_previews(file_bytes: bytes, threshold: int) -> Tuple[Image.Image, Image.Image]:
    _, mask_img = load_image_pair(file_bytes)

    arr = np.asarray(mask_img, dtype=np.float32)
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    keep_mask = lum > threshold

    rgb = np.asarray(mask_img.convert("RGB"), dtype=np.uint8)
    alpha = np.where(keep_mask, 255, 0).astype(np.uint8)

    removed_vis = rgb.copy()
    removed_vis[~keep_mask] = np.array([255, 0, 180], dtype=np.uint8)

    rgba = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    preview = composite_on_checkerboard(rgba)

    return Image.fromarray(removed_vis, "RGB"), preview


@st.cache_data(show_spinner=False)
def composite_mask_on_target_cached(mask_bytes: bytes, target_bytes: bytes, threshold: int) -> Image.Image:
    _, mask_img = load_image_pair(mask_bytes)
    _, target_img = load_image_pair(target_bytes)

    if mask_img.size != target_img.size:
        mask_img = mask_img.resize(target_img.size, Image.Resampling.LANCZOS)

    arr = np.asarray(mask_img, dtype=np.float32)
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    keep_mask = lum > threshold

    rgb = np.asarray(mask_img.convert("RGB"), dtype=np.uint8)
    alpha = np.where(keep_mask, 255, 0).astype(np.uint8)
    mask_rgba = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")

    return Image.alpha_composite(target_img.convert("RGBA"), mask_rgba).convert("RGB")


def make_checkerboard(size: Tuple[int, int], cell: int = 20) -> np.ndarray:
    w, h = size
    yy, xx = np.indices((h, w))
    pattern = ((xx // cell + yy // cell) % 2).astype(np.uint8)
    light = np.array([238, 238, 238], dtype=np.uint8)
    dark = np.array([196, 196, 196], dtype=np.uint8)
    return np.where(pattern[..., None] == 0, light, dark)


def composite_on_checkerboard(rgba: Image.Image) -> Image.Image:
    arr = np.asarray(rgba.convert("RGBA"), dtype=np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3:4] / 255.0
    board = make_checkerboard((rgba.width, rgba.height)).astype(np.float32)
    out = rgb * alpha + board * (1.0 - alpha)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# GPS / metadata helpers
# ============================================================
def rational_to_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except Exception:
        return None


def gps_dms_to_decimal(dms: Any, ref: str) -> Optional[float]:
    if not isinstance(dms, (tuple, list)) or len(dms) != 3:
        return None

    deg = rational_to_float(dms[0])
    minute = rational_to_float(dms[1])
    sec = rational_to_float(dms[2])

    if deg is None or minute is None or sec is None:
        return None

    decimal = deg + minute / 60.0 + sec / 3600.0
    if ref in {"S", "W"}:
        decimal *= -1
    return decimal


def decimal_to_dms_rationals(value: float) -> Tuple[Any, Any, Any]:
    value = abs(float(value))
    deg = int(value)
    minutes_full = (value - deg) * 60.0
    minute = int(minutes_full)
    seconds = (minutes_full - minute) * 60.0
    return (
        TiffImagePlugin.IFDRational(deg, 1),
        TiffImagePlugin.IFDRational(minute, 1),
        TiffImagePlugin.IFDRational(int(round(seconds * 10000)), 10000),
    )


def extract_gps_decimal(img: Image.Image) -> Optional[Tuple[float, float]]:
    exif = img.getexif()
    if not exif or 34853 not in exif:
        return None

    try:
        gps_ifd = exif.get_ifd(34853)
    except Exception:
        gps_ifd = {}

    if not isinstance(gps_ifd, dict):
        return None

    lat_ref = gps_ifd.get(1)
    lat_dms = gps_ifd.get(2)
    lon_ref = gps_ifd.get(3)
    lon_dms = gps_ifd.get(4)

    if isinstance(lat_ref, bytes):
        lat_ref = lat_ref.decode(errors="ignore")
    if isinstance(lon_ref, bytes):
        lon_ref = lon_ref.decode(errors="ignore")

    if not lat_ref or not lon_ref:
        return None

    lat = gps_dms_to_decimal(lat_dms, str(lat_ref))
    lon = gps_dms_to_decimal(lon_dms, str(lon_ref))

    if lat is None or lon is None:
        return None

    return lat, lon


def set_exif_gps(exif: Image.Exif, lat: float, lon: float) -> Image.Exif:
    gps_ifd = {
        1: "N" if lat >= 0 else "S",
        2: decimal_to_dms_rationals(lat),
        3: "E" if lon >= 0 else "W",
        4: decimal_to_dms_rationals(lon),
    }
    exif[34853] = gps_ifd
    return exif


def remove_exif_gps(exif: Image.Exif) -> Image.Exif:
    if 34853 in exif:
        del exif[34853]
    return exif


def stringify_metadata_value(value: Any) -> str:
    if isinstance(value, bytes):
        return f"<bytes: {len(value):,} bytes>"
    if isinstance(value, dict):
        return json.dumps({str(k): stringify_metadata_value(v) for k, v in value.items()}, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return ", ".join(stringify_metadata_value(v) for v in value)
    return str(value)


@st.cache_data(show_spinner=False)
def metadata_rows_from_bytes(file_bytes: bytes) -> List[Dict[str, str]]:
    raw, _ = load_image_pair(file_bytes)
    rows: List[Dict[str, str]] = []

    rows.append({"group": "Image", "tag_id": "", "tag_name": "format", "value": str(raw.format)})
    rows.append({"group": "Image", "tag_id": "", "tag_name": "mode", "value": str(raw.mode)})
    rows.append({"group": "Image", "tag_id": "", "tag_name": "size", "value": f"{raw.width} × {raw.height}"})

    for key, value in sorted((raw.info or {}).items(), key=lambda x: str(x[0])):
        rows.append(
            {
                "group": "Pillow info",
                "tag_id": "",
                "tag_name": str(key),
                "value": stringify_metadata_value(value),
            }
        )

    exif = raw.getexif()
    if not exif:
        return rows

    for tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))

        if tag_id == 34853:
            rows.append({"group": "EXIF", "tag_id": str(tag_id), "tag_name": "GPSInfo", "value": "see GPS rows"})
            try:
                gps_ifd = exif.get_ifd(34853)
            except Exception:
                gps_ifd = value if isinstance(value, dict) else {}
            if isinstance(gps_ifd, dict):
                for gps_id, gps_value in gps_ifd.items():
                    gps_name = ExifTags.GPSTAGS.get(gps_id, str(gps_id))
                    rows.append(
                        {
                            "group": "GPS",
                            "tag_id": str(gps_id),
                            "tag_name": gps_name,
                            "value": stringify_metadata_value(gps_value),
                        }
                    )
            continue

        if tag_id == 34665:
            rows.append({"group": "EXIF", "tag_id": str(tag_id), "tag_name": "ExifOffset", "value": "see EXIF sub-IFD rows"})
            try:
                exif_ifd = exif.get_ifd(34665)
            except Exception:
                exif_ifd = {}
            if isinstance(exif_ifd, dict):
                for sub_id, sub_value in exif_ifd.items():
                    sub_name = ExifTags.TAGS.get(sub_id, str(sub_id))
                    rows.append(
                        {
                            "group": "EXIF sub-IFD",
                            "tag_id": str(sub_id),
                            "tag_name": sub_name,
                            "value": stringify_metadata_value(sub_value),
                        }
                    )
            continue

        rows.append(
            {
                "group": "EXIF",
                "tag_id": str(tag_id),
                "tag_name": tag_name,
                "value": stringify_metadata_value(value),
            }
        )

    return rows


@st.cache_data(show_spinner=False)
def make_final_jpg_bytes(
    composite_png_bytes: bytes,
    metadata_source_bytes: Optional[bytes],
    gps_mode: str,
    manual_lat: Optional[float],
    manual_lon: Optional[float],
    quality: int = JPEG_QUALITY,
) -> bytes:
    composite_img = Image.open(io.BytesIO(composite_png_bytes)).convert("RGB")

    metadata_source = None
    if metadata_source_bytes is not None:
        metadata_source, _ = load_image_pair(metadata_source_bytes)

    exif = Image.Exif()
    icc_profile = None

    if metadata_source is not None:
        exif = metadata_source.getexif()
        icc_profile = metadata_source.info.get("icc_profile")

    if exif:
        exif[274] = 1
        exif[40962] = composite_img.width
        exif[40963] = composite_img.height

    if gps_mode == "manual" and manual_lat is not None and manual_lon is not None:
        exif = set_exif_gps(exif, manual_lat, manual_lon)
    elif gps_mode == "remove":
        exif = remove_exif_gps(exif)

    buf = io.BytesIO()
    save_kwargs: Dict[str, Any] = {
        "format": "JPEG",
        "quality": quality,
        "subsampling": 0,
        "optimize": True,
    }

    if exif:
        save_kwargs["exif"] = exif.tobytes()
    if icc_profile is not None:
        save_kwargs["icc_profile"] = icc_profile

    composite_img.save(buf, **save_kwargs)
    return buf.getvalue()


# ============================================================
# Share store helpers
# ============================================================
@st.cache_resource(show_spinner=False)
def get_share_store() -> Dict[str, List[Dict[str, Any]]]:
    return {}


def now_ts() -> float:
    return time.time()


def password_key(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def cleanup_share_store(store: Dict[str, List[Dict[str, Any]]]) -> None:
    empty_keys = []
    current = now_ts()
    for key, items in store.items():
        store[key] = [item for item in items if item["expires_at"] is None or item["expires_at"] > current]
        if not store[key]:
            empty_keys.append(key)
    for key in empty_keys:
        del store[key]


def expiration_to_ts(mode: str, custom_minutes: int) -> Optional[float]:
    if mode == "계속 보관":
        return None
    preset = {
        "10분": 10,
        "30분": 30,
        "1시간": 60,
        "6시간": 360,
        "24시간": 1440,
    }
    minutes = preset.get(mode, max(1, int(custom_minutes)))
    return now_ts() + minutes * 60


def expires_label(expires_at: Optional[float]) -> str:
    if expires_at is None:
        return "앱 실행 중"
    remaining = int(expires_at - now_ts())
    if remaining <= 0:
        return "만료됨"
    if remaining < 60:
        return f"{remaining}초"
    if remaining < 3600:
        return f"{remaining // 60}분"
    return f"{remaining // 3600}시간 {(remaining % 3600) // 60}분"


def guess_mime(filename: str) -> str:
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def add_shared_bytes(
    storage_password: str,
    filename: str,
    data: bytes,
    expires_at: Optional[float],
    kind: str,
    mime: Optional[str] = None,
) -> Tuple[bool, str]:
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_SHARE_FILE_MB:
        return False, f"{filename}: {MAX_SHARE_FILE_MB}MB를 초과했습니다."

    store = get_share_store()
    cleanup_share_store(store)
    key = password_key(storage_password)
    file_id = hashlib.sha256(f"{filename}-{datetime.now(timezone.utc).isoformat()}".encode("utf-8")).hexdigest()[:16]

    store.setdefault(key, []).append(
        {
            "file_id": file_id,
            "name": filename,
            "size": len(data),
            "data": data,
            "mime": mime or guess_mime(filename),
            "kind": kind,
            "uploaded_at_ts": now_ts(),
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "expires_at": expires_at,
        }
    )
    return True, f"{filename}: 저장 완료"


def list_shared_files(storage_password: str) -> List[Dict[str, Any]]:
    store = get_share_store()
    cleanup_share_store(store)
    return list(store.get(password_key(storage_password), []))


def delete_shared_file(storage_password: str, file_id: str) -> None:
    store = get_share_store()
    key = password_key(storage_password)
    if key in store:
        store[key] = [item for item in store[key] if item["file_id"] != file_id]
        if not store[key]:
            del store[key]


def clear_shared_files(storage_password: str) -> None:
    store = get_share_store()
    key = password_key(storage_password)
    if key in store:
        del store[key]


def global_store_stats() -> Dict[str, Any]:
    store = get_share_store()
    cleanup_share_store(store)
    all_items = [item for items in store.values() for item in items]
    return {
        "stores": len(store),
        "files": len(all_items),
        "bytes": sum(item["size"] for item in all_items),
        "items": all_items,
    }


def upload_times_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["hour", "files"])
    df = pd.DataFrame([{"uploaded_at": pd.to_datetime(item["uploaded_at_ts"], unit="s", utc=True)} for item in items])
    df["hour"] = df["uploaded_at"].dt.floor("h")
    return df.groupby("hour").size().reset_index(name="files")


# ============================================================
# UI helpers
# ============================================================
def render_histogram(file_bytes: bytes, threshold: int) -> None:
    hist, bins = luminance_histogram(file_bytes)
    x = np.arange(256)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=hist, name="pixels", hovertemplate="brightness=%{x}<br>pixels=%{y}<extra></extra>"))
    fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"threshold {threshold}")
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Brightness",
        yaxis_title="Pixels",
        showlegend=False,
    )
    fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)


def gps_picker(default_lat: float, default_lon: float, key: str) -> Tuple[float, float]:
    lat_key = f"{key}_lat"
    lon_key = f"{key}_lon"

    if lat_key not in st.session_state:
        st.session_state[lat_key] = float(default_lat)
    if lon_key not in st.session_state:
        st.session_state[lon_key] = float(default_lon)

    col1, col2 = st.columns(2)
    with col1:
        st.session_state[lat_key] = st.number_input(
            "Latitude",
            -90.0,
            90.0,
            float(st.session_state[lat_key]),
            format="%.6f",
            key=f"{key}_lat_input",
        )
    with col2:
        st.session_state[lon_key] = st.number_input(
            "Longitude",
            -180.0,
            180.0,
            float(st.session_state[lon_key]),
            format="%.6f",
            key=f"{key}_lon_input",
        )

    lat = float(st.session_state[lat_key])
    lon = float(st.session_state[lon_key])

    if folium is not None and st_folium is not None:
        use_click_map = st.checkbox("지도에서 클릭으로 좌표 지정", value=False, key=f"{key}_use_click_map")

        if use_click_map:
            m = folium.Map(location=[lat, lon], zoom_start=13)
            folium.Marker([lat, lon], tooltip="Current GPS").add_to(m)

            # st_folium naturally reruns once when map interaction is sent back to Streamlit.
            # Do not call st.rerun() here; otherwise a click can feel like a rerun loop.
            result = st_folium(
                m,
                height=360,
                use_container_width=True,
                key=f"{key}_map",
                returned_objects=["last_clicked"],
            )
            clicked = result.get("last_clicked") if isinstance(result, dict) else None

            if clicked:
                clicked_lat = float(clicked["lat"])
                clicked_lon = float(clicked["lng"])

                if abs(clicked_lat - lat) > 1e-7 or abs(clicked_lon - lon) > 1e-7:
                    st.session_state[lat_key] = clicked_lat
                    st.session_state[lon_key] = clicked_lon
                    st.success(f"좌표가 선택되었습니다: {clicked_lat:.6f}, {clicked_lon:.6f}")
                    lat = clicked_lat
                    lon = clicked_lon
        else:
            st.map(pd.DataFrame([{"lat": lat, "lon": lon}]), latitude="lat", longitude="lon")
    else:
        st.map(pd.DataFrame([{"lat": lat, "lon": lon}]), latitude="lat", longitude="lon")
        st.caption("지도 클릭 지정은 streamlit-folium 설치 시 활성화됩니다. 현재는 수동 입력 좌표만 표시합니다.")

    return float(st.session_state[lat_key]), float(st.session_state[lon_key])


def render_file_preview(item: Dict[str, Any]) -> None:
    mime = item.get("mime", "application/octet-stream")
    data = item["data"]
    name = item["name"]
    if mime.startswith("image/"):
        st.image(data, caption=name, use_container_width=True)
    elif mime.startswith("text/") or name.lower().endswith((".txt", ".md", ".csv", ".json", ".py", ".toml")):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        st.text_area("Preview", value=text[:10000], height=180, disabled=True, key=f"preview_{item['file_id']}")
    else:
        st.write("Preview unavailable")


# ============================================================
# App gate
# ============================================================
password_gate("APP_PASSWORD", "app_unlocked", "App login")

st.title("Mask Tool")
page = st.radio("View", ["Mask", "Share"], horizontal=True, label_visibility="collapsed")

# ============================================================
# Page: Mask
# ============================================================
if page == "Mask":
    st.subheader("Threshold Mask")

    upload_col1, upload_col2 = st.columns(2)
    with upload_col1:
        mask_file = st.file_uploader("Mask image", type=SUPPORTED_TYPES, key="mask_uploader")
    with upload_col2:
        target_file = st.file_uploader("Target image", type=SUPPORTED_TYPES, key="target_uploader")

    if mask_file is None:
        st.info("Mask 이미지를 업로드해 주세요.")
        st.stop()

    mask_bytes = mask_file.getvalue()
    mask_raw, mask_img = load_image_pair(mask_bytes)

    with st.form("threshold_form"):
        threshold = st.slider("Threshold", 0, 255, DEFAULT_THRESHOLD, 1)
        apply_threshold = st.form_submit_button("Apply")

    render_histogram(mask_bytes, threshold)
    removed_vis, checker_preview = build_mask_previews(mask_bytes, threshold)

    preview_col1, preview_col2, preview_col3 = st.columns(3)
    with preview_col1:
        st.markdown("**Original mask**")
        st.image(mask_img, use_container_width=True)
    with preview_col2:
        st.markdown("**Removed area**")
        st.image(removed_vis, use_container_width=True)
    with preview_col3:
        st.markdown("**Transparent preview**")
        st.image(checker_preview, use_container_width=True)

    if target_file is None:
        st.info("Target 이미지를 업로드하면 최종 합성 결과가 표시됩니다.")
        st.stop()

    target_bytes = target_file.getvalue()
    target_raw, target_img = load_image_pair(target_bytes)
    composite_img = composite_mask_on_target_cached(mask_bytes, target_bytes, threshold)
    composite_png_bytes = pil_to_png_bytes(composite_img)

    st.markdown("### Result")
    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.markdown("**Target**")
        st.image(target_img, use_container_width=True)
    with result_col2:
        st.markdown("**Final**")
        st.image(composite_img, use_container_width=True)

    st.markdown("### Metadata & GPS")
    metadata_source_choice = st.radio("Metadata source", ["없음", "mask", "target"], horizontal=True, index=2)
    metadata_source_bytes = None
    metadata_source_raw = None
    if metadata_source_choice == "mask":
        metadata_source_bytes = mask_bytes
        metadata_source_raw = mask_raw
    elif metadata_source_choice == "target":
        metadata_source_bytes = target_bytes
        metadata_source_raw = target_raw

    source_gps = extract_gps_decimal(metadata_source_raw) if metadata_source_raw is not None else None
    gps_choice = st.radio("GPS", ["metadata 유지", "직접 지정", "GPS 제거"], horizontal=True, index=0)
    gps_mode = "keep"
    manual_lat = None
    manual_lon = None

    if gps_choice == "직접 지정":
        gps_mode = "manual"
        default_lat, default_lon = source_gps if source_gps else (37.5665, 126.9780)
        manual_lat, manual_lon = gps_picker(default_lat, default_lon, key="manual_gps")
    elif gps_choice == "GPS 제거":
        gps_mode = "remove"
    elif source_gps:
        st.map(pd.DataFrame([{"lat": source_gps[0], "lon": source_gps[1]}]), latitude="lat", longitude="lon")

    final_jpg_bytes = make_final_jpg_bytes(
        composite_png_bytes=composite_png_bytes,
        metadata_source_bytes=metadata_source_bytes,
        gps_mode=gps_mode,
        manual_lat=manual_lat,
        manual_lon=manual_lon,
        quality=JPEG_QUALITY,
    )

    st.download_button("Download final JPG", data=final_jpg_bytes, file_name="final_result.jpg", mime="image/jpeg")

    metadata_view = st.radio("Metadata table", ["숨김", "mask", "target"], horizontal=True, index=0)
    if metadata_view == "mask":
        rows = metadata_rows_from_bytes(mask_bytes)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif metadata_view == "target":
        rows = metadata_rows_from_bytes(target_bytes)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============================================================
# Page: Share
# ============================================================
else:
    password_gate("SHARE_PAGE_PASSWORD", "share_page_unlocked", "Share login", fallback_secret_name="APP_PASSWORD")

    st.subheader("Mini Share")
    with st.expander("Storage statistics", expanded=False):
        stats = global_store_stats()

        m1, m2, m3 = st.columns(3)
        m1.metric("Stores", f"{stats['stores']:,}")
        m2.metric("Files", f"{stats['files']:,}")
        m3.metric("Stored size", f"{stats['bytes'] / 1024 / 1024:.2f} MB")

        time_df = upload_times_df(stats["items"])
        if not time_df.empty:
            st.line_chart(time_df.set_index("hour"), y="files")
        else:
            st.write("No uploads yet")

    st.markdown("### Storage")
    with st.form("storage_key_form"):
        storage_password = st.text_input("Storage password", type="password")
        connect = st.form_submit_button("Open storage")
    if connect:
        st.session_state["current_storage_password"] = storage_password

    storage_password = st.session_state.get("current_storage_password", "")
    if not storage_password:
        st.stop()

    items = list_shared_files(storage_password)
    st.markdown("### Add files")

    exp_col1, exp_col2 = st.columns([2, 1])
    with exp_col1:
        expiration_mode = st.selectbox("Expiration", ["계속 보관", "10분", "30분", "1시간", "6시간", "24시간", "직접 입력"], index=0)
    with exp_col2:
        custom_minutes = st.number_input("Minutes", min_value=1, max_value=10080, value=60, step=1, disabled=expiration_mode != "직접 입력")
    expires_at = expiration_to_ts(expiration_mode, int(custom_minutes))

    upload_tab, text_tab = st.tabs(["Upload file", "Create text file"])

    with upload_tab:
        files = st.file_uploader("Files", accept_multiple_files=True, key="share_upload_files")
        if st.button("Save uploaded files"):
            if not files:
                st.error("파일을 선택해 주세요.")
            else:
                messages = []
                for uploaded in files:
                    ok, msg = add_shared_bytes(storage_password, uploaded.name, uploaded.getvalue(), expires_at, "upload", uploaded.type or guess_mime(uploaded.name))
                    messages.append(msg)
                st.success(" / ".join(messages))
                st.rerun()

    with text_tab:
        text_name_col, ext_col = st.columns([2, 1])
        with text_name_col:
            text_filename = st.text_input("Filename", value="note")
        with ext_col:
            ext_choice = st.selectbox("Extension", ["txt", "md", "json", "csv", "py", "toml", "직접 입력"], index=0)
        custom_ext = ""
        if ext_choice == "직접 입력":
            custom_ext = st.text_input("Custom extension", value="log")
        text_body = st.text_area("Content", height=220)
        if st.button("Save text file"):
            extension = custom_ext.strip().lstrip(".") if ext_choice == "직접 입력" else ext_choice
            filename = f"{text_filename.strip() or 'note'}.{extension or 'txt'}"
            data = text_body.encode("utf-8")
            ok, msg = add_shared_bytes(storage_password, filename, data, expires_at, "text", guess_mime(filename))
            st.success(msg)
            st.rerun()

    st.markdown("### Stored files")
    items = list_shared_files(storage_password)
    if not items:
        st.info("이 storage password로 저장된 파일이 없습니다.")
        st.stop()

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "filename": item["name"],
                    "kind": item.get("kind", "file"),
                    "size": f"{item['size']:,} bytes",
                    "mime": item.get("mime", ""),
                    "uploaded_at_utc": item["uploaded_at"],
                    "expires": expires_label(item["expires_at"]),
                }
                for item in items
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Clear this storage"):
        clear_shared_files(storage_password)
        st.rerun()

    for item in items:
        with st.expander(item["name"]):
            render_file_preview(item)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.download_button("Download", data=item["data"], file_name=item["name"], mime=item.get("mime", "application/octet-stream"), key=f"download_{item['file_id']}")
            with c2:
                if st.button("Delete", key=f"delete_{item['file_id']}"):
                    delete_shared_file(storage_password, item["file_id"])
                    st.rerun()
