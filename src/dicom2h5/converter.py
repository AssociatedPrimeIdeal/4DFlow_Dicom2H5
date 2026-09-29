"""Convert 4D flow MRI DICOM series to HDF5 files."""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pydicom
from tqdm import tqdm


# DICOM conversion is memory-heavy. Keep process pools bounded instead of
# letting ProcessPoolExecutor create one worker per host CPU core.
MAX_WORKERS = max(1, min(8, int(os.environ.get("DICOM2H5_MAX_WORKERS", "8"))))


def _pool_workers(total):
    return max(1, min(MAX_WORKERS, int(total or 0)))


AXIS_CODE_TO_NAME = {
    0: "LR",
    1: "AP",
    2: "FH",
}
AXIS_ORDER = (1, 2, 3)
MAGNITUDE_KEY = ("mag", 0.0)
AXIS_TO_LABEL = {1: "LR", 2: "AP", 3: "FH"}
DIRECTION_VECTORS_LPS = {
    "LR": np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
    "RL": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    "AP": np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
    "PA": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    "FH": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    "HF": np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
}


def get_filtered_dcm_files(path):
    dcm_files = []
    for root, dirs, files in os.walk(path):
        del dirs
        for file in files:
            is_dcm_ext = file.lower().endswith(".dcm")
            is_uid_style = all(c in "0123456789." for c in file) and any(c.isdigit() for c in file) and "." in file
            has_im_prefix = "IM" in file
            is_i_digits = file.startswith("I") and file[1:].isdigit()
            is_pure_num = all(c in "0123456789." for c in file) and any(c.isdigit() for c in file)
            if (is_dcm_ext or is_uid_style or has_im_prefix or is_i_digits or is_pure_num) and file != "DICOMDIR":
                dcm_files.append(os.path.join(root, file))
    return dcm_files


def check_manufacturer(dcm_files):
    for file in dcm_files:
        try:
            ds = pydicom.dcmread(file, stop_before_pixels=True)
            if hasattr(ds, "Manufacturer"):
                return ds.Manufacturer
        except Exception:
            continue
    return None


def get_image_orientation_patient(ds):
    if hasattr(ds, "ImageOrientationPatient") and ds.ImageOrientationPatient is not None:
        return np.array(ds.ImageOrientationPatient, dtype=np.float32)

    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared:
        try:
            orientation = shared[0].PlaneOrientationSequence[0].ImageOrientationPatient
            return np.array(orientation, dtype=np.float32)
        except Exception:
            pass

    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame:
        for frame in per_frame:
            try:
                orientation = frame.PlaneOrientationSequence[0].ImageOrientationPatient
                return np.array(orientation, dtype=np.float32)
            except Exception:
                continue
    return None


def get_resolution_from_ds(ds):
    if hasattr(ds, "PixelSpacing") and ds.PixelSpacing is not None:
        pix_spacing_y, pix_spacing_x = ds.PixelSpacing
        thickness = ds.SliceThickness if hasattr(ds, "SliceThickness") else None
        return (
            float(pix_spacing_x),
            float(pix_spacing_y),
            float(thickness) if thickness is not None else None,
        )

    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if shared:
        try:
            pm = shared[0].PixelMeasuresSequence[0]
            return (
                float(pm.PixelSpacing[0]),
                float(pm.PixelSpacing[1]),
                float(pm.SliceThickness),
            )
        except Exception:
            pass

    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if per_frame:
        for frame in per_frame:
            try:
                pm = frame.PixelMeasuresSequence[0]
                return (
                    float(pm.PixelSpacing[0]),
                    float(pm.PixelSpacing[1]),
                    float(pm.SliceThickness),
                )
            except Exception:
                continue
    return None


def dominant_axis_name(vec):
    vec = np.asarray(vec, dtype=np.float32)
    axis_idx = int(np.argmax(np.abs(vec)))
    axis_name = AXIS_CODE_TO_NAME[axis_idx]
    return axis_name[::-1] if vec[axis_idx] < 0 else axis_name


def infer_spatial_order_from_orientation(orientation):
    if orientation is None:
        return None
    orientation = np.asarray(orientation, dtype=np.float32)
    if orientation.size < 6:
        return None
    row_dir = dominant_axis_name(orientation[:3])
    col_dir = dominant_axis_name(orientation[3:6])
    slice_dir = dominant_axis_name(np.cross(orientation[:3], orientation[3:6]))
    return [row_dir, col_dir, slice_dir]


def _dominant_patient_direction(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    axis = int(np.argmax(np.abs(vector)))
    # Match the historical h5schema SpatialOrder convention.  This mapping is
    # for image-array directions; velocity vectors use their own metadata path.
    positive = ("RL", "AP", "FH")
    negative = ("LR", "PA", "HF")
    return (positive if vector[axis] >= 0 else negative)[axis]


def derive_spatial_metadata(files):
    """Derive stored-array spatial labels and geometry from DICOM headers."""
    headers = []
    for filename in files[: min(len(files), 256)]:
        try:
            headers.append(pydicom.dcmread(filename, stop_before_pixels=True, force=True))
        except Exception:
            continue

    identity = np.eye(3, dtype=np.float32)
    if not headers:
        return {
            "SpatialOrder": ("AP", "LR", "FH"),
            "Origin": np.zeros(3, dtype=np.float32),
            "ImageOrientationPatient": np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32),
            "SliceDirectionLPS": identity[:, 2],
            "RotationMatrix": identity,
        }

    first = headers[0]
    orientation = get_image_orientation_patient(first)
    if orientation is None or np.asarray(orientation).size < 6:
        spatial_order = ("AP", "LR", "FH")
        slice_normal = np.asarray([0.0, 0.0, 1.0])
        row_direction = np.asarray([0.0, 1.0, 0.0])
        col_direction = np.asarray([1.0, 0.0, 0.0])
        image_orientation = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    else:
        orientation = np.asarray(orientation, dtype=np.float64).reshape(-1)[:6]
        col_index_direction = orientation[:3]
        row_index_direction = orientation[3:6]
        slice_normal = np.cross(col_index_direction, row_index_direction)
        row_direction = row_index_direction
        col_direction = col_index_direction
        image_orientation = orientation.astype(np.float32)
        # Match h5schema: DICOM's first IOP vector is the column direction,
        # while the second is the row direction in the stored XY array.
        spatial_order = (
            _dominant_patient_direction(row_index_direction),
            _dominant_patient_direction(col_index_direction),
            _dominant_patient_direction(slice_normal),
        )

    positioned = []
    for header in headers:
        position = getattr(header, "ImagePositionPatient", None)
        if position is None:
            continue
        point = np.asarray(position, dtype=np.float64).reshape(3)
        try:
            location = float(getattr(header, "SliceLocation"))
            if not np.isfinite(location):
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            location = float(np.dot(point, slice_normal))
        positioned.append((location, point))

    origin = np.zeros(3, dtype=np.float64)
    if positioned:
        positioned.sort(key=lambda item: item[0])
        origin = positioned[0][1]
        if len(positioned) >= 2:
            displacement = positioned[-1][1] - positioned[0][1]
            if np.dot(displacement, slice_normal) < 0:
                spatial_order = (*spatial_order[:2], _dominant_patient_direction(-slice_normal))
                slice_normal = -slice_normal

    return {
        "SpatialOrder": tuple(spatial_order),
        "Origin": origin.astype(np.float32),
        "ImageOrientationPatient": image_orientation,
        "SliceDirectionLPS": np.asarray(slice_normal, dtype=np.float32),
        "RotationMatrix": np.stack(
            [row_direction, col_direction, slice_normal], axis=1
        ).astype(np.float32),
    }


def _patient_age_years(value):
    match = re.fullmatch(r"\s*(\d{3})([DWMY])\s*", str(value or "").upper())
    if not match:
        return np.nan
    amount = float(match.group(1))
    return amount / {"D": 365.25, "W": 52.1775, "M": 12.0, "Y": 1.0}[match.group(2)]


def extract_dicom_metadata(ds):
    """Return standard patient, institution, scanner, and series metadata."""
    def text(keyword):
        value = getattr(ds, keyword, "")
        return str(value or "").strip()

    def number(keyword):
        try:
            value = float(getattr(ds, keyword))
            return value if np.isfinite(value) else np.nan
        except (AttributeError, TypeError, ValueError):
            return np.nan

    patient = {
        "PatientName": text("PatientName"),
        "PatientID": text("PatientID"),
        "PatientAge": text("PatientAge").upper(),
        "PatientAgeYears": _patient_age_years(getattr(ds, "PatientAge", None)),
        "PatientSex": text("PatientSex").upper(),
        "PatientHeightM": number("PatientSize"),
        "PatientWeightKg": number("PatientWeight"),
        "PatientPosition": text("PatientPosition").upper(),
    }
    scanner = {
        key: text(key)
        for key in (
            "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
            "StationName", "Manufacturer", "ManufacturerModelName",
            "DeviceSerialNumber", "SoftwareVersions",
        )
    }
    acquisition = {
        key: text(key)
        for key in ("Modality", "BodyPartExamined", "StudyDescription", "SeriesDescription", "ProtocolName")
    }
    return {"patient": patient, "scanner": scanner, "acquisition": acquisition}


def _write_metadata_group(parent, name, values):
    group = parent.create_group(name)
    text_dtype = h5py.string_dtype(encoding="utf-8")
    for key, value in values.items():
        if isinstance(value, str):
            group.create_dataset(key, data=value, dtype=text_dtype)
        else:
            group.create_dataset(key, data=value)


def normalize_axis_name(dir_name):
    if dir_name is None:
        return None
    sorted_dir = "".join(sorted(dir_name.upper()))
    mapping = {
        "LR": "LR",
        "AP": "AP",
        "FH": "FH",
    }
    return mapping.get(sorted_dir)


def axis_name_to_nv(axis_name):
    mapping = {
        "LR": 1,
        "AP": 2,
        "FH": 3,
    }
    return mapping.get(axis_name)


def extract_rr_interval(ds):
    if hasattr(ds, "HeartRate"):
        heart_rate = float(ds.HeartRate)
        if heart_rate > 0:
            return 60000 / heart_rate
    if hasattr(ds, "CardiacRate"):
        cardiac_rate = float(ds.CardiacRate)
        if cardiac_rate > 0:
            return 60000 / cardiac_rate
    if hasattr(ds, "CardiacRRIntervalSpecified"):
        return float(ds.CardiacRRIntervalSpecified)
    if hasattr(ds, "ImageComments"):
        match = re.search(r"RR\s+(\d+)", str(ds.ImageComments))
        if match:
            return float(match.group(1))
    return None


def sort_key(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        return (0, float(value))
    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


def canonical_venc(value):
    if value is None:
        return None
    return round(abs(float(value)), 6)


def scale_phase_data(data_array, slope, intercept, venc):
    data = data_array * slope + intercept
    if venc is None or intercept in (None, 0):
        return data
    return data / intercept * venc


def get_siemens_extra_uid(ds):
    try:
        if (0x5200, 0x9230) not in ds:
            return ""
        fg0 = ds[(0x5200, 0x9230)][0]
        if (0x0021, 0x10FE) not in fg0:
            return ""
        priv0 = fg0[(0x0021, 0x10FE)][0]
        if (0x0021, 0x1056) not in priv0:
            return ""
        return str(priv0[(0x0021, 0x1056)].value)
    except Exception:
        return ""


def get_siemens_acquisition_id(ds):
    """Return Siemens' private acquisition identifier when consistently set."""
    values = {
        str(element.value).strip()
        for element in ds.iterall()
        if element.tag == (0x0021, 0x1060) and str(element.value).strip()
    }
    return next(iter(values)) if len(values) == 1 else ""


def get_referenced_series_uid(ds, series_index):
    return (
        ds.ReferencedImageEvidenceSequence[0]
        .ReferencedSeriesSequence[series_index]
        .ReferencedSOPSequence[0]
        .ReferencedSOPInstanceUID
    )


def get_single_frame_siemens_group_uid(ds):
    if (0x0021, 0x1060) in ds:
        return str(ds[(0x0021, 0x1060)].value)
    return str(ds.FrameOfReferenceUID)


def infer_siemens_axis(sequence_name, orientation):
    axis_name = None
    sequence_name_lower = sequence_name.lower()
    if any(dir_tag in sequence_name_lower for dir_tag in ["rl", "lr"]):
        axis_name = "LR"
    elif any(dir_tag in sequence_name_lower for dir_tag in ["ap", "pa"]):
        axis_name = "AP"
    elif any(dir_tag in sequence_name_lower for dir_tag in ["hf", "fh"]):
        axis_name = "FH"
    elif any(dir_tag in sequence_name_lower for dir_tag in ["in", "th"]):
        spatial_order = infer_spatial_order_from_orientation(orientation)
        if spatial_order is not None:
            axis_name = normalize_axis_name(spatial_order[2])
    return axis_name_to_nv(axis_name) if axis_name is not None else 0


def extract_siemens_venc(sequence_name):
    parts = sequence_name.split("_")
    if len(parts) > 1:
        matches = re.findall(r"\d+", parts[1])
        if matches:
            return float(matches[0])
    matches = re.findall(r"\d+", sequence_name)
    return float(matches[0]) if matches else None


UIH_FLOWQ_RE = re.compile(
    r"(?:^|[_\-\s])v(?P<venc>\d+)[_\-\s]+"
    r"(?P<mode>through|inplane)[_\-\s]+"
    r"(?P<direction>rl|lr|ap|pa|hf|fh)(?:$|[_\-\s])",
    re.IGNORECASE,
)
UIH_DIRECTION_TO_AXIS = {
    "RL": 1,
    "LR": 1,
    "AP": 2,
    "PA": 2,
    "HF": 3,
    "FH": 3,
}


def parse_uih_flowq_label(value, *, px_py_pz_series=False):
    """Read UIH FlowQ's signed direction and VENC from its private label.

    The uMR 790-style ``Px/Py/Pz`` export writes ``inplane_AP`` for the
    in-plane polarity that the H5/CVI convention represents as ``PA``.  Apply
    that polarity override only when the caller has identified the
    ``Px/Py/Pz`` series.  Other UIH FlowQ exports and legacy RO/PE/SS paths
    keep the direction written in the private label.
    """
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    text = str(value or "").strip()
    match = UIH_FLOWQ_RE.search(text)
    if match is None:
        return None
    raw_direction = match.group("direction").upper()
    direction = "PA" if px_py_pz_series and raw_direction == "AP" else raw_direction
    direction_text = raw_direction if raw_direction == direction else f"{raw_direction}->{direction}"
    direction_source = f"UIH FlowQ private direction: {match.group('mode').lower()}_{direction_text}"
    if px_py_pz_series and raw_direction == "AP":
        direction_source += " (Px/Py/Pz QC polarity override)"
    return {
        "axis": UIH_DIRECTION_TO_AXIS[direction],
        "direction_label": direction,
        "direction_source": direction_source,
        "venc": float(match.group("venc")),
    }


def direction_label_from_vector(value):
    """Convert a signed DICOM patient-LPS vector to an anatomical label."""
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size < 3 or not np.all(np.isfinite(vector[:3])):
        return None
    vector = vector[:3]
    axis = int(np.argmax(np.abs(vector)))
    if abs(float(vector[axis])) <= 0:
        return None
    # DICOM patient coordinates are LPS: +X=L, +Y=P, +Z=H.
    positive = ("RL", "PA", "FH")
    negative = ("LR", "AP", "HF")
    return (positive if vector[axis] >= 0 else negative)[axis]


def siemens_velocity_direction(ds, orientation, venc_direction, slope, intercept):
    """Return the anatomical direction of Siemens phase values after scaling."""
    modern_label = None
    modern_through = False
    legacy_label = None
    legacy_through = False
    for element in ds.iterall():
        value = element.value.decode(errors="replace") if isinstance(element.value, bytes) else str(element.value)
        if element.tag == (0x0021, 0x1129):
            match = re.search(
                r"v\d+[_-](?:inplane[_-](?P<direction>rl|lr|ap|pa|hf|fh)|through)",
                value,
                re.IGNORECASE,
            )
            if match:
                if match.group("direction"):
                    modern_label = match.group("direction").upper()
                else:
                    modern_through = True
        elif element.tag == (0x0021, 0x1029):
            match = re.search(
                r"v\d+[_-](?:inplane[_-](?P<direction>rl|lr|ap|pa|hf|fh)|through)",
                value,
                re.IGNORECASE,
            )
            if match:
                if match.group("direction"):
                    legacy_label = match.group("direction").upper()
                else:
                    legacy_through = True
        elif element.tag == (0x0021, 0x1077):
            match = re.search(r"v\d+(?P<direction>rl|lr|ap|pa|hf|fh|in)(?:$|[_-])", value, re.IGNORECASE)
            if match:
                direction = match.group("direction").upper()
                if direction == "IN":
                    legacy_through = True
                else:
                    legacy_label = direction

    if modern_label:
        label = modern_label
    elif modern_through:
        if orientation is None or np.asarray(orientation).size < 6:
            return None
        iop = np.asarray(orientation, dtype=np.float64).reshape(-1)[:6]
        label = _dominant_patient_direction(np.cross(iop[:3], iop[3:6]))
    elif legacy_label:
        label = legacy_label
    elif legacy_through:
        # XA exports use the old ``(0021,1029)``/``(0021,1077)`` labels.  In
        # the Vida export represented by ``wang_li_feng_a006012235`` the
        # DICOM velocity vector is anti-parallel to the image slice normal,
        # while the private ``v...through`` label denotes that normal.  Use
        # the IOP normal when the vector is actually the through-plane vector;
        # retain the vector fallback for legacy files whose vector is not
        # collinear with the image normal.
        if orientation is not None and np.asarray(orientation).size >= 6:
            iop = np.asarray(orientation, dtype=np.float64).reshape(-1)[:6]
            slice_normal = np.cross(iop[:3], iop[3:6])
            vector = np.asarray(venc_direction, dtype=np.float64).reshape(-1)
            if vector.size >= 3:
                normal_norm = np.linalg.norm(slice_normal)
                vector_norm = np.linalg.norm(vector[:3])
                if normal_norm > 0 and vector_norm > 0:
                    collinear = abs(float(np.dot(slice_normal, vector[:3]))) / (
                        normal_norm * vector_norm
                    )
                    if collinear >= 0.95:
                        label = _dominant_patient_direction(slice_normal)
                    else:
                        label = direction_label_from_vector(vector)
                else:
                    label = direction_label_from_vector(vector)
            else:
                label = direction_label_from_vector(venc_direction)
        else:
            label = direction_label_from_vector(venc_direction)
    else:
        label = direction_label_from_vector(venc_direction)

    if label is None:
        return None
    if intercept not in (None, 0) and float(slope) / float(intercept) < 0:
        label = {"RL": "LR", "LR": "RL", "AP": "PA", "PA": "AP", "FH": "HF", "HF": "FH"}[label]
    return label


def siemens_single_frame_velocity_direction(sequence_name, orientation, slope, intercept):
    """Resolve Siemens single-frame phase polarity from its sequence name."""
    value = str(sequence_name or "")
    match = re.search(r"v\d+(?:[_-]?(?:inplane[_-])?(?P<direction>rl|lr|ap|pa|hf|fh)|[_-]?(?P<through>in|through))(?:$|[_-])", value, re.IGNORECASE)
    if match is None:
        return None
    direction = match.group("direction")
    if direction:
        label = direction.upper()
    else:
        if orientation is None or np.asarray(orientation).size < 6:
            return None
        iop = np.asarray(orientation, dtype=np.float64).reshape(-1)[:6]
        # Siemens' single-frame ``...v050in`` polarity is opposite to the
        # image-plane normal used by the multiframe VENC vector.
        label = direction_label_from_vector(-np.cross(iop[:3], iop[3:6]))
    if intercept not in (None, 0) and float(slope) / float(intercept) < 0:
        label = {"RL": "LR", "LR": "RL", "AP": "PA", "PA": "AP", "FH": "HF", "HF": "FH"}[label]
    return label


def make_record(
    axis,
    venc,
    data,
    rr_interval,
    resolution,
    orientation,
    slice_location=None,
    trigger_time=None,
    direction_label=None,
    direction_source=None,
):
    axis = int(axis) if axis else 0
    venc = canonical_venc(venc) if axis else None
    if axis:
        channel_key = (axis, venc)
    else:
        channel_key = MAGNITUDE_KEY
    return {
        "channel_key": channel_key,
        "axis": axis,
        "venc": venc,
        "slice": slice_location,
        "time": trigger_time,
        "data": np.asarray(data),
        "rr": rr_interval,
        "resolution": tuple(resolution) if resolution is not None else None,
        "orientation": orientation,
        "direction_label": direction_label,
        "direction_source": direction_source,
    }


def check_file_core(file, manuf):
    try:
        ds = pydicom.dcmread(file)
        data_array = ds.pixel_array
        if len(data_array.shape) == 3:
            group_dcm = 1
            if "siemens" in manuf.lower():
                if hasattr(ds, "ImageType") and hasattr(ds, "ProtocolName") and hasattr(ds, "SeriesDescription"):
                    image_type = ds.ImageType
                    series_description = ds.SeriesDescription
                    if (
                        "4dflow" in series_description.lower()
                        or ("4d" in series_description.lower() and "flow" in series_description.lower())
                        or "flow" in series_description.lower()
                    ):
                        base_uid = get_referenced_series_uid(ds, 1)
                        extra_uid = get_siemens_extra_uid(ds)
                        acquisition_id = get_siemens_acquisition_id(ds)
                        group_uid = base_uid + extra_uid
                        if acquisition_id:
                            group_uid += f"#{acquisition_id}"
                        if image_type[2] == "VELOCITY" and "P" in series_description:
                            return file, group_uid, group_dcm
                        if image_type[2] == "T1" or image_type[2] == "ANGIO":
                            return file, group_uid, group_dcm
                elif hasattr(ds, "ImageType") and hasattr(ds, "PulseSequenceName") and hasattr(
                    ds, "ComplexImageComponent"
                ):
                    image_type = ds.ImageType
                    sequence_name = ds.PulseSequenceName
                    complex_image_component = ds.ComplexImageComponent
                    if "3d1r4" in sequence_name:
                        if image_type[2] == "VELOCITY" and "PHASE" in complex_image_component:
                            return file, get_referenced_series_uid(ds, 1), group_dcm
                        if image_type[2] == "T1":
                            return file, get_referenced_series_uid(ds, 1), group_dcm
                return None
            if "philips" in manuf.lower():
                group_dcm = 2
                if not hasattr(ds, "ProtocolName") or not hasattr(ds, "ImageType"):
                    return None
                image_type = ds.ImageType
                protocol_name = ds.ProtocolName
                if image_type[2] == "FLOW_ENCODED" and "WIP" in protocol_name and "2D" not in protocol_name:
                    return (
                        file,
                        ds.ReferencedImageEvidenceSequence[0]
                        .ReferencedSeriesSequence[0]
                        .ReferencedSOPSequence[0]
                        .ReferencedSOPInstanceUID,
                        group_dcm,
                    )
            if "ge" in manuf.lower() or "uih" in manuf.lower():
                return None
        elif len(data_array.shape) >= 2 and all(dim > 1 for dim in data_array.shape):
            group_dcm = 0
            if "siemens" in manuf.lower():
                if not hasattr(ds, "ImageType") or not hasattr(ds, "SequenceName") or not hasattr(ds, "ProtocolName"):
                    return None
                image_type = ds.ImageType
                protocol_name = ds.ProtocolName
                if "flow" in protocol_name.lower() and (image_type[2] == "P" or image_type[2] == "M"):
                    return file, get_single_frame_siemens_group_uid(ds), group_dcm
            elif "philips" in manuf.lower():
                if hasattr(ds, "ProtocolName") and hasattr(ds, "ImageType"):
                    if ds.ImageType[-3] != "M_PCA":
                        protocol_name = ds.ProtocolName
                        if "DelRec" in protocol_name:
                            if (ds.ImageType[-2] == "M" and "AP" in protocol_name) or ds.ImageType[-2] == "P":
                                return file, ds.ReferencedImageSequence[1].ReferencedSOPInstanceUID, group_dcm
                        elif "WIP" in protocol_name:
                            if (ds.ImageType[-2] == "M" and ds.ImageType[-1] != "PCA") or ds.ImageType[-2] == "P":
                                return file, ds.ReferencedImageSequence[1].ReferencedSOPInstanceUID, group_dcm
                        return None
                return None
            elif "ge" in manuf.lower():
                if not hasattr(ds, "SeriesDescription"):
                    return None
                series_description = ds.SeriesDescription
                if any(tag in series_description for tag in ["SI", "AP", "LR", "Anatomy"]):
                    return file, ds.FrameOfReferenceUID + ds.SeriesTime, group_dcm
            elif "uih" in manuf.lower():
                if (
                    not hasattr(ds, "ImageType")
                    or not hasattr(ds, "SequenceName")
                    or not hasattr(ds, "SeriesDescription")
                ):
                    return None
                sequence_name = ds.SequenceName
                series_description = ds.SeriesDescription
                # VENC scout images can share UIH's ``gre_fq`` sequence name
                # and FrameOfReferenceUID with the actual 4D-flow series,
                # while having a different matrix size. Keep them out of the
                # flow group so they cannot be mixed into one volume.
                if (
                    "fq" in sequence_name
                    and "MRA" not in series_description
                    and "vencscout" not in str(series_description).lower()
                ):
                    return file, ds.FrameOfReferenceUID, group_dcm
    except Exception:
        pass
    return None


def get_filtered_flow_dcm_files(dcm_files, manuf):
    grouped_files = {}
    group_dcms = {}

    with ProcessPoolExecutor(max_workers=_pool_workers(len(dcm_files))) as executor:
        results = list(tqdm(executor.map(check_file_core, dcm_files, [manuf] * len(dcm_files)), total=len(dcm_files)))

    for result in results:
        if result is None:
            continue
        file, frame_of_reference_uid, group_dcm = result
        grouped_files.setdefault(frame_of_reference_uid, []).append(file)
        group_dcms.setdefault(frame_of_reference_uid, []).append(group_dcm)

    if "siemens" in manuf.lower():
        for key in list(grouped_files.keys()):
            has_t1 = False
            for file in grouped_files[key]:
                try:
                    ds_tmp = pydicom.dcmread(file, stop_before_pixels=True)
                    if hasattr(ds_tmp, "ImageType") and len(ds_tmp.ImageType) > 2 and ds_tmp.ImageType[2] == "T1":
                        has_t1 = True
                        break
                except Exception:
                    pass
            if not has_t1:
                continue

            filtered_files = []
            for file in grouped_files[key]:
                try:
                    ds_tmp = pydicom.dcmread(file, stop_before_pixels=True)
                    if hasattr(ds_tmp, "ImageType") and len(ds_tmp.ImageType) > 2 and ds_tmp.ImageType[2] == "ANGIO":
                        continue
                except Exception:
                    pass
                filtered_files.append(file)
            grouped_files[key] = filtered_files

        # Some Siemens exports reuse one FrameOfReferenceUID for multiple
        # acquisitions with different matrices. Keep those separate before
        # stacking so one channel cannot become a ragged array.
        split_files = {}
        split_dcms = {}
        for key, files in grouped_files.items():
            by_shape = {}
            for filename in files:
                try:
                    header = pydicom.dcmread(filename, stop_before_pixels=True, force=True)
                    shape = (
                        int(getattr(header, "Rows")),
                        int(getattr(header, "Columns")),
                        int(getattr(header, "NumberOfFrames", 1)),
                    )
                except (AttributeError, TypeError, ValueError, OSError):
                    shape = (0, 0, 0)
                by_shape.setdefault(shape, []).append(filename)
            if len(by_shape) <= 1:
                split_files[key] = files
                split_dcms[key] = group_dcms[key]
                continue
            for shape, shape_files in sorted(by_shape.items(), key=lambda item: item[0]):
                shape_key = f"{key}#shape-{shape[0]}x{shape[1]}x{shape[2]}"
                split_files[shape_key] = shape_files
                split_dcms[shape_key] = group_dcms[key]
        grouped_files = split_files
        group_dcms = split_dcms

    for key in group_dcms:
        group_dcms[key] = int(np.median(np.asarray(group_dcms[key], dtype=np.float32)))

    return grouped_files, group_dcms


def deduplicate_flow_files(flow_dcm_files):
    deduplicated = {}
    for key, files in flow_dcm_files.items():
        unique_files = []
        seen_uids = set()
        for file in files:
            try:
                ds = pydicom.dcmread(file, stop_before_pixels=True)
                sop_instance_uid = getattr(ds, "SOPInstanceUID", file)
            except Exception:
                sop_instance_uid = file
            if sop_instance_uid in seen_uids:
                continue
            seen_uids.add(sop_instance_uid)
            unique_files.append(file)
        deduplicated[key] = unique_files
    return deduplicated


def check_flow_file_core(file, manuf):
    ds = pydicom.dcmread(file)
    data_array = ds.pixel_array
    rr_interval = extract_rr_interval(ds)
    resolution = get_resolution_from_ds(ds)
    orientation = get_image_orientation_patient(ds)

    if "siemens" in manuf.lower():
        image_type = ds.ImageType
        sequence_name = ds.SequenceName
        slice_location = getattr(ds, "SliceLocation", 0)
        trigger_time = getattr(ds, "TriggerTime", 0)
        slope = float(ds.RescaleSlope) if hasattr(ds, "RescaleSlope") else 1.0
        intercept = float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else 0.0
        axis = 0
        venc = None
        direction_label = None
        direction_source = None
        if image_type[2] == "P":
            venc = extract_siemens_venc(sequence_name)
            axis = infer_siemens_axis(sequence_name, orientation)
            direction_label = siemens_single_frame_velocity_direction(
                sequence_name, orientation, slope, intercept
            )
            if direction_label:
                axis = {"RL": 1, "LR": 1, "AP": 2, "PA": 2, "HF": 3, "FH": 3}[direction_label]
                direction_source = "Siemens single-frame private sequence direction + PixelValueTransformation"
        data = scale_phase_data(data_array, slope, intercept, venc) if axis else (data_array * slope + intercept)
        return make_record(
            axis, venc, data, rr_interval, resolution, orientation,
            slice_location, trigger_time, direction_label, direction_source,
        )

    if "philips" in manuf.lower():
        protocol_name = ds.ProtocolName
        image_type = ds.ImageType
        axis = 0
        slice_location = ds[(0x2001, 0x100A)].value
        trigger_time = ds[(0x2001, 0x1008)].value
        slope = float(ds.RescaleSlope) if hasattr(ds, "RescaleSlope") else 1.0
        intercept = float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else 0.0
        venc = None
        direction_label = None
        direction_source = None
        if image_type[-2] == "P":
            try:
                pc_velocity = ds[(0x2001, 0x101A)].value
            except (KeyError, TypeError, AttributeError):
                pc_velocity = None
            if pc_velocity is not None:
                pc_velocity = np.asarray(pc_velocity, dtype=np.float32).reshape(-1)
                if pc_velocity.size >= 3 and np.any(np.abs(pc_velocity[:3]) > 0):
                    axis = int(np.argmax(np.abs(pc_velocity[:3]))) + 1
                    direction_label = direction_label_from_vector(pc_velocity[:3])
                    direction_source = "Philips private (2001,101A) PC Velocity"
            if axis == 0:
                protocol_match = re.search(
                    r"(?:^|[^A-Z])(RL|LR|AP|PA|HF|FH)(?:$|[^A-Z])",
                    str(protocol_name).upper(),
                )
                if protocol_match:
                    direction_label = protocol_match.group(1)
                    axis = UIH_DIRECTION_TO_AXIS[direction_label]
                    direction_source = "Philips ProtocolName direction token"
            venc = intercept
            data = data_array * slope + intercept
            return make_record(
                axis, venc, data, rr_interval, resolution, orientation,
                slice_location, trigger_time, direction_label, direction_source,
            )
        if image_type[-2] == "M":
            # Philips exports the magnitude component as M_FFE/M_PCA while
            # velocity components use P/PCA. Keep magnitude in the native
            # channel so the assembled result has [mag, vx, vy, vz].
            data = data_array * slope + intercept
            return make_record(
                0, None, data, rr_interval, resolution, orientation,
                slice_location, trigger_time,
            )

    if "ge" in manuf.lower():
        series_description = str(ds.SeriesDescription)
        axis = 0
        slope = 1.0
        intercept = 0.0
        venc = None
        direction_label = None
        direction_source = None
        if "Anatomy" not in series_description:
            if "LR" in series_description:
                axis = 1
                # GE's classic 4D-flow series names use the encoded gradient
                # axis token. In this export convention, LR Flow is the +L
                # (RL) velocity polarity and SI Flow is the -H (HF) polarity.
                direction_label = "RL"
            elif "AP" in series_description:
                axis = 2
                direction_label = "AP"
            elif "SI" in series_description:
                axis = 3
                direction_label = "HF"
            direction_source = "GE SeriesDescription axis convention"
            slope = 1 / 10
            venc = ds[(0x0019, 0x10CC)].value / 10
        slice_location = getattr(ds, "SliceLocation", 0)
        trigger_time = getattr(ds, "TriggerTime", 0)
        data = data_array * slope + intercept
        return make_record(
            axis, venc, data, rr_interval, resolution, orientation,
            slice_location, trigger_time, direction_label, direction_source,
        )

    if "uih" in manuf.lower():
        series_description = ds.SeriesDescription
        # UIH uMR 790 exports velocity series as Px/Py/Pz. The original
        # converter only recognized RO/PE/SS and therefore treated all three
        # phase series as magnitude. FlowQ's private label contains both the
        # direction and VENC, for example ``v150_inplane_hf``.
        private_flow_label = ""
        try:
            private_flow_value = ds[(0x0065, 0x1012)].value
            if isinstance(private_flow_value, bytes):
                private_flow_label = private_flow_value.decode(errors="replace")
            else:
                private_flow_label = str(private_flow_value)
        except Exception:
            pass
        axis = 0
        venc = None
        direction_label = None
        direction_source = None
        px_py_pz_series = bool(
            re.search(r"(?:^|[_\-\s])p[xyz](?:$|[_\-\s])", str(series_description), re.IGNORECASE)
        )
        flowq = parse_uih_flowq_label(
            private_flow_label,
            px_py_pz_series=px_py_pz_series,
        )
        if flowq is not None:
            axis = flowq["axis"]
            direction_label = flowq["direction_label"]
            direction_source = flowq["direction_source"]
            venc = flowq["venc"]
        elif re.search(r"[_-]px(?:[_-]|$)", private_flow_label, re.IGNORECASE):
            axis = 3
            direction_label = "HF"
            direction_source = "UIH legacy private axis: px"
        elif re.search(r"[_-]py(?:[_-]|$)", private_flow_label, re.IGNORECASE):
            axis = 2
            direction_label = "AP"
            direction_source = "UIH legacy private axis: py"
        elif re.search(r"[_-]pz(?:[_-]|$)", private_flow_label, re.IGNORECASE):
            axis = 1
            direction_label = "RL"
            direction_source = "UIH legacy private axis: pz"
        elif "RO" in series_description:
            axis = 1
        elif "PE" in series_description:
            axis = 2
        elif "SS" in series_description:
            axis = 3
        if venc is None:
            match = re.search(
                r"(?:VENC\s*|v)(\d+)",
                f"{series_description} {private_flow_label}",
                re.IGNORECASE,
            )
            venc = int(match.group(1)) if match else None
        slope = float(ds.RescaleSlope) if hasattr(ds, "RescaleSlope") else 1.0
        intercept = float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else 0.0
        slice_location = getattr(ds, "SliceLocation", 0)
        trigger_time = getattr(ds, "TriggerTime", 0)
        data = data_array * slope + intercept
        return make_record(
            axis, venc, data, rr_interval, resolution, orientation,
            slice_location, trigger_time, direction_label, direction_source,
        )

    return None


def check_flow_file_core_group_dcm(file, manuf):
    ds = pydicom.dcmread(file)
    data_array = ds.pixel_array
    rr_interval = extract_rr_interval(ds)
    resolution = get_resolution_from_ds(ds)
    orientation = get_image_orientation_patient(ds)

    if "siemens" not in manuf.lower():
        return None

    image_type = ds.ImageType
    dsp = ds.PerFrameFunctionalGroupsSequence[-1]
    axis = 0
    venc = None
    direction_label = None
    direction_source = None
    if image_type[2] == "VELOCITY":
        venc = dsp.MRVelocityEncodingSequence[0].VelocityEncodingMaximumValue
        venc_dir = dsp.MRVelocityEncodingSequence[0].VelocityEncodingDirection
        axis = int(np.argmax(np.abs(venc_dir))) + 1
    slope = float(dsp.PixelValueTransformationSequence[0].RescaleSlope)
    intercept = float(dsp.PixelValueTransformationSequence[0].RescaleIntercept)
    if image_type[2] == "VELOCITY":
        direction_label = siemens_velocity_direction(
            ds, orientation, venc_dir, slope, intercept
        )
        if direction_label:
            axis = {"RL": 1, "LR": 1, "AP": 2, "PA": 2, "HF": 3, "FH": 3}[direction_label]
            direction_source = (
                "Siemens private encoding label + IOP/vector + PixelValueTransformation"
                if any(element.tag in ((0x0021, 0x1029), (0x0021, 0x1077), (0x0021, 0x1129)) for element in ds.iterall())
                else "DICOM velocity direction + PixelValueTransformation"
            )
    slice_location = dsp.FrameContentSequence[0].InStackPositionNumber
    data = scale_phase_data(data_array, slope, intercept, venc) if axis else (data_array * slope + intercept)
    return make_record(
        axis, venc, data, rr_interval, resolution, orientation,
        slice_location=slice_location,
        direction_label=direction_label,
        direction_source=direction_source,
    )


def check_flow_file_core_group_dcm2(file, manuf):
    ds = pydicom.dcmread(file)
    data_array = ds.pixel_array
    rr_interval = extract_rr_interval(ds)
    resolution = get_resolution_from_ds(ds)
    orientation = get_image_orientation_patient(ds)

    if "philips" not in manuf.lower():
        return None

    protocol_name = ds.ProtocolName
    dsp = ds.PerFrameFunctionalGroupsSequence[-1]
    axis = 0
    venc = None
    direction_label = None
    direction_source = None
    if "DelRec" in protocol_name:
        venc = dsp.MRVelocityEncodingSequence[0].VelocityEncodingMaximumValue
        venc_dir = dsp.MRVelocityEncodingSequence[0].VelocityEncodingDirection
        axis = int(np.argmax(np.abs(venc_dir))) + 1
        direction_label = direction_label_from_vector(venc_dir)
        direction_source = "DICOM MRVelocityEncodingSequence.VelocityEncodingDirection"

    spe = int(dsp.FrameContentSequence[0].InStackPositionNumber)
    if axis == 0:
        dsp = ds.PerFrameFunctionalGroupsSequence[0]
        slope = float(dsp.PixelValueTransformationSequence[0].RescaleSlope)
        intercept = float(dsp.PixelValueTransformationSequence[0].RescaleIntercept)
        data = data_array.reshape(2, spe, -1, data_array.shape[1], data_array.shape[2])[0]
    else:
        slope = float(dsp.PixelValueTransformationSequence[0].RescaleSlope)
        intercept = float(dsp.PixelValueTransformationSequence[0].RescaleIntercept)
        data = data_array.reshape(3, spe, -1, data_array.shape[1], data_array.shape[2])[-1]
    data = data * slope + intercept
    return make_record(
        axis, venc, data, rr_interval, resolution, orientation,
        direction_label=direction_label,
        direction_source=direction_source,
    )


def update_metadata(metadata, record):
    rr_interval = record.get("rr")
    if rr_interval is not None and rr_interval not in metadata["rr_values"]:
        metadata["rr_values"].append(rr_interval)

    resolution = record.get("resolution")
    if resolution is not None and tuple(resolution) not in metadata["resolutions"]:
        metadata["resolutions"].append(tuple(resolution))
    if record.get("channel_key") != MAGNITUDE_KEY and record.get("direction_label"):
        metadata["direction_labels"][record["channel_key"]] = record["direction_label"]
        metadata["direction_sources"][record["channel_key"]] = record.get(
            "direction_source", ""
        )


def build_single_frame_channels(records, metadata):
    channel_entries = {}
    for record in records:
        if record is None:
            continue
        update_metadata(metadata, record)
        channel_entries.setdefault(record["channel_key"], []).append((record["slice"], record["time"], record["data"]))

    channel_arrays = {}
    for key, entries in channel_entries.items():
        if not entries:
            continue
        slice_values = sorted({entry[0] for entry in entries}, key=sort_key)
        time_values = sorted({entry[1] for entry in entries}, key=sort_key)
        sample = np.asarray(entries[0][2])
        volume = np.zeros((len(slice_values), len(time_values), *sample.shape), dtype=sample.dtype)
        slice_index = {value: idx for idx, value in enumerate(slice_values)}
        time_index = {value: idx for idx, value in enumerate(time_values)}
        for slice_location, trigger_time, data in sorted(entries, key=lambda item: (sort_key(item[0]), sort_key(item[1]))):
            volume[slice_index[slice_location], time_index[trigger_time]] = data
        channel_arrays[key] = volume
    return channel_arrays


def build_multiframe_slice_channels(records, metadata):
    channel_entries = {}
    for record in records:
        if record is None:
            continue
        update_metadata(metadata, record)
        channel_entries.setdefault(record["channel_key"], []).append((record["slice"], record["data"]))

    channel_arrays = {}
    for key, entries in channel_entries.items():
        if not entries:
            continue
        ordered_entries = sorted(entries, key=lambda item: sort_key(item[0]))
        channel_arrays[key] = np.asarray([entry[1] for entry in ordered_entries])

    if channel_arrays:
        min_slices = min(array.shape[0] for array in channel_arrays.values())
        channel_arrays = {key: array[:min_slices] for key, array in channel_arrays.items()}
    return channel_arrays


def build_multiframe_volume_channels(records, metadata):
    channel_entries = {}
    for record in records:
        if record is None:
            continue
        update_metadata(metadata, record)
        channel_entries.setdefault(record["channel_key"], []).append(record["data"])

    channel_arrays = {}
    for key, entries in channel_entries.items():
        if not entries:
            continue
        channel_arrays[key] = np.asarray(entries[0])
    return channel_arrays


def build_channel_layout(channel_arrays):
    axis_to_vencs = {axis: [] for axis in AXIS_ORDER}
    for key in channel_arrays:
        if key == MAGNITUDE_KEY:
            continue
        axis, venc = key
        if axis in axis_to_vencs and venc not in axis_to_vencs[axis]:
            axis_to_vencs[axis].append(venc)

    for axis in AXIS_ORDER:
        axis_to_vencs[axis] = sorted(axis_to_vencs[axis], key=sort_key)

    ordered_keys = []
    if MAGNITUDE_KEY in channel_arrays:
        ordered_keys.append(MAGNITUDE_KEY)

    venc_values = []
    max_levels = max((len(values) for values in axis_to_vencs.values()), default=0)
    for level_idx in range(max_levels):
        for axis in AXIS_ORDER:
            axis_vencs = axis_to_vencs[axis]
            if level_idx >= len(axis_vencs):
                venc_values.append(np.nan)
                continue
            key = (axis, axis_vencs[level_idx])
            venc_values.append(float(axis_vencs[level_idx]))
            if key in channel_arrays:
                ordered_keys.append(key)

    if not ordered_keys:
        ordered_keys = list(channel_arrays.keys())

    return ordered_keys, venc_values


def stack_flow_data(channel_arrays, ordered_keys):
    stacked = np.stack([channel_arrays[key] for key in ordered_keys], axis=0)
    return np.transpose(stacked, (3, 4, 1, 2, 0))


def get_flow_data(flow_dcm_files, manuf, group_dcm, return_metadata=False):
    metadata = {
        "rr_values": [],
        "resolutions": [],
        "direction_labels": {},
        "direction_sources": {},
    }

    if group_dcm == 0:
        with ProcessPoolExecutor(max_workers=_pool_workers(len(flow_dcm_files))) as executor:
            records = list(
                tqdm(
                    executor.map(check_flow_file_core, flow_dcm_files, [manuf] * len(flow_dcm_files)),
                    total=len(flow_dcm_files),
                )
            )
        channel_arrays = build_single_frame_channels(records, metadata)
    elif group_dcm == 1:
        with ProcessPoolExecutor(max_workers=_pool_workers(len(flow_dcm_files))) as executor:
            records = list(
                tqdm(
                    executor.map(check_flow_file_core_group_dcm, flow_dcm_files, [manuf] * len(flow_dcm_files)),
                    total=len(flow_dcm_files),
                )
            )
        channel_arrays = build_multiframe_slice_channels(records, metadata)
    elif group_dcm == 2:
        with ProcessPoolExecutor(max_workers=_pool_workers(len(flow_dcm_files))) as executor:
            records = list(
                tqdm(
                    executor.map(check_flow_file_core_group_dcm2, flow_dcm_files, [manuf] * len(flow_dcm_files)),
                    total=len(flow_dcm_files),
                )
            )
        channel_arrays = build_multiframe_volume_channels(records, metadata)
    else:
        raise ValueError(f"Unsupported GroupDCM value: {group_dcm}")

    if not channel_arrays:
        empty_resolution = (None, None, None)
        empty = (np.array([]), None, empty_resolution, [])
        return (*empty, {}) if return_metadata else empty

    ordered_keys, venc_values = build_channel_layout(channel_arrays)
    flow_data = stack_flow_data(channel_arrays, ordered_keys)
    final_rr = metadata["rr_values"][0] if metadata["rr_values"] else None
    final_resolution = metadata["resolutions"][0] if metadata["resolutions"] else (None, None, None)
    direction_labels = []
    direction_sources = []
    direction_vectors = []
    for key in ordered_keys:
        if key == MAGNITUDE_KEY:
            continue
        axis, _ = key
        label = metadata["direction_labels"].get(key, AXIS_TO_LABEL.get(axis, ""))
        direction_labels.append(label)
        direction_sources.append(
            metadata["direction_sources"].get(key, "Dicom2H5 numeric axis")
        )
        direction_vectors.append(
            DIRECTION_VECTORS_LPS.get(
                label, DIRECTION_VECTORS_LPS.get(AXIS_TO_LABEL.get(axis, "LR"))
            )
        )
    velocity_metadata = {
        "VENCOrder": direction_labels,
        "VelocityDirectionsLPS": np.asarray(direction_vectors, dtype=np.float32),
        "VelocityDirectionPolarityKnown": [
            bool(key in metadata["direction_labels"])
            for key in ordered_keys
            if key != MAGNITUDE_KEY
        ],
        "VelocityDirectionSource": direction_sources,
    }
    result = (flow_data, final_rr, final_resolution, venc_values)
    return (*result, velocity_metadata) if return_metadata else result


def write_h5_group(
    h5_file,
    key,
    flow_data,
    rr_interval,
    resolution,
    venc_values,
    venc_order=None,
    spatial_metadata=None,
    dicom_metadata=None,
):
    flow_data = np.asarray(flow_data)
    if flow_data.ndim != 5 or flow_data.shape[-1] != 4:
        raise ValueError(f"native mag/flow contract requires XYZT4 assembled data, got {flow_data.shape!r}")
    group = h5_file.create_group(str(key))
    text_dtype = h5py.string_dtype(encoding="utf-8")
    # Native H5 keeps the layout AutoFlow already consumes: magnitude and
    # velocity are separate datasets.  The velocity channels are in cm/s;
    # H52Dicom converts them to phase only when writing DICOM pixels.
    group.create_dataset("mag", data=np.asarray(flow_data[..., 0], dtype=np.float32))
    group.create_dataset("flow", data=np.asarray(flow_data[..., 1:4], dtype=np.float32))
    group.create_dataset("RR", data=rr_interval if rr_interval is not None else np.nan)

    if resolution and all(value is not None for value in resolution):
        group.create_dataset("Resolution", data=np.array(resolution, dtype=np.float32))
    else:
        group.create_dataset("Resolution", data=np.array([np.nan, np.nan, np.nan], dtype=np.float32))

    if venc_values:
        group.create_dataset("VENC", data=np.array(venc_values, dtype=np.float32))
    else:
        group.create_dataset("VENC", data=np.array([np.nan, np.nan, np.nan], dtype=np.float32))
    if venc_order is not None:
        group.create_dataset("VENCOrder", data=np.asarray(venc_order, dtype=object), dtype=text_dtype)
    if spatial_metadata:
        group.create_dataset("SpatialOrder", data=np.asarray(spatial_metadata["SpatialOrder"], dtype=object), dtype=text_dtype)
        for key in ("Origin", "ImageOrientationPatient", "SliceDirectionLPS", "RotationMatrix"):
            group.create_dataset(key, data=spatial_metadata[key])
    if dicom_metadata:
        for name in ("patient", "scanner", "acquisition"):
            _write_metadata_group(group, name, dicom_metadata.get(name, {}))


def convert_dicom_to_h5(dicom_path, data_save_path, *, dicom_files=None, case_id=None):
    """Convert a DICOM tree (or an explicit file list) to canonical native H5.

    Each sequence group contains native image data, velocity/spatial order,
    geometry, and standard patient/scanner/acquisition metadata.
    """
    dcm_files = list(dicom_files) if dicom_files is not None else get_filtered_dcm_files(dicom_path)
    manuf = check_manufacturer(dcm_files)
    if not manuf:
        raise ValueError("Could not determine manufacturer from DICOM files.")

    flow_dcm_files, group_dcms = get_filtered_flow_dcm_files(dcm_files, manuf)
    flow_dcm_files = deduplicate_flow_files(flow_dcm_files)
    print(f"Found {len(flow_dcm_files)} different sequence groups from {manuf}")

    with h5py.File(data_save_path, "w") as h5_file:
        if case_id:
            h5_file.attrs["case_id"] = str(case_id)
        for key in list(flow_dcm_files.keys()):
            if not flow_dcm_files[key]:
                continue
            print(f"UID: {key}, File Nums: {len(flow_dcm_files[key])}")
            try:
                result = get_flow_data(
                    flow_dcm_files[key],
                    manuf,
                    group_dcms[key],
                    return_metadata=True,
                )
            except (ValueError, KeyError, IndexError) as exc:
                # A case can contain scout/2D groups with incompatible
                # matrices. Keep processing independent groups so one bad
                # auxiliary sequence does not discard valid 4D-flow data.
                print(f"Skipping invalid sequence group {key}: {exc}")
                continue
            flow_data, rr_interval, resolution, venc_values, velocity_metadata = result
            header = None
            for filename in flow_dcm_files[key]:
                try:
                    header = pydicom.dcmread(filename, stop_before_pixels=True, force=True)
                    break
                except Exception:
                    continue
            spatial_metadata = derive_spatial_metadata(flow_dcm_files[key])
            dicom_metadata = extract_dicom_metadata(header or pydicom.dataset.Dataset())
            dicom_metadata["scanner"]["Manufacturer"] = (
                dicom_metadata["scanner"].get("Manufacturer") or str(manuf)
            )
            print(
                f"UID: {key}, Data Shape: {flow_data.shape}, RR: {rr_interval}, "
                f"Resolution: {resolution}, VENC: {venc_values}"
            )
            if flow_data.size > 0:
                try:
                    write_h5_group(
                        h5_file,
                        key,
                        flow_data,
                        rr_interval,
                        resolution,
                        venc_values,
                        venc_order=velocity_metadata.get("VENCOrder"),
                        spatial_metadata=spatial_metadata,
                        dicom_metadata=dicom_metadata,
                    )
                except (ValueError, KeyError, IndexError) as exc:
                    print(f"Skipping invalid assembled group {key}: {exc}")

    return data_save_path


def native_group_names(data_path):
    """Return H5 groups containing the native ``mag``/``flow`` datasets."""
    names = []
    with h5py.File(data_path, "r") as handle:
        if isinstance(handle, h5py.Group) and "mag" in handle and "flow" in handle:
            names.append("")

        def visit(name, obj):
            if isinstance(obj, h5py.Group) and "mag" in obj and "flow" in obj:
                names.append(str(name).strip("/"))

        handle.visititems(visit)
    return sorted(set(names), key=lambda value: (value.count("/"), value))


def validate_native_h5(data_path):
    """Validate the canonical native H5 contract without rewriting it."""
    errors = []
    groups = native_group_names(data_path)
    required = (
        "mag", "flow", "RR", "Resolution", "VENC", "VENCOrder", "SpatialOrder",
        "Origin", "ImageOrientationPatient", "SliceDirectionLPS", "RotationMatrix",
        "patient", "scanner", "acquisition",
    )
    with h5py.File(data_path, "r") as handle:
        for name in groups:
            group = handle if not name else handle[name]
            missing = [key for key in required if key not in group]
            if missing:
                errors.append(f"{name or '<root>'}: missing {', '.join(missing)}")
                continue
            mag_shape = tuple(group["mag"].shape)
            flow_shape = tuple(group["flow"].shape)
            if len(mag_shape) != 4:
                errors.append(f"{name or '<root>'}: mag must be XYZT, got {mag_shape}")
            if len(flow_shape) != 5 or flow_shape[:4] != mag_shape or flow_shape[-1] != 3:
                errors.append(f"{name or '<root>'}: flow must be XYZT3, got {flow_shape}")
            if np.asarray(group["VENC"]).reshape(-1).size != 3:
                errors.append(f"{name or '<root>'}: VENC must contain 3 values")
            if np.asarray(group["VENCOrder"]).reshape(-1).size != 3:
                errors.append(f"{name or '<root>'}: VENCOrder must contain 3 values")
            if np.asarray(group["Resolution"]).reshape(-1).size != 3:
                errors.append(f"{name or '<root>'}: Resolution must contain 3 values")
            if np.asarray(group["SpatialOrder"]).reshape(-1).size != 3:
                errors.append(f"{name or '<root>'}: SpatialOrder must contain 3 values")
            if np.asarray(group["ImageOrientationPatient"]).reshape(-1).size != 6:
                errors.append(f"{name or '<root>'}: ImageOrientationPatient must contain 6 values")
            if np.asarray(group["Origin"]).reshape(-1).size != 3:
                errors.append(f"{name or '<root>'}: Origin must contain 3 values")
            if tuple(group["RotationMatrix"].shape) != (3, 3):
                errors.append(f"{name or '<root>'}: RotationMatrix must be 3x3")
            for subgroup in ("patient", "scanner", "acquisition"):
                if subgroup not in group or not isinstance(group[subgroup], h5py.Group):
                    errors.append(f"{name or '<root>'}: missing {subgroup} metadata group")
    if not groups:
        errors.append("no native mag/flow group found")
    return {"valid": not errors, "errors": errors, "groups": groups}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="4D Flow DICOM to HDF5")
    parser.add_argument(
        "--dicom-path",
        "--dicom_path",
        dest="dicom_path",
        type=str,
        default="./",
        help="Path to the DICOM directory.",
    )
    parser.add_argument(
        "--data-save-path",
        "--data_save_path",
        dest="data_save_path",
        type=str,
        default="./dcmarray.h5",
        help="Path to the output HDF5 file.",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        convert_dicom_to_h5(args.dicom_path, args.data_save_path)
    except ValueError as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")
    return 0
