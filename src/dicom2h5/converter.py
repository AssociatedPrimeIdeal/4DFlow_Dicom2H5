"""Convert 4D flow MRI DICOM series to HDF5 files."""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pydicom
from tqdm import tqdm


AXIS_CODE_TO_NAME = {
    0: "LR",
    1: "AP",
    2: "FH",
}
AXIS_ORDER = (1, 2, 3)
MAGNITUDE_KEY = ("mag", 0.0)


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


def venc_axis_to_name(nv):
    mapping = {
        1: "LR",
        2: "AP",
        3: "FH",
    }
    return mapping.get(int(nv))


def create_string_dataset(group, name, values):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.array(values, dtype=object), dtype=string_dtype)


def extract_rr_interval(ds):
    if hasattr(ds, "HeartRate"):
        return 60000 / float(ds.HeartRate)
    if hasattr(ds, "CardiacRate"):
        return 60000 / float(ds.CardiacRate)
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


def make_record(axis, venc, data, rr_interval, resolution, orientation, slice_location=None, trigger_time=None):
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
                        if image_type[2] == "VELOCITY" and "P" in series_description:
                            return file, base_uid + extra_uid, group_dcm
                        if image_type[2] == "T1" or image_type[2] == "ANGIO":
                            return file, base_uid + extra_uid, group_dcm
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
                if "fq" in sequence_name and "MRA" not in series_description:
                    return file, ds.FrameOfReferenceUID, group_dcm
    except Exception:
        pass
    return None


def get_filtered_flow_dcm_files(dcm_files, manuf):
    grouped_files = {}
    group_dcms = {}

    with ProcessPoolExecutor() as executor:
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
        if image_type[2] == "P":
            venc = extract_siemens_venc(sequence_name)
            axis = infer_siemens_axis(sequence_name, orientation)
        data = scale_phase_data(data_array, slope, intercept, venc) if axis else (data_array * slope + intercept)
        return make_record(axis, venc, data, rr_interval, resolution, orientation, slice_location, trigger_time)

    if "philips" in manuf.lower():
        protocol_name = ds.ProtocolName
        image_type = ds.ImageType
        axis = 0
        slice_location = ds[(0x2001, 0x100A)].value
        trigger_time = ds[(0x2001, 0x1008)].value
        slope = float(ds.RescaleSlope) if hasattr(ds, "RescaleSlope") else 1.0
        intercept = float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else 0.0
        venc = None
        if image_type[-2] == "P":
            if any(dir_tag in protocol_name for dir_tag in ["RL", "LR"]):
                axis = 1
            elif any(dir_tag in protocol_name for dir_tag in ["AP", "PA"]):
                axis = 2
            elif any(dir_tag in protocol_name for dir_tag in ["HF", "FH"]):
                axis = 3
            venc = intercept
        data = data_array * slope + intercept
        return make_record(axis, venc, data, rr_interval, resolution, orientation, slice_location, trigger_time)

    if "ge" in manuf.lower():
        series_description = ds.SeriesDescription
        axis = 0
        slope = 1.0
        intercept = 0.0
        venc = None
        if "Anatomy" not in series_description:
            if "LR" in series_description:
                axis = 1
            elif "AP" in series_description:
                axis = 2
            elif "SI" in series_description:
                axis = 3
            slope = 1 / 10
            venc = ds[(0x0019, 0x10CC)].value / 10
        slice_location = getattr(ds, "SliceLocation", 0)
        trigger_time = getattr(ds, "TriggerTime", 0)
        data = data_array * slope + intercept
        return make_record(axis, venc, data, rr_interval, resolution, orientation, slice_location, trigger_time)

    if "uih" in manuf.lower():
        series_description = ds.SeriesDescription
        axis = 0
        venc = None
        if "RO" in series_description:
            axis = 1
            match = re.search(r"VENC\s*(\d+)", series_description)
            venc = int(match.group(1)) if match else None
        elif "PE" in series_description:
            axis = 2
            match = re.search(r"VENC\s*(\d+)", series_description)
            venc = int(match.group(1)) if match else None
        elif "SS" in series_description:
            axis = 3
            match = re.search(r"VENC\s*(\d+)", series_description)
            venc = int(match.group(1)) if match else None
        slope = float(ds.RescaleSlope) if hasattr(ds, "RescaleSlope") else 1.0
        intercept = float(ds.RescaleIntercept) if hasattr(ds, "RescaleIntercept") else 0.0
        slice_location = getattr(ds, "SliceLocation", 0)
        trigger_time = getattr(ds, "TriggerTime", 0)
        data = data_array * slope + intercept
        return make_record(axis, venc, data, rr_interval, resolution, orientation, slice_location, trigger_time)

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
    if image_type[2] == "VELOCITY":
        venc = dsp.MRVelocityEncodingSequence[0].VelocityEncodingMaximumValue
        venc_dir = dsp.MRVelocityEncodingSequence[0].VelocityEncodingDirection
        axis = int(np.argmax(np.abs(venc_dir))) + 1
    slope = float(dsp.PixelValueTransformationSequence[0].RescaleSlope)
    intercept = float(dsp.PixelValueTransformationSequence[0].RescaleIntercept)
    slice_location = dsp.FrameContentSequence[0].InStackPositionNumber
    data = scale_phase_data(data_array, slope, intercept, venc) if axis else (data_array * slope + intercept)
    return make_record(axis, venc, data, rr_interval, resolution, orientation, slice_location=slice_location)


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
    if "DelRec" in protocol_name:
        venc = dsp.MRVelocityEncodingSequence[0].VelocityEncodingMaximumValue
        venc_dir = dsp.MRVelocityEncodingSequence[0].VelocityEncodingDirection
        axis = int(np.argmax(np.abs(venc_dir))) + 1

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
    return make_record(axis, venc, data, rr_interval, resolution, orientation)


def update_metadata(metadata, record):
    rr_interval = record.get("rr")
    if rr_interval is not None and rr_interval not in metadata["rr_values"]:
        metadata["rr_values"].append(rr_interval)

    resolution = record.get("resolution")
    if resolution is not None and tuple(resolution) not in metadata["resolutions"]:
        metadata["resolutions"].append(tuple(resolution))

    if metadata["spatial_order"] is None:
        metadata["spatial_order"] = infer_spatial_order_from_orientation(record.get("orientation"))


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

    venc_order = [venc_axis_to_name(axis) or "" for axis in AXIS_ORDER]
    return ordered_keys, venc_order, venc_values


def stack_flow_data(channel_arrays, ordered_keys):
    stacked = np.stack([channel_arrays[key] for key in ordered_keys], axis=0)
    return np.transpose(stacked, (3, 4, 1, 2, 0))


def get_flow_data(flow_dcm_files, manuf, group_dcm):
    metadata = {
        "rr_values": [],
        "resolutions": [],
        "spatial_order": None,
    }

    if group_dcm == 0:
        with ProcessPoolExecutor() as executor:
            records = list(
                tqdm(
                    executor.map(check_flow_file_core, flow_dcm_files, [manuf] * len(flow_dcm_files)),
                    total=len(flow_dcm_files),
                )
            )
        channel_arrays = build_single_frame_channels(records, metadata)
    elif group_dcm == 1:
        with ProcessPoolExecutor() as executor:
            records = list(
                tqdm(
                    executor.map(check_flow_file_core_group_dcm, flow_dcm_files, [manuf] * len(flow_dcm_files)),
                    total=len(flow_dcm_files),
                )
            )
        channel_arrays = build_multiframe_slice_channels(records, metadata)
    elif group_dcm == 2:
        with ProcessPoolExecutor() as executor:
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
        return np.array([]), None, empty_resolution, metadata["spatial_order"], ["", "", ""], []

    ordered_keys, venc_order, venc_values = build_channel_layout(channel_arrays)
    flow_data = stack_flow_data(channel_arrays, ordered_keys)
    final_rr = metadata["rr_values"][0] if metadata["rr_values"] else None
    final_resolution = metadata["resolutions"][0] if metadata["resolutions"] else (None, None, None)
    return flow_data, final_rr, final_resolution, metadata["spatial_order"], venc_order, venc_values


def write_h5_group(h5_file, key, flow_data, rr_interval, resolution, spatial_order, venc_order, venc_values):
    group = h5_file.create_group(str(key))
    group.create_dataset("img", data=flow_data)
    group.create_dataset("RR", data=rr_interval if rr_interval is not None else np.nan)

    if resolution and all(value is not None for value in resolution):
        group.create_dataset("Resolution", data=np.array(resolution, dtype=np.float32))
    else:
        group.create_dataset("Resolution", data=np.array([np.nan, np.nan, np.nan], dtype=np.float32))

    if venc_values:
        group.create_dataset("VENC", data=np.array(venc_values, dtype=np.float32))
    else:
        group.create_dataset("VENC", data=np.array([np.nan, np.nan, np.nan], dtype=np.float32))

    create_string_dataset(group, "SpatialOrder", spatial_order if spatial_order is not None else ["", "", ""])
    create_string_dataset(group, "VENCOrder", venc_order if venc_order is not None else ["", "", ""])


def convert_dicom_to_h5(dicom_path, data_save_path):
    dcm_files = get_filtered_dcm_files(dicom_path)
    manuf = check_manufacturer(dcm_files)
    if not manuf:
        raise ValueError("Could not determine manufacturer from DICOM files.")

    flow_dcm_files, group_dcms = get_filtered_flow_dcm_files(dcm_files, manuf)
    flow_dcm_files = deduplicate_flow_files(flow_dcm_files)
    print(f"Found {len(flow_dcm_files)} different sequence groups from {manuf}")

    with h5py.File(data_save_path, "w") as h5_file:
        for key in list(flow_dcm_files.keys()):
            if not flow_dcm_files[key]:
                continue
            print(f"UID: {key}, File Nums: {len(flow_dcm_files[key])}")
            flow_data, rr_interval, resolution, spatial_order, venc_order, venc_values = get_flow_data(
                flow_dcm_files[key],
                manuf,
                group_dcms[key],
            )
            print(
                f"UID: {key}, Data Shape: {flow_data.shape}, RR: {rr_interval}, "
                f"Resolution: {resolution}, VENC: {venc_values}, "
                f"SpatialOrder: {spatial_order}, VENCOrder: {venc_order}"
            )
            if flow_data.size > 0:
                write_h5_group(
                    h5_file,
                    key,
                    flow_data,
                    rr_interval,
                    resolution,
                    spatial_order,
                    venc_order,
                    venc_values,
                )

    return data_save_path


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
