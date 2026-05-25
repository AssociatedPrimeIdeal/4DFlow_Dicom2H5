"""Convert 4D flow MRI DICOM series to HDF5 files."""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pydicom
from tqdm import tqdm


def get_filtered_dcm_files(path):
    dcm_files = []
    for root, dirs, files in os.walk(path):
        del dirs
        for file in files:
            if (file.endswith(".dcm") or "." not in file or file.endswith(".DCM")) and file != "DICOMDIR":
                dcm_files.append(os.path.join(root, file))
    return dcm_files


def check_manufacturer(dcm_files):
    for file in dcm_files:
        try:
            ds = pydicom.dcmread(file)
            if hasattr(ds, "Manufacturer"):
                return ds.Manufacturer
        except Exception:
            continue
    return None


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
                    if "4dflow" in series_description.lower():
                        if image_type[2] == "VELOCITY" and "P" in series_description:
                            return (
                                file,
                                ds.ReferencedImageEvidenceSequence[0]
                                .ReferencedSeriesSequence[0]
                                .ReferencedSOPSequence[0]
                                .ReferencedSOPInstanceUID,
                                group_dcm,
                            )
                        if image_type[2] == "T1":
                            return (
                                file,
                                ds.ReferencedImageEvidenceSequence[0]
                                .ReferencedSeriesSequence[0]
                                .ReferencedSOPSequence[0]
                                .ReferencedSOPInstanceUID,
                                group_dcm,
                            )
                elif hasattr(ds, "ImageType") and hasattr(ds, "PulseSequenceName") and hasattr(
                    ds, "ComplexImageComponent"
                ):
                    image_type = ds.ImageType
                    sequence_name = ds.PulseSequenceName
                    complex_image_component = ds.ComplexImageComponent
                    if "3d1r4" in sequence_name:
                        if image_type[2] == "VELOCITY" and "PHASE" in complex_image_component:
                            return (
                                file,
                                ds.ReferencedImageEvidenceSequence[0]
                                .ReferencedSeriesSequence[1]
                                .ReferencedSOPSequence[0]
                                .ReferencedSOPInstanceUID,
                                group_dcm,
                            )
                        if image_type[2] == "T1":
                            return (
                                file,
                                ds.ReferencedImageEvidenceSequence[0]
                                .ReferencedSeriesSequence[1]
                                .ReferencedSOPSequence[0]
                                .ReferencedSOPInstanceUID,
                                group_dcm,
                            )
                else:
                    return None
            elif "philips" in manuf.lower():
                group_dcm = 2
                if not hasattr(ds, "ProtocolName") or not hasattr(ds, "ImageType"):
                    return None
                image_type = ds.ImageType
                protocol_name = ds.ProtocolName
                if image_type[2] == "FLOW_ENCODED" and "WIP" in protocol_name and "2D" not in protocol_name:
                    return (
                        file,
                        ds.ReferencedImageEvidenceSequence[1]
                        .ReferencedSeriesSequence[0]
                        .ReferencedSOPSequence[0]
                        .ReferencedSOPInstanceUID,
                        group_dcm,
                    )
            elif "ge" in manuf.lower():
                pass
            elif "uih" in manuf.lower():
                pass
        elif len(data_array.shape) >= 2 and all(dim > 1 for dim in data_array.shape):
            group_dcm = 0
            if "siemens" in manuf.lower():
                if not hasattr(ds, "ImageType") or not hasattr(ds, "SequenceName") or not hasattr(ds, "ProtocolName"):
                    return None
                image_type = ds.ImageType
                sequence_name = ds.SequenceName
                protocol_name = ds.ProtocolName
                del sequence_name
                if "flow" in protocol_name.lower() and (image_type[2] == "P" or image_type[2] == "M"):
                    return file, ds.FrameOfReferenceUID, group_dcm
            elif "philips" in manuf.lower():
                if not hasattr(ds, "ProtocolName") or not hasattr(ds, "ImageType") or ds.ImageType[-3] == "M_PCA":
                    return None
                protocol_name = ds.ProtocolName
                if "WIP DelRec" in protocol_name:
                    if (ds.ImageType[-2] == "M" and "AP" in protocol_name) or ds.ImageType[-2] == "P":
                        return file, ds.ReferencedImageSequence[1].ReferencedSOPInstanceUID, group_dcm
            elif "ge" in manuf.lower():
                if not hasattr(ds, "SeriesDescription"):
                    return None
                series_description = ds.SeriesDescription
                if any(tag in series_description for tag in ["SI", "AP", "LR", "Anatomy"]):
                    return file, ds.FrameOfReferenceUID, group_dcm
            elif "uih" in manuf.lower():
                if (
                    not hasattr(ds, "ImageType")
                    or not hasattr(ds, "SequenceName")
                    or not hasattr(ds, "SeriesDescription")
                ):
                    return None
                image_type = ds.ImageType
                sequence_name = ds.SequenceName
                series_description = ds.SeriesDescription
                del image_type
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
            if result is not None:
                file, frame_of_reference_uid, group_dcm = result
                if frame_of_reference_uid not in grouped_files:
                    grouped_files[frame_of_reference_uid] = []
                if frame_of_reference_uid not in group_dcms:
                    group_dcms[frame_of_reference_uid] = []
                group_dcms[frame_of_reference_uid].append(group_dcm)
                grouped_files[frame_of_reference_uid].append(file)
        for key in group_dcms.keys():
            group_dcms[key] = np.median(np.array(group_dcms[key]))
    return grouped_files, group_dcms


def check_flow_file_core(file, manuf):
    ds = pydicom.dcmread(file)
    data_array = ds.pixel_array
    rr_interval = None
    if hasattr(ds, "HeartRate"):
        heart_rate = ds.HeartRate
        rr_interval = 60000 / heart_rate
    elif hasattr(ds, "CardiacRate"):
        heart_rate = ds.CardiacRate
        rr_interval = 60000 / heart_rate
    elif hasattr(ds, "CardiacRRIntervalSpecified"):
        rr_interval = ds.CardiacRRIntervalSpecified
    elif hasattr(ds, "ImageComments"):
        rr_interval = ds.ImageComments
        rr_interval = int(re.search(r"RR\s+(\d+)", rr_interval).group(1))
    resolution = None

    if "siemens" in manuf.lower():
        image_type = ds.ImageType
        sequence_name = ds.SequenceName
        slice_location = ds.SliceLocation
        trigger_time = ds.TriggerTime
        slope = ds.RescaleSlope if hasattr(ds, "RescaleSlope") else 1
        intercept = ds.RescaleIntercept if hasattr(ds, "RescaleIntercept") else 0
        nv = 0
        venc_pair = None
        pix_spacing_y, pix_spacing_x = ds.PixelSpacing
        thickness = ds.SliceThickness if hasattr(ds, "SliceThickness") else None
        resolution = (pix_spacing_x, pix_spacing_y, thickness)

        if image_type[2] == "P":
            venc = float(re.findall(r"\d+", sequence_name.split("_")[1])[0])
            if any(dir_tag in sequence_name.lower() for dir_tag in ["rl", "lr"]):
                nv = 1
            elif any(dir_tag in sequence_name.lower() for dir_tag in ["ap", "pa"]):
                nv = 2
            elif any(dir_tag in sequence_name.lower() for dir_tag in ["hf", "fh"]):
                nv = 3
            elif any(dir_tag in sequence_name.lower() for dir_tag in ["in", "th"]):
                nv = 4
            venc_pair = (nv, abs(venc))
        data = (data_array * slope + intercept) / intercept * venc if nv else (data_array * slope + intercept)
        return nv, slice_location, trigger_time, data, rr_interval, resolution, venc_pair

    if "philips" in manuf.lower():
        protocol_name = ds.ProtocolName
        image_type = ds.ImageType
        nv = 0

        pix_spacing_y, pix_spacing_x = ds.PixelSpacing
        thickness = ds.SliceThickness if hasattr(ds, "SliceThickness") else None
        resolution = (pix_spacing_x, pix_spacing_y, thickness)
        slice_location = ds[0x2001, 0x100A].value
        trigger_time = ds[0x2001, 0x1008].value
        slope = ds.RescaleSlope if hasattr(ds, "RescaleSlope") else 1
        intercept = ds.RescaleIntercept if hasattr(ds, "RescaleIntercept") else 0

        venc_pair = None
        if image_type[-2] == "P":
            if any(dir_tag in protocol_name for dir_tag in ["RL", "LR"]):
                nv = 1
            elif any(dir_tag in protocol_name for dir_tag in ["AP", "PA"]):
                nv = 2
            elif any(dir_tag in protocol_name for dir_tag in ["HF", "FH"]):
                nv = 3
            venc_pair = (nv, abs(intercept))

        data = data_array * slope + intercept
        return nv, slice_location, trigger_time, data, rr_interval, resolution, venc_pair

    if "ge" in manuf.lower():
        series_description = ds.SeriesDescription
        nv = 0
        pix_spacing_y, pix_spacing_x = ds.PixelSpacing
        thickness = ds.SliceThickness if hasattr(ds, "SliceThickness") else None
        resolution = (pix_spacing_x, pix_spacing_y, thickness)

        slope = 1
        intercept = 0
        venc_pair = None
        if "Anatomy" not in series_description:
            if "LR" in series_description:
                nv = 1
            elif "AP" in series_description:
                nv = 2
            elif "SI" in series_description:
                nv = 3
            slope = 1 / 10
            venc_pair = (nv, abs(ds[0x0019, 0x10CC].value / 10))
        slice_location = ds.SliceLocation
        trigger_time = ds.TriggerTime
        data = data_array * slope + intercept
        return nv, slice_location, trigger_time, data, rr_interval, resolution, venc_pair

    if "uih" in manuf.lower():
        series_description = ds.SeriesDescription
        nv = 0
        pix_spacing_y, pix_spacing_x = ds.PixelSpacing
        thickness = ds.SliceThickness if hasattr(ds, "SliceThickness") else None
        resolution = (pix_spacing_x, pix_spacing_y, thickness)
        venc_pair = None
        if "RO" in series_description:
            nv = 1
            match = re.search(r"VENC\s*(\d+)", series_description)
            venc = int(match.group(1))
            venc_pair = (nv, venc)
        elif "PE" in series_description:
            nv = 2
            match = re.search(r"VENC\s*(\d+)", series_description)
            venc = int(match.group(1))
            venc_pair = (nv, venc)
        elif "SS" in series_description:
            nv = 3
            match = re.search(r"VENC\s*(\d+)", series_description)
            venc = int(match.group(1))
            venc_pair = (nv, venc)
        slope = ds.RescaleSlope if hasattr(ds, "RescaleSlope") else 1
        intercept = ds.RescaleIntercept if hasattr(ds, "RescaleIntercept") else 0
        slice_location = ds.SliceLocation
        trigger_time = ds.TriggerTime
        data = data_array * slope + intercept
        return nv, slice_location, trigger_time, data, rr_interval, resolution, venc_pair

    return None, None, None, None, None, None, None


def check_flow_file_core_group_dcm(file, manuf):

    ds = pydicom.dcmread(file)
    data_array = ds.pixel_array

    venc_pair=None

    rr_interval=None

    if hasattr(ds,"HeartRate"):
        rr_interval=60000/ds.HeartRate

    elif hasattr(ds,"CardiacRate"):
        rr_interval=60000/ds.CardiacRate

    elif hasattr(ds,"CardiacRRIntervalSpecified"):
        rr_interval=ds.CardiacRRIntervalSpecified

    resolution=None

    if "siemens" in manuf.lower():

        image_type=ds.ImageType

        dsp=ds.PerFrameFunctionalGroupsSequence[-1]

        pix_spacing_x=(
            dsp.PixelMeasuresSequence[0]
            .PixelSpacing[0]
        )

        pix_spacing_y=(
            dsp.PixelMeasuresSequence[0]
            .PixelSpacing[1]
        )

        thickness=(
            dsp.PixelMeasuresSequence[0]
            .SliceThickness
        )

        resolution=(
            pix_spacing_x,
            pix_spacing_y,
            thickness
        )

        nv=0

        if image_type[2]=="VELOCITY":

            venc=(
                dsp.MRVelocityEncodingSequence[0]
                .VelocityEncodingMaximumValue
            )

            venc_dir=(
                dsp.MRVelocityEncodingSequence[0]
                .VelocityEncodingDirection
            )

            nv=np.argmax(
                np.abs(venc_dir)
            )+1

            venc_pair=(
                nv,
                abs(float(venc))
            )

        slope=(
            dsp.PixelValueTransformationSequence[0]
            .RescaleSlope
        )

        intercept=(
            dsp.PixelValueTransformationSequence[0]
            .RescaleIntercept
        )

        slice_location=(
            dsp.FrameContentSequence[0]
            .InStackPositionNumber
        )

        data=(
            data_array*slope+intercept
        )

        return (
            nv,
            slice_location,
            data,
            rr_interval,
            resolution,
            venc_pair
        )

    return None,None,None,None,None,None
def check_flow_file_core_group_dcm2(file, manuf):
    ds = pydicom.dcmread(file)
    data_array = ds.pixel_array
    venc_pair = None
    rr_interval = None
    if hasattr(ds, "HeartRate"):
        heart_rate = ds.HeartRate
        rr_interval = 60000 / heart_rate
    elif hasattr(ds, "CardiacRate"):
        heart_rate = ds.CardiacRate
        rr_interval = 60000 / heart_rate
    elif hasattr(ds, "CardiacRRIntervalSpecified"):
        rr_interval = ds.CardiacRRIntervalSpecified

    resolution = None
    if "siemens" in manuf.lower():
        return None, None, None, None, None

    if "philips" in manuf.lower():
        protocol_name = ds.ProtocolName
        dsp = ds.PerFrameFunctionalGroupsSequence[-1]
        if "DelRec" not in protocol_name:
            nv = 0
        else:
            venc = dsp.MRVelocityEncodingSequence[0].VelocityEncodingMaximumValue
            venc_dir = dsp.MRVelocityEncodingSequence[0].VelocityEncodingDirection
            nv = np.argmax(np.abs(venc_dir)) + 1
            venc_pair = (nv, abs(venc))
        pix_spacing_x = dsp.PixelMeasuresSequence[0].PixelSpacing[0]
        pix_spacing_y = dsp.PixelMeasuresSequence[0].PixelSpacing[1]
        thickness = dsp.PixelMeasuresSequence[0].SliceThickness
        resolution = (pix_spacing_x, pix_spacing_y, thickness)
        spe = dsp.FrameContentSequence[0].InStackPositionNumber
        if nv == 0:
            dsp = ds.PerFrameFunctionalGroupsSequence[0]
            slope = dsp.PixelValueTransformationSequence[0].RescaleSlope
            intercept = dsp.PixelValueTransformationSequence[0].RescaleIntercept
            data = data_array.reshape(2, spe, -1, data_array.shape[1], data_array.shape[2])[0]
        else:
            dsp = ds.PerFrameFunctionalGroupsSequence[-1]
            slope = dsp.PixelValueTransformationSequence[0].RescaleSlope
            intercept = dsp.PixelValueTransformationSequence[0].RescaleIntercept
            data = data_array.reshape(3, spe, -1, data_array.shape[1], data_array.shape[2])[-1]
        data = data * slope + intercept
        return nv, data, rr_interval, resolution, venc_pair

    if "ge" in manuf.lower():
        return None, None, None, None, None
    return None, None, None, None, None

def get_flow_data(
        flow_dcm_files,
        manuf,
        group_dcm
):

    rr_values=[]

    resolutions=[]

    venc_list=[]

    if group_dcm==0:

        flow_data=[[] for _ in range(5)]

        spe_values=[[] for _ in range(5)]

        nt_values=[[] for _ in range(5)]

        with ProcessPoolExecutor() as executor:

            results=list(
                tqdm(
                    executor.map(
                        check_flow_file_core,
                        flow_dcm_files,
                        [manuf]*len(flow_dcm_files)
                    ),
                    total=len(flow_dcm_files)
                )
            )

        for nv,slice_location,\
            trigger_time,data,\
            rr_interval,res,\
            venc_pair in results:

            if nv is None:
                continue

            spe_values[nv].append(
                slice_location
            )

            nt_values[nv].append(
                trigger_time
            )

            flow_data[nv].append(
                data
            )

            venc_list.append(
                venc_pair
            )

            if rr_interval is not None:
                if rr_interval not in rr_values:
                    rr_values.append(
                        rr_interval
                    )

            if res is not None:
                if tuple(res) not in resolutions:
                    resolutions.append(
                        tuple(res)
                    )

        didx=next(
            i for i in range(5)
            if len(flow_data[i])==0
        )

        del flow_data[didx]
        del spe_values[didx]
        del nt_values[didx]

        for i in range(len(flow_data)):

            paired=list(
                zip(
                    spe_values[i],
                    nt_values[i],
                    flow_data[i]
                )
            )

            paired.sort(
                key=lambda x:
                (x[0],x[1])
            )

            spe_values[i],\
            nt_values[i],\
            flow_data[i]=zip(*paired)

            flow_data[i]=np.array(
                flow_data[i]
            )

        flow_data=np.array(
            flow_data
        )

        flow_data=flow_data.reshape(
            -1,
            len(set(
                spe_values[0]
            )),
            len(set(
                nt_values[0]
            )),
            *flow_data.shape[-2:]
        )

        flow_data=np.transpose(
            flow_data,
            (3,4,1,2,0)
        )


    elif group_dcm==1:

        flow_dict={}
        spe_dict={}

        with ProcessPoolExecutor() as executor:

            results=list(
                tqdm(
                    executor.map(
                        check_flow_file_core_group_dcm,
                        flow_dcm_files,
                        [manuf]*len(flow_dcm_files)
                    ),
                    total=len(
                        flow_dcm_files
                    )
                )
            )

        for nv,\
            slice_location,\
            data,\
            rr_interval,\
            res,\
            venc_pair in results:

            if nv is None:
                continue

            if venc_pair is None:

                key=(0,0)

            else:

                key=venc_pair

            if key not in flow_dict:

                flow_dict[key]=[]

                spe_dict[key]=[]

            flow_dict[key].append(
                data
            )

            spe_dict[key].append(
                slice_location
            )

            venc_list.append(
                venc_pair
            )

            if rr_interval is not None:

                if rr_interval not in rr_values:

                    rr_values.append(
                        rr_interval
                    )

            if res is not None:

                if tuple(res) not in resolutions:

                    resolutions.append(
                        tuple(res)
                    )

        ordered_keys=sorted(
            flow_dict.keys(),
            key=lambda x:
            (x[0],x[1])
        )

        final_data=[]

        for key in ordered_keys:

            paired=list(
                zip(
                    spe_dict[key],
                    flow_dict[key]
                )
            )

            paired.sort(
                key=lambda x:
                x[0]
            )

            _,arr=zip(
                *paired
            )

            arr=np.array(
                arr
            )

            print(
                key,
                arr.shape
            )

            final_data.append(
                arr
            )

        min_n=min(
            x.shape[0]
            for x in final_data
        )

        final_data=[
            x[:min_n]
            for x in final_data
        ]

        flow_data=np.stack(
            final_data,
            axis=0
        )

        flow_data=np.transpose(
            flow_data,
            (3,4,1,2,0)
        )


    elif group_dcm==2:

        flow_data=[[] for _ in range(4)]

        with ProcessPoolExecutor() as executor:

            results=list(
                tqdm(
                    executor.map(
                        check_flow_file_core_group_dcm2,
                        flow_dcm_files,
                        [manuf]*len(flow_dcm_files)
                    ),
                    total=len(
                        flow_dcm_files
                    )
                )
            )

        for nv,data,\
            rr_interval,\
            res,\
            venc_pair in results:

            if nv is None:
                continue

            flow_data[nv].append(
                data
            )

            venc_list.append(
                venc_pair
            )

        flow_data=np.array(
            flow_data
        )[:,0]

        flow_data=np.transpose(
            flow_data,
            (3,4,1,2,0)
        )

    else:

        raise ValueError


    final_rr=(
        rr_values[0]
        if rr_values
        else None
    )

    final_res=(
        resolutions[0]
        if resolutions
        else (None,None,None)
    )

    aggregated={}

    for item in venc_list:

        if item is None:
            continue

        nv=item[0]

        venc=item[1]

        if nv not in aggregated:

            aggregated[nv]=[]

        if venc not in aggregated[nv]:

            aggregated[nv].append(
                venc
            )

    venc_output=[]

    for k in sorted(
        aggregated.keys()
    ):

        venc_output.append(
            sorted(
                aggregated[k]
            )
        )

    return (
        flow_data,
        final_rr,
        final_res,
        venc_output
    )


def write_h5_group(
        h5_file,
        key,
        flow_files,
        flow_data,
        rr_interval,
        resolution,
        venc_list
):

    group=h5_file.create_group(
        str(key)
    )

    group.create_dataset(
        "img",
        data=flow_data
    )

    group.create_dataset(
        "Paths",
        data=[
            p.encode("utf-8")
            for p in flow_files
        ]
    )

    if rr_interval is None:

        group.create_dataset(
            "RR",
            data=np.nan
        )

    else:

        group.create_dataset(
            "RR",
            data=rr_interval
        )

    if resolution:

        group.create_dataset(
            "Resolution",
            data=np.array(
                resolution,
                dtype=np.float32
            )
        )

    else:

        group.create_dataset(
            "Resolution",
            data=np.array(
                [np.nan]*3,
                dtype=np.float32
            )
        )

    venc_flat=[]

    for x in venc_list:

        venc_flat.extend(x)

    group.create_dataset(
        "Venc",
        data=np.array(
            venc_flat,
            dtype=np.float32
        )
    )
def convert_dicom_to_h5(dicom_path, data_save_path):
    dcm_files = get_filtered_dcm_files(dicom_path)
    manuf = check_manufacturer(dcm_files)
    if not manuf:
        raise ValueError("Could not determine manufacturer from DICOM files.")

    flow_dcm_files, group_dcms = get_filtered_flow_dcm_files(dcm_files, manuf)
    num_keys = len(flow_dcm_files)
    print(f"Found {num_keys} different sequence groups from {manuf}")

    with h5py.File(data_save_path, "w") as h5_file:
        for key in list(flow_dcm_files.keys()):
            print(f"UID: {key}, File Nums: {len(flow_dcm_files[key])}")
            flow_data, rr_interval, resolution, venc_list = get_flow_data(
                flow_dcm_files[key], manuf, group_dcms[key]
            )
            print(
                f"UID: {key}, Data Shape: {flow_data.shape}, RR: {rr_interval}, "
                f"Resolution: {resolution}, Venc: {venc_list}"
            )

            if flow_data.size > 0:
                write_h5_group(h5_file, key, flow_dcm_files[key], flow_data, rr_interval, resolution, venc_list)

    return data_save_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="4D Flow DICOM to HDF5 (with heart rate and resolution)")
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
