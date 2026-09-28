# 4DFlow_Dicom2H5
Converting 4D flow MRI DICOM to HDF5 file

## Install

```bash
pip install .
```

## Usage

After installation, use the console command:

```bash
dicom2h5 --dicom-path <path_to_dicom_directory> --data-save-path <path_to_save_h5_file>
```

You can also run it as a module:

```bash
python -m dicom2h5 --dicom-path <path_to_dicom_directory> --data-save-path <path_to_save_h5_file>
```

The legacy script entry point is still kept for compatibility:

```bash
python Dicom2H5.py --dicom_path <path_to_dicom_directory> --data_save_path <path_to_save_h5_file>
```

## Code Explanation
The code will recursively search for all subfolders and files under the specified dicom_path, storing all identified 4D flow sequences in a single HDF5 file. The resulting matrix dimensions will be X, Y, Z, T, V, where:

X, Y, Z represent spatial dimensions.

T represents the temporal dimension.

V represents the velocity encoding dimension.

Different sequences will be stored in the HDF5 file using distinct UIDs as keys.

Each sequence group is the native contract consumed by AutoFlow and H52Dicom:

```text
mag          XYZT (magnitude)
flow         XYZT3 (velocity channels, cm/s)
RR           cardiac interval in ms
Resolution   three voxel sizes in mm
VENC         three velocity-encoding values in cm/s
VENCOrder    three directed labels such as RL, PA, FH
SpatialOrder three directed spatial-axis labels derived from ImageOrientationPatient
Origin       DICOM patient-LPS origin
patient/     PatientName, PatientID, age, sex, height, and weight
scanner/     Institution, address, station, manufacturer, model, and device details
acquisition/ Modality, body part, study/series description, and protocol
```

`VENCOrder` is parsed directly from DICOM velocity metadata (including vendor
private fields). `SpatialOrder` follows the source array axes and is inferred
from DICOM orientation and slice positions. Geometry and standard patient,
institution, scanner, and acquisition fields are retained in each sequence
group for downstream conversion and audit.

This script currently supports the majority of 4D flow DICOM formats from Siemens, Philips, GE, and UIH.
