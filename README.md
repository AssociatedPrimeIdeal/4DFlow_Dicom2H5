# 4DFlow_Dicom2H5
Converting 4D flow MRI DICOM to HDF5 file

## Usage

To use the `Dicom2H5.py` script for conversion, please follow these steps:

1. **Ensure Dependencies are Installed**:
   Make sure your environment has the required Python libraries, such as `numpy`, `h5py`, and `pydicom`.

2. **Run the Command**:
   Use the following command format to call the script:

   ```bash
   python Dicom2H5.py --dicom_path <path_to_dicom_directory> --data_save_path <path_to_save_h5_file>

## Code Explanation
The code will recursively search for all subfolders and files under the specified dicom_path, storing all identified 4D flow sequences in a single HDF5 file. The resulting matrix dimensions will be X, Y, Z, T, V, where:

X, Y, Z represent spatial dimensions.

T represents the temporal dimension.

V represents the velocity encoding dimension.

Different sequences will be stored in the HDF5 file using distinct UIDs as keys.

This script currently supports the majority of 4D flow DICOM formats from Siemens, Philips, GE, and UIH.