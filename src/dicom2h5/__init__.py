"""Public package interface for 4D flow DICOM to HDF5 conversion."""

from .converter import convert_dicom_to_h5, main, native_group_names, validate_native_h5

__all__ = ["convert_dicom_to_h5", "native_group_names", "validate_native_h5", "main"]
__version__ = "0.1.0"
