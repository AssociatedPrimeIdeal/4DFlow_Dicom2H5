import unittest
from unittest.mock import patch

import numpy as np
from pydicom.dataset import Dataset

from dicom2h5.converter import (
    _uses_philips_classic_flow_slice_number,
    derive_spatial_metadata,
    get_referenced_series_uid,
)


def header(model="Ingenia CX", protocol="WIP DelRec - AP", frames=1, number=1):
    ds = Dataset()
    ds.Manufacturer = "Philips"
    ds.ManufacturerModelName = model
    ds.ProtocolName = protocol
    ds.NumberOfFrames = frames
    ds.ImageOrientationPatient = [1, 0, 0, 0, 0, -1]
    ds.ImagePositionPatient = [0, number, 0]
    ds.SliceLocation = -number
    ds.add_new((0x2001, 0x100A), "IS", number)
    return ds


class PhilipsSpatialMetadataTests(unittest.TestCase):
    def derive(self, headers):
        with patch("dicom2h5.converter.pydicom.dcmread", side_effect=lambda name, **kw: headers[int(name)]):
            return derive_spatial_metadata([str(i) for i in range(len(headers))])

    def test_private_slice_order_matches_stored_array_on_multiple_models(self):
        for model in ["Ingenia CX", "Ingenia Elition X", "Ingenia"]:
            for protocol in ["WIP DelRec - AP", "WIP 4D_Qflow_retro_sen2"]:
                with self.subTest(model=model, protocol=protocol):
                    meta = self.derive([header(model, protocol, number=n) for n in [3, 1, 2]])
                    self.assertEqual(meta["SpatialOrder"], ("HF", "RL", "AP"))
                    np.testing.assert_array_equal(meta["Origin"], [0, 1, 0])
                    np.testing.assert_array_equal(meta["SliceDirectionLPS"], [0, 1, 0])
                    np.testing.assert_array_equal(meta["RotationMatrix"][:, 2], [0, 1, 0])

    def test_multiframe_and_other_protocols_keep_existing_geometry_path(self):
        for protocol, frames in [("WIP DelRec - AP", 20), ("WIP 2D_PC_TIPS_AP", 1), ("WIP 4DFlow-Liver", 1)]:
            with self.subTest(protocol=protocol, frames=frames):
                ds = header(protocol=protocol, frames=frames)
                self.assertFalse(_uses_philips_classic_flow_slice_number(ds))
                meta = self.derive([header(protocol=protocol, frames=frames, number=n) for n in [3, 1, 2]])
                self.assertEqual(meta["SpatialOrder"], ("HF", "RL", "PA"))
                np.testing.assert_array_equal(meta["Origin"], [0, 3, 0])

    def test_missing_private_slice_number_does_not_activate_rule(self):
        ds = header()
        del ds[(0x2001, 0x100A)]
        self.assertFalse(_uses_philips_classic_flow_slice_number(ds))

    def test_other_manufacturers_do_not_activate_rule(self):
        for vendor in ["Siemens", "GE MEDICAL SYSTEMS", "UIH"]:
            ds = header()
            ds.Manufacturer = vendor
            self.assertFalse(_uses_philips_classic_flow_slice_number(ds))


class ReferencedSeriesTests(unittest.TestCase):
    def dataset(self, count):
        ds, evidence = Dataset(), Dataset()
        series = []
        for i in range(count):
            item, sop = Dataset(), Dataset()
            sop.ReferencedSOPInstanceUID = f"1.2.3.{i + 1}"
            item.ReferencedSOPSequence = [sop]
            series.append(item)
        evidence.ReferencedSeriesSequence = series
        ds.ReferencedImageEvidenceSequence = [evidence]
        return ds

    def test_single_series_falls_back_to_available_reference(self):
        self.assertEqual(get_referenced_series_uid(self.dataset(1), 1), "1.2.3.1")

    def test_multiple_series_preserve_requested_reference(self):
        self.assertEqual(get_referenced_series_uid(self.dataset(2), 1), "1.2.3.2")


if __name__ == "__main__":
    unittest.main()
