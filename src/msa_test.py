# Dependencies
from src.msa import MSA
from Bio import AlignIO
import numpy as np
import unittest


class TestMSA(unittest.TestCase):

    # Constructor
    def __init__(self, *args, **kwargs):
        # Call parent constructor
        super().__init__(*args, **kwargs)
        # Define path to alignment files
        self.path_fa = 'data/sample.fa'
        self.path_afa = 'data/sample.afa'
        self.path_sth = 'data/sample.sto'
        # Open connection to fasta file
        with open(self.path_afa, 'r') as file:
            # Parse alignment
            self.alignment_raw = AlignIO.read(file, MSA.FORMAT_FASTA)
        # Load small sample MSA from file
        self.small_msa = MSA.load_fasta('data/small.afa')

    def test_parse_fasta(self):
        # Parse alignment
        msa = MSA.parse_alignment(self.alignment_raw)
        # Get alignment shape
        n, m = msa.shape
        # Check shape of parsed alignment
        self.assertEqual(n, 1036, 'There should be 1033 aligned sequences')
        self.assertEqual(m, 1078, 'There should be 1078 aligned positions')

    def test_load_stockholm(self):
        # Load alignment from file
        msa = MSA.load_sth(self.path_sth)
        # Get alignment shape
        n, m = msa.shape
        # Check shape of loaded alignment
        self.assertEqual(n, 1036, 'There should be 1036 aligned sequences')
        self.assertEqual(m, 1078, 'There should be 1078 aligned positions')

    # Test attributes on small sample alignment
    def test_attributes_small(self):
        # Get small sample alignment
        msa = self.small_msa
        # Check attributes in first row
        self.assertEqual('A123', msa.acc[0], 'Should be equal to `A123`')
        self.assertEqual(msa.begin[0], 1, 'Should be equal to 1')
        self.assertEqual(msa.end[0], 7, 'Should be equal to 7')
        self.assertEqual('--KNQlghl..'.upper(), b''.join(msa.residues[0, :]).decode('utf-8'), 'Should be equal to `--KNQLGHL..`')
        # Check attributes in last row
        self.assertEqual('C789', msa.acc[2], 'Should be equal to `C789`')
        self.assertEqual(msa.begin[2], 24, 'Should be equal to 24')
        self.assertEqual(msa.end[2], 32, 'Should be equal to 32')
        self.assertEqual('..FQILEIQnk'.upper(), b''.join(msa.residues[2, :]).decode('utf-8'), 'Should be equal to `..FQILEIQnk`')

    # Test logo on small sample alignment
    def test_get_logo_small(self):
        # Get sample alignment
        msa = self.small_msa
        # Compute logo
        logo = msa.get_logo()
        # Get shape of the logo
        n, m = logo.shape
        # Check shape of the logo
        self.assertEqual(n, 21, 'There should be 20 columns for residues and 1 for gaps, 21 in total')
        self.assertEqual(m, 11, 'There should be 11 aligned positions')
        # Check content of the logo
        self.assertEqual(logo[-1, 0], 3, 'There should be 3 gaps in the first column')
        self.assertEqual(logo[-1, -1], 2, 'There should be 2 gaps in the last column')

    # Test frequencies on small sample alignment
    def test_get_frequencies(self):
        # Get small alignment
        msa = self.small_msa
        # Get frequencies values
        frequencies = msa.get_frequencies()
        # Get frequencies shape
        n, m = frequencies.shape
        # Test frequencies shape and values
        self.assertEqual(n, 21, 'Should be equal to 21')
        self.assertEqual(m, 11, 'Should be equal to 11')
        self.assertEqual(frequencies[0, 0], 0.0, 'Should be equal to 0.0')
        self.assertEqual(frequencies[-1, 0], 1, 'Should be equal to 1')
        self.assertEqual(frequencies[4, 2], 1/3, 'Should be equal to 0.25')

    # Test occupancy on small sample alignment
    def test_get_occupancy_small(self):
        # Get sample alignment
        msa = self.small_msa
        # Compute occupancy
        occupancy = msa.get_occupancy()
        # Get shape of the occuoancy array
        m = len(occupancy)
        # Check shape of the occupancy array
        self.assertEqual(m, 11, 'There should be 11 aligned positions')
        # Check content of the occupancy array
        self.assertEqual(occupancy[0], 0, 'It should be equal to 0')
        self.assertEqual(occupancy[2], 1, 'It should be equal to 1')
        self.assertEqual(occupancy[8], 2/3, 'It should be equal to 0.33')
        self.assertEqual(occupancy[-1], 1/3, 'It should be equal to 0.66')

    # Test KL divergence on small sample alignment
    def test_get_kl_divergence_small(self):
        # Get sample alignment
        msa = self.small_msa
        # Compute conservation
        conservation = msa.get_kl_divergence(b=np.e)
        # Get conservation shape
        m = len(conservation)
        # Test conservation
        self.assertEqual(m, 11, 'There should be 11 aligned positions')
        self.assertAlmostEqual(conservation[2], 5.812, 3, 'Should be equal to 5.781')
        self.assertAlmostEqual(conservation[8], 3.395, 3, 'Should be equal to 3.395')

    # Test Shannon entropy on small sample alignment
    def test_get_shannon_entropy(self):
        # Get sample alignment
        msa = self.small_msa
        # Compute conservation
        conservation = msa.get_shannon_entropy(b=np.e)
        # Get conservation shape
        m = len(conservation)
        # Test conservation
        self.assertEqual(m, 11, 'There should be 11 aligned positions')
        self.assertAlmostEqual(conservation[2], 1.099, 3, 'Should be equal to 1.099')
        self.assertAlmostEqual(conservation[8], 0.732, 3, 'Should be equal to 0.732')

if __name__ == '__main__':
    # Execute tests
    unittest.main()