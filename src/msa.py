# Dependencies
from Bio import AlignIO
import pandas as pd
import numpy as np
import json
import sys
import re


class MSA(object):
    """ Define multiple sequence alignment class
    
    This class handles multiple sequence alignments (MSA) and allows to compute some statistics on them.
    A multiple sequence alignment can be represented as a table, with many rows as the number of aligned
    sequences. Each row represents either a sequence or a subsequence in the target database. It is identified
    by the accession number of the sequence itself, followed by the indices of the initial and final residues
    within it, in the latter case. All other columns but these three, define aligned positions and contain a
    single character identifying a protein residue or a gap.

    Attributes
    ----------
    accession : list(str)
        List of accession numbers for each row in the alignment
    acc : list(str)
        Just a shortcut for accession
    begin : list(int)
        List of positions within each sequence identified by the accession number where the aligned region begins
    end : list(int)
        List of positions within each sequence identified by the accession number where the aligned region ends
    residues : np.ndarray
        Matrix containing the aligned residue characters
    res : np.ndarray
        Just a shortcut for residues
    
    Methods
    -------
    get_identity : np.array
        Returns single dimension array containing the identity of each position in the alignment
    get_similarity : np.array
        Returns ingle dimension 
    get_logo : np.ndarray
        Retrieves the percentage of each residue in each position
    to_numpy : np.ndarray
        Returns the whole alignment as Numpy matrix
    to_dataframe : pd.DataFrame
        Returns the whole alignment as Pandas DataFrame
    load_file : MSA
        Load alignment from user defined file format
    load_fasta : MSA
        Load alignment from fasta file format
    load_sth : MSA
        Load alignment from stockholm file format
    """

    # (Static) Available file formats
    FORMAT_FASTA = 'fasta'
    FORMAT_STH = 'stockholm'

    # (Static) Available amino-acids
    ALL_RESIDUES = np.char.array(list('ACDEFGHIKLMNPQRSTVWY'), itemsize=1, unicode=False)

    # (Static) Path to file containing amino acid frequencies
    FREQUENCIES_PATH = './data/freqs_tr_2021_02.json'
     
    # Constructor
    def __init__(self, accession, begin, end, residues):
        # Store attributes
        self.accession =  accession
        self.begin = begin
        self.end = end
        self.residues = residues
        # Open file containing amino acid frequencies
        with open(self.FREQUENCIES_PATH) as file:
            # Load amino acid frequencies
            self.frequencies = {
                residue_code.encode(): residue_freq
                for residue_code, residue_freq
                in json.load(file).items()
                if residue_code.encode() in self.ALL_RESIDUES
            }

    @property
    def acc(self):
        return self.accession

    @property
    def res(self):
        return self.residues

    @property
    def shape(self):
        return self.res.shape

    def get_occupancy(self):
        """ Compute occupancy
        
        Compute the rate of aligned residues in each column, excluding gaps
        """
        # Compute logo for current alignment
        logo = self.get_logo()
        # Compute occupancy denominator by summing number of occurriencies
        den = np.sum(logo, axis=0)
        # Compute occupancy numerator by subtracting gaps from total
        num = den - logo[-1, :]
        # Compute occupancy as fraction of non-gap aligned residues
        return num / den

    def get_conservation(self, b=2):
        return self.get_kl_divergence(b=b)

    def get_kl_divergence(self, b=2):
        # Get alignment shape
        _, m = self.shape
        # Initialize Kullback-Leibler divergence
        conservation = np.zeros(shape=(m, ), dtype=np.float)
        # Get observed and expected amino-acid frequencies
        observed, expected = self.get_frequencies(), self.frequencies
        # Loop trhough each amino acid
        for i, residue_code in enumerate(self.ALL_RESIDUES):
            # Get current odds, exclude cases where logarithm is zero
            odds = np.where(observed[i, :] > 0, observed[i, :] / expected[residue_code], 1)
            # Update conservation
            conservation += observed[i, :] * (np.log(odds) / np.log(b))
        # Return Kullback-Leibler divergence
        return conservation

    def get_shannon_entropy(self, b=2):
        # Get alignment shape
        _, m = self.shape
        # Get observed amino-acid frequencies
        observed = self.get_frequencies()
        # Initialize Shannon's entropy
        conservation = np.zeros(shape=(m, ), dtype=np.float)
        # Loop trhough each amino acid
        for i, _ in enumerate(self.ALL_RESIDUES):
            # Compute logarithm multiplier
            frequencies = np.where(observed[i, :] > 0, observed[i, :], 1)
            # Update conservation
            conservation -= frequencies * (np.log(frequencies) / np.log(b))
        # Return Kullback-Leibler divergence
        return conservation

    def get_logo(self):
        # Get number of reisdues
        p = len(self.ALL_RESIDUES)
        # Get shape of current alignment
        n, m = self.shape
        # Initialize logo as numpy array
        logo = np.zeros(shape=(p + 1, m), dtype=np.float)
        # Loop through each available residue
        for i, residue_code in enumerate(self.ALL_RESIDUES):
            # Get occurrencies for current residues in each column
            logo[i, :] = np.sum(self.residues == residue_code, axis=0)
        # Compute gaps as difference between total and counted occurrencies
        logo[-1, :] = n - np.sum(logo[:p, :], axis=0)
        # Return logo
        return logo

    def get_frequencies(self):
        # Get logo of current alignment
        logo = self.get_logo()
        # Get shape of the alignment
        n, _ = self.shape
        # Compute and return frequencies
        return logo / n

    def to_numpy(self):
        # Cast attributes to numpy
        acc = np.array(self.acc, dtype=np.unicode)
        beg = np.array(self.begin, dtype=np.int)
        end = np.array(self.end, dtype=np.int)
        # Get residues as numpy matrix
        res = self.res
        # Return all the attributes
        return acc, beg, end, res

    def to_dataframe(self):
        # Get shape of inner residues matrix
        n, m = self.shape
        # Initialize dataframe columns
        columns = ['entry_acc', 'region_beg', 'region_end']
        # Update dataframe columns with those of the alignment
        columns = [*columns, *['resid_%d' % i for i in range(1, m+1)]]
        # Create dataframe from residues
        df = pd.DataFrame(self.residues, columns=columns[3:])
        # Add other columns
        df.loc[:, columns[0]] = self.accession
        df.loc[:, columns[1]] = self.begin
        df.loc[:, columns[2]] = self.end
        # Return dataframe with sorted columns
        return df.loc[:, columns]

    @classmethod
    def parse_alignment(cls, alignment):
        # Initialize shape of the alignment
        shape = n, m = 0, 0
        # Initialize attributes
        accession, begin, end, residues = [], [], [], []
        # Loop though each record in parsed file
        for i, record in enumerate(alignment):
            # Get sequence and identifier for current record
            identifier, sequence = str(record.id), list(record.seq.upper())
            # Check identifier of current record
            match = re.search(r'^([^\s]+)/(\d+)-(\d+)\s*', identifier)
            # Case identifier matches expected format set accession, begin, end
            if match:
                # Set accession, begin, end
                accession.append(str(match[1]))
                begin.append(int(match[2]))
                end.append(int(match[3]))
            # Otherwise set accession only
            else:
                # Set accession only
                accession.append(identifier)
                # Set default values for begin and end
                begin.append(0)
                end.append(len(sequence))
            # Store sequence for current record
            residues.append(sequence)
            # Define shape of the residues matrix
            shape = n, m = n + 1, len(sequence)
        # Cast residues to matrix of characters
        residues = np.char.array(residues, itemsize=1, unicode=False)
        # Reshape matrix to shape
        residues = residues.reshape(shape)
        # Return parsed alignment
        return cls(accession, begin, end, residues)

    @classmethod
    def load_file(cls, file_or_path, file_format):
        # Define whether input is path or file
        is_path = isinstance(file_or_path, str)
        # Create new file handler
        handler = open(file_or_path, 'r') if is_path else file_or_path
        # Parse alignment from fasta file
        alignment = cls.parse_alignment(AlignIO.read(handler, file_format))
        # Close handler if input was path
        if is_path: handler.close()
        # Return loaded alignment
        return alignment

    @classmethod
    def load_fasta(cls, file_or_path):
        return cls.load_file(file_or_path, cls.FORMAT_FASTA)

    @classmethod
    def load_sth(cls, file_or_path):
        return cls.load_file(file_or_path, cls.FORMAT_STH)


# Example
if __name__ == '__main__':
    # Define path to example alignment in FASTA format
    path = 'data/sample.afa'
    # Load alignment from FASTA file
    msa = MSA.load_fasta(path)
    # Check shape of residues matrix
    print('Alignment has shape %s' % str(msa.shape))

    # Define both unicode and bytes matrices
    res_utf8 = np.char.array(msa.res, itemsize=1, unicode=True)
    res_bytes = np.char.array(msa.res, itemsize=1, unicode=False)
    # Check alignment size in memory
    print('Alignment (utf-8) has size %.02f' % (sys.getsizeof(res_utf8) / 1e06))
    print('Alignment (bytes) has size %.02f' % (sys.getsizeof(res_bytes) / 1e06))