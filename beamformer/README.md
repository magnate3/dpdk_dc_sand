# Pre-Beamformer reordering.
This code will perform the necessary data reordering required prior to beamforming.
The primary reordering is performed on a GPU with unit test verification done on the CPU.

Input:  Data as ingest by an X-Engine core. This data is assumed to a subset of the full F-Enginw output.
	uint16_t [n_batches][n_antennas][n_channels][n_samples_per_channel][polarizations][complexity]

Output: Data suitable for beamforming.
	uint16_t [n_batches][polarizations][n_channels][n_blocks][n_samples_per_block][n_ants][complexity]
