from typing import List, Sequence, Tuple

import numpy as np
from katsdpsigproc import accel, fft
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext
import config

# from .. import N_POLS
# from . import SAMPLE_BITS, pfb, postproc
N_POLS = config.N_POLS
SAMPLE_BITS = config.SAMPLE_BITS

class ComputeTemplate:
    """Template for the channelisation operation sequence.

    The reason for doing things this way can be read in the relevant
    `katsdpsigproc docs`_.

    .. _katsdpsigproc docs: https://katsdpsigproc.readthedocs.io/en/latest/user/operations.html#operation-templates

    Parameters
    ----------
    context
        The GPU context that we'll operate in.
    taps
        The number of taps that you want the resulting PFB-FIRs to have.
    """

    def __init__(self, context: AbstractContext) -> None:
        self.context = context

    def instantiate(
        self, command_queue: AbstractCommandQueue, samples: int, spectra: int, spectra_per_heap: int, channels: int
    ) -> "Compute":  # We have to put the return type in quotes because we haven't declared the `Compute` class yet.
        """Generate a :class:`Compute` object based on the template."""
        return Compute(self, command_queue, samples, spectra, spectra_per_heap, channels)

class Compute(accel.OperationSequence):
    """The DSP processing pipeline handling F-engine operation.

    The actual running of this operation isn't done through the :meth:`_run`
    method or by calling it directly, if you're familiar with the usual method
    of `composing operations`_. Fgpu's compute is streaming rather than
    batched, i.e. we have to coordinate the receiving of new data and the
    transmission of processed data along with the actual processing operation.

    Currently, no internal checks for consistency of the parameters are
    performed. The following constraints are assumed, Bad Things(TM) may happen
    if they aren't followed:

    - spectra_per_heap <= spectra - i.e. a chunk of data must be enough to send out at
      least one heap.
    - spectra % spectra_per_heap == 0
    - samples >= taps*channels*2.  An input chunk requires at least enough
      samples to output a single spectrum. The factor of 2 is because the
      PFB input is real, so 2*channels samples are needed for each output
      spectrum.
    - samples % 8 == 0

    .. _composing operations: https://katsdpsigproc.readthedocs.io/en/latest/user/operations.html#composing-operations

    Parameters
    ----------
    template
        Template for the channelisation operation sequence.
    command_queue
        The GPU command queue (typically this will be a CUDA Stream) on which
        actual processing operations are to be scheduled.
    samples
        Number of samples in each input chunk (per polarisation), including
        padding samples.
    spectra
        Number of spectra in each output chunk.
    spectra_per_heap
        Number of spectra to send in each output heap.
    channels
        Number of channels into which the input data will be decomposed.
    """

    def __init__(
        self,
        template: ComputeTemplate,
        command_queue: AbstractCommandQueue,
        samples: int,
        spectra: int,
        spectra_per_heap: int,
        channels: int,
    ) -> None:
        self.sample_bits = SAMPLE_BITS
        self.template = template
        self.channels = channels
        self.samples = samples
        self.spectra = spectra
        self.spectra_per_heap = spectra_per_heap

        # PFB-FIR and FFT each happen for each polarisation.
        # self.pfb_fir = [
        #     template.pfb_fir.instantiate(command_queue, samples, spectra, channels) for pol in range(N_POLS)
        # ]
        fft_template = fft.FftTemplate(
            template.context,
            1,
            (spectra, 2 * channels),
            np.float32,
            np.complex64,
            (spectra, 2 * channels),
            (spectra, channels + 1),
        )
        self.fft = [fft_template.instantiate(command_queue, fft.FftMode.FORWARD) for pol in range(N_POLS)]

        # Postproc is single though because it involves the corner turn which
        # combines the two pols.
        # self.postproc = template.postproc.instantiate(command_queue, spectra, spectra_per_heap, channels)

        operations: List[Tuple[str, accel.Operation]] = []
        # for pol in range(N_POLS):
        #     operations.append((f"pfb_fir{pol}", self.pfb_fir[pol]))
        for pol in range(N_POLS):
            operations.append((f"fft{pol}", self.fft[pol]))
        # operations.append(("postproc", self.postproc))

        compounds = {
            # fft0:work_area and fft1:work_area are just scratchpad memory.
            # Since the FFTs are run sequentially they won't interfere with
            # each other, i.e., fft0 is finished by the time fft1 starts.
            "fft_work": [f"fft{pol}:work_area" for pol in range(N_POLS)],
            # We expect the weights on the PFB-FIR taps to be the same for both
            # pols so they can share memory.
            # "weights": [f"pfb_fir{pol}:weights" for pol in range(N_POLS)],
            # "out": ["postproc:out"],
            # "fine_delay": ["postproc:fine_delay"],
            # "phase": ["postproc:phase"],
            # "gains": ["postproc:gains"],
        }
        # for pol in range(N_POLS):
        #     compounds[f"in{pol}"] = [f"pfb_fir{pol}:in"]
        #     compounds[f"fft_in{pol}"] = [f"pfb_fir{pol}:out", f"fft{pol}:src"]
        #     compounds[f"fft_out{pol}"] = [f"fft{pol}:dest", f"postproc:in{pol}"]
    
        for pol in range(N_POLS):
            compounds[f"in{pol}"] = [f"fft{pol}:src"]
            compounds[f"fft_out{pol}"] = [f"fft{pol}:dest"]
        
        super().__init__(command_queue, operations, compounds)

    # def run_frontend(
    #     self, samples: Sequence[accel.DeviceArray], in_offsets: Sequence[int], out_offset: int, spectra: int
    # ) -> None:
    #     """Run the PFB-FIR on the received samples.

    #     Coarse delay also seems to be involved.

    #     Parameters
    #     ----------
    #     samples
    #         A pair of device arrays containing the samples, one for each pol.
    #     in_offsets
    #         Index of first sample in input array to process (one for each pol).
    #     out_offset
    #         TODO: Figure out what this is. Need to refer to the actual pfb_fir kernel.
    #     spectra
    #         How many spectra worth of samples to push through the PFB-FIR.
    #     """
    #     if len(samples) != N_POLS:
    #         raise ValueError(f"samples must contain {N_POLS} elements")
    #     if len(in_offsets) != N_POLS:
    #         raise ValueError(f"in_offsets must contain {N_POLS} elements")
    #     for pol in range(N_POLS):
    #         self.bind(**{f"in{pol}": samples[pol]})
    #     # TODO: only bind relevant slots for frontend
    #     self.ensure_all_bound()
    #     for pol in range(N_POLS):
    #         # TODO: could run these in parallel, but that would require two
    #         # command queues.
    #         self.pfb_fir[pol].in_offset = in_offsets[pol]
    #         self.pfb_fir[pol].out_offset = out_offset
    #         self.pfb_fir[pol].spectra = spectra
    #         self.pfb_fir[pol]()

    # def run_backend(self, out: accel.DeviceArray) -> None:
    #     """Run the FFT and postproc on the data which has been PFB-FIRed.

    #     Postproc incorporates fine-delay, requantisation and corner-turning.

    #     Parameters
    #     ----------
    #     out
    #         Destination for the processed data.
    #     """
    #     self.bind(out=out)
    #     # TODO: only bind relevant slots for backend
    #     self.ensure_all_bound()
    #     for fft_op in self.fft:
    #         fft_op()
    #     self.postproc()

    def run_fft(self, out: accel.DeviceArray) -> None:
        """Run the FFT and postproc on the data which has been PFB-FIRed.

        Postproc incorporates fine-delay, requantisation and corner-turning.

        Parameters
        ----------
        out
            Destination for the processed data.
        """
        self.bind(out=out)
        # TODO: only bind relevant slots for backend
        self.ensure_all_bound()
        for fft_op in self.fft:
            fft_op()
        self.postproc()