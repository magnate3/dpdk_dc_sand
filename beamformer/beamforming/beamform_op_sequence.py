import numpy as np
from katsdpsigproc import accel
from katsdpsigproc import cuda
from katsdpsigproc.accel import Operation, IOSlot, Dimension, build, roundup
from katsdpsigproc.accel import visualize_operation
import beamform
from beamform_reorder import prebeamform_reorder

class CoeffGenerator:
    def __init__(self, batches, pols, num_chan, n_blocks, samples_per_block, ants):
        self.batches = batches
        self.pols = pols
        self.num_chan = num_chan
        self.n_blocks = n_blocks
        self.samples_per_block = samples_per_block
        self.ants = ants
        self.complexity = 2
    
    def Coeffs(self, test_id):
        l = self.batches * self.pols * self.num_chan * self.n_blocks * self.samples_per_block

        coeffs = np.ones(((self.ants*2)*2*l),np.float32).reshape(l,2,self.ants * 2)
        real_value = 4
        imag_value = 1
        for i in range(coeffs.shape[0]):
            for j in range(coeffs.shape[1]):
                for k in range(coeffs.shape[2]):
                    if j == 0:
                        if k % 2:
                            coeffs[i,j,k] = -1 * imag_value
                        else:
                            coeffs[i,j,k] = real_value
                    else:
                        if k % 2:
                            coeffs[i,j,k] = real_value
                        else:
                            coeffs[i,j,k] = imag_value

        if test_id == 'kernel':
            return coeffs.reshape(self.batches, self.pols, self.num_chan, self.n_blocks, self.samples_per_block, 2, self.ants * self.complexity)
        else:
            return coeffs

class BeamformSeqTemplate:
    def __init__(self, context: cuda.Context, queue: accel.AbstractCommandQueue, n_ants: int, n_channels: int, n_samples_per_channel: int, n_batches: int):
        self.preBeamformReorder = prebeamform_reorder.PreBeamformReorderTemplate(context, n_ants, n_channels, n_samples_per_channel, n_batches)
        self.beamformMult = beamform.MultiplyTemplate(context, n_ants, n_channels, n_samples_per_channel, n_batches)

    def instantiate(self, queue, coeffs, test_id):
        return OpSequence(self, queue, coeffs, test_id)

class OpSequence(accel.OperationSequence):
    def __init__(self, template, queue, coeffs, test_id):
        self.prebeamformReorder = template.preBeamformReorder.instantiate(queue)
        self.beamformMult = template.beamformMult.instantiate(queue, coeffs, test_id)
        operations = [
            ('reorder', self.prebeamformReorder),
            ('beamformMult', self.beamformMult)
        ]
        compounds = {
            'bufin': ['reorder:inSamples' ],
            'bufint': ['reorder:outReordered','beamformMult:inData'],
            'bufout': ['beamformMult:outData']
        }
        super().__init__(queue, operations, compounds)
        self.template = template

    def __call__(self):
        super().__call__()


def print_debug(host_out):
    # Debug: Print out all the entries to verify values
    for b in range(host_out.shape[0]):
        for p in range(host_out.shape[1]):
            for c in range(host_out.shape[2]):
                for bl in range(host_out.shape[3]):
                    for s in range(host_out.shape[4]):
                        for cmplx in range(host_out.shape[5]):
                            print(host_out[b][p][c][bl][s][cmplx])

# Reorder Specs
batches = 3
ants = 4
num_chan = 64
n_samples_per_channel = 256
samples_per_block = 16
n_blocks = n_samples_per_channel // samples_per_block
pols = 2

# NOTE: test_id is a temporary inclusion meant to identify which complex multiply to call.
# Options:  'sgemm' for cublas matrix mult
#           'kernel' for numba-based complex multiplication kernel
test_id = 'kernel' 

# Generate coefficients
coeff_gen = CoeffGenerator(batches, pols, num_chan, n_blocks, samples_per_block, ants)
coeffs = coeff_gen.Coeffs(test_id)

ctx = accel.create_some_context()
queue = ctx.create_command_queue()
op_template = BeamformSeqTemplate(ctx, queue, ants, num_chan, n_samples_per_channel, batches)
op = op_template.instantiate(queue, coeffs, test_id)
op.ensure_all_bound()

bufin_device = op.prebeamformReorder.buffer("inSamples")
host_in = bufin_device.empty_like()

bufout_device = op.beamformMult.buffer("outData")
host_out = bufout_device.empty_like()

# --- Inject ones data for test ---
host_in[:] = np.ones(shape=host_in.shape, dtype=host_in.dtype)
# host_in[:] = np.ones(host_in.shape,np.float32)

# --- Inject random data for test ---
# rng = np.random.default_rng(seed=2021)
# host_in[:] = rng.uniform(
#     np.iinfo(host_in.dtype).min, np.iinfo(host_in.dtype).max, host_in.shape
# ).astype(host_in.dtype)

#....Or specific values...
# host_in[0][0][0][0][0][0] = 1
# host_in[0][0][0][1][0][0] = 2
# host_in[0][0][0][2][0][0] = 3
# host_in[0][0][0][3][0][0] = 4
# host_in[0][0][0][4][0][0] = 5
# host_in[0][0][0][5][0][0] = 6
# host_in[0][0][0][6][0][0] = 7
# host_in[0][0][0][7][0][0] = 8
# print(host_in)

bufin_device.set(queue, host_in)
op()
bufout_device.get(queue, host_out)

# Debug: Print out all the entries to verify values
print_debug(host_out)

# Visualise the operation (Just for interest)
accel.visualize_operation(op,'test_op_vis')