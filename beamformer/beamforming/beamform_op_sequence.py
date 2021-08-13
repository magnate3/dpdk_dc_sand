import numpy as np
from katsdpsigproc import accel
from katsdpsigproc import cuda
from katsdpsigproc.accel import Operation, IOSlot, Dimension, build, roundup
from katsdpsigproc.accel import visualize_operation
import beamform
from beamform_reorder import prebeamform_reorder

class BeamformSeqTemplate:
    def __init__(self, context: cuda.Context, queue: accel.AbstractCommandQueue, n_ants: int, n_channels: int, n_samples_per_channel: int, n_batches: int):
        self.preBeamformReorder = prebeamform_reorder.PreBeamformReorderTemplate(context, n_ants, n_channels, n_samples_per_channel, n_batches)
        self.beamformMult = beamform.MultiplyTemplate(context, queue)

    def instantiate(self, queue, scale):
        return OpSequence(self, queue, scale)

class OpSequence(accel.OperationSequence):
    def __init__(self, template, queue, scale):
        self.prebeamformReorder = template.preBeamformReorder.instantiate(queue)
        self.beamformMult = template.beamformMult.instantiate(queue, template.preBeamformReorder.outputDataShape, scale)
        operations = [
            ('reorder', self.prebeamformReorder),
            ('beamformMult', self.beamformMult)
        ]
        # compounds = {
        #     'bufin': ['reorder:inSamples' ],
        #     'bufint': ['reorder:outReordered', 'beamformMult:inData'],
        #     'bufout': ['beamformMult:outData']
        # }
        compounds = {
            'bufin': ['reorder:inSamples' ],
            'bufint': ['reorder:outReordered','beamformMult:inData'],
            'bufout': ['beamformMult:outData']
        }
        super().__init__(queue, operations, compounds)
        self.template = template

    def __call__(self):
        super().__call__()


# Reorder Specs

batches = 3
ants = 4
num_chan = 1024
n_samples_per_channel = 256

ctx = accel.create_some_context()
queue = ctx.create_command_queue()
op_template = BeamformSeqTemplate(ctx, queue, ants, num_chan, n_samples_per_channel, batches)
op = op_template.instantiate(queue, 3)
op.ensure_all_bound()

bufin_device = op.prebeamformReorder.buffer("inSamples")
host_in = bufin_device.empty_like()

bufout_device = op.beamformMult.buffer("outData")
host_out = bufout_device.empty_like()

# host_in[:] = np.ones(shape=host_in.shape, dtype=host_in.dtype)

# Inject random data for test.
# rng = np.random.default_rng(seed=2021)
# host_in[:] = rng.uniform(
#     np.iinfo(host_in.dtype).min, np.iinfo(host_in.dtype).max, host_in.shape
# ).astype(host_in.dtype)

host_in[0][0][0][0][0][0] = 1
host_in[0][0][0][1][0][0] = 2
host_in[0][0][0][2][0][0] = 3
host_in[0][0][0][3][0][0] = 4
host_in[0][0][0][4][0][0] = 5
host_in[0][0][0][5][0][0] = 6
host_in[0][0][0][6][0][0] = 7
host_in[0][0][0][7][0][0] = 8
#print(host_in)

bufin_device.set(queue, host_in)
op()
bufout_device.get(queue, host_out)
#print(host_out)
a = 1
# Visualise the operation
accel.visualize_operation(op,'test_op_vis')