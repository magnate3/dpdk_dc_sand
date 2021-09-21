import numpy as np
import katsdpsigproc.accel
from katsdpsigproc.accel import Operation, IOSlot, Dimension, build, roundup
from katsdpsigproc.accel import visualize_operation
import multiply

class MultSeqTemplate:
    def __init__(self, context, queue):
        self.Mult1 = multiply.MultiplyTemplate(context, queue)
        self.Mult2 = multiply.MultiplyTemplate(context, queue)

    def instantiate(self, queue, size, scale):
        return MultCombined(self, queue, size, scale)

class MultCombined(katsdpsigproc.accel.OperationSequence):
    def __init__(self, template, queue, size, scale):
        self.Mult1 = template.Mult1.instantiate(queue, size, 3)
        self.Mult2 = template.Mult2.instantiate(queue, size, 7)
        operations = [
            ('Mult1', self.Mult1),
            ('Mult2', self.Mult2)
        ]
        # operations = [
        #     ('Mult1', self.Mult1)
        # ]
        compounds = {
            'bufin': ['Mult1:inData' ],
            'bufint': ['Mult1:outData', 'Mult2:inData'],
            'bufout': ['Mult2:outData']
        }
        super().__init__(queue, operations, compounds)
        self.template = template

    def __call__(self):
        super().__call__()


ctx = katsdpsigproc.accel.create_some_context()
queue = ctx.create_command_queue()
op_template = MultSeqTemplate(ctx, queue)
op = op_template.instantiate(queue, 64, 3)
op.ensure_all_bound()

bufin_device = op.Mult1.buffer("inData")
host_in = bufin_device.empty_like()

bufout_device = op.Mult2.buffer("outData")
host_out = bufout_device.empty_like()

#host[:] = np.random.uniform(size=host_in.shape)
host_in[:] = np.ones(shape=host_in.shape)
print(host_in)

bufin_device.set(queue, host_in)
op()
bufout_device.get(queue, host_out)
print(host_out)
a = 1
# Visualise the operation
katsdpsigproc.accel.visualize_operation(op,'test_op_vis')