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
        self.Mult1 = template.Mult1.instantiate(queue, size, scale)
        self.Mult2 = template.Mult2.instantiate(queue, size, scale)
        operations = [
            ('Mult1', self.Mult1),
            ('Mult2', self.Mult2)
        ]
        compounds = {
            'bufin': ['Mult1:data', 'Mult2:data']
        }
        super().__init__(queue, operations, compounds)
        self.template = template

    def __call__(self):
        super().__call__()


ctx = katsdpsigproc.accel.create_some_context()
queue = ctx.create_command_queue()
op_template = MultSeqTemplate(ctx, queue)
op = op_template.instantiate(queue, 50, 3.0)
op.ensure_all_bound()
buf = op.Mult1.buffer('data')
host = buf.empty_like()
host[:] = np.random.uniform(size=host.shape)
#host[:] = np.ones(shape=host.shape)
print(host)
buf.set(queue, host)
op()
buf.get(queue, host)
print(host)
a = 1
# Visualise the operation
katsdpsigproc.accel.visualize_operation(op,'test_op_vis')