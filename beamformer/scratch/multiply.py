import numpy as np
import katsdpsigproc.accel
from katsdpsigproc.accel import Operation, IOSlot, Dimension, build, roundup

SOURCE = """
<%include file="/port.mako"/>

KERNEL void multiply(GLOBAL float *data, float scale) {
    data[get_global_id(0)] *= scale;
}
"""
class MultiplyTemplate:
    def __init__(self, context, queue):
        self.context = context

    def instantiate(self, queue, size, scale):
        return Multiply(queue, size, scale)

class Multiply(Operation):
    WGS = 32

    def __init__(self, queue, size, scale):
        super().__init__(queue)
        program = build(queue.context, '', source=SOURCE)
        self.kernel = program.get_kernel('multiply')
        self.scale = np.float32(scale)
        self.slots['data'] = IOSlot((Dimension(size, self.WGS),), np.float32)

    def _run(self):
        data = self.buffer('data')
        self.command_queue.enqueue_kernel(
            self.kernel,
            [data.buffer, self.scale],
            global_size=(roundup(data.shape[0], self.WGS),),
            local_size=(self.WGS,)
        )