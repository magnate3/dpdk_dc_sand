import numpy as np
import pkg_resources
import katsdpsigproc.accel
from katsdpsigproc.accel import Operation, IOSlot, Dimension, build, roundup

# SOURCE = """
# <%include file="/port.mako"/>

# KERNEL void multiply(GLOBAL float *data, float scale) {
#     data[get_global_id(0)] *= scale;
# }
# """
class MultiplyTemplate:
    def __init__(self, context, queue):
        self.context = context

    def instantiate(self, queue, shape, scale):
        return Multiply(queue, shape, scale)

class Multiply(Operation):
    WGS = 32

    def __init__(self, queue, shape, scale):
        super().__init__(queue)

        program = build(
            queue.context,
            "kernels/beamform_kernel.mako",
            {
                "scale": scale,
            },
            extra_dirs=[pkg_resources.resource_filename(__name__, "")],
        )
        
        self.kernel = program.get_kernel("multiply")

        self.scale = np.uint16(scale)
        self.slots["inData"] = IOSlot(dimensions=shape, dtype=np.uint8)
        self.slots["outData"] = IOSlot(dimensions=shape, dtype=np.uint8)

    def _run(self):
        inData_buffer = self.buffer("inData")
        outData_buffer = self.buffer("outData")
        self.command_queue.enqueue_kernel(
            self.kernel,
            [inData_buffer.buffer, outData_buffer.buffer, self.scale],
            global_size=(roundup(inData_buffer.shape[0], self.WGS),),
            local_size=(self.WGS,)
        )