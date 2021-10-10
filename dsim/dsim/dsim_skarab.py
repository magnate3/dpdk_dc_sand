"""Control for SKARAB-based DSim."""

import re
import time

from casperfpga import CasperFpga
from casperfpga.attribute_container import AttributeContainer
from casperfpga.register import Register
from casperfpga.transport_skarab import SkarabTransport

from .utils import StreamAddress, get_prefixed_name, parse_config_file, remove_nones

# for DM in [pc cm ^{ -3}] , time in [s] , and frequency in [Hz]
alpha = 2.410e-16

# region --- Signal Sources ---


class Source(object):
    """Base Source class from which to inherit.

    :param register:
        casperfpga Register object for ease of interfacing.
    :param name:
        Name of Source.
    """

    def __init__(self, register: Register, name: str):
        self.parent = register.parent
        self.name = name


class SineSource(Source):
    """Sinusoid Source abstraction from DSim design.

    :param freq_register:
    :param scale_register:
    :param name:
    :param repeat_en_register:
    :param repeat_len_register:
    :param repeat_len_field_name:
    """

    def __init__(
        self,
        freq_register: Register,
        scale_register: Register,
        name: str,
        repeat_en_register: Register = None,
        repeat_len_register: Register = None,
        repeat_len_field_name: Register = None,
    ) -> None:
        super(SineSource, self).__init__(freq_register, name)

        self.freq_register = freq_register
        self.scale_register = scale_register

        self.repeat_len_register = repeat_len_register
        self.repeat_len_field_name = repeat_len_field_name
        self.repeat_en_register = repeat_en_register

        self.sample_rate_hz = float(self.parent.config["sample_rate_hz"])
        freq_field = self.freq_register.field_get_by_name("frequency")
        self.nr_freq_steps = 2 ** freq_field.width_bits

        self.max_freq = self.sample_rate_hz / 2.0
        self.delta_freq = self.max_freq / self.nr_freq_steps

    @property
    def frequency(self):
        """Frequency in Hz."""
        return self.freq_register.read()["data"]["frequency"] * self.delta_freq

    @property
    def scale(self):
        """Scale factor for Source."""
        return self.scale_register.read()["data"]["scale"]

    @property
    def repeat(self):
        """Repeat counter."""
        if self.repeat_len_register is None or self.repeat_en_register is None:
            return None
        else:
            return self.repeat_len_register.read()["data"][self.repeat_len_field_name]

    def set(self, scale: float = 0, frequency: float = 0, repeat_n: int = 0):
        """Set source parameters.

        :param scale:
            Scaling factor for source - between 0 and 1.
        :param frequency:
            Frequency of source in Hz - from 0 to the Nyquist freq.
        :param repeat_n:
            Forces output to be periodic every N samples, or disables repeat if set to 0.

        """
        if 0 <= scale <= 1:
            self.scale_register.write(scale=scale)
        if frequency >= 0:
            freq_steps = int(round(frequency / self.delta_freq))
            self.freq_register.write(frequency=freq_steps)
        if repeat_n > 0:
            if self.repeat_len_register is None:
                raise NotImplementedError("Required repeat length register not found")
            if self.repeat_len_field_name is None:
                raise NotImplementedError("Required repeat length field name not specified")
            if self.repeat_en_register is None:
                raise NotImplementedError("Required repeat enable register not found")

            self.repeat_len_register.write(**{self.repeat_len_field_name: repeat_n})
        if repeat_n == 0:
            self.repeat_en_register.write(en=0)
        else:
            self.repeat_en_register.write(en=1)


class NoiseSource(Source):
    """A collection of items representing a Noise Source in a DSim design.

    :param scale_register:
        Register attribute from CasperFpga object.
    :param name:
        Name of Noise Source.
    """

    def __init__(self, scale_register, name: str) -> None:
        self.scale_register = scale_register
        super(NoiseSource, self).__init__(scale_register, name)

    @property
    def scale(self):
        """Get the scale of the Noise Source."""
        return self.scale_register.read()["data"]["scale"]

    def set(self, scale=None):
        """Set the scale of the Noise Source."""
        # TODO work in 'human' units such W / Hz?
        write_vars = remove_nones(dict(scale=scale))
        self.scale_register.write(**write_vars)


# endregion


class Output(object):
    """Class to collate output data type properties.

    :param name:
    :param scale_register:
    :param control_register:
    """

    def __init__(self, name: str, scale_register: Register, control_register: Register) -> None:
        self.name = name
        self.scale_register = scale_register
        self.control_register = control_register
        self.tgv_select_field = "tvg_select" + self.name

    @property
    def output_type(self) -> str:
        """Currently-selected output type."""
        if self.control_register.read()["data"][self.tgv_select_field] == 0:
            return "test_vectors"
        else:
            return "signal"

    @property
    def scale(self) -> None:
        """Scale factor of Output."""
        return self.scale_register.read()["data"]["scale"]

    def select_output(self, output_type) -> None:
        """Switch between Test Vector Generator and Signal output types."""
        if output_type == "test_vectors":
            self.control_register.write(**{self.tgv_select_field: 0})
        elif output_type == "signal":
            self.control_register.write(**{self.tgv_select_field: 1})
        else:
            raise ValueError('Valid output_type values: "test_vectors" and "signal"')

    def scale_output(self, scale: int) -> None:
        """Apply scale to Output signal.

        #TODO: Probably need to rename this method, or change naming conventions.

        :param scale:

        """
        self.scale_register.write(scale=scale)


class FpgaDsimHost(CasperFpga):
    """
    An FpgaHost that acts as a Digitiser unit.

    :param host:
        Hostname or IP address of SKARAB to program as a DSim.
    :param katcp_port:
        Port on which SKARAB receives communications - defaults to 7147.
    :param fpgfilename:
        Name of fpg file to program.
    :param config_file:
        config.ini file to parse for DSim configuration details.
    :param config_dict:
        Config dict that might already be parsed - would certainly save us some time/memory.
        - If present, will override the presence of a config_file.
    """

    def __init__(self, host, katcp_port=7147, fpgfilename=None, config_file=None, config_dict=None, **kwargs):
        super().__init__(self, host=host, katcp_port=katcp_port, transport=SkarabTransport, **kwargs)
        self.config_dict = None
        if config_dict is not None:
            self.config_dict = config_dict
        else:
            self.config_dict = parse_config_file(config_file)["dsimengine"]
        # Although can't we get the fpg file from the config?
        self.fpgfilename = fpgfilename

        self.sine_sources = AttributeContainer()
        self.noise_sources = AttributeContainer()
        self.outputs = AttributeContainer()

    def get_system_information(self):
        """Get system information and build D-engine sources."""
        self.get_system_information(self.fpgfilename)

        self.sine_sources.clear()
        self.noise_sources.clear()
        self.outputs.clear()

        for reg in self.registers:
            sin_name = get_prefixed_name("freq_cwg", reg.name)
            noise_name = get_prefixed_name("scale_wng", reg.name)
            output_scale_name = get_prefixed_name("scale_out", reg.name)
            if sin_name is not None:
                scale_reg_postfix = "_" + sin_name if reg.name.endswith("_" + sin_name) else sin_name
                scale_reg = getattr(self.registers, "scale_cwg" + scale_reg_postfix)

                repeat_en_reg_name = "rpt_en_cwg" + scale_reg_postfix
                repeat_len_reg_name = "rpt_length_cwg{}".format(scale_reg_postfix)

                repeat_en_reg = getattr(self.registers, repeat_en_reg_name, None)
                repeat_len_reg = getattr(self.registers, repeat_len_reg_name, None)
                repeat_len_field_name = "repeat_length"

                setattr(
                    self.sine_sources,
                    "sin_" + sin_name,
                    SineSource(
                        reg,
                        scale_reg,
                        sin_name,
                        repeat_len_register=repeat_len_reg,
                        repeat_en_register=repeat_en_reg,
                        repeat_len_field_name=repeat_len_field_name,
                    ),
                )
            elif noise_name is not None:
                setattr(self.noise_sources, "noise_" + noise_name, NoiseSource(reg, noise_name))
            elif output_scale_name is not None:
                # TODO TEMP hack due to misnamed register
                if output_scale_name.startswith("arb"):
                    continue
                setattr(
                    self.outputs, "out_" + output_scale_name, Output(output_scale_name, reg, self.registers.control)
                )

    def initialise(self) -> None:
        """Program and initialise the DSim host."""
        if not self.is_connected():
            self.connect()
        if self.fpgfilename:
            self._program()
        else:
            self.logger.info("Not programming host {} since no fpgfilename is configured".format(self.host))

        if not self.is_running():
            raise RuntimeError("D-engine {host} not running".format(**self.__dict__))

        self.get_system_information(self.fpgfilename)
        self.setup_gbes()

        # Set digitizer polarisation IDs, 0 - h, 1 - v
        self.registers.receptor_id.write(pol0_id=0, pol1_id=1)
        self.data_resync()

        # Default to generating test-vectors
        for output in self.outputs:
            output.select_output("test_vectors")

    def reset(self) -> None:
        """Reset DSim."""
        self.registers.control.write(mrst="pulse")

    def data_resync(self) -> None:
        """Start the local timer on the test d-engine - mrst, then a fake sync."""
        self.reset()
        self.registers.control.write(msync="pulse")

    def enable_data_output(self, enabled: bool = True) -> None:
        """En/disable 10GbE data output."""
        pol_tx_reg = self.registers.pol_tx_always_on
        reg_vals = {n: enabled for n in pol_tx_reg.field_names() if n.endswith("_tx_always_on")}
        pol_tx_reg.write(**reg_vals)

        if "gbe_control" in self.registers.names():
            self.registers.gbe_control.write_int(15 if enabled else 0)
        elif "gbecontrol" in self.registers.names():
            self.registers.gbecontrol.write_int(15 if enabled else 0)

        if enabled:
            self.registers.control_output.write(load_en_time="pulse")

    def pulse_data_output(self, n_pkts: int) -> None:
        """
        Produce a data output pulse of n_pkts per polarisation.

        Does nothing if data is already being transmitted.
        """
        register_list = [r for r in self.registers if re.match(r"^pol\d_num_pkts$", r.name)]
        for r in register_list:
            r.write(**{r.name: n_pkts})
        reg_field_names = self.registers.pol_traffic_trigger.field_names()
        self.registers.pol_traffic_trigger.write(**{n: "pulse" for n in reg_field_names})

    def _program(self) -> None:
        """Program the fpg file and ensure 40GbE core is not transmitting."""
        self.logger.info(f"Programming {self.host} with file {self.fpgfilename}")
        stime = time.time()
        self.upload_to_ram_and_program(self.fpgfilename)
        self.logger.info("Programmed {} in {:.2f} seconds.".format(self.host, time.time() - stime))

        # Ensure data is not sent before the gbes are configured
        self.enable_data_output(False)

    def setup_gbes(self) -> None:
        """Set up the 40GbE core on a SKARAB DSim."""
        port = StreamAddress.from_address_string(self.config_dict["pol0_destination_ips"].strip()).port
        gbe = self.gbes[self.gbes.names()[0]]
        if gbe.get_port() != port:
            gbe.set_port(port)

        self.write_int("gbe_porttx", port)
        for pol in [0, 1]:
            addr = StreamAddress.from_address_string(self.config_dict["pol%1i_destination_ips" % pol].strip())
            for index in range(addr.ip_range):
                # The interleave-by-pairs of IP addresses is implemented by this formula.
                self.write_int("gbe_iptx%1i" % (2 * index - index % 2 + 2 * pol), addr.ip_address.ip_int + index)

        self.registers.control.write(gbe_rst=False)


# end
