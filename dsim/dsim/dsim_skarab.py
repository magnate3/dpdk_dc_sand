"""Control for SKARAB-based DSim."""

import os
import re
from time import sleep, time

from casperfpga import CasperFpga
from casperfpga.attribute_container import AttributeContainer
from casperfpga.register import Register
from casperfpga.skarab_definitions import SkarabProgrammingError
from casperfpga.skarab_fileops import FpgProcessor
from casperfpga.transport_skarab import SkarabTransport

from .utils import StreamAddress, get_prefixed_name, parse_config_file, remove_nones

# import subprocess

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

        self.sample_rate_hz = float(self.parent.config_dict["sample_rate_hz"])
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

    def __init__(self, host, katcp_port=7147, fpgfilename=None, config_file=None, config_dict=None):
        super().__init__(host=host, katcp_port=katcp_port, transport=SkarabTransport)

        # Just adding it to test type-hinting
        self.transport: SkarabTransport

        self.config_dict = None
        if config_dict is not None:
            # But how do we confirm this is the dsimengine section of the Config file?
            self.config_dict = config_dict
        else:
            self.config_dict = parse_config_file(config_file)["dsimengine"]
        # Although can't we get the fpg file from the config?
        # self.fpgfilename = self.config_dict['bistream']
        self.fpgfilename = fpgfilename

        self.sine_sources = AttributeContainer()
        self.noise_sources = AttributeContainer()
        self.outputs = AttributeContainer()

    def get_system_info(self) -> None:
        """Get system information and build D-engine sources."""
        # Run the usual/core get_system_information to populate the Yellow Block objects
        # - Although, this is quite likely to be done during self._program()
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
            if not self._program():
                # TODO: Think of a better error message.
                raise SkarabProgrammingError("Failed to program DSim.")
        else:
            self.logger.info("Not programming host {} since no fpgfilename is configured".format(self.host))

        if not self.is_running():
            raise RuntimeError("D-engine {host} not running".format(**self.__dict__))

        self.get_system_info()
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

    def _program(self) -> bool:
        """Program the fpg file and ensure 40GbE core is not transmitting.

        :return:
            Boolean - True/False - Success/Fail.
        """
        self.logger.info(f"Programming {self.host} with file {self.fpgfilename}")

        # * File extension should really already have been checked

        # Moving this up here so we can store this tmp_binfile.bin in the same directory
        # as the programming utility.
        cwd = os.path.dirname(__file__)
        progska_dir = f"{cwd}/progska"
        # The utility call wasn't working with the absolute path
        os.chdir(progska_dir)

        upload_start_time = time()
        # binfilename = '/home/apatel/binfiles/fpgstream_' + str(os.getpid()) + '.bin'
        binfilename = "fpgstream_" + str(os.getpid()) + ".bin"
        fpg_processor = FpgProcessor(self.fpgfilename, bin_name=binfilename)
        _ = fpg_processor.make_bin()

        # * Clear SDRAM in preparation for programming
        self.transport.clear_sdram()

        # * progska
        # - Need to get CWD, as we can't use relative paths to access the built-utility
        # cwd = os.path.dirname(__file__)
        # progska_dir = f"{cwd}/progska"
        # # The utility call wasn't working with the absolute path
        # os.chdir(progska_dir)
        progska_utility = "./progska"
        binfile_arg = f"-f {binfilename}"
        chunk_size_arg = "-s 1988"

        # - Command-line call should resemble something like (-v for verbose logging):
        #   ./progska /path/to/prog_file.fpg -s 1988 -v <hostname_or_ipaddr>
        progska_cmdline_call = f"{progska_utility} {binfile_arg} {chunk_size_arg} -v {self.host}"  # noqa: F541
        # progska_cmdline_call_list = [progska_utility, binfile_arg, chunk_size_arg, "-v", self.host]

        # result = subprocess.run(progska_cmdline_call, check=True)
        # result = subprocess.run(progska_cmdline_call_list, capture_output=True, check=True)
        # result = subprocess.run(progska_cmdline_call_list, stderr=subprocess.STDOUT)
        result = os.system(progska_cmdline_call)

        if not result:
            # result = 0 when it runs successfully
            # - At least the few times I tried it manually
            if not self.finish_programming(upload_start_time):
                return False
            # else: Continue!
            os.remove(binfilename)
            return True
        else:
            return False

    def finish_programming(self, upload_start_time: time, timeout: int = 60) -> bool:
        """Carry out final steps once progska has uploaded the binary file.

        Lifted straight from casperfpga.transport_skarab.SkarabTransport.upload_to_ram_and_program.

        :param upload_start_time:
            The time at which the upload to SDRAM was started.
        :param timeout:
            Timeout (seconds) to wait before calling the Time of Death on the programming process.

        :return:
            Boolean - True/False - Success/Fail.
        """
        self.transport._sdram_programmed = True
        self.transport.boot_from_sdram()

        upload_time_total = time() - upload_start_time
        timeout = timeout + time()
        reboot_start_time = time()

        while timeout > time():
            if self.is_connected(retries=1):
                result, firmware_version = self.transport.check_running_firmware()
                if result:
                    reboot_time = time() - reboot_start_time
                    # TODO: Reformat into f-string
                    self.logger.info(
                        "Skarab is back up, in %.1f seconds (%.1f + %.1f) with FW ver "
                        "%s" % (upload_time_total + reboot_time, upload_time_total, reboot_time, firmware_version)
                    )
                    break
                else:
                    return False
            sleep(0.1)

        # Might be better to put this inside the if-statement once the board is back online
        self.get_system_information(self.fpgfilename)

        # Ensure data is not sent before the gbes are configured
        self.enable_data_output(False)
        return True

    def setup_gbes(self) -> None:
        """Set up the 40GbE core on a SKARAB DSim."""
        port = StreamAddress.from_address_string(self.config_dict["pol0_destination_ips"].strip()).port
        gbe = self.gbes[self.gbes.names()[0]]
        if gbe.get_port() != port:
            gbe.set_port(port)

        self.write_int("gbe_porttx", port)
        for pol_num in [0, 1]:
            pol_config_key = f"pol{pol_num}_destination_ips"
            addr = StreamAddress.from_address_string(self.config_dict[pol_config_key].strip())
            for index in range(addr.ip_range):
                # The interleave-by-pairs of IP addresses is implemented by this formula.
                # TODO: Reformat into f-string
                self.write_int("gbe_iptx%1i" % (2 * index - index % 2 + 2 * pol_num), addr.ip_address.ip_int + index)

        self.registers.control.write(gbe_rst=False)


# end
