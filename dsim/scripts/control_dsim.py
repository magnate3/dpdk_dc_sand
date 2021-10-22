################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Control script for DSim operation."""

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

DEFAULT_KATCP_PORT = 7147
DEFAULT_CONFIG_FILENAME = "test_dsim_config.json"


@dataclass
class Band:
    """Holds presets for a known frequency band."""

    adc_sample_rate: float
    centre_frequency: float


BANDS = {
    "l": Band(adc_sample_rate=1712e6, centre_frequency=1284e6),
    "u": Band(adc_sample_rate=1088e6, centre_frequency=816e6),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command-line arguments (which may be specified as a parameter)."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        type=str,
        help="Already-generated JSON config file to configure DSim. \
            If this is present it overrides other command-line args.",
    )

    parser.add_argument("--host", type=str, help="Hostname or IP address of the DSim.")
    # TODO: Change the --use-skarab default to False once in production
    parser.add_argument("--use-skarab", action="store_true", default=True, help="Use SKARAB as the DSim.")
    parser.add_argument("--use-cpu", action="store_true", default=False, help="Use host server as the DSim.")

    parser.add_argument(
        "--sine-source",
        action="append",
        default=[],
        nargs=3,
        help="Choose which sine to source, sin_0 or sin_1. Set Scale and Frequency.",
    )
    parser.add_argument(
        "--noise-source",
        action="append",
        default=[],
        nargs=2,
        help="Choose which Noise to source, noise_0 or noise_1. Set Noise Scale.",
    )
    parser.add_argument(
        "--output-select",
        action="append",
        default=[],
        nargs=3,
        help="Choose which Output to source from, Output_0 or Output_1. "
        "Output types, choose from signal or test_vectors."
        "Output scale, choose a decimal value from 0-1.",
    )
    # parser.add_argument(
    #     "--output-scale",
    #     action="append",
    #     default=[],
    #     nargs=2,
    #     help="Choose which Output to source from, Output 0 or Output 1. Output Scale, choose between 0 - 1.",
    # )

    parser.add_argument("--band", default="l", choices=["l", "u"], help="Band ID [%(default)s].")
    parser.add_argument("--adc-sample-rate", type=float, help="ADC sample rate in Hz [from --band]")
    parser.add_argument(
        "-f", "--fpg", dest="fpgfilename", default=None, action="store", help="fpg filename to program to SKARAB DSim."
    )
    parser.add_argument(
        "--initialise",
        action="store_true",
        default=False,
        help="Initialise DSim host with input configuration, e.g. Program SKARAB DSim with input --fpg file.",
    )
    # --start and --stop are mainly to be explicit in Enabling/Disabling data Tx
    # - i.e. It's a bit clearer straight away vs --enable-tx True/False
    parser.add_argument("--start", action="store_true", default=False, help="Start data Tx from DSim [%(default)s].")
    parser.add_argument("--stop", action="store_true", default=False, help="Stop data Tx from DSim [%(default)s].")
    parser.add_argument(
        "--ipython", action="store_true", default=False, help="Enter an IPython session once configuration is complete."
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="DEBUG",
        help="Log level to use for logging DSim operations, options: {info, debug, error}.",
    )
    parser.add_argument(
        "-w",
        "--write",
        # default=DEFAULT_CONFIG_FILENAME,
        type=str,
        help="Write to configuration data to file in JSON format.",
    )

    args = parser.parse_args(argv)

    if args.config_file:
        abs_path = os.path.abspath(args.config_file)
        if not os.path.isfile(abs_path):
            raise ValueError("Config file provided is not valid.")
        # else: Continue!
        args.config_file = abs_path
        # TODO: It's likely that I'll need to unpack this JSON Config to run it through the checks below
        return args

    # Might as well do some rudimentary error-checking of inputs here before moving on
    if args.host is None:
        # TODO: Don't be so passive-aggressive
        raise ValueError("Host identifier not provided, where do I run the DSim?")
    if args.use_skarab and args.use_cpu:
        # Problem
        raise ValueError("Cannot use SKARAB and CPU-based DSim at the same time.")
    if args.adc_sample_rate is None:
        args.adc_sample_rate = BANDS[args.band].adc_sample_rate
    if args.start and args.stop:
        raise ValueError("Cannot Start AND Stop data Tx, please pick one and re-run.")

    if args.write:
        abs_dest_path = os.path.abspath(args.write)
        abs_path, filename = os.path.split(abs_dest_path)
        if not os.path.exists(abs_path):
            raise ValueError("Destination path for config file is not valid.")
        # else: Make sure the filename is legit
        json_filename = f"{filename.split('.')[0]}.json"
        args.write = f"{abs_path}/{json_filename}"

    return args


def generate_config(args: argparse.Namespace) -> dict:
    """Produce the configuration dict from the parsed command-line arguments.

    Values should be error-checked before arriving to be generated as JSON data.

    Returns
    -------
    dict
        Config dictionary in JSON format.
    """
    config: dict = {
        "dsim_config": {},
        "sine_source": {},
        "noise_source": {},
        "outputs": {},
    }

    config["dsim_config"]["host"] = args.host

    if args.use_skarab:
        config["dsim_config"]["use_skarab"] = args.use_skarab
        config["dsim_config"]["fpgfilename"] = args.fpgfilename
    else:
        # I think this logic is sound?
        config["dsim_config"]["use_cpu"] = args.use_cpu

    config["dsim_config"]["adc_sample_rate"] = args.adc_sample_rate
    config["dsim_config"]["initialise"] = args.initialise

    # Do need to have both items populated
    config["dsim_config"]["start"] = args.start
    config["dsim_config"]["stop"] = args.stop

    if len(args.sine_source) > 0:
        for sine_id, scale_str, freq_str in args.sine_source:
            config["sine_source"][sine_id] = (scale_str, freq_str)
    if len(args.noise_source) > 0:
        for noise_id, noise_scale in args.noise_source:
            config["noise_source"][noise_id] = noise_scale
    if len(args.output_select) > 0:
        for output_id, output_type, output_scale in args.output_select:
            config["outputs"][output_id] = (output_type, output_scale)

    return config


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Launch the DSim.

    Well, not just yet. Currently ensuring input command-line args are sound.
    """
    args = parse_args(argv)

    config = generate_config(args)

    if args.ipython:
        import IPython

        IPython.embed()

    if args.write:
        with contextlib.ExitStack() as exit_stack:
            f = exit_stack.enter_context(open(args.write, "w"))
            json.dump(config, f, indent=4)
            f.write("\n")  # json.dump doesn't write a final newline
        return 0
    else:
        # TODO: Start DSim activities!
        raise NotImplementedError("Not doing anything else just yet...")


if __name__ == "__main__":
    sys.exit(main())
