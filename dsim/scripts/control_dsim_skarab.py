#!/usr/bin/python3
"""Script to launch and control a SKARAB Dsim."""

import argparse
import os
import sys
import time

from dsim.dsim_skarab import SkarabDsim
from dsim.utils import parse_config_file

parser = argparse.ArgumentParser(
    description="Control the dsim-engine (fake digitiser).", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "-c", "--config", dest="config_filename", default=None, type=str, action="store", help="corr2 config file."
)
parser.add_argument("--host", dest="host", action="store", help="Host name or IP address of intended SKARAB DSim.")
parser.add_argument(
    "-f", "--fpg", dest="fpgfilename", default=None, action="store", help="fpg filename (Optional unless debugging)"
)
parser.add_argument(
    "--program", dest="program", action="store_true", default=False, help="(re)program the fake digitiser"
)
parser.add_argument(
    "--deprogram", dest="deprogram", action="store_true", default=False, help="deprogram the fake digitiser"
)
parser.add_argument(
    "--start", dest="start", action="store_true", default=False, help="start the fake digitiser transmission"
)
parser.add_argument(
    "--status", dest="status", action="store_true", default=False, help="Checks the status of digitiser transmission"
)
parser.add_argument(
    "--pulse",
    action="store_true",
    default=False,
    help=(
        "Send a pulse of fake packets. Does nothing if digitiser "
        "is already transmitting, use --stop first. It will only "
        "work once until the dsim-engine is reset, e.g using "
        "--resync which can be used with --pulse (--resync will "
        "be applied first)."
    ),
)
parser.add_argument(
    "--pulse-packets",
    action="store",
    default=100,
    type=int,
    help="Send out out this many data packets per polarisation" " if --pulse is specified (default: %(default)d.",
)
parser.add_argument("--resync", action="store_true", default=False, help="Restart digitiser time and then sync")
parser.add_argument(
    "--stop", dest="stop", action="store_true", default=False, help="stop the fake digitiser transmission"
)
parser.add_argument(
    "--loglevel",
    dest="loglevel",
    action="store",
    default="DEBUG",
    help="log level to use, default None, options: {INFO, DEBUG, ERROR}",
)
parser.add_argument(
    "--ipython",
    action="store_true",
    default=False,
    help="Launch an embedded ipython shell once all is said " "and done",
)
parser.add_argument(
    "--sine-source",
    action="append",
    default=[],
    nargs=3,
    help="Choose which sine to source, sin_0 or sin_1. " "Set Scale and Frequency",
)
parser.add_argument(
    "--noise-source",
    action="append",
    default=[],
    nargs=2,
    help="Choose which Noise to source, noise_0 or noise_1. " "Set noise scale.",
)
parser.add_argument(
    "--output-type",
    action="append",
    default=[],
    nargs=2,
    help="Choose which Output to source from, Output_0 or "
    "Output_1. Output types, choose from signal or "
    "test_vectors.",
)
parser.add_argument(
    "--output-scale",
    action="append",
    default=[],
    nargs=2,
    help="Choose which Output to source from, Output 0 or " "Output 1. Output Scale, choose between 0 - 1.",
)
parser.add_argument(
    "--zeros-sine", action="store_true", default=False, help="Sets all sine sources to 0 scale and 0 Frequency."
)
parser.add_argument("--zeros-noise", action="store_true", default=False, help="Sets all  noise sources to 0 scale.")
parser.add_argument(
    "--repeat-sine",
    action="append",
    default=[],
    nargs=2,
    help="Force a sin source to repeat exactly every N "
    "samples. Setting N to 0 disables repeating. "
    "Arguments: sin_name, N. E.g --repeat-sin sin_0 4096",
)
args = parser.parse_args()


try:
    # if args.config_filename is not None:
    if not os.path.isfile(args.config_filename):
        # Problem
        raise ValueError(f"Config file {args.config_filename} is not a valid config file.")

    config_dict = parse_config_file(config_file=args.config_filename)
    dsim_config_dict = config_dict.get("dsimengine")

    if args.fpgfilename:
        print("Starting Digitiser with fpgfilename: {}".format(args.fpgfilename))

    dfpga = SkarabDsim(dsim_config_dict["host"], fpgfilename=args.fpgfilename, config_dict=dsim_config_dict)
    try:
        import logging  # noqa: F401

        dfpga.logger.setLevel(level=eval(f"logging.{args.loglevel.upper()}"))
    except AttributeError:
        # Default to DEBUG for now
        dfpga.logger.setLevel(10)

    print("Connected to {}.".format(dfpga.host))
except TypeError:
    raise RuntimeError("Config template was not parsed!!!")

if args.start and args.stop:
    raise RuntimeError("Start and stop? You must be crazy!")

something_happened = False
if args.program:
    dfpga.initialise()
    something_happened = True
else:
    # dfpga already has access to the fpg-file
    dfpga.get_system_info()

# # TODO HACK
# if 'gbecontrol' in dfpga.registers.names():
#     dfpga.registers.gbecontrol.write_int(15)
# if 'gbecontrol' in dfpga.registers.names():
#     dfpga.registers.gbecontrol.write_int(15)

# TODO HACK
if "cwg0_en" in dfpga.registers.names():
    dfpga.registers.cwg0_en.write(en=1)
    dfpga.registers.cwg1_en.write(en=1)
# /HACK

if args.deprogram:
    dfpga.deprogram()
    something_happened = True
    print("Deprogrammed {}.".format(dfpga.host))

if args.resync:
    dfpga.resync()
    sync_epoch = time.time()
    something_happened = True
    print("Reset digitiser timer and sync timer: {}".format(sync_epoch))

if args.start:
    # start tx
    print("Starting TX on {}".format(dfpga.host))
    sys.stdout.flush()
    dfpga.enable_data_output(enabled=True)
    dfpga.registers.control.write(gbe_txen=True)
    print("done.")
    sys.stdout.flush()
    something_happened = True

if args.status:
    # start tx
    sys.stdout.flush()
    errmsg = "function is broken! Missing register: forty_gbe_txctr"
    assert hasattr(dfpga.registers, "forty_gbe_txctr"), errmsg
    before = dfpga.registers.forty_gbe_txctr.read()["data"]["reg"]
    time.sleep(0.5)
    after = dfpga.registers.forty_gbe_txctr.read()["data"]["reg"]

    if before == after:
        print("Digitiser tx raw data failed.")
    else:
        print("Digitiser tx raw data success.")
    sys.stdout.flush()
    something_happened = True

if args.stop:
    dfpga.enable_data_output(enabled=False)
    print("Stopped transmission on {}.".format(dfpga.host))
    something_happened = True

if args.pulse:
    dfpga.pulse_data_output(args.pulse_packets)
    print("Pulsed {} packets per polarisation".format(args.pulse_packets))
    something_happened = True

if args.sine_source:
    for sine_name, xscale_s, yfreq_s in args.sine_source:
        xscale = float(xscale_s)
        yfreq = float(yfreq_s)
        try:
            sine_source = getattr(dfpga.sine_sources, "sin_{}".format(sine_name))
        except AttributeError:
            print("You can only select between sine sources: {}".format([ss.name for ss in dfpga.sine_sources]))
            sys.exit(1)
        try:
            sine_source.set(scale=xscale, frequency=yfreq)
        except ValueError:
            print("\nError, verify your inputs for sin_{}".format(sine_source.name))
            print("Max Frequency should be {}MHz".format(sine_source.max_freq / 1e6))
            print("Scale should be between 0 and 1")
            sys.exit(1)
        print("")
        print("sine source: {}".format(sine_source.name))
        print("scale: {}".format(sine_source.scale))
        print("frequency: {}".format(sine_source.frequency))
    something_happened = True

if args.zeros_sine:
    """
    Set all sine sources to zeros
    """
    sources_names = dfpga.sine_sources.names()
    for source in sources_names:
        try:
            sine_source = getattr(dfpga.sine_sources, "{}".format(source))
            sine_source.set(0, 0)
            print("sine source {}, set to {}.".format(sine_source.name, sine_source.scale))
        except Exception as exc:
            print("An error occured: {}".format(str(exc)))
            sys.exit(1)
    something_happened = True

if args.noise_source:
    for noise_sources, noise_scale_s in args.noise_source:
        noise_scale = float(noise_scale_s)
        try:
            source_from = getattr(dfpga.noise_sources, "noise_{}".format(noise_sources))
        except AttributeError:
            print("You can only select between noise sources:" " {}".format(dfpga.noise_sources.names)())
            sys.exit(1)
        try:
            source_from.set(scale=noise_scale)
        except ValueError:
            print("Valid scale input is between 0 - 1.")
            sys.exit(1)
        print("")
        print("noise source: {}".format(source_from.name))
        print("noise scale: {}".format(source_from.scale))
    something_happened = True

if args.zeros_noise:
    """
    Set all sine sources to zeros
    """
    sources_names = dfpga.noise_sources.names()
    for source in sources_names:
        try:
            noise_source = getattr(dfpga.noise_sources, "{}".format(source))
            noise_source.set(0)
            print("noise source {}, set to {}.".format(noise_source.name, noise_source.scale))
        except Exception as exc:
            raise RuntimeError("An error occurred: {}".format(str(exc)))
    something_happened = True

if args.output_type:
    for output_type, output_type_s in args.output_type:
        try:
            type_from = getattr(dfpga.outputs, "out_{}".format(output_type))
        except AttributeError:
            print("You can only select between, Output_0 or Output_1.")
            sys.exit(1)
        try:
            type_from.select_output(output_type_s)
        except ValueError:
            print("Valid output_type values: 'test_vectors' and 'signal'")
            sys.exit(1)
        print("\noutput selected: {}".format(type_from.name))
        print("output type: {}".format(type_from.output_type))
    something_happened = True
# ---------------------------------------------
if args.output_scale:
    for output_scale, output_scale_s in args.output_scale:
        scale_value = float(output_scale_s)
        try:
            scale_from = getattr(dfpga.outputs, "out_{}".format(output_scale))
        except AttributeError:
            print("You can only select between, {}".format(dfpga.outputs.names()))
            sys.exit(1)
        try:
            scale_from.scale_output(scale_value)
        except ValueError:
            print("Valid scale input is between 0 - 1.")
            sys.exit(1)
        """
        Check if it can read what was written to it!
        """
        print("")
        print("output selected: {}".format(scale_from.name))
        print("output scale: {}".format(scale_from.scale_register.read()["data"]["scale"]))
    something_happened = True

if args.repeat_sine:
    something_happened = True
    for sine_name, repeat in args.repeat_sine:
        try:
            sine = getattr(dfpga.sine_sources, "sin_{}".format(sine_name))
        except AttributeError:
            print("You can only select between sine sources: {}".format([ss.name for ss in dfpga.sine_sources]))
            sys.exit(1)
        try:
            sine.set(repeatN=int(repeat))
        except NotImplementedError:
            print(("Source repeat not implemented for source '{sine_name}'.".format(**locals())))
            sys.exit(1)

# f = dfpga
# f.registers.control.write(tvg_select0=1, tvg_select1=1)
# f.registers.pol_tx_always_on.write(pol0_tx_always_on=1, pol1_tx_always_on=1)
# f.registers.src_sel_cntrl.write(src_sel_0=0, src_sel_1=0)

if args.ipython:
    # Embedding ipython for debugging
    import IPython

    IPython.embed()
    something_happened = True

if not something_happened:
    parser.print_help()
# dfpga.disconnect()
