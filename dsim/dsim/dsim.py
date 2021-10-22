"""Top-level interface to DSim control for katgpucbf."""

from . import dsim_skarab


class DSim:
    """The top-level interface to DSim implementations used for katgpucbf's integration testing.

    Parameters
    ----------
    host
        Hostname or IP Address from which DSim data will be transmitted.
    port
        Port on which DSim data will be transmitted.
    adc_sample_rate
        DSim sample rate (Hz).
    mcast_addrs
        Multicast addresses that DSim data will be available on.
    sine_source_config
        A dictionary of the Sine Source configuration containing
        - Source ID, scale and frequency.
    noise_source_config
        A dictionary of the Noise Source configuration containing
        - ID and scale.
    """

    def __init__(
        self,
        host: str,
        port: int,
        adc_sample_rate: float,
        use_skarab: bool = True,
        **kwargs,
    ) -> None:

        # List class attributes for type hinting, largely for readability

        self.host = host
        self.port = port
        self.adc_sample_rate = adc_sample_rate

        self.transport = None
        if use_skarab:
            self.transport = dsim_skarab.SkarabDsim(host, **kwargs)
        else:
            # CPU implementation is not supported, sorry!
            raise NotImplementedError("CPU implementation is not supported just yet.")

    def initialise(self) -> None:
        """Initialise the DSim with the provided configuration.

        It should have all the required fields populated by now.
        """
        self.transport.initialise()

    def reset(self) -> None:
        """Reset the DSim."""
        self.transport.reset()

    def start_data_tx(self) -> None:
        """Start DSim data transmission."""
        self.transport.enable_data_output(enabled=True)

    def stop_data_tx(self) -> None:
        """Stop DSim data transmission."""
        self.transport.enable_data_output(enabled=False)

    def resync(self) -> None:
        """Resynchronise data transmission from the DSim."""
        self.transport.resync()

    def shutdown(self) -> None:
        """Shutdown the DSim."""
        raise NotImplementedError
