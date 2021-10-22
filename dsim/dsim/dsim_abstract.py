"""Abstract Base Class for DSims to implement."""

from abc import ABC, abstractmethod


class AbstractDsim(ABC):
    """Abstract DSim class to be implemented by a desired platform.

    This Abstract Base Class provides no error-checking of parameters, and requires
    the implementing Class to ensure

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
    """

    def __init__(self, host: str, port: int, adc_sample_rate: float, **kwargs) -> None:

        self.host = host
        self.port = port
        self.adc_sample_rate = adc_sample_rate

    @abstractmethod
    def initialise(self) -> bool:
        """Initialise the DSim with the provided configuration."""
        pass

    @abstractmethod
    def reset(self) -> bool:
        """Reset the DSim.

        Returns
        -------
        boolean
            Success/Fail - True/False.
        """
        pass

    @abstractmethod
    def enable_data_output(self, enable: bool) -> bool:
        """Enable/Disable DSim data transmission.

        Parameters
        ----------
        enable
            Boolean to indicate enabling or disabling of data Tx.

        Returns
        -------
        boolean
            Success/Fail - True/False.
        """
        pass

    @abstractmethod
    def resync(self) -> bool:
        """Resynchronise data transmission from the DSim.

        Returns
        -------
        boolean
            Success/Fail - True/False.
        """
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """Halt the DSim's operations and shut it down.

        Returns
        -------
        boolean
            Success/Fail - True/False.
        """
        pass

    @abstractmethod
    def get_dsim_status(self) -> None:
        """Get the status of the DSim.

        The manner in which this is done is left entirely to the implementor.
        """
        pass
