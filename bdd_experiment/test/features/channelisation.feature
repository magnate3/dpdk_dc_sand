Feature: channelisation

  Scenario: Requirement ABC

    Given a SKARAB dsim
    And a random channel
    When sweeping frequencies across a channel and capturing heaps
    Then the peak is in the centre of the channel
    And response outside the channel is below -62 dB
