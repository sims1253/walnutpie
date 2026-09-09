C++ API
=======

Sampling functionality
----------------------

Top-level call
______________

This function will spawn threads to perform end-to-end sampling

.. doxygenfunction:: walnutpie::walnuts


Iterator-style samplers
_______________________

These classes implement Walnuts in an iteration-per-call style iterator.

.. doxygenclass:: walnutpie::WalnutsSampler
   :members:
.. doxygenclass:: walnutpie::AdaptiveWalnuts
   :members:

Configuration
-------------

The following classes (and their builders) are used to configure Walnuts.

.. doxygenclass:: walnutpie::WalnutsConfig
   :members:

.. doxygenclass:: walnutpie::InitConfigBuilder
   :members:
.. doxygenclass:: walnutpie::WarmupConfigBuilder
   :members:
.. doxygenclass:: walnutpie::SamplingConfigBuilder
   :members:


.. doxygenclass:: walnutpie::InitConfig
.. doxygenclass:: walnutpie::WarmupConfig
.. doxygenclass:: walnutpie::SamplingConfig


Concepts
--------

The following concepts describe the types expected by `walnutpie`.

.. doxygenconcept:: walnutpie::LogpGrad
.. doxygenconcept:: walnutpie::ErrorCallback
.. doxygenconcept:: walnutpie::SampleHandler
.. doxygenconcept:: walnutpie::ChainHandler
.. doxygenconcept:: walnutpie::GlobalHandler
.. doxygenconcept:: walnutpie::InterruptCallback

Experimental ridge-budget cap
-----------------------------

The opt-in ``ridge_guard`` compares final chain positions, not chain means.
A large dispersion ratio does not establish a likelihood-null ridge or
convergence failure. The default threshold of zero disables the guard.

``ridge_min_micro`` is the positive cap on the replacement minimum micro
steps per macro step when the guard fires (default 128). Zero is invalid.
The nominal floor of 16 is clipped to the cap, so caps from 1 through 15
remain valid. Very large demands saturate before conversion to an integer.
The replacement can lower the adapted budget; it does not always raise it.
This cap is not a global limit on adaptation or total trajectory cost.

The cap correction does not establish improved ESS/s or ridge recovery.
The detector remains experimental.
