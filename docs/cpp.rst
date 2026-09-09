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

Adaptation exit statistics
---------------------------

``walnutpie::detail::adapt_with_stats(init, warmup, adapters, interrupt)``
returns the controller's mass/step aggregates and ``log_mass_dispersion``.
The latter is the mean, over coordinates, of the sample variance of log mass
across chains. One finite chain gives zero. Nonfinite log masses or an
unrepresentable variance give NaN. Empty or mismatched inputs are rejected.

These are snapshots read when the controller exits, not necessarily the final
worker states after joining. The stopping rule and per-chain tuning are
unchanged. Collecting statistics can still affect scheduling near an early-stop
boundary; exact multi-chain draw identity is not promised for that case.
The legacy ``adapt`` path does not compute this diagnostic. This is an internal
C++ interface (``detail``), not a new high-level or Python sampling option.
Scale disagreement is not proof of multimodality or convergence.
