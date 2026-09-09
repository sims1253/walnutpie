C++ API
=======

Sampling functionality
----------------------

Top-level call
______________

This function will spawn threads to perform end-to-end sampling.
The target is shared across chain threads and must support concurrent calls.
The generic callable API cannot inspect captured model state or prove thread safety.
For Stan targets, ``STAN_THREADS=true`` is a build prerequisite, not a guarantee
that custom functions, callbacks or other shared state are safe.
``DynamicStanModel::stan_threads()`` returns an optional boolean; check for
known true explicitly, not just whether metadata is present.

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
