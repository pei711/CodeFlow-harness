"""Evolution plugin for the small-real subject.

Everything in this package is measurement code: task definitions, the grading
subprocess, the result readers, the designer, and the ``BenchBundle`` entry
point. ``codeflow.evolver.applier.path_guard.IMMUTABLE_PATTERNS`` lists
``benchmarks/appworld/evolve/`` as evolver-immutable, so no candidate can edit
the code that grades it.
"""
