"""Subject package named ``appworld`` to match the registered bench key.

``codeflow.evolver.launch.registry.BENCHES`` maps the bench name ``appworld`` to the
module path ``benchmarks.appworld.evolve.entry:build`` and imports it from the
*subject* repo, so a subject that wants to run under that key must carry this
package layout. Nothing here talks to the real AppWorld environment.
"""
