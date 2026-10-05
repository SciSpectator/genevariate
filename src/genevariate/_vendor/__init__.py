"""Vendored third-party packages.

This file exists so setuptools' ``packages.find`` treats ``_vendor`` as a
package and therefore ships ``geo_label_extractor`` with the distribution.
Without it the directory is skipped entirely at install time, and
``geo_extract_driver._ensure_path`` — which locates the vendored pipeline
relative to its own installed path — has nothing to point at, so both
``genevariate --llm-extract`` and the ``genevariate-llm-extract`` console
script fail on any pip-installed copy while still working from a source
checkout.
"""
