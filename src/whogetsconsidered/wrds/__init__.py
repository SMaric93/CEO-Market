"""Optional WRDS ingestion tools for bootstrapping canonical research inputs.

The main package is designed around cleaned CSV/Parquet extracts supplied by the researcher.
This subpackage adds a separate, opt-in bootstrap path that can pull project-relevant WRDS
tables, stage them locally, and materialize whatever canonical inputs WRDS can credibly
support without pretending that all required research inputs live on WRDS.
"""

from whogetsconsidered.wrds.client import WrdsClientProtocol, build_wrds_client
from whogetsconsidered.wrds.puller import pull_wrds_bundle

__all__ = ["WrdsClientProtocol", "build_wrds_client", "pull_wrds_bundle"]
