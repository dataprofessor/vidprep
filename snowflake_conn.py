"""Snowflake connection helper — uses Streamlit's managed connection, which
reads `.streamlit/secrets.toml` locally and embedded identity when hosted in
Snowflake (Streamlit in Snowflake). Same code path both places."""

import streamlit as st


def run_query(sql: str, params=None):
    """Execute a single SQL statement and return all rows."""
    cur = st.connection("snowflake").cursor()
    cur.execute(sql, params)
    return cur.fetchall()


def put_file(local_path: str, stage_fqn: str) -> str:
    """Upload a local file to the given stage via PUT, overwriting if present.

    Returns the relative path (file name) as it appears on the stage.
    """
    import os

    cur = st.connection("snowflake").cursor()
    # PUT requires a file:// URI and does not accept bind params for the path.
    abs_path = os.path.abspath(local_path).replace("'", "\\'")
    cur.execute(
        f"PUT 'file://{abs_path}' '@{stage_fqn}' AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
    )
    return os.path.basename(local_path)
