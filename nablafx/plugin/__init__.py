"""End-to-end CLAP plugin driver: checkpoint -> ONNX bundle -> .clap.

The heavy lifting (ONNX export, metadata composition) lives in
``nablafx.export``. This module just orchestrates the hand-off to the native
build step in ``native/clap/`` on macOS, and prints rsync-style instructions
on other platforms.
"""
