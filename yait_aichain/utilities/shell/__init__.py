"""``Shell(folder)``: a folder as a utility, with four tools sharing it —
``browse``, ``read``, ``write`` and ``run``. The folder defaults to the working
directory; ``run`` starts there but is not confined by it.
See docs/utilities/index.md.
"""

from ._browse import LocalBrowseTool
from ._read import LocalReadTool
from ._write import LocalWriteTool
from ._run import LocalRunTool

__all__ = ["Shell", "LocalBrowseTool", "LocalReadTool", "LocalWriteTool", "LocalRunTool"]




# ── The utility ────────────────────────────────────────────────────────────
from ._utility import Shell
