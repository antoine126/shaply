"""Backports for standard-library features not available on Python 3.10.

Kept in one place so the rest of the codebase can do
``from shaply._compat import StrEnum`` without scattering version checks.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum as StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of :class:`enum.StrEnum` (added in Python 3.11).

        Reproduces its behaviour on Python 3.10, where plain ``(str, Enum)``
        members format inconsistently: this makes ``str(member)`` and
        f-string formatting both return the plain value instead of
        ``"ClassName.MEMBER"``, matching the native 3.11+ type exactly.
        """

        def __str__(self) -> str:
            return str(self.value)
