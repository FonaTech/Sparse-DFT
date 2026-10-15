class SparseDFTError(Exception):
    """Base Sparse-DFT error with a machine-readable code."""

    code = "SDFT_ERROR"

    def __init__(self, message: str, *, details=None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def as_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}


class InputError(SparseDFTError):
    code = "INPUT_ERROR"


class ArchiveIntegrityError(SparseDFTError):
    code = "PAW_ARCHIVE_INTEGRITY_ERROR"


class UnsupportedError(SparseDFTError):
    code = "UNSUPPORTED"
