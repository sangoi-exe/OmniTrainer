from enum import Enum


class ValidationTimestepMode(Enum):
    AUTO = "AUTO"
    FIXED = "FIXED"
    STRATIFIED = "STRATIFIED"

    def __str__(self):
        return self.value
