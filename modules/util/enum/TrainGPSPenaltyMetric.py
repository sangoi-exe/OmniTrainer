from enum import Enum


class TrainGPSPenaltyMetric(Enum):
    MSE = "MSE"
    COSINE = "COSINE"

    def __str__(self):
        return self.value
