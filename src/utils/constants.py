"""Project-wide constants, enums, and configuration values.

This module is the single source of truth for all shared constants.
Every team member imports from here — never hardcode these values.

Owner: P1 (Data & Config Lead)
Consumers: P2, P3, P4, P5
"""

from enum import Enum

# ---------------------------------------------------------------------------
# Domain Enums
# ---------------------------------------------------------------------------

class ModalityType(str, Enum):
    """Imaging modality types available in the radiology department."""

    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"


class VisitType(str, Enum):
    """How the patient arrived at the department."""

    SCHEDULED = "scheduled"
    WALK_IN = "walk_in"
    EMERGENCY = "emergency"


class UrgencyLevel(str, Enum):
    """Clinical urgency classification."""

    EMERGENCY = "emergency"
    URGENT = "urgent"
    ROUTINE = "routine"


class ShiftType(str, Enum):
    """Staff shift classifications."""

    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"


class AgeGroup(str, Enum):
    """Patient age group buckets."""

    PEDIATRIC = "0-18"
    YOUNG_ADULT = "18-40"
    MIDDLE_AGED = "40-60"
    SENIOR = "60+"


class InsuranceType(str, Enum):
    """Insurance/payment classification."""

    GOVERNMENT = "government"
    PRIVATE = "private"
    SELF_PAY = "self_pay"


class ExamComplexity(str, Enum):
    """Imaging exam complexity level."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class WeatherCategory(str, Enum):
    """India-specific seasonal weather categories."""

    SUMMER = "summer"
    MONSOON = "monsoon"
    WINTER = "winter"


class DistanceCategory(str, Enum):
    """Patient travel distance classification."""

    LOCAL = "local"
    CITY = "city"
    OUTSTATION = "outstation"


class SchedulingPolicy(str, Enum):
    """Available scheduling policies for simulation comparison."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    SHORTEST_JOB_FIRST = "sjf"
    WAVE = "wave"
    RADQUEUE_AI = "radqueue_ai"
    RADQUEUE_NOSHOW = "radqueue_noshow"


class HospitalTier(str, Enum):
    """Hospital size/type classification (Indian context)."""

    TIER_1 = "tier_1"  # Government Teaching Hospital (250+ patients/day)
    TIER_2 = "tier_2"  # District Hospital (100-150 patients/day)
    TIER_3 = "tier_3"  # Private Multi-Specialty (80-120 patients/day)


# ---------------------------------------------------------------------------
# Service Time Parameters (from Indian hospital studies — §6.1)
# Log-normal distribution parameters: (mu, sigma)
# ---------------------------------------------------------------------------

SERVICE_TIME_PARAMS: dict[ModalityType, dict[str, float]] = {
    ModalityType.XRAY: {"mu": 1.6, "sigma": 0.35, "median_minutes": 6.0},
    ModalityType.CT: {"mu": 3.0, "sigma": 0.4, "median_minutes": 23.0},
    ModalityType.MRI: {"mu": 3.7, "sigma": 0.35, "median_minutes": 45.0},
    ModalityType.ULTRASOUND: {"mu": 1.8, "sigma": 0.4, "median_minutes": 7.0},
}

# ---------------------------------------------------------------------------
# Scheduling Constants
# ---------------------------------------------------------------------------

URGENCY_WEIGHTS: dict[UrgencyLevel, int] = {
    UrgencyLevel.EMERGENCY: 10,
    UrgencyLevel.URGENT: 5,
    UrgencyLevel.ROUTINE: 1,
}

FAIRNESS_PENALTY_RATE: float = 0.1  # Per minute over average wait
EMERGENCY_BUFFER_FRACTION: float = 0.10  # Reserve 10% capacity for emergencies

MAX_WAIT_ROUTINE_MINUTES: int = 90
MAX_WAIT_URGENT_MINUTES: int = 30
MAX_WAIT_EMERGENCY_MINUTES: int = 10

# Book a wait-listed standby patient when an appointment's no-show risk is at least this
OVERBOOKING_PROBABILITY_THRESHOLD: float = 0.30

# ---------------------------------------------------------------------------
# Operational Constants
# ---------------------------------------------------------------------------

REGISTRATION_TIME_MEAN_MINUTES: float = 5.0
REGISTRATION_TIME_STD_MINUTES: float = 2.0

RADIOLOGIST_REPORTING_TIME_MEAN_MINUTES: float = 12.0
RADIOLOGIST_REPORTING_TIME_STD_MINUTES: float = 5.0

TECHNOLOGIST_PREP_TIME_MEAN_MINUTES: float = 4.0
TECHNOLOGIST_PREP_TIME_STD_MINUTES: float = 2.0

# ---------------------------------------------------------------------------
# Indian OPD Patterns
# ---------------------------------------------------------------------------

PEAK_HOURS: list[tuple[int, int]] = [(9, 11), (14, 16)]  # 9-11 AM, 2-4 PM
WALK_IN_RATIO_GOVERNMENT: float = 0.70
WALK_IN_RATIO_PRIVATE: float = 0.30
NOSHOW_RATE_RANGE: tuple[float, float] = (0.15, 0.25)
EMERGENCY_RATIO_RANGE: tuple[float, float] = (0.08, 0.12)

AVG_DAILY_PATIENTS_TIER_1: int = 200  # Midpoint of 150-250
AVG_DAILY_PATIENTS_TIER_2: int = 125  # Midpoint of 100-150
AVG_DAILY_PATIENTS_TIER_3: int = 100  # Midpoint of 80-120

# ---------------------------------------------------------------------------
# Model Training Constants
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

OPTUNA_N_TRIALS_DEFAULT: int = 100
EARLY_STOPPING_PATIENCE: int = 50

# ---------------------------------------------------------------------------
# Evaluation Targets (from §12)
# ---------------------------------------------------------------------------

TARGET_MAE_MINUTES: float = 5.0
TARGET_RMSE_MINUTES: float = 8.0
TARGET_R_SQUARED: float = 0.85
TARGET_MAPE_PERCENT: float = 15.0

TARGET_NOSHOW_AUC_ROC: float = 0.82
TARGET_NOSHOW_AUC_PR: float = 0.65
TARGET_NOSHOW_F1: float = 0.70

TARGET_WAIT_REDUCTION_PERCENT: float = 30.0
TARGET_UTILIZATION_PERCENT: float = 80.0

# ---------------------------------------------------------------------------
# Data Generation Constants
# ---------------------------------------------------------------------------

SIMULATION_DAYS_DEFAULT: int = 180
DEPARTMENT_OPEN_HOUR: int = 8  # 8 AM
DEPARTMENT_CLOSE_HOUR: int = 20  # 8 PM
MINUTES_PER_DAY: int = 1_440
SECONDS_PER_MINUTE: int = 60
