class BankError(Exception):
    """Expected operational failure with a process exit code."""

    exit_code = 1


class UsageError(BankError):
    exit_code = 2


class DeviceSelectionError(UsageError):
    pass


class RecipeConflict(UsageError):
    pass


class PilotBlocked(BankError):
    """Full batch refused because the engineering pilot gate is not accepted."""

    exit_code = 3


class EngineeringGateFailed(BankError):
    """Pilot finished, models retained, engineering gate not accepted."""

    exit_code = 4
