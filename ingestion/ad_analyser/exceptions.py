class AdAnalyserError(Exception):
    pass

class DataValidationError(AdAnalyserError):
    pass

class ContentDownloadError(AdAnalyserError):
    pass

class IntegrationError(AdAnalyserError):
    pass

class ConfigError(AdAnalyserError):
    pass
