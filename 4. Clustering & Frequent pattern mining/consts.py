NUMBER_OF_CLUSTERS = 7
COLS_TO_MINE = ["Injury Severity", "Agency Name", "ACRS Report Type", "Route Type", "Cross-Street Type"]
FPGROWTH_PARAMS = {
    'ASSOCIATION_NUM_ATTRIBUTES': 4,
    'SUPPORT_NUM_ATTRIBUTES': 5,
    'ASSOCIATION_RULE_THRESHOLD': 0.8,
    'MIN_LENGTH_OF_PATTERN': 3,
    'SUPPORT_THRESHOLD_PERCENTAGE': 40,
    'MAX_NUM_OF_FREQUENT_PATTERNS': 30
}