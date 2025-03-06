import pandas as pd
import pyfpgrowth
import math

def mine_frequent_patterns(df, columns_to_mine, **kwargs):
    df = df[columns_to_mine]
    df = df.astype(str)
    # print(len(df))
    # print(df.dtypes)
    # df = df.dropna()
    df = df.values.tolist()

    # Check that the association_rule_attributes and support_num_attributes are less than the length of the columns_to_mine
    if kwargs['ASSOCIATION_NUM_ATTRIBUTES'] > len(columns_to_mine):
        print("association_num_attributes is greater than the length of columns_to_mine")
        exit(1)
    if kwargs['SUPPORT_NUM_ATTRIBUTES'] > len(columns_to_mine):
        print("support_num_attributes is greater than the length of columns_to_mine")
        exit(1)
    if kwargs['MIN_LENGTH_OF_PATTERN'] > len(columns_to_mine):
        print("The minimum number of attributes in frequent pattern is greater than the length of columns_to_mine")
        exit(1)

    # Calculate support_threshold from SUPPORT_THRESHOLD_PERCENTAGE
    support_threshold = math.floor(len(df) * (kwargs['SUPPORT_THRESHOLD_PERCENTAGE'] / 100))

    patterns = pyfpgrowth.find_frequent_patterns(df, support_threshold)
    rules = pyfpgrowth.generate_association_rules(patterns, kwargs['ASSOCIATION_RULE_THRESHOLD'])

    top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop Frequent Patterns: {len(top_patterns)}\n")
    
    freq_pattern_count = 0
    frequent_patterns = {}

    for pattern, support in top_patterns:
        
        if freq_pattern_count <= kwargs['MAX_NUM_OF_FREQUENT_PATTERNS']:
            frequent_patterns[pattern] = support
            if len(pattern) == kwargs['MIN_LENGTH_OF_PATTERN']:
                print(f"{pattern}: support = {support}")
                freq_pattern_count += 1
        else:
            frequent_patterns = {}
            break    

    if freq_pattern_count < kwargs['MAX_NUM_OF_FREQUENT_PATTERNS']:
        print(f"\nOnly {freq_pattern_count} frequent patterns found with \n\tMIN_LENGTH_OF_PATTERN = {kwargs['MIN_LENGTH_OF_PATTERN']}\n\tMAX_NUM_OF_FREQUENT_PATTERNS = {kwargs['MAX_NUM_OF_FREQUENT_PATTERNS']}")
    else:
        print(f"\nTop {kwargs['MAX_NUM_OF_FREQUENT_PATTERNS']} Frequent Patterns printed")

    print("\n______________________________________________________\n")

    # sorted_rules = sorted(rules.items(), key=lambda x: (len(x[0]), x[1][1]), reverse=True)
    # print("\nAssociation Rules:")
    # for rule, (consequent, confidence) in sorted_rules:
    #     print(f"{rule} -> {consequent}: confidence = {round(confidence, 2)}\n")

    print("\n______________________________________________________\n")
    print("\n______________________________________________________\n\n")

    return frequent_patterns

# Usage example:
# df = pd.read_csv('data/crash_reporting_drivers_data_sanitized.csv')
# columns_to_mine = ["Injury Severity", "Agency Name", "ACRS Report Type", "Route Type", "Cross-Street Type"]
# kwargs = {
#     'ASSOCIATION_NUM_ATTRIBUTES': 4,
#     'SUPPORT_NUM_ATTRIBUTES': 5,
#     'MIN_LENGTH_OF_PATTERN': 3,
#     'SUPPORT_THRESHOLD_PERCENTAGE': 10,
#     'MAX_NUM_OF_FREQUENT_PATTERNS': 30,
#     'ASSOCIATION_RULE_THRESHOLD': 0.8
# }

# mine_frequent_patterns(df, columns_to_mine, **kwargs)

# frequent_patterns[pattern] = support
#         if counter != kwargs['MAX_NUM_OF_FREQUENT_PATTERNS']:
#             pass
#         else: 
#             if len(pattern) == kwargs['MIN_LENGTH_OF_PATTERN']:
#                 print(f"{pattern}: support = {support}")
#                 counter += 1