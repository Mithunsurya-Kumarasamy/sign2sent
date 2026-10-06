# Predefined vocabulary for Sign2Sent
VOCABULARY = [
    "HELLO",
    "GOOD",
    "MORNING",
    "NIGHT",
    "THANK YOU",
    "PLEASE",
    "HELP",
    "WATER",
    "FOOD",
    "HOME",
    "SCHOOL",
    "COLLEGE",
    "FRIEND",
    "NAME",
    "WHAT",
    "WHERE",
    "HOW",
    "YES",
    "NO",
    "I",
    "YOU",
    "WE",
    "NEED",
    "WANT"
]

# Mapping from word to index
LABEL_MAP = {word: idx for idx, word in enumerate(VOCABULARY)}

# Mapping from index to word
IDX_MAP = {idx: word for word, idx in LABEL_MAP.items()}

def get_num_classes():
    return len(VOCABULARY)
