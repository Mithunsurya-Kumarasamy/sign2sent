from typing import List

class SentenceGenerator:
    """
    Translates a sequence of individual sign language tokens into a grammatically
    correct English sentence using template matching and fallback rules.
    """
    def __init__(self):
        # Predefined grammar templates mapping a tuple of signs to an English sentence.
        self.templates = {
            ("I", "NEED", "WATER"): "I need water.",
            ("I", "WANT", "FOOD"): "I want food.",
            ("HELLO", "HOW", "YOU"): "Hello, how are you?",
            ("MY", "NAME", "IS"): "My name is ", # Will require dynamic appending
            ("WHAT", "IS", "YOUR", "NAME"): "What is your name?",
            ("GOOD", "MORNING"): "Good morning!",
            ("GOOD", "NIGHT"): "Good night!",
            ("THANK YOU",): "Thank you.",
            ("PLEASE", "HELP"): "Please help me.",
            ("WHERE", "IS", "COLLEGE"): "Where is the college?",
            ("YES",): "Yes.",
            ("NO",): "No."
        }

    def generate_sentence(self, sequence: List[str]) -> str:
        """
        Attempts to match the current sequence to a template.
        If no exact match, falls back to a best-effort concatenation.
        """
        if not sequence:
            return ""

        seq_tuple = tuple(sequence)

        # 1. Exact Match
        if seq_tuple in self.templates:
            return self.templates[seq_tuple]

        # 2. Sub-sequence Matching / Dynamic Patterns
        # e.g., "MY", "NAME", "X"
        if len(sequence) >= 3 and sequence[0] == "MY" and sequence[1] == "NAME":
            name = " ".join(sequence[2:]).title()
            return f"My name is {name}."

        if len(sequence) >= 2 and sequence[0] == "I" and sequence[1] == "WANT":
            rest = " ".join(sequence[2:]).lower()
            return f"I want {rest}."

        if len(sequence) >= 2 and sequence[0] == "I" and sequence[1] == "NEED":
            rest = " ".join(sequence[2:]).lower()
            return f"I need {rest}."

        # 3. Fallback: Capitalize first word, lowercase the rest, add period.
        fallback_sentence = " ".join(sequence).capitalize()
        return fallback_sentence + "."
        
    def reset(self):
        """
        Reset method in case the generator needs to hold state in future expansions.
        Currently stateless.
        """
        pass

if __name__ == "__main__":
    # Test cases
    generator = SentenceGenerator()
    print(generator.generate_sentence(["HELLO", "HOW", "YOU"]))
    print(generator.generate_sentence(["I", "NEED", "WATER"]))
    print(generator.generate_sentence(["MY", "NAME", "MITHUN"]))
    print(generator.generate_sentence(["FRIEND", "SCHOOL"]))
