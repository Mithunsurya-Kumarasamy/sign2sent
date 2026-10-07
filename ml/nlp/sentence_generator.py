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
            ("HELLO", "HOW", "ARE", "YOU"): "Hello, how are you?",
            ("MY", "NAME", "IS"): "My name is...",
            ("WHAT", "IS", "YOUR", "NAME"): "What is your name?",
            ("GOOD", "MORNING"): "Good morning!",
            ("GOOD", "NIGHT"): "Good night!",
            ("THANK YOU",): "Thank you.",
            ("PLEASE", "HELP"): "Please help me.",
            ("WHERE", "IS", "COLLEGE"): "Where is the college?",
            ("YES",): "Yes.",
            ("NO",): "No.",
            ("HELLO",): "Hello!"
        }

    def generate_sentence(self, sequence: List[str]) -> str:
        """
        Attempts to match the current sequence to a template.
        If no exact match, falls back to a best-effort concatenation.
        """
        if not sequence:
            return ""

        # Normalize tokens (trim whitespace and uppercase)
        clean_seq = [s.strip().upper() for s in sequence if s and s.strip()]
        if not clean_seq:
            return ""

        seq_tuple = tuple(clean_seq)

        # 1. Exact Match
        if seq_tuple in self.templates:
            return self.templates[seq_tuple]

        # 2. Greeting + Introduction Pattern
        # e.g., ["HELLO", "MY", "NAME", "IS", "JOHN"] or ["HELLO", "MY", "NAME", "JOHN"]
        if len(clean_seq) >= 3 and clean_seq[0] == "HELLO" and clean_seq[1] == "MY" and clean_seq[2] == "NAME":
            name_tokens = clean_seq[3:]
            if name_tokens and name_tokens[0] == "IS":
                name_tokens = name_tokens[1:]
            if name_tokens:
                name = " ".join(name_tokens).title()
                return f"Hello, my name is {name}."
            return "Hello, my name is..."

        # 3. Introduction Pattern without greeting
        # e.g., ["MY", "NAME", "MITHUN"] or ["MY", "NAME", "IS", "JOHN"]
        if len(clean_seq) >= 2 and clean_seq[0] == "MY" and clean_seq[1] == "NAME":
            name_tokens = clean_seq[2:]
            if name_tokens and name_tokens[0] == "IS":
                name_tokens = name_tokens[1:]
            if name_tokens:
                name = " ".join(name_tokens).title()
                return f"My name is {name}."
            return "My name is..."

        # 4. "I WANT / NEED" Patterns
        if len(clean_seq) >= 2 and clean_seq[0] == "I" and clean_seq[1] == "WANT":
            rest = " ".join(clean_seq[2:]).lower()
            return f"I want {rest}." if rest else "I want..."

        if len(clean_seq) >= 2 and clean_seq[0] == "I" and clean_seq[1] == "NEED":
            rest = " ".join(clean_seq[2:]).lower()
            return f"I need {rest}." if rest else "I need..."

        # 5. Question patterns: "WHERE [IS] <PLACE>"
        if len(clean_seq) >= 2 and clean_seq[0] == "WHERE":
            rest_tokens = clean_seq[1:]
            if rest_tokens and rest_tokens[0] == "IS":
                rest_tokens = rest_tokens[1:]
            if rest_tokens:
                place = " ".join(rest_tokens).lower()
                return f"Where is the {place}?"

        # 6. Fallback: Capitalize first word, lowercase the rest, add period.
        fallback_sentence = " ".join(clean_seq).capitalize()
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
