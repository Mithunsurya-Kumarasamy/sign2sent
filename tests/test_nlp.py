import pytest
from ml.nlp.sentence_generator import SentenceGenerator

def test_exact_template_match():
    gen = SentenceGenerator()
    
    sentence = gen.generate_sentence(["HELLO", "HOW", "YOU"])
    assert sentence == "Hello, how are you?"
    
    sentence = gen.generate_sentence(["I", "NEED", "WATER"])
    assert sentence == "I need water."

def test_dynamic_pattern_match():
    gen = SentenceGenerator()
    
    sentence = gen.generate_sentence(["MY", "NAME", "MITHUN"])
    assert sentence == "My name is Mithun."
    
    sentence = gen.generate_sentence(["I", "WANT", "COFFEE"])
    assert sentence == "I want coffee."

def test_fallback_grammar():
    gen = SentenceGenerator()
    
    # Should capitalize the first word and add a period
    sentence = gen.generate_sentence(["DOG", "RUN", "FAST"])
    assert sentence == "Dog run fast."

def test_empty_sequence():
    gen = SentenceGenerator()
    assert gen.generate_sentence([]) == ""
