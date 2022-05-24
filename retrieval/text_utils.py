import random

def repeat(text, n_max_gram=3, n_max_repeat=3):
    """repeat n-grams"""
    tokens = text.split()

    n_gram = random.randint(1, n_max_gram)

    repeat_token_idx = random.randint(0, len(tokens) - n_gram)

    repeated_tokens = tokens[repeat_token_idx:repeat_token_idx+n_gram]

    n_repeat = random.randint(1, n_max_repeat)
    for _ in range(n_repeat):
        insert_idx = random.randint(0, len(tokens))
        tokens = tokens[:insert_idx] + \
            repeated_tokens + tokens[insert_idx:]

    new_text = " ".join(tokens)
    return new_text

def remove(text, n_max_gram=3):
    """remove n-grams"""
    tokens = text.split()

    n_gram = random.randint(1, n_max_gram)

    remove_token_idx = random.randint(0, len(tokens) - n_gram)

    tokens = tokens[:remove_token_idx] + tokens[remove_token_idx + n_gram:]

    new_text = " ".join(tokens)
    return new_text

def insert(text, vocab, n_max_tokens=3):
    """Insert tokens"""
    tokens = text.split()

    n_insert_token = random.randint(1, n_max_tokens)

    for _ in range(n_insert_token):
        insert_token_idx = random.randint(0, len(tokens) - 1)
        insert_token = random.choice(vocab)
        tokens = tokens[:insert_token_idx] + [insert_token] + tokens[insert_token_idx:]

    new_text = " ".join(tokens)
    return new_text

def swap(text, vocab, n_max_tokens=3):
    """Swap tokens"""
    tokens = text.split()

    n_swap_tokens = random.randint(1, n_max_tokens)

    for _ in range(n_swap_tokens):
        swap_token_idx = random.randint(0, len(tokens) - 1)

        swap_token = random.choice(vocab)
        while swap_token == tokens[swap_token_idx]:
            swap_token = random.choice(vocab)

        tokens[swap_token_idx] = swap_token

    new_text = " ".join(tokens)
    return new_text

def shuffle(text):
    """shuffle tokens"""
    tokens = text.split()

    random.shuffle(tokens)

    new_text = " ".join(tokens)
    return new_text
