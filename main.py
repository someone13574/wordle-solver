import os
import numpy as np
from PIL import Image

def feedback_to_pattern(feedback: str) -> int:
    pattern = 0
    for idx, letter in enumerate(feedback):
        if letter == "2":
            pattern += 3 ** idx * 2
        elif letter == "1":
            pattern += 3 ** idx
    return pattern

def generate_pairwise_patterns(word_list):
    # load if pre-generated
    if os.path.exists("pattern-lookup.npy"):
        with open("pattern-lookup.npy", "rb") as f:
            return np.load(f)
    print("generating lookup")

    # equality check
    a = np.broadcast_to(word_list[:, np.newaxis, :], (word_list.shape[0], word_list.shape[0], 5))
    b = np.broadcast_to(word_list[np.newaxis, :, :], (word_list.shape[0], word_list.shape[0], 5))
    equality = np.equal(a, b)
    pattern = equality.astype(np.uint8) * (3 ** np.arange(5) * 2).astype(np.uint8)[np.newaxis, np.newaxis, :]
    pattern = pattern.sum(axis=2)

    # remove consumed letters (partial matching doesn't work with matched)
    a = np.where(equality, np.zeros_like(a), a)
    b = np.where(equality, np.full_like(b, '0'), b)

    # check each letters of b in order so letters can be consumed
    for idx in range(5):
        partial_match = a == b[:, :, idx][:, :, np.newaxis]
        match_indices = partial_match.argmax(axis=2)[:, :, np.newaxis]
        has_match = partial_match.sum(axis=2) != 0

        original_vals = np.take_along_axis(a, match_indices, axis=2).squeeze(2)
        set_val = np.where(has_match, np.zeros_like(original_vals), original_vals)[:, :, np.newaxis]
        np.put_along_axis(a, match_indices, set_val, axis=2)
        pattern += has_match.astype(np.uint8) * 3 ** idx
    
    # save
    with open("pattern-lookup.npy", "wb") as f:
        np.save(f, pattern)

    return pattern


word_list = []
with open("valid_words.txt", "r") as reader:
    for line in reader:
        word_list.append(list(line.strip()))

word_list = np.array(word_list)
pattern = generate_pairwise_patterns(word_list)

valid_words = word_list
guess_idx = 0
while True:
    if guess_idx == 0:
        max_word_idx = 10800
    else:
        max_word_idx = 0
        max_entropy = 0.0
        print(f"{pattern.shape[0]} possible solutions remain")
        for word_idx in range(word_list.shape[0]):
            solution_prob = (word_list[word_idx] == valid_words).all(1).any(0) / valid_words.shape[0]

            _, counts = np.unique(pattern[:, word_idx], return_counts=True)
            probs = counts.astype(np.float32)
            probs /= pattern.shape[0]
            probs = -np.log2(probs) * probs
            entropy = probs.sum() + solution_prob

            if entropy > max_entropy:
                max_entropy = entropy
                max_word_idx = word_idx
    guess_idx += 1
    print(valid_words)
    print("try:", word_list[max_word_idx])
    feedback_pattern = feedback_to_pattern(input("feedback: "))
    if feedback_pattern == 242:
        quit()
    
    indices, = np.where(pattern[:, max_word_idx] == feedback_pattern)

    pattern = np.take_along_axis(pattern, indices[:, np.newaxis], axis=0)
    valid_words = np.take_along_axis(valid_words, indices[:, np.newaxis], axis=0)
