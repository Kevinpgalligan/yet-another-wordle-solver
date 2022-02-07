# For comparison...
#   https://news.ycombinator.com/item?id=30050231

import collections
import sys
import argparse
import time
import math
import tqdm

parser = argparse.ArgumentParser(description="""A Wordle solver.

Sample usage:

  python3 wordle.py --continue-from '' --output-to /tmp/blah secrets.txt guesses.txt
  python3 wordle.py --continue-from 'guess,BGYBB;hello,BBGGG' secrets.txt guesses.txt
  python3 wordle.py --evaluate 'guess,hello' secrets.txt guesses.txt""")

parser.add_argument("secrets", help="Path to file that contains possible secrets, one per line. Secrets can also be used as guesses.")
parser.add_argument("guesses", help="Path to file that contains possible guesses, one per line.")
parser.add_argument("--continue-from", help="Figure out the best next guess, given guesses so far in the form 'guess,BGYBB;hello,BBGGG'.")
parser.add_argument("--output-to",
    help="If a path is provided via this flag command, FULL OUTPUT from certain commands is written to the corresponding file.")
parser.add_argument("--evaluate",
    help="Evaluate the performance of a greedy algorithm after it uses the comma-separated initial guesses that you provide.")
parser.add_argument("--debug", action="store_true", default=False)
# Use this to run test cases.
parser.add_argument("--test", action="store_true", default=False, help=argparse.SUPPRESS)
args = parser.parse_args()

debug = args.debug

def load_words(path):
    with open(path, "r") as f:
        return f.read().strip().split("\n")

def count_letters(word):
    letter_counts = collections.defaultdict(int)
    for c in word:
        letter_counts[c] += 1
    return letter_counts

secrets = set(load_words(args.secrets))
guesses = set(load_words(args.guesses)) | secrets

secrets_counts = collections.defaultdict(set)
secrets_exact = collections.defaultdict(set)

# Cache info about about secret words.
for secret in secrets:
    letter_counts = count_letters(secret)
    for letter, count in letter_counts.items():
        for k in range(1, count+1):
            secrets_counts[(letter, k)].add(secret)
    for i, letter in enumerate(secret):
        secrets_exact[(letter, i)].add(secret)

def expected_reduced_set_size(guess, possibles):
    # The remaining words can be bucketed by the type
    # of hint they give for this guess. The size of a bucket
    # is the number of remaining possibilities if that hint
    # is returned. And the size of a bucket also determines
    # its probability, since the words are equally probable.
    hint_counts = collections.defaultdict(int)
    for secret in possibles:
        hint_counts[make_hint(guess, secret)] += 1
    result = 0
    for count in hint_counts.values():
        result += (count/len(possibles)) * count
    return result

def make_hint(guess, secret):
    secret_counts = count_letters(secret)
    sig = ["B" for _ in range(5)]
    # Need to make 2 passes so that the exact matches reduce the
    # count first. Otherwise, we could get YBGBB when it should
    # be BBGBB, guessing 'total' for 'peter'.
    for i, (l_guess, l_secret) in enumerate(zip(guess, secret)):
        if l_guess == l_secret:
            sig[i] = "G"
            secret_counts[l_guess] -= 1
    for i, (l_guess, l_secret) in enumerate(zip(guess, secret)):
        if l_guess != l_secret and l_guess in secret and secret_counts[l_guess] > 0:
            sig[i] = "Y"
            secret_counts[l_guess] -= 1
    return "".join(sig)

def reduce_set_inplace(guess, secret, possibles):
    reduce_set_with_hint_inplace(guess, make_hint(guess, secret), possibles)

def reduce_set_with_hint_inplace(guess, hint, possibles):
    exact_count_chars = set()
    match_counts = collections.defaultdict(int)
    for i, (l, h) in enumerate(zip(guess, hint)):
        if h == "G":
            possibles &= secrets_exact[(l, i)]
            match_counts[l] += 1
        elif h == "Y":
            match_counts[l] += 1
            possibles -= secrets_exact[(l, i)]
        elif h == "B":
            # Just to make sure it shows up with a count
            # of at least 0.
            match_counts[l] = match_counts[l]
            possibles -= secrets_exact[(l, i)]
            exact_count_chars.add(l)
        else:
            raise Exception("Unexpected hint type: " + h)
    for l, c in match_counts.items():
        if c > 0:
            possibles &= secrets_counts[(l, c)]
        if l in exact_count_chars:
            possibles -= secrets_counts[(l, c+1)]

def rank_guesses(possibles, print_progress=False):
    guess_ranking = []
    t0 = time.time()
    for i, guess in enumerate(guesses):
        guess_ranking.append((guess, expected_reduced_set_size(guess, possibles)))
        if i % math.ceil(len(guesses)/10) == 0 and print_progress:
            print(f"({time.time()-t0} sec) Finished evaluating", i, "out of", len(guesses), "guesses.")
    # First sort by the expected size of the reduced set, break ties
    # based on whether the word is in the set of possibile words.
    guess_ranking.sort(key=lambda p: (p[1], p[0] not in possibles))
    return guess_ranking

def evaluate_after_initial_guesses(secret, initial_guesses, fout=False):
    possibles = secrets.copy()
    guesses_so_far = initial_guesses.copy()
    for g in initial_guesses:
        reduce_set_inplace(g, secret, possibles)
    while (not guesses_so_far or guesses_so_far[-1] != secret) and len(guesses_so_far) < 6:
        guess_ranking = rank_guesses(possibles)
        next_guess = guess_ranking[0][0]
        reduce_set_inplace(next_guess, secret, possibles)
        guesses_so_far.append(next_guess)
    success = (guesses_so_far[-1] == secret)
    if fout:
        if success: fout.write("[WIN]")
        else: fout.write("[FAIL]")
        fout.write(f"[{secret}] ")
        fout.write(" -> ".join(guesses_so_far))
        fout.write("\n")
    return success, len(guesses_so_far)

if args.test:
    assert "GGGBG" == make_hint("swirl", "swill")
    assert "BBBYB" == make_hint("swill", "loopy")
    assert "GGGGG" == make_hint("hello", "hello")
    print("All tests passed.")
    sys.exit(0)

if args.continue_from is not None:
    possibles = secrets.copy()
    if args.continue_from:
        so_far = [pair.split(",") for pair in args.continue_from.split(";")]
    else:
        so_far = []
    for (guess, hint) in so_far:
        if guess not in guesses or len(guess) != len(hint):
            print("[ERROR] Invalid guess:", guess)
            sys.exit(1)
        reduce_set_with_hint_inplace(guess, hint, possibles)
    print("Remaining words:", possibles)
    if not possibles:
        print("[ERROR] Impossible sequence of hints, no words remaining.")
        sys.exit(1)
    
    guess_ranking = rank_guesses(possibles, print_progress=True)
    print("Best 5 guesses are:", guess_ranking[:5])
    print("Worst guess is:", guess_ranking[-1])
    if args.output_to:
        with open(args.output_to, "w") as f:
            for guess, erss in guess_ranking:
                f.write(guess)
                f.write(" ")
                f.write(str(erss))
                f.write("\n")
elif args.evaluate is not None:
    initial_guesses = args.evaluate.split(",")
    assert len(initial_guesses) < 6
    success_sum = 0
    total_guesses = 0
    guess_distr = collections.defaultdict(int)
    t0 = time.time()
    f = open(args.output_to, "w") if args.output_to else False
    for secret in tqdm.tqdm(secrets):
        success, num_guesses = evaluate_after_initial_guesses(secret, initial_guesses, fout=f)
        if success:
            success_sum += 1
            total_guesses += num_guesses
            guess_distr[num_guesses] += 1
    print("Successes:", success_sum, "out of", len(secrets), f"({100*(success_sum/len(secrets))}%)")
    print("In the success cases, there was a mean of", total_guesses/success_sum, "guesses.")
    print("Guess distribution:")
    for ng, count in sorted(guess_distr.items()):
        print("*", ng, "guess(es),", count, f"({100*(count/success_sum)}%)")
    if f:
        f.close()
else:
    print("[ERROR] No instructions provided!")
    sys.exit(1)

sys.exit(0)
