## Task

Marvin's three-class problem: given a piece of text, decide whether it is not a movie/TV review (0), a positive review (1), or a negative review (2). The dataset has roughly 70k training rows and 17.6k test rows; submissions are scored by macro-F1 on Kaggle.

The training distribution is mildly skewed toward class 0:

| label | meaning           | count  | share |
|-------|-------------------|--------|-------|
| 0     | not a review      | 32,289 | 45.9% |
| 1     | positive review   | 19,139 | 27.2% |
| 2     | negative review   | 18,889 | 26.9% |

## Statistical Analysis

Two quick analyses on the training set turn out to be very informative for modeling decisions.

**Document length**: Class 0 is dramatically shorter than the two review classes; classes 1 and 2 are almost identical on length:

| class            | mean words | median words | mean chars | median chars |
|------------------|-----------:|-------------:|-----------:|-------------:|
| 0 (non-review)   |         77 |           33 |        459 |          203 |
| 1 (positive)     |        207 |          152 |      1,171 |          842 |
| 2 (negative)     |        208 |          157 |      1,164 |          869 |

So length is a strong cue for *review vs. non-review*, but carries essentially zero information about *positive vs. negative*. The model has to rely on lexical content for the latter distinction.

**Distinguishing vocabulary**: Top unigrams per class by smoothed log-odds vs. the rest of the corpus (`min_df=20`, `max_df=0.5`, top-10 shown):

- 0: `kaj`, `行者道`, `руб`, `сколько`, `по`, `на`, `八戒道`, `說著`, `estis`, `raskolnikov` 
- 1: `fido`, `kinnear`, `eisenstein`, `matthau`, `sammo`, `visconti`, `bakshi`, `kazan`, `kieslowski`, `grint`
- 2: `boll`, `bloodrayne`, `uwe`, `hobgoblins`, `manos`, `snooze`, `odious`, `unwatchable`, `awfulness`, `stinker`

The class 0 reviews are mostly non-English text and excerpts from public-domain literature. Class 1's top tokens are proper nouns of acclaimed directors and actors. Class 2 splits cleanly into infamous "bad movie" landmarks and explicit negative-sentiment vocabulary.

This shapes the modeling choices that follow:

- Word n-grams are needed for sentiment phrases and the proper-noun clusters that signal critical praise.
- Character n-grams earn their keep on class 0, where the word tokenizer struggles with mixed-script content and rare proper names.
- Predicting class 1 vs. class 2 is the genuinely hard sub-task — they are length-equivalent and stylistically similar; only word-level sentiment cues separate them.

## Approach

A single scikit-learn `Pipeline`:

1. Light text cleaning: strip HTML residue (`<br />` is everywhere in this corpus) and collapse whitespace; nothing else.
2. `FeatureUnion` of two TF-IDF vectorizers: word 1–2 grams and character 3–5 grams (`analyzer="char_wb"`).
3. `LogisticRegression(class_weight="balanced", max_iter=1000)`: multinomial via the `lbfgs` solver.

The two vectorizers complement each other well. Word n-grams capture obvious sentiment cues like “not recommend” or “waste of time”, while character n-grams pick up morphology (`disappoint*`), misspellings, and stylistic features that the word tokenizer misses entirely (review punctuation like `!!!`, ratings like `10/10`). Stacking them via `FeatureUnion` lets logistic regression weigh both views jointly.

## Preprocessing

Deliberately minimal:

- `re.sub(r"<[^>]+>", " ", text)` — drop all HTML tags, replace with a space (so adjacent words don't get glued).
- `re.sub(r"\s+", " ", text).strip()` — collapse whitespace and embedded newlines.
- `fillna("")` — guard against missing values.

Things I intentionally did not do:

- No stopword removal. `not`, `don't`, `never` carry the entire signal for class 2; tossing them is asking to mispredict negatives.
- No stemming/lemmatization. With sublinear TF and char n-grams, the model already handles morphological variation; explicit stemmers tend to plateau at break-even.
- No punctuation/digit stripping. `!!!`, `???`, `10/10`, `2 stars` all carry sentiment. it lets TF-IDF down-weight what isn't useful.

The vectorizers themselves do an additional layer of normalization: `lowercase=True`, `strip_accents="unicode"`, `min_df=2`, `max_df=0.95` (corpus-adaptive stop-word filter), and `sublinear_tf=True` (log-scale term frequencies, which helps a lot with long IMDB-style reviews).

## Results

On a 6,000-row stratified subsample, 5-fold CV macro-F1 was 0.8859 ± 0.0124 (folds: 0.9092 / 0.8734 / 0.8861 / 0.8823 / 0.8787). Training on the full 70k took ~3 minutes on CPU. Predicted-label distribution on the test set (45.4% / 28.0% / 26.6%) tracks the training distribution almost exactly.

Final Kaggle leaderboard score: 0.92932

## Code

Repository: <https://github.com/uazhlt-ms-program/grad-level-term-project-kaggle-competition-zhex-ua>

## Replication

The whole pipeline runs in a Docker container. no local Python setup required. On a Linux/macOS machine with Docker installed:

**1. Clone the repo.**

```bash
git clone https://github.com/uazhlt-ms-program/grad-level-term-project-kaggle-competition-zhex-ua.git
cd grad-level-term-project-kaggle-competition-zhex-ua
```

**2. Get the dataset.**

Download `train.csv` and `test.csv` from the Kaggle competition page and place them under `data/`. 

The `data/` directory is intentionally untracked — Docker mounts it at runtime, so the files never need to live inside the image.

**3. Build the image.**

```bash
docker build -t marvin-baseline .
```

This pulls `python:3.10-slim` (~150 MB) and installs the four pinned dependencies from [`requirements.txt`](requirements.txt) (`numpy`, `scipy`, `pandas`, `scikit-learn`). First build takes ~1 minute.

**4. Train and produce the submission.**

```bash
docker run --rm -v "$PWD/data:/app/data" marvin-baseline \
    --no-cv --output data/submission.csv
```

This loads `data/train.csv`, fits the pipeline on all ~70k rows (~3 minutes on CPU), predicts `data/test.csv`, and writes `data/submission.csv` (the `data/` mount means the file persists on the host after the container exits).

With cross-validation: For a local macro-F1 estimate before submitting, drop `--no-cv` — the script then runs 5-fold stratified CV on the full training set first, then refits on everything and writes the same submission.

```bash
# 5-fold CV
docker run --rm -v "$PWD/data:/app/data" marvin-baseline --output data/submission.csv

# 3-fold CV instead
docker run --rm -v "$PWD/data:/app/data" marvin-baseline \
    --cv-folds 3 --output data/submission.csv

# to reproduce the 6k-subsample CV number reported in the Results section:
docker run --rm -v "$PWD/data:/app/data" marvin-baseline --quick 6000
```

