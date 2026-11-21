<div align="center">
  <h1>🎬 Movie Genre Story AI</h1>
  <h3>Movie Script Genre Prediction · Story Generation · Story Improvement</h3>
  <p>
    A Python NLP project that classifies movie-script text and provides
    Cohere-powered story-generation features through a Gradio interface.
  </p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/ML-scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/NLP-NLTK-8A2BE2?style=for-the-badge" alt="NLP">
    <img src="https://img.shields.io/badge/UI-Gradio-F97316?style=for-the-badge" alt="Gradio">
    <img src="https://img.shields.io/badge/Generation-Cohere-39594D?style=for-the-badge" alt="Cohere">
  </p>

  <p><strong>🍿 From movie scripts to genres—and story ideas to new narratives.</strong></p>
</div>

---

## Table of Contents

- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [How the Project Works](#how-the-project-works)
- [Dataset and Genre Labels](#dataset-and-genre-labels)
- [Model and Results](#model-and-results)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Using the Gradio App](#using-the-gradio-app)
- [Security and API Notes](#security-and-api-notes)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

---

## About the Project

**Movie Genre Story AI** is a movie-script text classification and storytelling project. It combines two separate features:

1. **Genre classification:** a locally trained machine-learning model predicts one genre from a movie-script excerpt.
2. **Story tools:** a Gradio interface sends prompts to the Cohere API to generate or improve story text.

The genre classifier is a traditional NLP and machine-learning pipeline; it does **not** generate stories. Story generation and story improvement are separate API-based features and are not trained using the genre-classification dataset.

> **Classification type:** The model predicts one genre per input. It is a single-label classifier, not a multi-label system that assigns several genres to the same script.

---

## Key Features

| Feature | What it does |
|:---|:---|
| 🎭 Genre prediction | Predicts one of nine genres from script text |
| 📊 Genre probabilities | Displays the classifier’s estimated probability for each genre |
| 📉 Confusion matrix | Compares actual and predicted labels in the notebook |
| ☁️ Word-cloud exploration | Creates sample word clouds for Drama, Action, and Comedy |
| ✍️ Story generation | Sends a genre, concept, tone, and complexity prompt to Cohere |
| 📝 Story improvement | Sends an existing story and a selected improvement focus to Cohere |
| 💾 Model persistence | Saves and loads the classifier, vectorizer, and genre mapping with Joblib |
| 🖥️ Gradio interface | Provides separate tabs for genre prediction, story generation, and story improvement |

The notebook’s word-cloud code combines only the first three available script examples for each of the selected genres. These visualizations are exploratory, not summaries of every script in those genres.

---

## How the Project Works

### 1. Genre Classification Pipeline

The classifier processes movie-script text through the following steps:

1. **Load the data** from `kaggle_movie_train.csv`.
2. **Remove the ID column** and map genre names to integer labels.
3. **Clean the text**:
   - Replace non-letter characters with spaces.
   - Convert text to lowercase.
   - Split text into words.
   - Remove English stopwords using NLTK.
   - Apply Porter stemming.
4. **Extract text features** using `CountVectorizer` with up to 10,000 features and unigram/bigram features.
5. **Train a classifier** using scikit-learn’s `MultinomialNB`.
6. **Predict a genre** and display the predicted label and estimated class probabilities.

```text
Movie-script text
       │
       ▼
Cleaning → lowercase → stopword removal → Porter stemming
       │
       ▼
CountVectorizer: up to 10,000 unigram and bigram features
       │
       ▼
Multinomial Naive Bayes
       │
       ▼
Predicted genre + estimated genre probabilities
```

### 2. Story Generation and Improvement

The Gradio app builds a prompt from the user’s selections and sends it to Cohere:

```text
Story idea / existing story
       │
       ▼
Gradio form and prompt construction
       │
       ▼
Cohere API
       │
       ▼
Generated or revised story text
```

The story-generation prompt requests a complete story of approximately 400–600 words. The actual output length and content can vary.

---

## Dataset and Genre Labels

The notebook expects these CSV files in its working directory, unless the file paths in the code are changed:

| File | Expected columns | Rows shown in the notebook | Purpose |
|:---|:---|---:|:---|
| `kaggle_movie_train.csv` | `id`, `text`, `genre` | 22,579 | Labeled data for training and evaluation |
| `kaggle_movie_test.csv` | `id`, `text` | 5,589 | Unlabeled examples used for sample predictions |

The 20% split used to evaluate the classifier is created from the labeled training data. It is **not** the same as the separate, unlabeled `kaggle_movie_test.csv` file.

The notebook maps the nine genres as follows:

| Genre | Label |
|:---|---:|
| other | 0 |
| action | 1 |
| adventure | 2 |
| comedy | 3 |
| drama | 4 |
| horror | 5 |
| romance | 6 |
| sci-fi | 7 |
| thriller | 8 |

The CSV files are not guaranteed to be included in this repository. Make sure you have the right to use and redistribute any dataset files you add.

---

## Model and Results

The notebook uses:

| Setting | Value |
|:---|:---|
| Feature extractor | `CountVectorizer` |
| Maximum features | 10,000 |
| N-gram range | Unigrams and bigrams `(1, 2)` |
| Classifier | `MultinomialNB` |
| Train/test split | 80% / 20%, `random_state=0` |
| Selected alpha in the notebook | `0.1` |

The notebook reports the following accuracy results:

| Experiment | Alpha | Reported accuracy |
|:---|---:|---:|
| Initial Naive Bayes model | 1.0 | 89.57% |
| Best result among the tested alpha values | 0.1 | 91.34% |

> **Evaluation caveat:** Treat 91.34% as an exploratory notebook result, not an independent final benchmark. The vectorizer is fitted on the full corpus before the split, and alpha is selected by repeatedly checking accuracy on the same held-out split. For a more reliable estimate, split the raw texts first, fit the vectorizer only on training data, tune alpha on a separate validation set, and evaluate once on an untouched test set.

The model-saving example refits the vectorizer and classifier on the full labeled corpus before saving the `.joblib` artifact for inference.

---

## Technology Stack

| Technology | Use |
|:---|:---|
| Python | Project programming language |
| NumPy and pandas | Data handling and numerical operations |
| NLTK | Stopword removal and Porter stemming |
| scikit-learn | Text vectorization, Naive Bayes, splitting, and evaluation |
| Matplotlib and Seaborn | Charts and confusion-matrix visualization |
| WordCloud | Exploratory word-cloud visualizations |
| Joblib | Saving and loading the trained model |
| Gradio | Interactive web interface |
| Cohere | External story-generation and improvement API |
| Jupyter Notebook | Interactive development and analysis |

---

## Repository Structure

The public repository root contains the README and the `labtherory/` project directory:

```text
movie-genre-classification/
├── README.md
└── labtherory/
    └── Project notebooks and Python source files
```

The code also expects or creates these runtime files:

| File | Role |
|:---|:---|
| `kaggle_movie_train.csv` | Labeled input used to train and evaluate the classifier |
| `kaggle_movie_test.csv` | Optional unlabeled input used for sample predictions |
| `movie_genre_classifier.joblib` | Saved classifier, vectorizer, and genre mapping loaded by the Gradio app |

The exact locations of the CSV files and model artifact must match the paths used by the notebook and app.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/krishnachaitanya941/movie-genre-classification.git
cd movie-genre-classification
```

### 2. Enter the project directory

```bash
cd labtherory
```

### 3. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 4. Install the packages used by the project

```bash
pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud joblib gradio cohere notebook
```

The notebook and app code download the NLTK English stopwords resource. If needed, download it manually:

```python
import nltk
nltk.download("stopwords")
```

---

## Running the Project

### Run the notebook

1. Put `kaggle_movie_train.csv` in the notebook’s working directory, or update the CSV path in the code.
2. Optionally put `kaggle_movie_test.csv` in the directory for the unlabeled prediction examples.
3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open the project notebook and run the cells in order.
5. Run the preprocessing, training, and model-saving cells to create `movie_genre_classifier.joblib`.

> **Important:** The model-saving `main()` example in the supplied code uses `corpus` and `df` created by earlier preprocessing steps. It is not standalone unless those variables are created in the same notebook or loaded in the script.

### Run the Gradio app

The app expects `movie_genre_classifier.joblib` at the configured model path. Run the Gradio code after creating that artifact.

If you save the Gradio code in a file named `app.py`, run:

```bash
python app.py
```

If the code is still in a notebook, run the cell that creates and launches the interface.

The supplied launch call uses `share=True`, which creates a shareable public app link. For local-only use, change it to:

```python
interface.launch(share=False)
```

---

## Using the Gradio App

### Genre Prediction

1. Open the **Genre Prediction** tab.
2. Paste a movie-script excerpt.
3. Click **Predict Genre**.
4. Review the predicted genre, estimated probabilities, and description.

The classifier’s probability values are model estimates; they should not be treated as guaranteed or calibrated confidence scores.

### Story Generator

1. Open the **Story Generator** tab.
2. Select a genre, tone, and complexity.
3. Enter a story concept.
4. Click **Generate Story**.

The prompt requests a 400–600-word narrative, but the returned length and details may differ.

### Story Improvement

1. Open the **Story Improvement** tab.
2. Paste the story you want to revise.
3. Select an improvement focus, such as character development or dialogue quality.
4. Click **Improve Story**.

---

## Security and API Notes

### Protect the Cohere API key

Never commit an API key to Python source code, a notebook, or this README. The app code should read the key from an environment variable instead of containing a literal key:

```python
import os
import cohere

co = cohere.Client(os.environ["COHERE_API_KEY"])
```

Set the variable in the terminal before launching the app.

**Windows PowerShell:**

```powershell
$env:COHERE_API_KEY = "YOUR_NEW_API_KEY"
```

**macOS / Linux:**

```bash
export COHERE_API_KEY="YOUR_NEW_API_KEY"
```

If a key has already been committed or shared, revoke it and create a replacement. Removing it from the latest file does not invalidate the exposed key.

### Cohere API compatibility

The supplied story functions call `co.generate(...)` and configure the model as `command-r-08-2024`. Cohere documents the Generate API as legacy and provides a migration path to the Chat API. The story-generation and story-improvement functions may need updating to work with the Cohere SDK/API version you install. See the [official Cohere documentation](https://docs.cohere.com/) for the currently supported endpoints, model names, and migration guidance.

The local genre classifier and its Joblib artifact are separate from Cohere; Cohere is used for the story features.

### Public Gradio links

Do not use `share=True` for private scripts or sensitive text. Anyone with access to the generated share link may be able to use the app while it is running.

---

## Limitations

- The classifier predicts **one genre only**.
- Bag-of-words features do not deeply understand plot, character, or narrative context.
- The dataset may have uneven numbers of examples per genre; accuracy alone may hide poor results on smaller classes.
- The reported accuracy is subject to the evaluation caveat described above.
- The notebook converts the feature matrix to a dense array with `.toarray()`, which can use substantial memory as the dataset grows.
- Cohere-backed story features require API access and may need code updates for SDK/API changes.
- The app’s CSS refers to a background image named `background .jpg`. Include an image at that path or update/remove the CSS reference.

---

## Future Enhancements

Possible next steps include:

- Use a scikit-learn `Pipeline` and fit the vectorizer only on training data.
- Tune model settings using a validation set and keep a separate test set untouched.
- Report macro-F1, precision, recall, and per-class performance alongside accuracy.
- Keep feature matrices sparse to reduce memory use.
- Compare CountVectorizer with TF-IDF or more advanced NLP models.
- Support multi-label genre prediction if the dataset provides multiple labels per movie.
- Migrate the Cohere story functions to the supported Chat API.
- Add a `requirements.txt` file with tested dependency versions.
- Add automated tests and clearer data/model path configuration.

These are potential improvements, not claims about features currently implemented.

---

## Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Clone your fork:

   ```bash
   git clone https://github.com/YOUR-USERNAME/movie-genre-classification.git
   ```

3. Create a branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. Make and test your changes.
5. Commit and push:

   ```bash
   git add .
   git commit -m "Describe your change"
   git push origin feature/your-feature-name
   ```

6. Open a pull request.

Please do not include API keys, private data, or dataset files unless their license permits redistribution.

---

## Acknowledgements

This project uses open-source Python tools and services, including NumPy, pandas, NLTK, scikit-learn, Matplotlib, Seaborn, WordCloud, Joblib, Jupyter, Gradio, and Cohere.

---

## Author

**Krishna Chaitanya**  
**Project:** Movie Genre Story AI

[![GitHub Repository](https://img.shields.io/badge/GitHub-Movie_Genre_Classification-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/krishnachaitanya941/movie-genre-classification)

[![Author Profile](https://img.shields.io/badge/GitHub-@krishnachaitanya941-3776AB?style=for-the-badge&logo=github)](https://github.com/krishnachaitanya941)

---

<div align="center">
  <h3>🎬 From Scripts to Genres. From Ideas to Stories.</h3>
  <p><em>Built with Python, machine learning, NLP, and creativity.</em></p>
  <p>⭐ If you find this project interesting, consider starring the repository.</p>
</div>
