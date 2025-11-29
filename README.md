# Twitter Multi-Label Classification with Word2Vec, GloVe, and RandomForest

---

## Overview
This project performs **multi-label text classification** on Twitter data using **Word2Vec** and **GloVe** embeddings. The embeddings are used to train machine learning classifiers such as **Logistic Regression** and **RandomForest** via `MultiOutputClassifier` to predict multiple labels per tweet.

The workflow includes:  
- Data loading and cleaning  
- Tokenization and text preprocessing  
- Multi-label binarization  
- Generating **Word2Vec** and **GloVe** embeddings  
- Combining embeddings for better representation  
- Training **RandomForestClassifier** or **LogisticRegression**  
- Model evaluation using **Micro F1** and **Macro F1** scores  
- Visualization of results  

---

## Dataset
The dataset is a **Twitter Multi-Label Classification Dataset** (CSV format) with two main columns:  

| Column   | Description                             |
|----------|-----------------------------------------|
| tweets   | The text of the tweet                    |
| labels   | Multi-label tags separated by commas, e.g., `hate,sarcasm` |

**Example:**

| tweets                    | labels           |
|----------------------------|----------------|
| "I hate this product"      | hate           |
| "This is so funny lol"     | humor,sarcasm  |

---

## Requirements
- Python 3.8+  
- Libraries:

```bash
pip install numpy pandas scikit-learn scikit-multilearn gensim matplotlib

