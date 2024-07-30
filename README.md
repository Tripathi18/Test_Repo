# Quora Question Answering Model

This repository contains a state-of-the-art question-answering model leveraging the Quora Question Answer Dataset. The objective is to create an AI system capable of understanding and generating accurate responses to a variety of user queries, mimicking a human-like interaction.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Tokenization](#tokenization)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run the code, you'll need to install the required packages. You can do this using pip:

```bash
pip install datasets
pip install transformers
pip install nltk
pip install torch
pip install matplotlib
pip install seaborn
```

## Dataset

The dataset used is the Quora Question Answer Dataset. It can be loaded using the `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset('toughdata/quora-question-answer-dataset')
print(dataset['train'][0])
```

## Preprocessing

Text preprocessing is performed using NLTK. This includes tokenization, removal of stopwords, stemming, and lemmatization.

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.lower() not in stop_words]
    tokens = [ps.stem(w) for w in tokens]  # For stemming
    tokens = [lemmatizer.lemmatize(w) for w in tokens]  # For lemmatization
    return ' '.join(tokens)

dataset = dataset.map(lambda x: {'question': preprocess(x['question']), 'answer': preprocess(x['answer'])})
```

## Tokenization

The tokenizer from the `transformers` library is used to tokenize the dataset. 

```python
from transformers import AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    inputs = tokenizer(
        examples['question'],
        examples['answer'],
        padding='max_length',
        truncation=True,
        return_offsets_mapping=True
    )
    offset_mapping = inputs.pop('offset_mapping')
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        input_ids = inputs['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sep_index = input_ids.index(tokenizer.sep_token_id)

        answer_offsets = offset[sep_index + 1:]
        answer_text = examples['answer'][i]

        start_char = 0
        end_char = len(answer_text)

        start_token = None
        end_token = None
        for idx, (start, end) in enumerate(answer_offsets):
            if start_token is None and start <= start_char < end:
                start_token = sep_index + 1 + idx
            if start <= end_char <= end:
                end_token = sep_index + 1 + idx
                break

        if start_token is None:
            start_token = cls_index
        if end_token is None:
            end_token = cls_index

        start_positions.append(start_token)
        end_positions.append(end_token)

    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

## Training

The model used for training is a pre-trained BERT model fine-tuned for question answering.

```python
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=0.4,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['train'],
)

trainer.train()
```

## Evaluation

Evaluation is conducted after each epoch using the training dataset.

```python
trainer.evaluate()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

---

Feel free to customize this README further based on your project's specifics and any additional details you want to include.
