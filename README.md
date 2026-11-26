# Sentiment Analysis Using RNN (C++)

A simple character-level Recurrent Neural Network (RNN) implemented from scratch in C++ to perform binary sentiment analysis on text.

## 1. Project Overview

This project implements a Simple Recurrent Neural Network (RNN) in C++ that analyzes text entered by the user and predicts its sentiment.

The model processes text character by character, maintains a hidden state, and uses a sigmoid output layer to generate a sentiment score.

The project demonstrates fundamental concepts of Artificial Intelligence and Natural Language Processing (NLP), including:

- Recurrent Neural Networks
- Sequential text processing
- Forward propagation
- Backpropagation Through Time (BPTT)
- Activation functions
- Gradient-based learning
- Sentiment classification

The program is implemented using standard C++ libraries without requiring external machine learning frameworks.

## 2. Features

- Character-level text processing.
- Simple RNN implemented from scratch.
- 16-dimensional hidden state.
- Tanh activation function.
- Sigmoid output function.
- Training using manually labelled examples.
- Backpropagation Through Time.
- Interactive command-line interface.
- Positive, negative and threshold-based neutral output.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| C++ | Core programming language |
| C++17 | Language standard |
| STL | Data structures and containers |
| `<vector>` | Weight matrices and hidden states |
| `<string>` | Text input processing |
| `<cmath>` | Mathematical operations |
| g++ | C++ compiler |
| Ubuntu / Linux | Development environment |

## 4. RNN Architecture

The model follows a simple recurrent architecture:

**Input Text → Character Encoding → Recurrent Hidden State → Sigmoid Output → Sentiment Prediction**

### Model Parameters

| Parameter | Value |
|---|---:|
| Input size | 256 |
| Hidden size | 16 |
| Learning rate | 0.01 |
| Training epochs | 700 |
| Hidden activation | tanh |
| Output activation | Sigmoid |

### Hidden State Calculation

The hidden state is updated for each character using:

\[
h_{t+1}=\tanh(b_h+W_{xh}[:,x_t]+W_{hh}h_t)
\]

Where:

- \(h_t\): Previous hidden state.
- \(W_{xh}\): Input-to-hidden weights.
- \(W_{hh}\): Recurrent weights.
- \(b_h\): Hidden bias.
- \(x_t\): Current character index.

### Output Calculation

The final hidden state is passed through a sigmoid output layer:

\[
\hat y=\frac{1}{1+e^{-z}}
\]

The resulting score lies between 0 and 1.

## 5. Training Dataset

The model is trained using eight manually created examples.

| Text | Target | Sentiment |
|---|---:|---|
| great | 1 | Positive |
| excellent job | 1 | Positive |
| i love it | 1 | Positive |
| good | 1 | Positive |
| bad | 0 | Negative |
| terrible app | 0 | Negative |
| i hate it | 0 | Negative |
| poor performance | 0 | Negative |

The training process runs for 700 epochs with a learning rate of 0.01.

**Note:** This is a small demonstration dataset, not a representative real-world sentiment dataset.

## 6. Sentiment Classification

The program uses the following thresholds:

| Prediction Score | Output |
|---|---|
| Greater than 0.6 | POSITIVE 😊 |
| Less than 0.4 | NEGATIVE 😡 |
| 0.4 to 0.6 | NEUTRAL 😐 |

The neutral category is a threshold-based uncertainty range. The network is trained only on positive and negative examples; it does not learn neutral sentiment as a separate class.

## 7. Installation and Requirements

### Requirements

- Ubuntu or another Linux distribution (or a compatible C++ environment).
- g++ compiler.
- C++17 or later.
- Basic command-line knowledge.

### Install g++

Open your terminal and execute:

```bash
sudo apt update
sudo apt install g++
```

Check the compiler version:

```bash
g++ --version
```

## 8. Compilation and Execution

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace the placeholder with your actual GitHub repository URL.

### Step 2: Enter the Project Directory

```bash
cd Sentiment-analysis-using-RNN
```

### Step 3: Compile the Program

```bash
g++ -std=c++17 -O2 main.cpp -o sentiment_rnn
```

### Step 4: Run the Program

```bash
./sentiment_rnn
```

The program trains the RNN and then starts accepting user input.

## 9. Example Program Interaction

```text
Training the C++ RNN on Ubuntu...

Training complete!

Enter text to analyze (or type 'exit' to quit): great

-> Confidence Score: [model output] | Sentiment: [predicted label]

Enter text to analyze (or type 'exit' to quit): terrible app

-> Confidence Score: [model output] | Sentiment: [predicted label]

Enter text to analyze (or type 'exit' to quit): exit
```

The scores and labels depend on the actual execution. The placeholders above are illustrative, not measured results.

## 10. Project Structure

A suggested repository organization is:

```text
Sentiment-analysis-using-RNN/
│
├── main.cpp
│
├── README.md
│
├── report/
│   └── Project_Report.pdf
│
└── screenshots/
    └── program_output.png
```

The current implementation can be maintained in a single C++ source file. The additional folders are optional and can be created when supporting documentation and screenshots are added.

## 11. Limitations

Although the project demonstrates the basic working of an RNN, it has several limitations:

1. **Small Dataset:** Only eight manually labelled examples are used for training.
2. **Character-Level Processing:** The model processes bytes rather than complete words or tokens.
3. **Unicode Handling:** Multi-byte UTF-8 characters may be processed as multiple time steps.
4. **Limited Generalization:** The model may not reliably classify sentences containing unfamiliar vocabulary.
5. **No Independent Evaluation:** A separate test dataset and measured accuracy, precision, recall and F1-score are not included.
6. **Neutral Classification:** Neutral is a threshold-based output, not a separately trained class.
7. **Basic RNN Architecture:** Long sequences can suffer from vanishing or exploding gradients.
8. **Weight Initialization:** The model uses identical constant-valued initial weights, which can limit the diversity of hidden-unit representations.
9. **Gradient Update Ordering:** Some parameters are updated before all backward gradient calculations are complete, which should be corrected for mathematically consistent training.

This project should therefore be considered an educational prototype rather than a production-ready sentiment analysis system.

## 12. Future Improvements

Possible improvements include:

- Expand the training dataset.
- Introduce randomized weight initialization.
- Correct gradient accumulation and parameter update ordering.
- Add gradient clipping.
- Implement a proper training, validation and testing pipeline.
- Calculate accuracy, precision, recall and F1-score.
- Add Unicode-aware text processing.
- Implement word embeddings.
- Compare SimpleRNN with LSTM and GRU architectures.
- Add a genuine neutral sentiment class.
- Save and load trained model weights.
- Develop a graphical user interface or web application.

## 13. Learning Outcomes

Through this project, the following concepts can be explored:

- Understanding recurrent neural network architecture.
- Implementing neural networks using C++.
- Processing sequential data.
- Understanding activation functions.
- Learning forward propagation and backpropagation.
- Applying machine learning concepts to Natural Language Processing.
- Working with vectors, matrices and numerical calculations.
- Developing an interactive command-line application.

## 14. Author

**Shivam Vaid**

- Registration Number: 12504760
- University: Lovely Professional University
- Course: CSE276 – Artificial Intelligence
- Academic Session: 2026–27

## 15. References

1. Goodfellow, I., Bengio, Y., and Courville, A. *Deep Learning*. MIT Press, 2016.
2. C++ Reference: https://en.cppreference.com/
3. Recurrent Neural Networks – Deep Learning fundamentals.
4. Supplied C++ SimpleRNN implementation.

---

**If you find this project useful for learning, feel free to explore the code, experiment with the model and implement the suggested improvements.**
