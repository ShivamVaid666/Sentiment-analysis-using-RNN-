#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <map>

// Hyperparameters
const int HIDDEN_SIZE = 16;
const int INPUT_SIZE = 256; 
const double LEARNING_RATE = 0.01;

double tanh_activation(double x) {
    return std::tanh(x);
}

double dtanh(double x) {
    return 1.0 - x * x;
}

class SimpleRNN {
public:
    std::vector<std::vector<double>> W_xh; 
    std::vector<std::vector<double>> W_hh; 
    std::vector<double> W_hy;              
    std::vector<double> b_h;               
    double b_y;                            

    SimpleRNN() {
        W_xh.resize(HIDDEN_SIZE, std::vector<double>(INPUT_SIZE, 0.01));
        W_hh.resize(HIDDEN_SIZE, std::vector<double>(HIDDEN_SIZE, 0.02));
        W_hy.resize(HIDDEN_SIZE, 0.03);
        b_h.resize(HIDDEN_SIZE, 0.0);
        b_y = 0.0;
    }

    std::pair<double, std::vector<std::vector<double>>> forward(const std::string& text) {
        int T = text.length();
        std::vector<std::vector<double>> h(T + 1, std::vector<double>(HIDDEN_SIZE, 0.0));

        for (int t = 0; t < T; ++t) {
            int char_idx = static_cast<unsigned char>(text[t]);
            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                double val = b_h[i] + W_xh[i][char_idx]; 
                for (int j = 0; j < HIDDEN_SIZE; ++j) {
                    val += W_hh[i][j] * h[t][j];      
                }
                h[t + 1][i] = tanh_activation(val);
            }
        }

        double y_raw = b_y;
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            y_raw += W_hy[i] * h[T][i];
        }
        double y_pred = 1.0 / (1.0 + std::exp(-y_raw)); 
        return {y_pred, h};
    }

    void train_step(const std::string& text, int target) {
        int T = text.length();
        auto [y_pred, h] = forward(text);
        double dy = y_pred - target;

        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            W_hy[i] -= LEARNING_RATE * (dy * h[T][i]);
        }
        b_y -= LEARNING_RATE * dy;

        std::vector<double> dh_next(HIDDEN_SIZE, 0.0);
        for (int i = 0; i < HIDDEN_SIZE; ++i) {
            dh_next[i] = dy * W_hy[i];
        }

        for (int t = T - 1; t >= 0; --t) {
            int char_idx = static_cast<unsigned char>(text[t]);
            std::vector<double> dtanh_val(HIDDEN_SIZE, 0.0);

            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                dtanh_val[i] = dtanh(h[t + 1][i]) * dh_next[i];
                b_h[i] -= LEARNING_RATE * dtanh_val[i];
                W_xh[i][char_idx] -= LEARNING_RATE * dtanh_val[i];
            }

            std::vector<double> dh_prev(HIDDEN_SIZE, 0.0);
            for (int i = 0; i < HIDDEN_SIZE; ++i) {
                for (int j = 0; j < HIDDEN_SIZE; ++j) {
                    W_hh[i][j] -= LEARNING_RATE * dtanh_val[i] * h[t][j];
                    dh_prev[j] += W_hh[i][j] * dtanh_val[i];
                }
            }
            dh_next = dh_prev;
        }
    }
};

int main() {
    SimpleRNN rnn;

    // Base training data
    std::vector<std::pair<std::string, int>> dataset = {
        {"great", 1}, {"excellent job", 1}, {"i love it", 1}, {"good", 1},
        {"bad", 0}, {"terrible app", 0}, {"i hate it", 0}, {"poor performance", 0}
    };

    std::cout << "Training the C++ RNN on Ubuntu..." << std::endl;
    for (int epoch = 0; epoch < 700; ++epoch) { // Increased epochs slightly for better accuracy
        for (const auto& data : dataset) {
            rnn.train_step(data.first, data.second);
        }
    }
    std::cout << "Training complete!\n" << std::endl;

    // --- Interactive User Input Section ---
    std::string userInput;
    while (true) {
        std::cout << "Enter text to analyze (or type 'exit' to quit): ";
        std::getline(std::cin, userInput);

        if (userInput == "exit") {
            break;
        }
        if (userInput.empty()) {
            continue;
        }

        // Run forward pass on what the user typed
        auto [prediction, hidden] = rnn.forward(userInput);
        
        std::string sentiment;
        if (prediction > 0.6) {
            sentiment = "POSITIVE 😊";
        } else if (prediction < 0.4) {
            sentiment = "NEGATIVE 😡";
        } else {
            sentiment = "NEUTRAL 😐";
        }

        std::cout << "-> Confidence Score: " << prediction << " | Sentiment: " << sentiment << "\n\n";
    }

    return 0;
}
