#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <thread>
#include <mutex>

// Function to generate a random word of random length
std::string generateRandomWord(int maxLength) {
    int length = rand() % maxLength + 1;
    std::string word;
    static const char alphabet[] =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    for (int i = 0; i < length; ++i) {
        char randomChar = alphabet[rand() % (sizeof(alphabet) - 1)];
        word += randomChar;
    }

    return word;
}

// Function to read words from the file in parallel
void readWordsFromFile(const std::string& filename, std::vector<std::string>& words, int start, int end, std::mutex& mtx) {
    std::ifstream inputFile(filename);

    if (inputFile.is_open()) {
        std::string word;
        for (int i = 0; i < start; ++i) {
            inputFile >> word;
        }

        for (int i = start; i <= end; ++i) {
            inputFile >> word;

            // Use a lock to protect access to the words vector
            std::lock_guard<std::mutex> lock(mtx);
            words[i] = word;
        }

        inputFile.close();
    } else {
        std::cerr << "Error opening input file." << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Check for the correct number of command-line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [parallel|sequential]" << std::endl;
        return 1;
    }

    // Set the number of words and maximum word length
    const int numWords = 10000;
    const int maxWordLength = 1000;

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    const std::string mode = argv[1];

    if (mode == "parallel") {
        // Create a vector to store the read words
        std::vector<std::string> words(numWords);

        // Measure the time taken for reading in parallel
        auto startReadParallel = std::chrono::high_resolution_clock::now();

        // Create multiple threads for parallel file reading
        const int numThreads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(numThreads);
        std::mutex mtx;

        int wordsPerThread = numWords / numThreads;
        for (int i = 0; i < numThreads; ++i) {
            int start = i * wordsPerThread;
            int end = (i == numThreads - 1) ? numWords - 1 : (i + 1) * wordsPerThread - 1;
            threads[i] = std::thread(readWordsFromFile, "random_words_parallel.txt", std::ref(words), start, end, std::ref(mtx));
        }

        // Join the threads
        for (int i = 0; i < numThreads; ++i) {
            threads[i].join();
        }

        // Measure the time taken for reading in parallel
        auto endReadParallel = std::chrono::high_resolution_clock::now();
        auto durationReadParallel = std::chrono::duration_cast<std::chrono::microseconds>(endReadParallel - startReadParallel);

        // Verify and print a few words to ensure correctness
        if (numWords >= 10) {
            std::cout << "First 10 words: ";
            for (int i = 0; i < 10; ++i) {
                std::cout << words[i] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Reading in " << mode << " mode took " << durationReadParallel.count() << " microseconds." << std::endl;
    } else if (mode == "sequential") {
        // Create an output file stream for the sequential version
        std::ofstream outputFileSequential("random_words_sequential.txt");

        // Check if the file is open
        if (!outputFileSequential.is_open()) {
            std::cerr << "Error opening file." << std::endl;
            return 1;
        }

        // Measure the time taken for the sequential version
        auto startSequential = std::chrono::high_resolution_clock::now();

        // Generate and write random words sequentially
        for (int i = 0; i < numWords; ++i) {
            std::string word = generateRandomWord(maxWordLength);
            outputFileSequential << word << std::endl;
        }

        // Measure the time taken for the sequential version
        auto endSequential = std::chrono::high_resolution_clock::now();
        auto durationSequential = std::chrono::duration_cast<std::chrono::microseconds>(endSequential - startSequential);

        // Close the sequential file
        outputFileSequential.close();

        std::cout << "Sequential version took " << durationSequential.count() << " microseconds." << std::endl;
    } else {
        std::cerr << "Invalid mode. Usage: " << argv[0] << " [parallel|sequential]" << std::endl;
        return 1;
    }

    return 0;
}
