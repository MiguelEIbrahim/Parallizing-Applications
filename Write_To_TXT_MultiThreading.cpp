#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <omp.h>

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

int main() {
    // Set the number of words and maximum word length
    const int numWords = 10000;
    const int maxWordLength = 1000;

    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Create an output file stream for both versions
    std::ofstream outputFile("random_words.txt");

    // Check if the file is open
    if (!outputFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    // Measure the time taken for the parallel version
    auto startParallel = std::chrono::high_resolution_clock::now();

    // Generate and write random words in parallel using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < numWords; ++i) {
        std::string word = generateRandomWord(maxWordLength);
        
        // Write the word to the file while protecting access with a lock
        #pragma omp critical
        {
            outputFile << word << std::endl;
        }
    }

    // Measure the time taken for the parallel version
    auto endParallel = std::chrono::high_resolution_clock::now();
    auto durationParallel = std::chrono::duration_cast<std::chrono::microseconds>(endParallel - startParallel);

    // Close the file
    outputFile.close();

    // Reopen the file for the sequential version
    outputFile.open("random_words.txt");

    if (!outputFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    // Measure the time taken for the sequential version
    auto startSequential = std::chrono::high_resolution_clock::now();

    // Generate and write random words sequentially
    for (int i = 0; i < numWords; ++i) {
        std::string word = generateRandomWord(maxWordLength);
        outputFile << word << std::endl;
    }

    // Measure the time taken for the sequential version
    auto endSequential = std::chrono::high_resolution_clock::now();
    auto durationSequential = std::chrono::duration_cast<std::chrono::microseconds>(endSequential - startSequential);

    // Close the file
    outputFile.close();

    std::cout << "Parallel version took " << durationParallel.count() << " microseconds." << std::endl;
    std::cout << "Sequential version took " << durationSequential.count() << " microseconds." << std::endl;

    return 0;
}
