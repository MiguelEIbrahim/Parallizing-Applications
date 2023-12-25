#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <omp.h>
//Open MP Variant -- Bad Speedup
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
        // Create a vector of output file streams for parallel version
        std::vector<std::ofstream> outputFilesParallel(omp_get_max_threads());
        for (int i = 0; i < omp_get_max_threads(); ++i) {
            outputFilesParallel[i].open("random_words_parallel_" + std::to_string(i) + ".txt");
            if (!outputFilesParallel[i].is_open()) {
                std::cerr << "Error opening file." << std::endl;
                return 1;
            }
        }

        // Measure the time taken for the parallel version
        auto startParallel = std::chrono::high_resolution_clock::now();

        // Generate and write random words in parallel
        #pragma omp parallel
        {
            int threadID = omp_get_thread_num();
            for (int i = threadID; i < numWords; i += omp_get_num_threads()) {
                std::string word = generateRandomWord(maxWordLength);
                outputFilesParallel[threadID] << word << std::endl;
            }
        }

        // Measure the time taken for the parallel version
        auto endParallel = std::chrono::high_resolution_clock::now();
        auto durationParallel = std::chrono::duration_cast<std::chrono::microseconds>(endParallel - startParallel);

        // Close the parallel files
        for (int i = 0; i < omp_get_max_threads(); ++i) {
            outputFilesParallel[i].close();
        }

        // Merge the parallel files into a single file
        std::ofstream outputFileParallel("random_words_parallel.txt");
        for (int i = 0; i < omp_get_max_threads(); ++i) {
            std::ifstream input("random_words_parallel_" + std::to_string(i) + ".txt");
            outputFileParallel << input.rdbuf();
            input.close();
        }
        outputFileParallel.close();

        std::cout << "Parallel version took " << durationParallel.count() << " microseconds." << std::endl;
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
