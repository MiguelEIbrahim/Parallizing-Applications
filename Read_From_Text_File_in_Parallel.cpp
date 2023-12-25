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

    // Now read from the files in parallel and measure the time

    // Vector to store the read words
    std::vector<std::string> words(numWords);

    // Measure the time taken for reading in parallel
    auto startReadParallel = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < numWords; ++i) {
        std::ifstream inputFile;
        if (mode == "parallel") {
            inputFile.open("random_words_parallel.txt");
        } else if (mode == "sequential") {
            inputFile.open("random_words_sequential.txt");
        }

        if (inputFile.is_open()) {
            std::string word;
            for (int j = 0; j <= i; ++j) {
                inputFile >> word;
            }
            words[i] = word;
            inputFile.close();
        } else {
            std::cerr << "Error opening input file." << std::endl;
        exit;
        }
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

    return 0;
}
//Using Open MP
//test using:
//g++ -o ReadMP Read_From_Text_File_in_Parallel.cpp -fopenmp
//./ReadMP Parallel
//./ReadMP Sequential
