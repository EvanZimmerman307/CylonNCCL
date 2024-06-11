#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// Define structures for the data
struct Data1 {
    int id;
    float value1;
};

struct Data2 {
    int id;
    float value2;
};

// Function to read Data1 from CSV
std::vector<Data1> read_data1_from_csv(const std::string& filename) {
    std::vector<Data1> data;
    std::ifstream file(filename);
    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        Data1 entry;
        std::string temp;
        
        std::getline(ss, temp, ',');
        entry.id = std::stoi(temp);
        std::getline(ss, temp, ',');
        entry.value1 = std::stof(temp);
        
        data.push_back(entry);
    }

    return data;
}

// Function to read Data2 from CSV
std::vector<Data2> read_data2_from_csv(const std::string& filename) {
    std::vector<Data2> data;
    std::ifstream file(filename);
    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        Data2 entry;
        std::string temp;
        
        std::getline(ss, temp, ',');
        entry.id = std::stoi(temp);
        std::getline(ss, temp, ',');
        entry.value2 = std::stof(temp);
        
        data.push_back(entry);
    }

    return data;
}

int main() {
    // Full paths to the CSV files
    std::string data1_file = "/Users/mihirsavkar/Desktop/NCCL/data1.csv";
    std::string data2_file = "/Users/mihirsavkar/Desktop/NCCL/data2.csv";

    // Read the CSV files
    std::vector<Data1> data1 = read_data1_from_csv(data1_file);
    std::vector<Data2> data2 = read_data2_from_csv(data2_file);

    // Print the data to verify
    std::cout << "Data1:" << std::endl;
    for (const auto& entry : data1) {
        std::cout << "id: " << entry.id << ", value1: " << entry.value1 << std::endl;
    }

    std::cout << "Data2:" << std::endl;
    for (const auto& entry : data2) {
        std::cout << "id: " << entry.id << ", value2: " << entry.value2 << std::endl;
    }

    return 0;
}
