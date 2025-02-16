#include <dawn/webgpu_cpp.h>
#include <iostream>

int main() {
    std::cout << "Dawn is set up successfully!" << std::endl;
    
    // Example: Create a null WebGPU instance (just to verify linking works)
    wgpu::Instance instance = wgpu::CreateInstance();
    if (instance) {
        std::cout << "WebGPU instance created successfully." << std::endl;
    } else {
        std::cout << "Failed to create WebGPU instance." << std::endl;
    }

    return 0;
}