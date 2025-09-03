import cv2
import cupy as cp
import numpy as np

# Load grayscale image
img = cv2.imread("assets/Macross_5.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Check the path and filename.")

# Transfer to GPU
img_gpu = cp.asarray(img)

# Define output array
output_gpu = cp.zeros_like(img_gpu)

# CUDA kernel as a string (Define kernel, grid, block, etc.) (THIS IS THE LESSON PLAN)
sobel_kernel = cp.RawKernel(r'''
extern "C" __global__
void sobel_filter(const unsigned char* img, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int gx = -img[(y-1)*width + (x-1)] - 2*img[y*width + (x-1)] - img[(y+1)*width + (x-1)]
                 + img[(y-1)*width + (x+1)] + 2*img[y*width + (x+1)] + img[(y+1)*width + (x+1)];

        int gy = -img[(y-1)*width + (x-1)] - 2*img[(y-1)*width + x] - img[(y-1)*width + (x+1)]
                 + img[(y+1)*width + (x-1)] + 2*img[(y+1)*width + x] + img[(y+1)*width + (x+1)];

        int magnitude = min(255, abs(gx) + abs(gy));
        output[y*width + x] = magnitude;
    }
}
''', 'sobel_filter')


# Launch kernel
height, width = img.shape
block = (16, 16)
grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])
sobel_kernel(grid, block, (img_gpu, output_gpu, np.int32(width), np.int32(height)))

# Transfer result back to CPU
output_cpu = cp.asnumpy(output_gpu)

# Sanity Check (This helps catch issues early, especially when debugging raw kernels or edge cases.)
if output_cpu is None or output_cpu.size == 0:
    raise ValueError("Output image is empty. Check kernel execution and input data.")

# Save or display result
cv2.imwrite("assets/Macross_5_sobel.png", output_cpu)