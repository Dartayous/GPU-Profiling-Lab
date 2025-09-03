import cv2
import cupy as cp
import numpy as np
import nvtx

# Define the fused CUDA kernel (VITALLY IMPORTANT STEP FOR NSIGHT TO WORK)
sobel_blur_color_kernel = cp.RawKernel(r'''
extern "C" __global__
void fused_filter(const unsigned char* img, unsigned char* output, int width, int height, int brightness) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Box Blur
        int blur_sum = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                blur_sum += img[(y + dy) * width + (x + dx)];
            }
        }
        int blurred = blur_sum / 9;

        // Brightness Adjustment
        int corrected = min(255, max(0, blurred + brightness));

        // Sobel Edge Detection
        int gx = -img[(y-1)*width + (x-1)] - 2*img[y*width + (x-1)] - img[(y+1)*width + (x-1)]
                 + img[(y-1)*width + (x+1)] + 2*img[y*width + (x+1)] + img[(y+1)*width + (x+1)];

        int gy = -img[(y-1)*width + (x-1)] - 2*img[(y-1)*width + x] - img[(y-1)*width + (x+1)]
                 + img[(y+1)*width + (x-1)] + 2*img[(y+1)*width + x] + img[(y+1)*width + (x+1)];

        int edge = min(255, abs(gx) + abs(gy));

        // Final blend
        output[y * width + x] = (corrected + edge) / 2;
    }
}
''', 'fused_filter')

# Load grayscale image
img = cv2.imread("assets/Macross_5.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Check the path and filename.")

# Preprocessing
with nvtx.annotate("Preprocessing", color="green"):
    img_gpu = cp.asarray(img)
    output_gpu = cp.zeros_like(img_gpu)

# Kernel launch setup
height, width = img.shape
block = (16, 16)
grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])
brightness = 30
stream = cp.cuda.Stream(non_blocking=True)

# Kernel execution
with nvtx.annotate("Fused Kernel in Stream", color="purple"):
    with stream:
        sobel_blur_color_kernel(grid, block, (
            img_gpu, output_gpu,
            np.int32(width), np.int32(height),
            np.int32(brightness)
        ))

# Postprocessing
with nvtx.annotate("Postprocessing", color="red"):
    stream.synchronize()
    output = cp.asnumpy(output_gpu)

    if output is None or output.size == 0:
        raise ValueError("Output image is empty. Check kernel execution and input data.")

    cv2.imwrite("assets/Macross_5_fused.png", output)
