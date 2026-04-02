# Reflection & Learning Plan

## Reflection
This project helped us better understand how CUDA can be used for real-time image processing on an embedded GPU platform. We applied the reduction parallel pattern to estimate urban green cover from live webcam frames by extracting green-channel values and summing them efficiently on the GPU. Through this work, we improved our understanding of CUDA kernels, memory transfers, and optimization methods such as minimizing divergence and reducing global memory access.

One thing that went well in this project was building a working workflow from live camera capture to GPU processing and output. It was useful to see how the same task could be improved through different optimization approaches. A challenge in the project was organizing the code and understanding how each optimization affected performance. Working through these steps gave us more confidence in parallel programming and CUDA-based development.

## Learning Plan
If this project were scaled further, more knowledge in GPU profiling, embedded system tuning, and advanced CUDA optimization would be useful. It would also help to learn more about memory hierarchy, occupancy, and efficient kernel design in greater detail. To improve these skills, useful resources would include CUDA documentation, Jetson platform documentation, course materials, and additional hands-on testing with profiling tools.
