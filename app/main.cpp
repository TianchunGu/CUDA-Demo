
__constant__ float c_data;
__constant__ float c_data2 = 6.6f;

int main(int argc, char** argv) {
  std::cout << "Starting CUDA vector add task from main()..." << std::endl;

  // 调用 CUDA 任务
  run_vector_add();

  std::cout << "CUDA task finished." << std::endl;

  return 0;
}