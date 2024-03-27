// OnnxInference.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <stdio.h>
#include <vector>
#include <iostream>

#ifdef _DEBUG
    #pragma comment(lib,"opencv_world451d.lib")
#else
    #pragma comment(lib,"opencv_world451.lib")
#endif

#pragma comment(lib,"onnxruntime.lib")

using namespace std;

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

std::string CurrentTime()
{
    //
    //return "";
    //
    try {
        std::chrono::system_clock::time_point time_point = std::chrono::system_clock::now();
        std::time_t time_t = std::chrono::system_clock::to_time_t(time_point);
        struct tm time_s;
        //std::tm* tm = std::localtime(&time_t);
        errno_t err = localtime_s(&time_s, &time_t);

        std::stringstream stream;
        //
        stream << std::put_time(&time_s, "%Y%m%d%H%M%S_");
        //
        return stream.str();
        //
    } catch (cv::Exception& ex) {
        return "CurrentTime Error: " + ex.msg + " ";
    }
}

void ObjectInference()
{   
   Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
   Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
   Ort::SessionOptions session_options;
   Ort::Session session(env, L"D:\\Projects\\Projects.Python\\ObjectDetectionYoloV8\\runs\\detect\\train6\\weights\\best.onnx", session_options);
   //
   // 입력 데이터 준비 (예시: 이미지를 텐서로 변환)
   cv::Mat image = cv::imread("https://github.com/cybodime/OnnxInference/blob/main/20220927_000057_1011022.jpg"); // 입력 이미지 로드
   cv::resize(image, image, cv::Size(640, 640)); // 이미지 크기 조정
   cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // BGR을 RGB로 변환
   image.convertTo(image, CV_32F, 1.0 / 255);
   std::vector<float> input_data(image.data, image.data + image.total() * image.channels());


   // ONNX 모델 실행
   const int64_t shape[] = { 1, 3, 640, 640 };
   const char* outputName = "output0";
   Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mi, input_data.data(), input_data.size(), shape, 4);
   std::vector<Ort::Value> output_tensor = session.Run(Ort::RunOptions(), nullptr, &input_tensor, 1, &outputName, 0);

   // 결과 처리 및 사각형 그리기
   float* output_data = output_tensor[0].GetTensorMutableData<float>();
   // 여기에서 물체 인식 결과를 추출하고 사각형 정보를 얻어야 합니다.

   // 예시: 물체의 경계 상자 정보 (x, y, width, height)
   int x = 100;
   int y = 150;
   int width = 50;
   int height = 80;

   cv::rectangle(image, cv::Rect(x, y, width, height), cv::Scalar(0, 255, 0), 2); // 녹색 사각형 그리기
   cv::imshow("Detected Object", image); // 결과 이미지 표시
}

void run_ort_trt()
{
    const int IMG_SIZE_X = 640;
    const int IMG_SIZE_Y = 640;

    //Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
    const auto& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* tensorrt_options;

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const wchar_t* model_path = L"D:\\Projects\\Projects.Python\\ObjectDetectionYoloV8\\runs\\detect\\train10\\weights\\best.onnx";

    //Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
    //std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(tensorrt_options, api.ReleaseTensorRTProviderOptions);
    //Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(session_options), rel_trt_options.get()));

    std::cout << "Running ORT TRT EP with default provider options" << std::endl; 

    Ort::Session session(env, model_path, session_options);    

    //*************************************************************************
    // print model input layer (node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    // print number of model input nodes
    const size_t num_input_nodes = session.GetInputCount();
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<const char*> input_node_names;
    input_names_ptr.reserve(num_input_nodes);
    input_node_names.reserve(num_input_nodes);
    std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
    // Otherwise need vector<vector<>>

    std::cout << "Number of inputs = " << num_input_nodes << std::endl;

    // iterate over all input nodes
    for (size_t i = 0; i < num_input_nodes; i++) {
        // print input node names
        auto input_name = session.GetInputNameAllocated(i, allocator);
        std::cout << "Input " << i << " : name = " << input_name.get() << std::endl;
        input_node_names.push_back(input_name.get());
        input_names_ptr.push_back(std::move(input_name));

        // print input node types
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::cout << "Input " << i << " : type = " << type << std::endl;

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        std::cout << "Input " << i << " : num_dims = " << input_node_dims.size() << '\n';
        for (size_t j = 0; j < input_node_dims.size(); j++) {
            std::cout << "Input " << i << " : dim[" << j << "] = " << input_node_dims[j] << '\n';
        }
        std::cout << std::flush;
    }

    //constexpr size_t input_tensor_size = 640 * 640 * 3;  // simplify ... using known dim values to calculate size
    // use OrtGetTensorShapeElementCount() to get official size!

    cv::Mat inputImage = cv::imread("D:\\Storage\\Sample\\YOLOv8Test\\20220927_000057_101누1022.jpg", cv::ImreadModes::IMREAD_COLOR);    
    //
    size_t channels = inputImage.channels();
    size_t total = inputImage.total();
    size_t input_tensor_size = IMG_SIZE_X * IMG_SIZE_Y * channels;  // simplify ... using known dim values to calculate size
    //
    //cv::resize(inputImage, inputImage, cv::Size(IMG_SIZE_X, IMG_SIZE_Y), cv::InterpolationFlags::INTER_CUBIC);    
    // Assign to vector for 3 channel image
    // Souce: https://stackoverflow.com/a/56600115/2076973 

    std::vector<float> input_tensor_values(IMG_SIZE_X * IMG_SIZE_Y * channels);
    //
    //cv::cvtColor(inputImage, inputImage, cv::ColorConversionCodes::COLOR_BGR2RGB);  // cv::COLOR_RGB2BGR);    //겉으로는 이미지 변화가 없다.
    //inputImage.convertTo(inputImage, CV_32FC3, 0.003921568627451);  //그냥 까맣게 보인다.
    //inputImage.convertTo(inputImage, CV_32FC3);
    //inputImage /= (float)255.0;

    cv::Mat flat = inputImage.reshape(1, total * channels);    
    input_tensor_values = inputImage.isContinuous() ? flat : flat.clone();
    
    //위에랑 얘랑 값이 좀 틀리다. 어느게 맞는지는 모르겠는데, TensorServer 만들때, 위에 방법으로 했다. 
    //그리고, 아래 방법으로 해도, 결과가 제대로 안나오는건 마찬가지라...
    //cv::Mat input_mat = cv::dnn::blobFromImage(inputImage);
    //input_tensor_values.assign(input_mat.begin<float>(), input_mat.end<float>());
            
    // create input tensor object from data values
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
    assert(input_tensor.IsTensor());

    //std::vector<const char*> output_node_names = { outputNames.front() };
    std::vector<const char*> output_node_names = { "output0" };

    // score model & input tensor, get back output tensor
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

    // Get pointer to output tensor float values
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    //assert(abs(floatarr[0] - 0.000045) < 1e-6);

    Ort::TensorTypeAndShapeInfo info = output_tensors[0].GetTensorTypeAndShapeInfo();
    
    float x_factor = inputImage.cols / INPUT_WIDTH;
    float y_factor = inputImage.rows / INPUT_HEIGHT;

    // Mat 타입으로 변환 후 가장 큰 confidence를 가진 Id를 최종 결과로 출력
    cv::Mat1f result = cv::Mat1f(1000, 1, floatarr);
    cv::Point classIdPoint;
    float* data = (float*)result.data;
    float confidence = data[4];
    float* classes_scores = data + 5;
    float cx = data[0];
    float cy = data[1];
    // Box dimension.
    float w = data[2];
    float h = data[3];

    int left = int((cx - 0.5 * w) * x_factor);
    int top = int((cy - 0.5 * h) * y_factor);
    int width = int(w * x_factor);
    int height = int(h * y_factor);

    //minMaxLoc(result, 0, &confidence, 0, &classIdPoint);

    //int classId = classIdPoint.y;
    //std::cout << "confidence: " << confidence << std::endl;
    //std::cout << "class: " << classId << std::endl;

    //cv::Mat1f result_1f = cv::Mat1f(IMG_SIZE_Y, IMG_SIZE_X, floatarr);
    //result_1f *= 255;
    //cv::Mat result;
    //result_1f.convertTo(result, CV_8UC3);
    //cv::imshow("Result", result);

    //cv::imwrite("D:\\Storage\\Test\\" + CurrentTime() + "onnx_result.jpg", result);

    // score the model, and print scores for first 5 classes
    //for (int i = 0; i < 5; i++) {
    //    std::cout << "Score for class [" << i << "] =  " << floatarr[i] << '\n';
    //}
    std::cout << std::flush;

    // Results should be as below...
    // Score for class[0] = 0.000045
    // Score for class[1] = 0.003846
    // Score for class[2] = 0.000125
    // Score for class[3] = 0.001180
    // Score for class[4] = 0.001317

    std::cout << "Done!" << std::endl;
    

}

void post_process(cv::Mat& input_image, vector<cv::Mat>& outputs, const vector<string>& class_name)
{
    // Initialize vectors to hold respective outputs while unwrapping     detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<cv::Rect> boxes;
    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float* data = (float*)outputs[0].data;
    const int dimensions = 85;
    // 25200 for default size 640.
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i) {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float* classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD) {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        data += 85;
    }
}


int main()
{    
   
    run_ort_trt();
    //OnnxInference();

    std::cin.get();

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
