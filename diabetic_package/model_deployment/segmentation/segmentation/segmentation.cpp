#include "Segmentation.h"

// �ٽ���Դ������
std::mutex session_source;

// public

Segmentation::Segmentation(std::string configPath, 
    bool useMultithread, 
    std::vector<std::string> outputName = { "classes" }) {
    std::string modelPath = "";


    // ����·���ж������û�м�//,û�����Զ����
    char last = configPath[configPath.length() - 1];
    //std::string
    if (last != '/') {
        std::string doubleSlash = configPath.substr(configPath.length() - 1, 2);
        std::string dSlash = "\\";
        if (!(doubleSlash == dSlash)) {
            configPath.insert(configPath.length(), "/");
        }
    }

    // ���������ļ�
    char buffer[100];
    std::ifstream configFile(configPath + "model_config.txt");
    if (!configFile.is_open()) {
        std::cout << "���������ļ��Ƿ���ڣ�" << std::endl;
    }

    while (!configFile.eof()) {
        configFile.getline(buffer, 100);
        std::string keyName = std::strtok(buffer, ":");
        std::string ValueName = std::strtok(NULL, ":");
        if (keyName == "model_name") {
            modelPath = configPath + ValueName;
        }
        else if (keyName == "input_height") {
            inputHeight = std::atoi(ValueName.c_str());
        }
        else if (keyName == "input_width") {
            inputWidth = std::atoi(ValueName.c_str());
        }
        else if (keyName == "input_channel") {
            inputChannel = std::atoi(ValueName.c_str());
        }
    }
    checkParam();

    if (outputName.size() > 1) {
        outputLayerName = { "classes","softmax" };
    }
    else {
        outputLayerName = outputName;
    }
    useMultithreading = useMultithread;
    loadModel(modelPath);
}

Segmentation::~Segmentation() {
    if (pSession != nullptr) {
        pSession->Close();
        pSession = nullptr;
    }
}

int Segmentation::predict(std::vector<cv::Mat> imgs, std::vector<std::vector<cv::Mat>>* output) {
    int batchSize = imgs.size();
    std::vector<cv::Size> imgSizes;

    // ��������tensor
    tensorflow::Tensor inputTensor(tensorflow::DT_FLOAT,
        tensorflow::TensorShape({ batchSize,inputHeight,inputWidth,inputChannel }));

    cvMatTotfTensor(imgs, inputTensor, imgSizes);

    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    inputs.push_back(std::pair<std::string, tensorflow::Tensor>("input_img", inputTensor));

    // ��ȡ���������tensor
    std::vector<tensorflow::Tensor> outputs;
    if (useMultithreading) {
        session_source.lock();
    }
    tensorflow::Status status = pSession->Run(inputs, outputLayerName, {}, &outputs);
    if (useMultithreading) {
        session_source.unlock();
    }
    if (!status.ok()) {
        std::cout << "ǰ�򴫲�Ԥ��ʧ��" << std::endl;
        return -1;
    }

    // ���Ԥ���� 
    for (int i = 0; i < outputLayerName.size(); i++) {
        std::vector<cv::Mat> labelImg;
        tfTensorTocvMat(outputs[i], labelImg, imgSizes);
        output->push_back(labelImg);
    }
    return 0;
}

cv::Mat Segmentation::ucharTocvMat(uchar * src, int height, int width, int channel) {
    return cv::Mat(cv::Size(width, height), CV_32FC(channel), src);
}

cv::Mat Segmentation::ucharTocvMat(uchar* src, cv::Size size, int channel) {
    return cv::Mat(size, CV_32FC(channel), src);
}

// private

// �������
void Segmentation::checkParam() {
    if (inputHeight <= 0) {
        std::cout << "���������ļ����Ƿ��趨�˲���������ͼƬ��" << std::endl;
    }
    if (inputWidth <= 0) {
        std::cout << "���������ļ����Ƿ��趨�˲���������ͼƬ��" << std::endl;
    }
    if (inputChannel <= 0) {
        std::cout << "���������ļ����Ƿ��趨�˲���������ͼƬͨ����" << std::endl;
    }
}

// ģ�ͼ���
int Segmentation::loadModel(std::string modelPath) {
    tensorflow::GraphDef graph;

    // ģ�ͼ���
    tensorflow::Status ret = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), modelPath, &graph);
    if (!ret.ok()) {
        std::cout << "ģ�ͼ���ʧ�ܣ�" << std::endl;
        std::cout << "ģ��·����" << modelPath << std::endl;
        return -1;
    }

    // session����
    auto options = tensorflow::SessionOptions();
    NewSession(options, &pSession);

    // ģ��ͼ��session�����
    ret = pSession->Create(graph);
    if (!ret.ok()) {
        std::cout << "ģ����session����ʧ��,����session�����Ƿ�ɹ���" << std::endl;
        return -1;
    }
    return 0;
}

// Mat ת tensor
void Segmentation::cvMatTotfTensor(std::vector<cv::Mat>& input, tensorflow::Tensor & outputTensor, std::vector<cv::Size>& imgSizes) {
    // 4:(b,h,w,c)
    int batchSize = input.size();
    int depth = inputChannel;
    float * p = outputTensor.flat<float>().data();

    for (int b = 0; b < batchSize; b++) {
        // ��¼ͼƬԭʼ�ߴ�
        imgSizes.push_back(input[b].size());

        // matתtensor
        cv::resize(input[b], input[b], cv::Size(inputWidth, inputHeight));
        cv::Mat tempMat(cv::Size(inputWidth, inputHeight), CV_32FC(depth), p);
        input[b].convertTo(tempMat, CV_32FC(depth));
        p += inputWidth*inputHeight*depth;
    }

}

// tensor ת mat
void Segmentation::tfTensorTocvMat(tensorflow::Tensor& inputTensor, std::vector<cv::Mat>& output, std::vector<cv::Size>& imgSizes) {
    tensorflow::TensorShape inputTensorShape = inputTensor.shape();

    int batch = inputTensorShape.dim_size(0);
    int height = inputTensorShape.dim_size(1);
    int width = inputTensorShape.dim_size(2);
    int depth;
    if (inputTensorShape.dims() < 4) {
        // (b,h,w):��άtensorת��
        depth = 1;
        uchar* p = inputTensor.flat<uchar>().data();

        for (int b = 0; b < batch; ++b) {
            cv::Mat singleImg8 = cv::Mat(height, width, CV_8UC(depth), p);
            cv::resize(singleImg8, singleImg8, imgSizes[b], 0, 0, cv::INTER_NEAREST);
            output.push_back(singleImg8);
            p += height*width*depth;
        }
    }
    else {
        // (b,h,w,c):��άtensorת��
        depth = inputTensorShape.dim_size(3);
        float* p = inputTensor.flat<float>().data();

        for (int b = 0; b < batch; ++b) {
            cv::Mat singleImg = cv::Mat(cv::Size(height, width), CV_32FC(depth), p);
            cv::resize(singleImg, singleImg, imgSizes[b], 0, 0, cv::INTER_LINEAR);
            output.push_back(singleImg);
            p += height*width*depth;
        }
    }
}