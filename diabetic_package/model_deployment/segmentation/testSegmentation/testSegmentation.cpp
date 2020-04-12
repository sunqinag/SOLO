#include "testSegmentation.h"
#include "../../classification/classification/classification.h"
#include <opencv2/imgproc/types_c.h>

void testSegmentation();

int main() {
    testSegmentation();
    return 0;
}

void testSegmentation() {
    // ��ʱ����
    clock_t start, finish, startAllTime, endAllTime;
    double costTime, allTime, allTimeMultiThread = 0;

    // ���ݻ�ȡ
    string testPath = "../../data";
    string filepath = testPath + "/*.png";
    vector<std::string> files;
    cv::glob(filepath, files, false);
    int imgNum = files.size();

    // Ԥ����ر���
    string configPath = "../../model/segmentation";
    bool useMultiThreading = false;
    Segmentation testPb = Segmentation(configPath,
        useMultiThreading,//�Ƿ�ʹ�ö��߳�
        { "softmax","classes" });
    int batchSize = 2;// ��������ͼƬ����

    startAllTime = clock();
    for (int i = 0; i < imgNum / batchSize; i++) {
        // mat vector����
        vector<cv::Mat> imgs;
        for (int b = 0; b < batchSize; b++) {
            imgs.push_back(cv::imread(files[i*batchSize + b], 0));
        }

        // ͼ��Ԥ��
        start = clock();
        vector<vector<cv::Mat>> res;
        if (useMultiThreading) {
            thread t(&Segmentation::predict, &testPb, imgs, &res);
            t.join();
        }
        else {
            testPb.predict(imgs, &res);
        }
        finish = clock();

        // ͼƬд��
        string path = "../results/";
        for (int b = 0; b < batchSize; b++) {
            string name = files[i * batchSize + b].substr(strlen(testPath.c_str()) + 1);
            cout << name << endl;
            cv::Mat temp = res[0][b];
            cv::imwrite(path + name, temp);//res[0][b]
        }

        costTime = (double)(finish - start) / CLOCKS_PER_SEC;
        allTime += costTime;
        cout << "��" << i + 1 << "��batch" << "��ʱ" << costTime << endl;
    }
    endAllTime = clock();
    double meanTime = (endAllTime - startAllTime) / CLOCKS_PER_SEC / double(imgNum);
    double meanPicTime = allTime / (double)imgNum;

    std::cout << "���߳�ƽ��ʱ��:" << meanTime << std::endl;
    std::cout << "����ͼƬԤ��ƽ��ʱ�䣺" << meanPicTime << endl;

    system("pause");
}