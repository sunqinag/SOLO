#include "testClassification.h"

int main() {
    // ��ʱ����
    clock_t start, finish, startAllTime, endAllTime;
    double costTime = 0, allTime = 0, allTimeMultiThread = 0;

    // ���ݻ�ȡ
    string testPath = "..\\data";
    string filepath = testPath + "/*.bmp";
    vector<std::string> files;
    cv::glob(filepath, files, false);
    int imgNum = files.size();

    // Ԥ����ر���
    string configPath = "../../model/classification";
    bool useMultiThreading = false;
    Classification testPb = Classification(configPath,
        useMultiThreading, // �Ƿ�ʹ�ö��߳�
        { "softmax","classes" });
    int batchSize = 3;// ��������ͼƬ����

    startAllTime = clock();
    for (int i = 0; i < imgNum / batchSize; i++) {
        // mat vector����
        vector<cv::Mat> imgs;
        for (int b = 0; b < batchSize; b++) {
            // ��������ͼƬ�ǲ�ɫͼ��
            imgs.push_back(cv::imread(files[i*batchSize + b], 0));
        }

        // ͼ��Ԥ��
        start = clock();
        vector<cv::Mat> res;
        if (useMultiThreading) {
            thread t(&Classification::predict, &testPb, imgs, &res);
            t.join();
        }
        else {
            testPb.predict(imgs, &res);
        }
        finish = clock();

        // ������
        vector<int> classes = testPb.cvMat2Int(res[0]);
        vector<vector<float>> softmax = testPb.cvMat2Float(res[1]);

        for (int b = 0; b < batchSize; b++) {
            cout << "�����ε�" << b << "��ͼ���" << classes[b] << endl;
            cout << "�����ε�" << b << "��ͼ�ĸ���ֵ��" << softmax[b][0]
                << " " << softmax[b][1] << endl;
        }

        costTime = (double)(finish - start) / CLOCKS_PER_SEC;
        allTime += costTime;
        cout << "��" << i + 1 << "��batch" << "��ʱ" << costTime << endl;
    }

    endAllTime = clock();
    double meanTime = (endAllTime - startAllTime) / CLOCKS_PER_SEC / double(imgNum / batchSize);
    double meanPicTime = allTime / (double)imgNum;

    std::cout << "��batch����ƽ��ʱ��:" << meanTime << std::endl;
    std::cout << "����ͼƬԤ��ƽ��ʱ�䣺" << meanPicTime << endl;

    system("pause");
    return 0;
}