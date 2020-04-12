#include "testSegmentationClassification.h"
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;



int main() {
    // ʱ�����
    clock_t start, finish;
    double costTime = 0, allTime = 0;

    // ͼƬ���ݶ�ȡ
    string testPath = "../../data/";
    string filePath = testPath + "*.png";
    vector<string> files;
    glob(filePath, files, false);
    int imgNum = files.size();
    string path = "../results/";//�������·��

    // �ָ����
    string segmentationConfigPath = "../../model/segmentation/";
    bool useMultiThreading = false;


    // �������
    string classificationConfigPath = "../../model/classification";
 
    segmentationClassification segmentationClassificationModel = segmentationClassification(
        classificationConfigPath,
        segmentationConfigPath,
        useMultiThreading
    );


    // ���Ԥ��
    int batchSize = 1;
    for (int i = 0; i < imgNum / batchSize; ++i) {
        // �ָ�����ͼ����������
        vector<Mat> imgs;
        for (int b = 0; b < batchSize; b++) {
            if ((i*batchSize + b) > imgNum) {
                // Խ���ж�
                break;
            }
            // �ָ�Ҫ��Ҷ�ͼ������
            imgs.push_back(imread(files[i*batchSize + b], 0));
        }

        // ͼ��Ԥ��
        start = clock();
        vector<Mat> results;
        results = segmentationClassificationModel.segmentationClassificationPredict(imgs, batchSize);
      
        finish = clock();

        // �������
        for (int b = 0; b < batchSize; ++b) {
            string name = files[i*batchSize + b].substr(strlen(testPath.c_str()));
            cout << "�ļ���" << name << endl;
            imwrite(path + name, results[b]);
        }

        costTime = (double)(finish - start) / CLOCKS_PER_SEC;
        allTime += costTime;
        cout << "��" << i << "��batch���ѵ�ʱ��" << costTime << endl;
    }
    double meanTime = allTime / double(imgNum);
    double meanTimePerBatch = allTime / double(imgNum / batchSize);
    cout << "����ͼƬԤ��ƽ����ʱ��" << meanTime << endl;
    cout << "batchΪ" << batchSize << "ʱ��ÿ��batchԤ��ƽ����ʱ" << meanTimePerBatch << endl;
    
    system("pause");
    return 0;
}
