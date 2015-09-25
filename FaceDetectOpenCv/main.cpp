#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
using namespace std;
using namespace cv;

char *FACES_TXT_PATH = "faceCSV.txt";
char *HARR_XML_PATH = "haarcascade_frontalface_alt.xml";
char *FACES_MODEL = "face.yaml";
char *POTRAITS ="potraits.jpg";
int DEVICE_ID = 0;

int FACE_WIDHT=92;
int FACE_HEIGHT=112;
int POTRITE_WIDTH = 100;
int POTRITE_HEIGHT = 100;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "找不到文件，请核对路径";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }

}

/*利用csv文件读取数据集并训练对应模型*/
void train_data(String fn_csv)
{    
    vector<Mat> images;
    vector<int> labels;
    //获取数据集，如果出错抛出异常
    try {
        read_csv(fn_csv, images, labels);        
    }
    catch (cv::Exception& e) {
        cerr << "打开文件失败 \"" << fn_csv << "\". 原因： " << e.msg << endl;
        exit(1);
    }

    // 如果训练集数量不够退出
    if(images.size() <= 1) {
        string error_message = "训练集图片少于2";
        CV_Error(CV_StsError, error_message);
    }

    //训练模型
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
    model->train(images, labels);
    model->save(FACES_MODEL);
}

void show_portrait(Mat &potrait, Mat &frame) {
    int channels = potrait.channels();
    int nRows = potrait.rows;
    int nCols = potrait.cols*channels;
    
    uchar *p_p, *p_f;
    for(auto i=0; i<nRows; i++) {
        p_p = potrait.ptr<uchar>(i);
        p_f = frame.ptr<uchar>(i);
        for(auto j=0; j<nCols; j++) {
            p_f[j*3] = p_p[j];
            p_f[j*3+1] = p_p[j+1];
            p_f[j*3+2] = p_p[j+2];
        }
    }
    
}

void makePotraitImages(vector<Mat> potraits) {
    int rows = potraits.size()/6;
    if(potraits.size()-rows *6>0)rows++;
    rows *= POTRITE_HEIGHT;
    int cols = 6*POTRITE_HEIGHT;
    Mat potrait_s = Mat(rows,cols,CV_8UC3);
    rows = POTRITE_HEIGHT;
    cols = POTRITE_WIDTH;
    uchar *p_ps, *p_p;
    for(auto i=0; i<potraits.size(); i++) {
        for(auto j=0; j<rows; j++) {
            p_ps = potrait_s.ptr<uchar>(i/6*POTRITE_HEIGHT+j)+3*(i%6)*POTRITE_WIDTH;
            p_p = potraits[i].ptr<uchar>(j);
            for(auto k=0; k<cols; k++) {
                p_ps[k*3] = p_p[k];
                p_ps[k*3+1] = p_p[k+1];
                p_ps[k*3+2] = p_p[k+2];
            }
        }
    }
    imwrite(POTRAITS, potrait_s);
}

void loadPortraits(const string& filename, vector<Mat>& images, char separator = ';') {
    string fn_csv = string(FACES_TXT_PATH);
    std::ifstream file(fn_csv.c_str(), ifstream::in);
    if (!file) {
        string error_message = "找不到文件，请核对路径.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    int label(0);
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            if(atoi(classlabel.c_str()) != label) {
                Mat potrait = imread(path, 0);
                resize(potrait, potrait,Size(POTRITE_WIDTH, POTRITE_HEIGHT));
                images.push_back(potrait);
                label = atoi(classlabel.c_str());
            }
        }
    }
}




int main(){
	// 保存图像和对应标签的向量，要求同一个人的图像必须对应相同的标签
    string fn_csv = string(FACES_TXT_PATH);
    string fn_haar = string(HARR_XML_PATH);

    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
    FileStorage model_file(FACES_MODEL, FileStorage::READ);    
    if(!model_file.isOpened()){
        cout<<"无法找到模型，训练中..."<<endl;
        train_data(fn_csv);//训练数据集，1表示EigenFace 2表示FisherFace 3表示LBPHFace
    }
    model->load(model_file);
    model_file.release();
    vector<Mat> potraits;
    loadPortraits(FACES_MODEL,potraits);
    makePotraitImages(potraits);
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);




	VideoCapture cap(DEVICE_ID);
	if(!cap.isOpened()) {
		cerr << "设备 " << DEVICE_ID << "无法打开" << endl;
		return -1;
	}

	Mat frame;
	for(;;) {
		cap >> frame;
		if(!frame.data)continue;
		// 拷贝现有frame
		Mat original = frame.clone();
		// 灰度化
		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		// 识别frame中的人脸
		vector< Rect_<int> > faces;
		haar_cascade.detectMultiScale(gray, faces);
        
		if(faces.size() != 0)
		{
			int max_area_rect=0;
			for(int i = 0; i < 1; i++) {
				if(faces[i].area() > faces[max_area_rect].area()){
					max_area_rect = i;
				}
            
			}

			// 顺序处理
			Rect face_i = faces[max_area_rect];

			Mat face = gray(face_i);
			rectangle(original, face_i, CV_RGB(0, 0, 255), 1);
			int pridicted_label = -1;
			double predicted_confidence = 0.0;
			model->predict(face, pridicted_label, predicted_confidence);
			string result_text = format("Prediction = %d confidence=%f", pridicted_label, predicted_confidence);
			int text_x = std::max(face_i.tl().x - 10, 0);
			int text_y = std::max(face_i.tl().y - 10, 0);
			putText(original,result_text,  Point(text_x, text_y),FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255), 2.0);
			if(pridicted_label >0)
				show_portrait(potraits[pridicted_label], original);
		}
		// 显示结果:
		imshow("face_recognizer", original);

		char key = (char) waitKey(20);
		if(key == 32)
			exit(0);;
	}
	return 0;
}