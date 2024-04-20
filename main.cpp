//#include <torch/torch.h>

#include <torch/script.h> // PyTorch C++ API
#include <opencv2/opencv.hpp> // OpenCV 
#include <iostream>


class Box {
    public:
    cv::Rect2d rect;
    float conf;
    Box(int x1, int y1, int w, int h, float conf){
        this->rect = cv::Rect2d(x1,y1,w,h);
        this->conf = conf;
    }
};

float iou(Box &fb, Box &sb) {
    float inter = (fb.rect & sb.rect).area();
    float union_ = fb.rect.area() + sb.rect.area() - inter;
    return inter/union_;
}

std::vector<Box> nms(std::vector<Box> &boxes, float iouThres){
    std::vector<Box> supBoxes;
    for (Box box: boxes) {
        bool valid = true;
        for (Box supBox: supBoxes) {
            if (&box!=&supBox && iou(box,supBox)>iouThres){
                valid = false;
                break;
            }
        }
        if (valid == true){
            supBoxes.push_back(box);
        }
    }
    return supBoxes;
}

std::vector<Box> getBoxes(at::Tensor &outputs, float confThres = 0.5, float iouThres = 0.15){
    std::vector<Box> candidates;
    for (unsigned short ibatch = 0; ibatch<outputs.sizes()[0]; ibatch++){
        for(unsigned short ibox = 0; ibox<outputs.sizes()[2]; ibox++){
            float conf = outputs[ibatch][4][ibox].item<float>();
            if (conf>= confThres) {
                int cx = outputs[ibatch][0][ibox].item<int>(),
                cy = outputs[ibatch][1][ibox].item<int>(),
                w = outputs[ibatch][2][ibox].item<int>(),
                h = outputs[ibatch][3][ibox].item<int>();

                int x1 = cx - w/2,
                y1 = cy - h/2;

                candidates.push_back(Box(x1,y1,w,h,conf));
            }
        }
    }
    sort(candidates.begin(),candidates.end(), [](Box b1, Box b2){return b1.conf>b2.conf;});
    std::vector<Box> boxes = nms(candidates,iouThres);

    return boxes;
}

void DrawBoxes(cv::Mat &img, std::vector<Box> &boxes) {
    cv::Scalar rectColor (0,0,192);
    int fontScale = 2, confPrecis = 2;
    for (Box box: boxes) {
        std::string text = std::to_string(box.conf);
        cv::rectangle(img, box.rect, rectColor,2);
        cv::Rect2d conf_f = cv::Rect2d(box.rect.x,box.rect.y-fontScale*12,(unsigned short)text.length()* fontScale*10,fontScale*12);
        cv::rectangle(img, conf_f, rectColor, 1);

        cv::putText(img, text, {int(box.rect.x),int(box.rect.y)}, cv::FONT_HERSHEY_PLAIN, fontScale, {255,255,255}, 2);

    }

}


int main() {
    torch::jit::script::Module model = torch::jit::load("../../best.torchscript");

    cv::VideoCapture cap("../../test_vid.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video file." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::Mat inputFrame;
        cv::cvtColor(frame, inputFrame, cv::COLOR_BGR2RGB);
        cv::normalize(frame,frame, 0.0,1.0, cv::NORM_MINMAX, CV_32F);
        cv::Mat resizedFrame;
        cv::resize(inputFrame, resizedFrame, cv::Size(640, 640));
        cv::Mat tensorFrame;
        resizedFrame.convertTo(tensorFrame, CV_32FC3, 1.0f / 255.0f);
        torch::Tensor inputTensor = torch::from_blob(tensorFrame.data, {640, 640, 3},torch::kFloat32).permute({2,0,1}).unsqueeze(0);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(inputTensor);
        at::Tensor outputs = model.forward(inputs).toTensor();
        std::vector<Box> boxes = getBoxes(outputs);
        DrawBoxes(resizedFrame,boxes);
        cv::namedWindow("OUT", cv::WINDOW_NORMAL);
        cv::cvtColor(resizedFrame, resizedFrame, cv::COLOR_RGB2BGR);
        cv::imshow("OUT", resizedFrame);
        cv::resizeWindow("OUT", 640,640);
        if (cv::waitKey(1) == 27) // Press ESC to close window
            break;
    }

    return 0;
}
