#include <algorithm>
#include <string>
#include <vector>
#include <iostream>

#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <fstream>

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"

#include <dirent.h>
#include <sys/types.h>

#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace std;

bool chekcIfFileExists(const std::string& name);
bool endswith(std::string const &base, std::string const &ending); //check if a string ends with other string for eg. .jpg
std::vector <std::string> listdir(const std::string& path); // generate a sorted list of files in a directory
std::vector <std::string> listdir(const std::string& path, std::string const &ending);
// generate a sorted list of files ending with specifc string for. eg. files ending with .jpg/.mp4 in a directory

void getFlowAsImage(const Mat &flow_x, const Mat &flow_y, Mat &flowimg, int bound);
