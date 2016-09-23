#include "utils.h"

bool endswith(std::string const &base, std::string const &ending) {
    if (base.length() >= ending.length()) {
        return (0 == base.compare(base.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}


void getFlowAsImage(const Mat &flow_x, const Mat &flow_y, Mat &flowimg, int bound)
{
        cv::Size szt = flow_x.size();
        int width = szt.width, height = szt.height;

        float tmp[3];
        float scale = 128/bound;
        //const float* px,py;
        for (int y = 0; y < height; y++)
        {
                //int64 start = getTickCount();

                const float* px = flow_x.ptr<float>(y); // get pointer to a row of flow x
                const float* py = flow_y.ptr<float>(y); // get pointer to a row of flow y

                for (int x = 0; x < width; x++)
                {
                        tmp[0] = px[x]; // access flow_x
                        tmp[1] = py[x]; // access flow_x
                        tmp[2] = std::sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1])*scale+128; // compute magnitute, scale it and offset it by 128 
                        tmp[0] = tmp[0]*scale+128; //scale it and offset it by 128
                        tmp[1] = tmp[1]*scale+128; //scale it and offset it by 128
                        for(int i=0;i<3;i++) // loop to set BGR values
                        {
                                if(tmp[i]<0) tmp[i]=0; // set to 0 if  less than 0
                                if(tmp[i]>255) tmp[i]=255; // set to 255 if more tha 255
                                flowimg.at<cv::Vec3b>(y,x)[2-i] = cvRound(tmp[i]);
                        }

                }
                //double timeSec = (getTickCount() - start) / getTickFrequency();
                //cout << "Time to loop over one row : " << timeSec*1000 << " ms" << endl;
        }
}

bool chekcIfFileExists(const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}


std::vector <std::string> listdir(const std::string& path)
{

  std::vector <std::string> result;
  dirent* de;
  DIR* dp;
  dp = opendir( path.empty() ? "." : path.c_str() );

  if (dp)
    {
    while (true)
      {

      de = readdir( dp );

      if (de == NULL) break;
      std::string s( de->d_name );
      if (!(s[0] == '.')) result.push_back(s);
      }

    closedir( dp );
    std::sort( result.begin(), result.end() );

    }
  else
    {
      cout<<"couldn't open directory"<<endl;
    }

 return result;
}

std::vector <std::string> listdir(const std::string& path, std::string const &ending)
{

  std::vector <std::string> result;  

  std::vector <std::string> filelist = listdir(path); // get filelsit out of directory

  for(int i=0;i<filelist.size();i++)
    {
        if(endswith(filelist[i],".jpg"))  result.push_back(filelist[i]); // add filename to list if it ends with ending
    }
// cout<<"Number of files found in "<< path <<" ending with "<<ending<<" are "<<result.size()<<endl;

 return result;

}

