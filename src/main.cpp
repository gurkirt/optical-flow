#include "utils.h"
#include "opencv2/imgproc/imgproc.hpp"


bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

float computeFlowForImages(string& inputdir, string& outdir, string& format, int method, int verbosity, int device_id, int bound,int imgscale)
{
	
	
	//	string inputdir = "/mnt/jupiter-beta/DALY/images/SsazwkVoOT4/"; 
	std::vector <std::string> filelist = listdir(inputdir,format); // get list of image in input directory endding with '.jpg'
	//	std::string outdir = "/mnt/jupiter-beta/DALY/motion-images/SsazwkVoOT4/";
	//	int method = 0;

	system(("mkdir -p "+outdir).c_str()); // create directory



	std::string outfilename; 

	int frame_num = 0;
	int nb_frames = filelist.size();

	if(verbosity>0) cout<<"Start computing flow for "<<nb_frames<<" frames in "<<outdir<<endl;


	Mat prev_image, next_image, flow_x, flow_y,flowimg;
	GpuMat frame_0, frame_1, d_frame0f, d_frame1f, flow;

	setDevice(device_id);
	Ptr<cuda::BroxOpticalFlow> brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);  // instantiate Brox optical flow object
	Ptr<cuda::FarnebackOpticalFlow> farn = cuda::FarnebackOpticalFlow::create();  // instantiate Farneback optical flow object
	Ptr<cuda::DensePyrLKOpticalFlow> lk = cuda::DensePyrLKOpticalFlow::create(Size(7, 7));  // instantiate LK optical flow object
	Ptr<cuda::OpticalFlowDual_TVL1> tvl1 = cuda::OpticalFlowDual_TVL1::create(0.25,0.15,0.3,5,5,0.01,300,0.8,0.0);  // instantiate TVL1 optical flow object

	//OpticalFlowDual_TVL1_GPU alg_tvl1;
	//BroxOpticalFlow alg_brox(0.197f, 50.0f, 0.8f, 10, 77, 10);
	int64 fstart,start; 
	double totaltime = 0.0;
	int cstart = 0;
	for(frame_num = 0; frame_num<nb_frames; frame_num++) {

		fstart = getTickCount();
		start = getTickCount();
		
		bool doresize = false;
		if (imgscale>1) doresize = true;
		cv::Size dstSize;
		outfilename = outdir+filelist[frame_num];
		if (!chekcIfFileExists(outfilename))
		{

			//if (verbosity>1) cout<<"start computting for image "<<filelist[frame_num];
			if((frame_num == 0) || (cstart==0))
			{ 
				prev_image = cv::imread(inputdir+filelist[frame_num], IMREAD_GRAYSCALE);
				dstSize.height = int(prev_image.rows/imgscale);
				dstSize.width = int(prev_image.cols/imgscale);
				
				printf("orignal %d %d \n",prev_image.rows,prev_image.cols);
				if (doresize) cv::resize(prev_image,prev_image,dstSize);
				printf("after %d %d \n",prev_image.rows,prev_image.cols);
				next_image = cv::imread(inputdir+filelist[frame_num+1], IMREAD_GRAYSCALE);
				if (doresize) cv::resize(next_image,next_image,dstSize);

			}
			else if(frame_num == nb_frames-1){
				prev_image = cv::imread(inputdir+filelist[frame_num-1], IMREAD_GRAYSCALE);
				dstSize.height = int(prev_image.rows/imgscale);
				dstSize.width = int(prev_image.cols/imgscale);
				cv::resize(prev_image,prev_image,dstSize);
				next_image = cv::imread(inputdir+filelist[frame_num], IMREAD_GRAYSCALE);
				if (doresize) cv::resize(next_image,next_image,dstSize);
			}
			else{
				//				cout<<"we are inlast else castart = "<<cstart<<endl;
				next_image = cv::imread(inputdir+filelist[frame_num+1], IMREAD_GRAYSCALE);
				dstSize.height = int(next_image.rows/imgscale);
				dstSize.width = int(next_image.cols/imgscale);
				if (doresize) cv::resize(next_image,next_image,dstSize);
			}
			//printf("Next %d %d \n",next_image.rows,next_image.cols);
			double timeSec = (getTickCount() - start) / getTickFrequency();

			if (verbosity>2) cout << "Time to load images : " << timeSec*1000 << " ms" << endl;

			start = getTickCount();
			frame_0.upload(prev_image);
			frame_1.upload(next_image);

			timeSec = (getTickCount() - start) / getTickFrequency();

			if (verbosity>3) cout << "Time to upload images to GPU : " << timeSec*1000 << " ms" << endl;

			start = getTickCount();	

			switch(method){
			case 0:
				frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
				frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
				brox->calc(d_frame0f, d_frame1f, flow);
				break;
			case 1:
				farn->calc(frame_0,frame_1,flow);
				break;
			case 2:
				lk->calc(frame_0,frame_1,flow);
				break;
			case 3:
				tvl1->calc(frame_0,frame_1,flow);
				break;
			}

			timeSec = (getTickCount() - start) / getTickFrequency();
			if (verbosity>2) cout << "Time to compute flow : " << timeSec*1000 << " ms" << endl;

			start = getTickCount();
			GpuMat planes[2];
			cuda::split(flow, planes);
			planes[0].download(flow_x);
			planes[1].download(flow_y);

			timeSec = (getTickCount() - start) / getTickFrequency();
			if (verbosity>3) cout << "Time to post process : " << timeSec*1000 << " ms" << endl;

			start = getTickCount();
			if ((frame_num ==0) || cstart==0) {flowimg = cv::Mat(flow_x.size(),CV_8UC3);cstart=1;}
			getFlowAsImage(flow_x, flow_y, flowimg, bound);

			timeSec = (getTickCount() - start) / getTickFrequency();
			if (verbosity>2) cout << "Time to convert : " << timeSec*1000 << " ms" << endl;

			start = getTickCount();

			imwrite(outfilename,flowimg);

			timeSec = (getTickCount() - start) / getTickFrequency();
			if (verbosity>2) cout << "Time to write : " << timeSec*1000 << " ms" << endl;

			start = getTickCount();
			std:swap(prev_image,next_image); // prev_image = next_image

			timeSec = (getTickCount() - start) / getTickFrequency();
			if (verbosity>3) cout << "Time to swap : " << timeSec*1000 << " ms" << endl;

			timeSec = (getTickCount() - fstart) / getTickFrequency();
			if (verbosity>1) cout << " total time to compute "<<timeSec*1000 << " ms" << " ["<<frame_num<<"/"<<nb_frames<<"]"<<endl;
			totaltime = totaltime + timeSec;
		}
		
	}

	return float(totaltime/nb_frames);

}

int computeFlowVideos(string& inputdir, string& outdir, string& format, int method, int verbosity, int device_id, int bound, int start, int end)
{

	std::vector <std::string> dirlist = listdir(inputdir);
	//for(int i=dirlist.size()-1;i>=0;i--)
	//for(int i=0;i<dirlist.size();i++)
	//for(int i=start; i>=0; i--)
	for(int i=start; i<dirlist.size(); i++)
	{
		string basevidedir = dirlist[i];
		replace(basevidedir,"\(","\\(");
		replace(basevidedir,"\)","\\)");
		string inputBasedir = inputdir+basevidedir+"/";
		string outBasedir = outdir+basevidedir+"/";

		float avg_time =  computeFlowForImages(inputBasedir, outBasedir, format, method, verbosity, device_id, bound,1);
		if(verbosity>0) cout<<"Done computting for directory number "<<i<<" with avergae time of "<<avg_time<<" seconds per image"<<endl<<endl;

	}
}

int computeFlowVideosOnActions(string& inputdir, string& outdir, string& format, int method, int verbosity, int device_id, int bound ,int imgscale)
{
	std::vector <std::string> actiondirlist = listdir(inputdir);
	for(int i=0; i<actiondirlist.size();i++)
	{
		string actiondir = inputdir+actiondirlist[i]+"/";
		string outactiondir = outdir+actiondirlist[i]+"/";
		std::vector <std::string> dirlist = listdir(actiondir);
		for(int i=dirlist.size()-1;i>=0;i--)
		//for(int i=start; i<end; i++)
		{
			
			string basevidedir = dirlist[i];
			replace(basevidedir,"\(","\\(");
			replace(basevidedir,"\)","\\)");
			string inputBasedir = actiondir+basevidedir+"/";
			string outBasedir = outactiondir+basevidedir+"/";
	
			float avg_time =  computeFlowForImages(inputBasedir, outBasedir, format, method, verbosity, device_id, bound,imgscale);
			if(verbosity>0) cout<<"Done computting for directory number "<<i<<" with avergae time of "<<avg_time<<" seconds per image"<<endl<<endl;
	
		}
	}
}
int main(int argc, char **argv)
{

	const char* keys =
	{
			"{ m   method     | 0 | specify the optical flow algorithm }"
			"{ md  multivideos | 0 | if doing it for muliple videos multivideos }"
			"{ act action     | 0 | if directory structure is based on action }"
			"{ id  inputDir   | /mnt/jupiter-beta/THUMOS/UCF101-rgb-Images/ | path of directory which contain images of folders with images where each folder belog to one video }"
			"{ od  outputDir  | /mnt/jupiter-beta/THUMOS/UCF101-motion-Images/ | Name of the directory where flow images will be stroed }"
			"{ f   format     | .jpg | format of input images }"
			"{ b   bound      | 8 | specify the maximum of optical flow }"
			"{ d   device_id  | 0 | set gpu id }"
			"{ v   verbosity  | 1 | verbosity }"
			"{ sv  start  | 0 | start video number }"
			"{ ev  end  | 19994 | end video number }"
			"{ ims imgscale  | 1 | imgscale }"
	};

	CommandLineParser cmd(argc, argv, keys);
	string inputBasedir = cmd.get<string>("inputDir"); // inputer diretory where images (folder with images) are present from a video or multiple videos
	string outBaseDir = cmd.get<string>("outputDir");  // save flow image in this directory
	string format = cmd.get<string>("format");

	int mv = cmd.get<int>("multivideos"); // if doing for multiple folders it would be 1 and inputDir should be base directory
	int method = cmd.get<int>("method");
	int bound = cmd.get<int>("bound");
	int device_id = cmd.get<int>("device_id");
	int verbosity = cmd.get<int>("verbosity");
	int start = cmd.get<int>("start");
	int end = cmd.get<int>("end");
	int isact = cmd.get<int>("action");
	int imgscale = cmd.get<int>("imgscale");
	cout<<"Done reading parameters\n";
	if (isact>0) {computeFlowVideosOnActions(inputBasedir, outBaseDir, format, method, verbosity, device_id, bound,imgscale);}
	else {computeFlowVideos(inputBasedir, outBaseDir, format, method, verbosity, device_id, bound, start, end);}

}
