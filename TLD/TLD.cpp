/*
 * TLD.cpp
 *
 *  Created on: Jun 9, 2011
 *      Author: alantrrs
 */
#include "stdafx.h"
#include <TLD.h>
#include <stdio.h>
using namespace cv;
using namespace std;


TLD::TLD()
{
}
TLD::TLD(const FileNode& file){
  read(file);
}

void TLD::read(const FileNode& file){
  ///Bounding Box Parameters
  min_win = (int)file["min_win"];//������������Сֵ
  ///Genarator Parameters
  //initial parameters for positive examples
  patch_size = (int)file["patch_size"];
  num_closest_init = (int)file["num_closest_init"];
  num_warps_init = (int)file["num_warps_init"];
  noise_init = (int)file["noise_init"];
  angle_init = (float)file["angle_init"];
  shift_init = (float)file["shift_init"];
  scale_init = (float)file["scale_init"];
  //update parameters for positive examples
  num_closest_update = (int)file["num_closest_update"];
  num_warps_update = (int)file["num_warps_update"];
  noise_update = (int)file["noise_update"];
  angle_update = (float)file["angle_update"];
  shift_update = (float)file["shift_update"];
  scale_update = (float)file["scale_update"];
  //parameters for negative examples
  bad_overlap = (float)file["overlap"];
  bad_patches = (int)file["num_patches"];
  classifier.read(file);
}

void TLD::init(const Mat& frame1,const Rect& box,FILE* bb_file){
  //bb_file = fopen("bounding_boxes.txt","w");
  //Get Bounding Boxes
	//�˺������ݴ����box��Ŀ��߽���ڴ����ͼ��frame1�й���ȫ����ɨ�贰�ڣ��������ص���  
    buildGrid(frame1,box);
    printf("Created %d bounding boxes\n",(int)grid.size());
  ///Preparation
  //allocationΪ���ֱ����������������ڴ�ռ䣻
	////����ͼ�����Լ���2bitBP������������haar�����ļ��㣩 
	//Mat�Ĵ�������ʽ�����֣�1.����create���У��У����ͣ�2.Mat���У��У����ͣ�ֵ������ 
  iisum.create(frame1.rows+1,frame1.cols+1,CV_32F);
  iisqsum.create(frame1.rows+1,frame1.cols+1,CV_64F);
  //Detector data�ж��壺std::vector<float> dconf;  ���ȷ�Ŷȣ���
  //vector ��reserve������vector��capacity����������sizeû�иı䣡��resize�ı���vector
  //��capacityͬʱҲ����������size��reserve������Ԥ���ռ䣬���ڿռ��ڲ���������Ԫ�ض���
  //������û������µĶ���֮ǰ���������������ڵ�Ԫ�ء�
  //�����ǵ���resize����reserve�����߶�����ԭ�е�Ԫ�ض�û��Ӱ�졣
  dconf.reserve(100);// ��Ԫ�ػ�û�й���, ��ʱ������[]����Ԫ��
  dbb.reserve(100);
  bbox_step =7;
  //tmp.conf.reserve(grid.size());
  //������Detector data�ж�����������������grid.size()��С�������һ��ͼ����ȫ����ɨ�贰�ڸ�����������
  //Detector data�ж���TempStruct tmp;  
  tmp.conf = vector<float>(grid.size());
  tmp.patt = vector<vector<int> >(grid.size(),vector<int>(10,0));
  //tmp.patt.reserve(grid.size());
  dt.bb.reserve(grid.size());
  good_boxes.reserve(grid.size());//�ص��ȺõĴ�
  bad_boxes.reserve(grid.size());//�ص��Ȳ�Ĵ�
  pEx.create(patch_size,patch_size,CV_64F);//TLD�ж��壺cv::Mat pEx;  //positive NN example ��СΪ15*15ͼ��Ƭ
  //Init Generator
  //TLD�ж��壺cv::PatchGenerator generator;  //PatchGenerator��������ͼ��������з���任
  /*
  cv::PatchGenerator::PatchGenerator (
  double     _backgroundMin,
  double     _backgroundMax,
  double     _noiseRange,
  bool     _randomBlur = true,
  double     _lambdaMin = 0.6,
  double     _lambdaMax = 1.5,
  double     _thetaMin = -CV_PI,
  double     _thetaMax = CV_PI,
  double     _phiMin = -CV_PI,
  double     _phiMax = CV_PI
  )
  һ����÷����ȳ�ʼ��һ��PatchGenerator��ʵ����Ȼ��RNGһ��������ӣ��ٵ��ã������������һ���任�����������
  */
  generator = PatchGenerator (0,0,noise_init,true,1-scale_init,1+scale_init,-angle_init*CV_PI/180,angle_init*CV_PI/180,-angle_init*CV_PI/180,angle_init*CV_PI/180);

  //�˺������ݴ����box��Ŀ��߽�򣩣�����֡ͼ���е�ȫ��������Ѱ�����box������С���������ƣ�
  //�ص�����󣩵�num_closest_init�����ڣ�Ȼ�����Щ���� ����good_boxes����
  //ͬʱ�����ص���С��0.2�ģ����� bad_boxes ����
  //���ȸ���overlap�ı�����Ϣѡ���ظ������������60%����ǰnum_closet_init= 10������ӽ�box��RectBox��
  //�൱�ڶ�RectBox����ɸѡ����ͨ��BBhull�����õ���ЩRectBox�����߽硣
  getOverlappingBoxes(box,num_closest_init);
  printf("Found %d good boxes, %d bad boxes\n",(int)good_boxes.size(),(int)bad_boxes.size());
  printf("Best Box: %d %d %d %d\n",best_box.x,best_box.y,best_box.width,best_box.height);
  printf("Bounding box hull: %d %d %d %d\n",bbhull.x,bbhull.y,bbhull.width,bbhull.height);
  //Correct Bounding Box
  lastbox=best_box;
  lastconf=1;
  lastvalid=true;
  //Print
  fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
  //Prepare Classifier
  classifier.prepare(scales);
  ///Generate Data
  // Generate positive data
  generatePositiveData(frame1,num_warps_init);
  // Set variance threshold
  Scalar stdev, mean;
  ////������Ҫ��ȡͼ��A��ĳ��ROI������Ȥ�����ɾ��ο򣩵Ļ�����Mat���B=img(ROI)������ȡ
  //frame1(best_box)�ͱ�ʾ��frame1����ȡbest_box����Ŀ�����򣩵�ͼ��Ƭ
  meanStdDev(frame1(best_box),mean,stdev);//ͳ��best_box�ľ�ֵ�ͱ�׼�var = pow(stdev.val[0],2) * 0.5;
  //���û���ͼ��ȥ����ÿ������ⴰ�ڵķ���
  //cvIntegral( const CvArr* image, CvArr* sum, CvArr* sqsum=NULL, CvArr* tilted_sum=NULL );
  //�������ͼ������ͼ��sum����ͼ��, W+1��H+1��sqsum������ֵƽ���Ļ���ͼ��tilted_sum��ת45�ȵĻ���ͼ��
  //���û���ͼ�񣬿��Լ�����ĳ���ص��ϣ��ҷ��Ļ�����ת�ľ��������н�����͡����ֵ�Լ���׼����ļ��㣬
  //���ұ�֤����ĸ��Ӷ�ΪO(1)��  
  integral(frame1,iisum,iisqsum);
  //����������ģ��һ��������ģ�飬���û���ͼ����ÿ������ⴰ�ڵķ���������var��ֵ��Ŀ��patch�����50%���ģ�
  //����Ϊ�京��ǰ��Ŀ�귽�var Ϊ��׼���ƽ��
  var = pow(stdev.val[0],2)*0.5; //getVar(best_box,iisum,iisqsum);��Ϊ�������������ֵ
  cout << "variance: " << var << endl;
  //check variance
  double vr =  getVar(best_box,iisum,iisqsum)*0.5;
  cout << "check variance: " << vr << endl;
  // Generate negative data
  //����TLD������һ��Ŀ�꣬��������ȷ����Ŀ����ˣ��ʳ�Ŀ����������ͼ���Ǹ��������������任��
  generateNegativeData(frame1);
  //Split Negative Ferns into Training and Testing sets (they are already shuffled)
  /*Ȼ��nEx��һ����Ϊѵ����nEx����һ����Ϊ���Լ�nExT��
  ͬ����nXҲ���Ϊѵ����nX�Ͳ��Լ�nXT��*/
  int half = (int)nX.size()*0.5f;
  nXT.assign(nX.begin()+half,nX.end());
  nX.resize(half);
  ///Split Negative NN Examples into Training and Testing sets
  half = (int)nEx.size()*0.5f;
  nExT.assign(nEx.begin()+half,nEx.end());
  nEx.resize(half);
  //Merge Negative Data with Positive Data and shuffle it
  //��������nX��������pX�ϲ���ferns_data[]�У����ڼ��Ϸ�������ѵ����
  vector<pair<vector<int>,int> > ferns_data(nX.size()+pX.size());
  vector<int> idx = index_shuffle(0,ferns_data.size());
  int a=0;
  for (int i=0;i<pX.size();i++){
      ferns_data[idx[a]] = pX[i];
      a++;
  }
  for (int i=0;i<nX.size();i++){
      ferns_data[idx[a]] = nX[i];
      a++;
  }
  //������õ���һ��������pEx��nEx�ϲ���nn_data[]�У���������ڷ�������ѵ����
  //Data already have been shuffled, just putting it in the same vector
  vector<cv::Mat> nn_data(nEx.size()+1);//����ڷ�����
  nn_data[0] = pEx;
  for (int i=0;i<nEx.size();i++){
      nn_data[i+1]= nEx[i];
  }
  ///Training
  //�����������ѵ����ѵ�� ���Ϸ�������ɭ�֣� �� ����ڷ�������
  classifier.trainF(ferns_data,2); //bootstrap = 2
  classifier.trainNN(nn_data);
  ///Threshold Evaluation on testing sets
  //�ò��Լ�������õ��� ���Ϸ�������ɭ�֣� �� ����ڷ������з��࣬���۲��޸ĵõ���õķ�������ֵ��
  classifier.evaluateTh(nXT,nExT);
}

/* Generate Positive data
 * Inputs:
 * - good_boxes (bbP)
 * - best_box (bbP0)
 * - frame (im0)
 * Outputs:
 * - Positive fern features (pX)
 * - Positive NN examples (pEx)
 */
void TLD::generatePositiveData(const Mat& frame, int num_warps){
	/*
	CvScalar����ɴ��1��4����ֵ����ֵ���������洢���أ���ṹ�����£�
	typedef struct CvScalar
	{
	double val[4];
	}CvScalar;
	���ʹ�õ�ͼ����1ͨ���ģ���s.val[0]�д洢����
	���ʹ�õ�ͼ����3ͨ���ģ���s.val[0]��s.val[1]��s.val[2]�д洢����
	*/
  Scalar mean;
  Scalar stdev;
  //��һ��best_box���򣬴���pEX��
  getPattern(frame(best_box),pEx,mean,stdev);
  //Get Fern features on warped patches
  Mat img;
  Mat warped;
  //void GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY=0, 
  //									int borderType=BORDER_DEFAULT ) ;
  //���ܣ��������ͼ��src���и�˹�˲�����dst�����
  //src��dst��Ȼ�ֱ�������ͼ������ͼ��KsizeΪ��˹�˲���ģ���С��sigmaX��sigmaY�ֱ�Ϊ��˹��
  //���ں����������˲�ϵ����borderTypeΪ��Ե��չ���ֵ���͡�
  //��9*9��˹��ģ������֡������img  ȥ�룿��
  GaussianBlur(frame,img,Size(9,9),1.5);//��˹�˲�
  //��imgͼ���н�ȡbbhull��Ϣ��bbhull�ǰ�����λ�úʹ�С�ľ��ο򣩵�ͼ�񸳸�warped
  //������Ҫ��ȡͼ��A��ĳ��ROI������Ȥ�����ɾ��ο򣩵Ļ�����Mat���B=img(ROI)������ȡ
  warped = img(bbhull);//ȡ�ص���ͼ��
  RNG& rng = theRNG();//���������
  Point2f pt(bbhull.x+(bbhull.width-1)*0.5f,bbhull.y+(bbhull.height-1)*0.5f);//bbhull�е�
  vector<int> fern(classifier.getNumStructs());
  pX.clear();//������
  Mat patch;
  //ÿ��box������num_warps�����ּ��α任������good_boxes.size��box
  //pXΪ������RectBox���߽紦����������Ϣ��pEx����ڵ�RectBox��Pattern��bbP0Ϊ����ڵ�RectBox��
  if (pX.capacity()<num_warps*good_boxes.size())
    pX.reserve(num_warps*good_boxes.size());
  int idx;
  for (int i=0;i<num_warps;i++){
     if (i>0)//��һ��good_box����Ҫ����任
		 //������ͼ��������з���任����RNGһ��������ӣ��ٵ��ã������������һ���任�����������
       generator(frame,pt,warped,bbhull.size(),rng);
    for (int b=0;b<good_boxes.size();b++){
        idx=good_boxes[b];//good_boxes����������� grid ������
		patch = img(grid[idx]); //��img�� grid[idx] ����Ҳ����bounding box�ص��ȸߵģ���һ��ͼ��Ƭ��ȡ����
		//�����õ������patch������fern��13λ�Ķ����ƴ��룩��
		classifier.getFeatures(patch, grid[idx].sidx, fern); //getFeatures�����õ������patch���������Ľڵ㣬Ҳ���������������fern��13λ�Ķ����ƴ��룩
		//positive ferns <features, labels=1>Ȼ����Ϊ������������pX�����ڼ��Ϸ����������������������⣻
        pX.push_back(make_pair(fern,1));//positive ferns <features, labels=1>  ������
     }
  }
  printf("Positive examples generated: ferns:%d NN:1\n",(int)pX.size());
}

//�˺�����frameͼ��best_box�����ͼ��Ƭ��һ��Ϊ��ֵΪ0��15 * 15��С��patch��
//����pEx����������ڷ����������������������У�����ڵ�box��Pattern������������ֻ��һ����
void TLD::getPattern(const Mat& img, Mat& pattern,Scalar& mean,Scalar& stdev){
  //Output: resized Zero-Mean patch
  resize(img,pattern,Size(patch_size,patch_size));
  meanStdDev(pattern,mean,stdev);
  pattern.convertTo(pattern,CV_32F);
  pattern = pattern-mean.val[0];
}

/*����֮ǰ�ص���С��0.2�ģ������� bad_boxes�ˣ���������ͦ�࣬
�ѷ������var*0.5f��bad_boxes�����븺������ͬ����һ����
��Ҫclassifier.getFeatures(patch, grid[idx].sidx, fern);��nX.push_back(make_pair(fern, 0));
�õ���Ӧ��fern�����ͱ�ǩ��nX�����������ڼ��Ϸ������ĸ���������*/
void TLD::generateNegativeData(const Mat& frame){
/* Inputs:
 * - Image
 * - bad_boxes (Boxes far from the bounding box)
 * - variance (pEx variance)
 * Outputs
 * - Negative fern features (nX)
 * - Negative NN examples (nEx)
 */
  random_shuffle(bad_boxes.begin(),bad_boxes.end());//Random shuffle bad_boxes indexes
  int idx;
  //Get Fern Features of the boxes with big variance (calculated using integral images)
  int a=0;
  //int num = std::min((int)bad_boxes.size(),(int)bad_patches*100); //limits the size of bad_boxes to try
  printf("negative data generation started.\n");
  vector<int> fern(classifier.getNumStructs());
  nX.reserve(bad_boxes.size());
  Mat patch;
  for (int j=0;j<bad_boxes.size();j++){
      idx = bad_boxes[j];
	  if (getVar(grid[idx], iisum, iisqsum)<var*0.5f)//�ѷ������var*0.5f��bad_boxes�����븺����
            continue;
      patch =  frame(grid[idx]);
	  classifier.getFeatures(patch,grid[idx].sidx,fern);
      nX.push_back(make_pair(fern,0));
      a++;
  }
  printf("Negative examples generated: ferns: %d ",a);
  //random_shuffle(bad_boxes.begin(),bad_boxes.begin()+bad_patches);//Randomly selects 'bad_patches' and get the patterns for NN;
  /* Ȼ������������bad_boxes��ȡbad_patches��100������box��
  Ȼ���� getPattern������frameͼ��bad_box�����ͼ��Ƭ��һ����15*15��С��patch��
  ����nEx����������ڷ������ĸ��������������С�*/
  Scalar dum1, dum2;
  nEx=vector<Mat>(bad_patches);
  for (int i=0;i<bad_patches;i++){
      idx=bad_boxes[i];
	  patch = frame(grid[idx]);
      getPattern(patch,nEx[i],dum1,dum2);
  }
  printf("NN: %d\n",(int)nEx.size());
}

double TLD::getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum){
  double brs = sum.at<int>(box.y+box.height,box.x+box.width);
  double bls = sum.at<int>(box.y+box.height,box.x);
  double trs = sum.at<int>(box.y,box.x+box.width);
  double tls = sum.at<int>(box.y,box.x);
  double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
  double blsq = sqsum.at<double>(box.y+box.height,box.x);
  double trsq = sqsum.at<double>(box.y,box.x+box.width);
  double tlsq = sqsum.at<double>(box.y,box.x);
  double mean = (brs+tls-trs-bls)/((double)box.area());
  double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
  return sqmean-mean*mean;//D(x) = E(x*x) - E(x)*E(x)���㷽��
}

//processFrame�������ĸ�ģ�飨���δ���������ģ�顢���ģ�顢�ۺ�ģ���ѧϰģ�飻
void TLD::processFrame(const cv::Mat& img1,const cv::Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2,
	BoundingBox& bbnext,bool& lastboxfound, bool tl, FILE* bb_file){
  vector<BoundingBox> cbb;//���п��ܵĴ�
  vector<float> cconf;//���п��ܴ������ƶ�
  int confident_detections=0;
  int didx; //detection index
  ///Track
  if(lastboxfound && tl){
      track(img1,img2,points1,points2);//track�������ǰһ֡img1��������points1����ǰ֡img2��������points2�ĸ���Ԥ��
  }
  else{
      tracked = false;
  }
  ///Detect
  detect(img2);
  ///Integration
  if (tracked){
      bbnext=tbb;
      lastconf=tconf;
      lastvalid=tvalid;
      printf("Tracked\n");
      if(detected){                                               //   if Detected
		  //
          clusterConf(dbb,dconf,cbb,cconf);                       //   cluster detections
          printf("Found %d clusters\n",(int)cbb.size());
          for (int i=0;i<cbb.size();i++){
			  //tbb���ٴ���tconf���ƶ����ҵ�����������ٵ���box����Ƚ�Զ���ࣨ�������⵽��box������������������ƶȱȸ�������Ҫ��
			  //��¼��������������Ҳ���ǿ��ŶȱȽϸߵ�Ŀ��box�ĸ�����
              if (bbOverlap(tbb,cbb[i])<0.5 && cconf[i]>tconf){  //  Get index of a clusters that is far from tracker and are more confident than the tracker
                  confident_detections++;
                  didx=i; //detection index
              }
          }
          if (confident_detections==1){                                //if there is ONE such a cluster, re-initialize the tracker
              printf("Found a better match..reinitializing tracking\n");
			  //�ж����ֻ��һ����������������box����ô�������Ŀ��box�����³�ʼ��������
              bbnext=cbb[didx];
              lastconf=cconf[didx];
              lastvalid=false;
          }
          else {
              printf("%d confident cluster was found\n",confident_detections);
              int cx=0,cy=0,cw=0,ch=0;
              int close_detections=0;
			  //�����������������box��ֻһ������ô���ҵ��������⵽��box��
			  //������Ԥ�⵽��box����ܽ����ص��ȴ���0.7��������box����������ʹ�С�����ۼӣ�
              for (int i=0;i<dbb.size();i++){
                  if(bbOverlap(tbb,dbb[i])>0.7){                     // Get mean of close detections
                      cx += dbb[i].x;
                      cy +=dbb[i].y;
                      cw += dbb[i].width;
                      ch += dbb[i].height;
                      close_detections++;
                      printf("weighted detection: %d %d %d %d\n",dbb[i].x,dbb[i].y,dbb[i].width,dbb[i].height);
                  }
              }
              if (close_detections>0){
				  //���������Ԥ�⵽��box����ܽ���box �� ����������Ԥ�⵽��box 
				  //�����������С��ƽ����Ϊ���յ�Ŀ��bounding box�����Ǹ�������Ȩֵ�ϴ�
                  bbnext.x = cvRound((float)(10*tbb.x+cx)/(float)(10+close_detections));   // weighted average trackers trajectory with the close detections
                  bbnext.y = cvRound((float)(10*tbb.y+cy)/(float)(10+close_detections));
                  bbnext.width = cvRound((float)(10*tbb.width+cw)/(float)(10+close_detections));
                  bbnext.height =  cvRound((float)(10*tbb.height+ch)/(float)(10+close_detections));
                  printf("Tracker bb: %d %d %d %d\n",tbb.x,tbb.y,tbb.width,tbb.height);
                  printf("Average bb: %d %d %d %d\n",bbnext.x,bbnext.y,bbnext.width,bbnext.height);
                  printf("Weighting %d close detection(s) with tracker..\n",close_detections);
              }
              else{
                printf("%d close detections were found\n",close_detections);

              }
          }
      }
  }
  else{ //���������û�и��ٵ�Ŀ��         //   If NOT tracking
      printf("Not tracking..\n");
      lastboxfound = false;
      lastvalid = false;
      if(detected){  //�������⵽��һЩ���ܵ�Ŀ��box  and detector is defined
          clusterConf(dbb,dconf,cbb,cconf);   //  cluster detections
          printf("Found %d clusters\n",(int)cbb.size());
          if (cconf.size()==1){
              bbnext=cbb[0];//��ֻ�Ǽ򵥵Ľ������cbb[0]��Ϊ�µĸ���Ŀ��box
              lastconf=cconf[0];
              printf("Confident detection..reinitializing tracker\n");
              lastboxfound = true;
          }
      }
  }
  lastbox=bbnext;
  if (lastboxfound)
    fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
  else
    fprintf(bb_file,"NaN,NaN,NaN,NaN,NaN\n");
  if (lastvalid && tl)
    learn(img2);//ѧϰģ��
}

//track�������ǰһ֡img1��������points1����ǰ֡img2��������points2�ĸ���Ԥ�⣻
void TLD::track(const Mat& img1, const Mat& img2,vector<Point2f>& points1,vector<Point2f>& points2){
  /*Inputs:
   * -current frame(img2), last frame(img1), last Bbox(bbox_f[0]).
   *Outputs:
   *- Confidence(tconf), Predicted bounding box(tbb),Validity(tvalid), points2 (for display purposes only)
   */
  //Generate points
  bbPoints(points1,lastbox);//����lastbox�о��Ȳ���10*10=100�������㣨����������㣩������points1��
  if (points1.size()<1){
      printf("BB= %d %d %d %d, Points not generated\n",lastbox.x,lastbox.y,lastbox.width,lastbox.height);
      tvalid=false;
      tracked=false;
      return;
  }
  vector<Point2f> points = points1;
  //Frame-to-frame tracking with forward-backward error cheking
  tracked = tracker.trackf2f(img1,img2,points,points2);
  if (tracked){
      //Bounding box prediction
	  //����ʣ�µ��ⲻ��һ��ĸ��ٵ�������Ԥ��bounding box�ڵ�ǰ֡��λ�úʹ�С tbb��
      bbPredict(points,points2,lastbox,tbb);
	  //���FB error����ֵ����10�����أ�����ֵ��������Ԥ�⵽�ĵ�ǰbox��λ���Ƴ�ͼ������Ϊ���ٴ��󣬴�ʱ������bounding box��
      if (tracker.getFB()>10 || tbb.x>img2.cols ||  tbb.y>img2.rows || tbb.br().x < 1 || tbb.br().y <1){
          tvalid =false; //too unstable prediction or bounding box out of image
          tracked = false;
          printf("Too unstable predictions FB error=%f\n",tracker.getFB());
          return;
      }
      //Estimate Confidence and Validity
      Mat pattern;
      Scalar mean, stdev;
      BoundingBox bb;
      bb.x = max(tbb.x,0);
      bb.y = max(tbb.y,0);
      bb.width = min(min(img2.cols-tbb.x,tbb.width),min(tbb.width,tbb.br().x));
      bb.height = min(min(img2.rows-tbb.y,tbb.height),min(tbb.height,tbb.br().y));
	  //��һ��img2(bb)��Ӧ��patch��size��������patch_size = 15*15��������pattern
      getPattern(img2(bb),pattern,mean,stdev);
      vector<int> isin;
      float dummy;
	  //����ͼ��Ƭpattern������ģ��M�ı������ƶ�
      classifier.NNConf(pattern,isin,dummy,tconf); //Conservative Similarity
      tvalid = lastvalid;
      if (tconf>classifier.thr_nn_valid){
          tvalid =true;
      }
  }
  else
    printf("No points tracked\n");

}

//����lastbox�о��Ȳ���10*10=100�������㣨����������㣩������points1��
void TLD::bbPoints(vector<cv::Point2f>& points,const BoundingBox& bb){
  int max_pts=10;
  int margin_h=0;
  int margin_v=0;
  int stepx = ceil((bb.width-2*margin_h)/max_pts);
  int stepy = ceil((bb.height-2*margin_v)/max_pts);
  for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy){
      for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx){
          points.push_back(Point2f(x,y));
      }
  }
}

////����ʣ�µ��ⲻ��һ��ĸ��ٵ�������Ԥ��bounding box�ڵ�ǰ֡��λ�úʹ�С tbb��
void TLD::bbPredict(const vector<cv::Point2f>& points1,const vector<cv::Point2f>& points2,
                    const BoundingBox& bb1,BoundingBox& bb2)    {
  int npoints = (int)points1.size();
  vector<float> xoff(npoints);
  vector<float> yoff(npoints);
  printf("tracked points : %d\n",npoints);
  for (int i=0;i<npoints;i++){
      xoff[i]=points2[i].x-points1[i].x;
      yoff[i]=points2[i].y-points1[i].y;
  }
  float dx = median(xoff);
  float dy = median(yoff);
  float s;
  if (npoints>1){
      vector<float> d;
      d.reserve(npoints*(npoints-1)/2);
      for (int i=0;i<npoints;i++){
          for (int j=i+1;j<npoints;j++){
              d.push_back(norm(points2[i]-points2[j])/norm(points1[i]-points1[j]));
          }
      }
      s = median(d);
  }
  else {
      s = 1.0;
  }
  float s1 = 0.5*(s-1)*bb1.width;
  float s2 = 0.5*(s-1)*bb1.height;
  printf("s= %f s1= %f s2= %f \n",s,s1,s2);
  bb2.x = round( bb1.x + dx -s1);
  bb2.y = round( bb1.y + dy -s2);
  bb2.width = round(bb1.width*s);
  bb2.height = round(bb1.height*s);
  printf("predicted bb: %d %d %d %d\n",bb2.x,bb2.y,bb2.br().x,bb2.br().y);
}

void TLD::detect(const cv::Mat& frame){
  //cleaning
  dbb.clear();
  dconf.clear();
  dt.bb.clear();
  double t = (double)getTickCount();
  Mat img(frame.rows,frame.cols,CV_8U);
  integral(frame,iisum,iisqsum);
  GaussianBlur(frame,img,Size(9,9),1.5);
  int numtrees = classifier.getNumStructs();
  float fern_th = classifier.getFernTh();
  vector <int> ferns(10);
  float conf;
  int a=0;
  Mat patch;
  for (int i=0;i<grid.size();i++){//FIXME: BottleNeck
	  //���û���ͼ����ÿ������ⴰ�ڵķ���������var��ֵ��Ŀ��patch�����50%���ģ�
	  //����Ϊ�京��ǰ��Ŀ�꣬ͨ����ģ��Ľ��뼯�Ϸ�����ģ�飺
      if (getVar(grid[i],iisum,iisqsum)>=var){
          a++;
		  patch = img(grid[i]);
          classifier.getFeatures(patch,grid[i].sidx,ferns);//�ȵõ���patch������ֵ
          conf = classifier.measure_forest(ferns);//�ټ��������ֵ��Ӧ�ĺ�������ۼ�ֵ��
          tmp.conf[i]=conf;
          tmp.patt[i]=ferns;
          if (conf>numtrees*fern_th){//�����Ϸ������ĺ�����ʵ�ƽ��ֵ������ֵfern_th����ѵ���õ���������Ϊ����ǰ��Ŀ�꣺
              dt.bb.push_back(i);//��ͨ�������������ģ���ɨ�贰�ڼ�¼��detect structure�У�
          }
      }
      else
        tmp.conf[i]=0.0;
  }
  int detections = dt.bb.size();
  printf("%d Bounding boxes passed the variance filter\n",a);
  printf("%d Initial detection from Fern Classifier\n",detections);
  if (detections>100){//���˳��ͨ�������������ģ���ɨ�贰��������100������ֻȡ������ʴ��ǰ100����
      nth_element(dt.bb.begin(),dt.bb.begin()+100,dt.bb.end(),CComparator(tmp.conf));
      dt.bb.resize(100);
      detections=100;
  }
//  for (int i=0;i<detections;i++){
//        drawBox(img,grid[dt.bb[i]]);
//    }
//  imshow("detections",img);
  if (detections==0){
        detected=false;
        return;
      }
  printf("Fern detector made %d detections ",detections);
  t=(double)getTickCount()-t;
  printf("in %gms\n", t*1000/getTickFrequency());
                                                                       //  Initialize detection structure
  dt.patt = vector<vector<int> >(detections,vector<int>(10,0));        //  Corresponding codes of the Ensemble Classifier
  dt.conf1 = vector<float>(detections);                                //  Relative Similarity (for final nearest neighbour classifier)
  dt.conf2 =vector<float>(detections);                                 //  Conservative Similarity (for integration with tracker)
  dt.isin = vector<vector<int> >(detections,vector<int>(3,-1));        //  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
  dt.patch = vector<Mat>(detections,Mat(patch_size,patch_size,CV_32F));//  Corresponding patches
  int idx;
  Scalar mean, stdev;
  float nn_th = classifier.getNNTh();
  for (int i=0;i<detections;i++){                                         //  for every remaining detection
      idx=dt.bb[i];                                                       //  Get the detected bounding box index
	  patch = frame(grid[idx]);
      getPattern(patch,dt.patch[i],mean,stdev);                //  Get pattern within bounding box
      classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dt.conf2[i]);  //  Evaluate nearest neighbour classifier
      dt.patt[i]=tmp.patt[idx];
      //printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
      if (dt.conf1[i]>nn_th){                                               //  idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
          dbb.push_back(grid[idx]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
          dconf.push_back(dt.conf2[i]);                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
      }
  }                                                                         //  end
  if (dbb.size()>0){
      printf("Found %d NN matches\n",(int)dbb.size());
      detected=true;
  }
  else{
      printf("No NN matches found.\n");
      detected=false;
  }
}

void TLD::evaluate(){
}

void TLD::learn(const Mat& img){
  printf("[Learning] ");
  ///Check consistency
  BoundingBox bb;
  bb.x = max(lastbox.x,0);
  bb.y = max(lastbox.y,0);
  bb.width = min(min(img.cols-lastbox.x,lastbox.width),min(lastbox.width,lastbox.br().x));
  bb.height = min(min(img.rows-lastbox.y,lastbox.height),min(lastbox.height,lastbox.br().y));
  Scalar mean, stdev;
  Mat pattern;
  getPattern(img(bb),pattern,mean,stdev);//��һ��img(bb)��Ӧ��patch��size��������patch_size = 15*15��������pattern
  vector<int> isin;
  float dummy, conf;
  classifier.NNConf(pattern,isin,conf,dummy);//��������ͼ��Ƭ����������Ŀ��box��������ģ��֮���������ƶ�conf
  if (conf<0.5) {//���ƶ�̫С��
      printf("Fast change..not training\n");
      lastvalid =false;
      return;
  }
  if (pow(stdev.val[0],2)<var){//����̫С��
      printf("Low variance..not training\n");
      lastvalid=false;
      return;
  }
  if(isin[2]==1){//����ʶ��Ϊ������
      printf("Patch in negative data..not traing");
      lastvalid=false;
      return;
  }
/// Data generation��������
  //���ȼ������е�ɨ�贰����Ŀǰ��Ŀ��box���ص��ȣ�
  for (int i=0;i<grid.size();i++){
      grid[i].overlap = bbOverlap(lastbox,grid[i]);
  }
  vector<pair<vector<int>,int> > fern_examples;
  good_boxes.clear();
  bad_boxes.clear();
  /*�ٸ��ݴ����lastbox������֡ͼ���е�ȫ��������Ѱ�����lastbox������С
  ���������ƣ��ص�����󣩵�num_closest_update�����ڣ�Ȼ�����Щ���ڹ���good_boxes����
  ��ֻ�ǰ�����������������룩ͬʱ�����ص���С��0.2�ģ����� bad_boxes ����*/
  getOverlappingBoxes(lastbox,num_closest_update);
  if (good_boxes.size()>0)
    generatePositiveData(img,num_warps_update);//Ȼ���÷���ģ�Ͳ����������������ڵ�һ֡�ķ�������ֻ����10*10=100����
  else{
    lastvalid = false;
    printf("No good boxes..Not training");
    return;
  }
  fern_examples.reserve(pX.size()+bad_boxes.size());
  fern_examples.assign(pX.begin(),pX.end());
  int idx;
  for (int i=0;i<bad_boxes.size();i++){
      idx=bad_boxes[i];
      if (tmp.conf[idx]>=1){//���븺���������ƶȴ���1???????
          
		  fern_examples.push_back(make_pair(tmp.patt[idx],0));
      }
  }
  //����ڷ�����������
  vector<Mat> nn_examples;
  nn_examples.reserve(dt.bb.size()+1);
  nn_examples.push_back(pEx);
  for (int i=0;i<dt.bb.size();i++){
      idx = dt.bb[i];
      if (bbOverlap(lastbox,grid[idx]) < bad_overlap)
        nn_examples.push_back(dt.patch[i]);
  }
  /// Classifiers update
  classifier.trainF(fern_examples,2);
  classifier.trainNN(nn_examples);
  classifier.show(); //���������⣨����ģ�ͣ�������������������ʾ�ڴ�����
}

void TLD::buildGrid(const cv::Mat& img, const cv::Rect& box){
  const float SHIFT = 0.1;//ɨ�贰�ڲ���Ϊ��ߵ� 10%
  const float SCALES[] = {0.16151,0.19381,0.23257,0.27908,0.33490,0.40188,0.48225,
                          0.57870,0.69444,0.83333,1,1.20000,1.44000,1.72800,
                          2.07360,2.48832,2.98598,3.58318,4.29982,5.15978,6.19174};//�߶�����ϵ��Ϊ1.2
  int width, height, min_bb_side;
  //Rect bbox;
  BoundingBox bbox;
  Size scale;
  int sc=0;
  for (int s=0;s<21;s++){
    width = round(box.width*SCALES[s]);
    height = round(box.height*SCALES[s]);
    min_bb_side = min(height,width);
    if (min_bb_side < min_win || width > img.cols || height > img.rows)
      continue;
    scale.width = width;
    scale.height = height;
    scales.push_back(scale);//ɨ�贰�ڴ�С
	//����ɨ�贰���������С
    for (int y=1;y<img.rows-height;y+=round(SHIFT*min_bb_side)){
      for (int x=1;x<img.cols-width;x+=round(SHIFT*min_bb_side)){
        bbox.x = x;
        bbox.y = y;
        bbox.width = width;
        bbox.height = height;
        bbox.overlap = bbOverlap(bbox,BoundingBox(box));//�ص���
        bbox.sidx = sc;//������
        grid.push_back(bbox);
      }
    }
    sc++;
  }
}

//������ÿһ��ɨ�贰���������Ŀ��box���ص���
//�ص��ȶ���Ϊ����box�Ľ��������ǵĲ����ı�
float TLD::bbOverlap(const BoundingBox& box1,const BoundingBox& box2){
  if (box1.x > box2.x+box2.width) { return 0.0; }
  if (box1.y > box2.y+box2.height) { return 0.0; }
  if (box1.x+box1.width < box2.x) { return 0.0; }
  if (box1.y+box1.height < box2.y) { return 0.0; }

  float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
  float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

  float intersection = colInt * rowInt;
  float area1 = box1.width*box1.height;
  float area2 = box2.width*box2.height;
  return intersection / (area1 + area2 - intersection);
}

//�˺������ݴ����box��Ŀ��߽�򣩣�����֡ͼ���е�ȫ��ɨ�贰����
//Ѱ�����box������С���������ƣ��ص�����󣩵�num_closest_init��10�������ڣ�
//Ȼ�����Щ���ڹ���good_boxes������ͬʱ�����ص���С��0.2�ģ�����bad_boxes������
//�൱�ڶ�ȫ����ɨ�贰�ڽ���ɸѡ����ͨ��BBhull�����õ���Щɨ�贰�ڵ����߽硣
void TLD::getOverlappingBoxes(const cv::Rect& box1,int num_closest){
  float max_overlap = 0;
  for (int i=0;i<grid.size();i++){
      if (grid[i].overlap > max_overlap) {
          max_overlap = grid[i].overlap;
          best_box = grid[i];//�ص������
      }
      if (grid[i].overlap > 0.6){
          good_boxes.push_back(i);//�ص��ȸߵ�
      }
      else if (grid[i].overlap < bad_overlap){
          bad_boxes.push_back(i);//�ص��ȵ͵�
      }
  }
  //Get the best num_closest (10) boxes and puts them in good_boxes
  //�õ��������ܳ���10��
  if (good_boxes.size()>num_closest){
	 //����ֻ�ǽ���nth���Ԫ���ź���λ�ã�
	  //STD�е�nth_element()�����ҳ�һ��������������n������Ϊ��num_closest�����Ǹ���������������к�
	  //��good_boxes[num_closest]ǰ��num_closest������������Ҳ�����ҵ���õ�num_closest��box��
    std::nth_element(good_boxes.begin(),good_boxes.begin()+num_closest,good_boxes.end(),OComparator(grid));//�˹��̵����˺���������Ϊν�ʺ���
	//����ѹ��good_boxesΪnum_closest��С
    good_boxes.resize(num_closest);
  }
  //��ȡgood_boxes �� Hull�ǣ�Ҳ���Ǵ��ڵı߿�
  getBBHull();
}

//��ͨ��BBhull�����õ���Щɨ�贰�ڵ����߽�
void TLD::getBBHull(){
  int x1=INT_MAX, x2=0;
  int y1=INT_MAX, y2=0;
  int idx;
  for (int i=0;i<good_boxes.size();i++){
      idx= good_boxes[i];
      x1=min(grid[idx].x,x1);
      y1=min(grid[idx].y,y1);
      x2=max(grid[idx].x+grid[idx].width,x2);
      y2=max(grid[idx].y+grid[idx].height,y2);
  }
  bbhull.x = x1;
  bbhull.y = y1;
  bbhull.width = x2-x1;
  bbhull.height = y2 -y1;
}

bool bbcomp(const BoundingBox& b1,const BoundingBox& b2){
  TLD t;
    if (t.bbOverlap(b1,b2)<0.5)
      return false;
    else
      return true;
}
int TLD::clusterBB(const vector<BoundingBox>& dbb,vector<int>& indexes){
  //FIXME: Conditional jump or move depends on uninitialised value(s)
  const int c = dbb.size();
  //1. Build proximity matrix
  Mat D(c,c,CV_32F);
  float d;
  for (int i=0;i<c;i++){
      for (int j=i+1;j<c;j++){
        d = 1-bbOverlap(dbb[i],dbb[j]);
        D.at<float>(i,j) = d;
        D.at<float>(j,i) = d;
      }
  }
  //2. Initialize disjoint clustering
  float *L = new float[c - 1]; //Level
  int (*nodes)[2] = new int[c - 1][2];
 int *belongs = new int[c];
 int m=c;
 for (int i=0;i<c;i++){
    belongs[i]=i;
 }
 for (int it=0;it<c-1;it++){
 //3. Find nearest neighbor
     float min_d = 1;
     int node_a, node_b;
     for (int i=0;i<D.rows;i++){
         for (int j=i+1;j<D.cols;j++){
             if (D.at<float>(i,j)<min_d && belongs[i]!=belongs[j]){
                 min_d = D.at<float>(i,j);
                 node_a = i;
                 node_b = j;
             }
         }
     }
     if (min_d>0.5){
         int max_idx =0;
         bool visited;
         for (int j=0;j<c;j++){
             visited = false;
             for(int i=0;i<2*c-1;i++){
                 if (belongs[j]==i){
                     indexes[j]=max_idx;
                     visited = true;
                 }
             }
             if (visited)
               max_idx++;
         }
         return max_idx;
     }

 //4. Merge clusters and assign level
     L[m]=min_d;
     nodes[it][0] = belongs[node_a];
     nodes[it][1] = belongs[node_b];
     for (int k=0;k<c;k++){
         if (belongs[k]==belongs[node_a] || belongs[k]==belongs[node_b])
           belongs[k]=m;
     }
     m++;
 }
 delete(L);
 delete(nodes);
 delete(belongs);
 return 1;

}

void TLD::clusterConf(const vector<BoundingBox>& dbb,const vector<float>& dconf,vector<BoundingBox>& cbb,vector<float>& cconf){
  int numbb =dbb.size();
  vector<int> T;
  float space_thr = 0.5;
  int c=1;
  switch (numbb){
  case 1:
    cbb=vector<BoundingBox>(1,dbb[0]);
    cconf=vector<float>(1,dconf[0]);
    return;
    break;
  case 2:
    T =vector<int>(2,0);
    if (1-bbOverlap(dbb[0],dbb[1])>space_thr){
      T[1]=1;
      c=2;
    }
    break;
  default:
    T = vector<int>(numbb,0);
    c = partition(dbb,T,(*bbcomp));
    //c = clusterBB(dbb,T);
    break;
  }
  cconf=vector<float>(c);
  cbb=vector<BoundingBox>(c);
  printf("Cluster indexes: ");
  BoundingBox bx;
  for (int i=0;i<c;i++){
      float cnf=0;
      int N=0,mx=0,my=0,mw=0,mh=0;
      for (int j=0;j<T.size();j++){
          if (T[j]==i){
              printf("%d ",i);
              cnf=cnf+dconf[j];
              mx=mx+dbb[j].x;
              my=my+dbb[j].y;
              mw=mw+dbb[j].width;
              mh=mh+dbb[j].height;
              N++;
          }
      }
      if (N>0){
          cconf[i]=cnf/N;
          bx.x=cvRound(mx/N);
          bx.y=cvRound(my/N);
          bx.width=cvRound(mw/N);
          bx.height=cvRound(mh/N);
          cbb[i]=bx;
      }
  }
  printf("\n");
}

