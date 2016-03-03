/*
 * FerNNClassifier.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 */
#include "stdafx.h"
#include <FerNNClassifier.h>

using namespace cv;
using namespace std;

void FerNNClassifier::read(const FileNode& file){
  ///Classifier Parameters
  valid = (float)file["valid"];
  ncc_thesame = (float)file["ncc_thesame"];
  nstructs = (int)file["num_trees"];
  structSize = (int)file["num_features"];
  thr_fern = (float)file["thr_fern"];
  thr_nn = (float)file["thr_nn"];
  thr_nn_valid = (float)file["thr_nn_valid"];
}

/*TLD�ķ������������֣����������ģ�顢���Ϸ�����ģ�������ڷ�����ģ�飻
�������������Ǽ����ģ�ÿһ��ɨ�贰������ȫ��ͨ�������������������ű���Ϊ����ǰ��Ŀ�ꡣ
����prepare���������Ҫ�ǳ�ʼ�����Ϸ�����ģ�飻
���Ϸ����������ɭ�֣�����n����������������10��������
ÿ�����������������ǻ���һ��pixel comparisons����13�����رȽϼ����ģ�
Ҳ����˵ÿ������13���жϽڵ㣨���һ��pixel comparisons����
�����ͼ��Ƭ��ÿһ���жϽڵ㣨��Ӧ���ص㣩���бȽϣ�����0����1��
Ȼ����13��0����1����һ��13λ�Ķ�������x����2^13�ֿ��ܣ���
ÿһ��x��Ӧһ���������P(y|x)= #p/(#p+#n) ��Ҳ��2^13�ֿ��ܣ���
#p��#n�ֱ������͸�ͼ��Ƭ����Ŀ����ô��һ�����Ϸ���������10������������������10����������ˣ�
��10��������ʽ���ƽ�������������ֵ��һ��ʼ�辭��ֵ0.65��������ѵ���Ż����Ļ���
����Ϊ��ͼ��Ƭ����ǰ��Ŀ�ꣻ
prepare���������Ҫ�ǳ�ʼ�����Ϸ�����ģ��*/
void FerNNClassifier::prepare(const vector<Size>& scales){
  acum = 0;
  //Initialize test locations for features
  //nstructsΪ��ľ����һ�������鹹����ÿ����������ͼ���Ĳ�ͬ��ͼ��ʾ���ĸ���
  //structSizeΪÿ����������������Ҳ��ÿ�������жϽڵ����������ÿһ����������Ϊһ�����߽ڵ㣻
  int totalFeatures = nstructs*structSize;
  features = vector<vector<Feature> >(scales.size(),vector<Feature> (totalFeatures));//??????????
  RNG& rng = theRNG();
  float x1f,x2f,y1f,y2f;
  int x1, x2, y1, y2;
  for (int i=0;i<totalFeatures;i++){
      x1f = (float)rng;
      y1f = (float)rng;
      x2f = (float)rng;
      y2f = (float)rng;
      for (int s=0;s<scales.size();s++){
		  //�����������������ص����꣨���������������ص�Ƚϵõ�0����1�ģ�
          x1 = x1f * scales[s].width;
          y1 = y1f * scales[s].height;
          x2 = x2f * scales[s].width;
          y2 = y2f * scales[s].height;
          features[s][i] = Feature(x1, y1, x2, y2);
      }

  }
  //Thresholds
  thrN = 0.5*nstructs;

  //Initialize Posteriors
  for (int i = 0; i<nstructs; i++) {
	  //pCounter[i][j]��ʶ��i������j��Ҷ�ӽڵ������ж����������������
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));//������ʳ�ʼ��ʱ��ÿ��������ʶ��ó�ʼ��Ϊ0��
	  pCounter.push_back(vector<int>(pow(2.0, structSize), 0)); //��������Ŀ
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));//��������Ŀ
  }
}

void FerNNClassifier::getFeatures(const cv::Mat& image,const int& scale_idx, vector<int>& fern){
  int leaf;
  for (int t=0;t<nstructs;t++){
      leaf=0;
      for (int f=0; f<structSize; f++){
          leaf = (leaf << 1) + features[scale_idx][t*nstructs+f](image);//�жϵ�ǰ��֧�ڵ�(x1,x2)>(x2,y2)?
      }
      fern[t]=leaf;
  }
}

//���ظ���������������������ֵ��Ӧ�ĺ�������ۼ�ֵ��
float FerNNClassifier::measure_forest(vector<int> fern) {
  float votes = 0;
  for (int i = 0; i < nstructs; i++) {
      votes += posteriors[i][fern[i]];//n�����ĸ������
  }
  return votes;
}

//����ʱ�������淽ʽ���£�����֪����ǩ��������ѵ��������ͨ��n�����������з��࣬
//���������������ô��Ӧ��#p��#n�ͻ���£�����P(y | x)Ҳ��Ӧ�����ˡ�
void FerNNClassifier::update(const vector<int>& fern, int C, int N) {
  int idx;
  for (int i = 0; i < nstructs; i++) {
      idx = fern[i];
      (C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
      if (pCounter[i][idx]==0) {
          posteriors[i][idx] = 0;
      } else {
          posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
      }
  }
}

/*��ÿһ������ferns_data[i] �������������������ǩ��
����measure_forest�������ظ���������������������ֵ��Ӧ�ĺ�������ۼ�ֵ��
���ۼ�ֵ���С����������ֵ��0.6* nstructs����ͱ�ʾƽ��ֵ��Ҫ����0.6��0.6* nstructs / nstructs��
,0.6�ǳ����ʼ��ʱ���ļ��Ϸ���������ֵ��Ϊ����ֵ��������ò��Լ��������޸ģ��ҵ����ţ���
Ҳ�������������������ȴ������ɸ������ˣ������˷���������ԾͰѸ�������ӵ��������⣬
ͬʱ��update�������º�����ʡ����ڸ�������ͬ����������ָ�����������󣬾���ӵ��������⡣*/
void FerNNClassifier::trainF(const vector<std::pair<vector<int>,int> >& ferns,int resample){
  // Conf = function(2,X,Y,Margin,Bootstrap,Idx)
  //                 0 1 2 3      4         5
  //  double *X     = mxGetPr(prhs[1]); -> ferns[i].first
  //  int numX      = mxGetN(prhs[1]);  -> ferns.size()
  //  double *Y     = mxGetPr(prhs[2]); ->ferns[i].second
  //  double thrP   = *mxGetPr(prhs[3]) * nTREES; ->threshold*nstructs
  //  int bootstrap = (int) *mxGetPr(prhs[4]); ->resample
  thrP = thr_fern*nstructs;                                                          // int step = numX / 10;
  //for (int j = 0; j < resample; j++) {                      // for (int j = 0; j < bootstrap; j++) {
      for (int i = 0; i < ferns.size(); i++){               //   for (int i = 0; i < step; i++) {
                                                            //     for (int k = 0; k < 10; k++) {
                                                            //       int I = k*step + i;//box index
                                                            //       double *x = X+nTREES*I; //tree index
		  /*�����������������ǩ������measure_forest�������ظ���������������������ֵ��Ӧ�ĺ�������ۼ�ֵ��
		  ���ۼ�ֵ���С����������ֵ��0.6* nstructs����ͱ�ʾƽ��ֵ��Ҫ����0.6��0.6* nstructs / nstructs��,
		  0.6�ǳ����ʼ��ʱ���ļ��Ϸ���������ֵ��Ϊ����ֵ��������ò��Լ��������޸ģ��ҵ����ţ���
		  Ҳ�������������������ȴ������ɸ������ˣ������˷���������ԾͰѸ�������ӵ��������⣬
		  ͬʱ��update�������º�����ʡ�*/
          if(ferns[i].second==1){                           //       if (Y[I] == 1) {
              if(measure_forest(ferns[i].first)<=thrP)      //         if (measure_forest(x) <= thrP)
                update(ferns[i].first,1,1);                 //             update(x,1,1);
		 //���ڸ�������ͬ����������ָ�����������󣬾���ӵ��������⡣
          }else{                                            //        }else{
              if (measure_forest(ferns[i].first) >= thrN)   //         if (measure_forest(x) >= thrN)
                update(ferns[i].first,0,1);                 //             update(x,0,1);
          }
      }
  //}
}

/*  ��ÿһ������nn_data�������ǩ����������ͨ��NNConf(nn_examples[i], isin, conf, dummy);
��������ͼ��Ƭ������ģ��֮���������ƶ�conf�����������ƶ�С��0.65 ��
����Ϊ�䲻����ǰ��Ŀ�꣬Ҳ���Ƿ�������ˣ���ʱ��Ͱ����ӵ��������⡣
Ȼ���ͨ��pEx.push_back(nn_examples[i]);����������ӵ�pEx���������У�
ͬ����������ָ�����������󣬾���ӵ��������⡣*/
void FerNNClassifier::trainNN(const vector<cv::Mat>& nn_examples){
  float conf,dummy;
  vector<int> y(nn_examples.size(),0);
  y[0]=1;//��һ��Ϊ������
  vector<int> isin;
  for (int i=0;i<nn_examples.size();i++){                          //  For each example
      NNConf(nn_examples[i],isin,conf,dummy);                      //  Measure Relative similarity
      if (y[i]==1 && conf<=thr_nn){                                //    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65
          if (isin[1]<0){                                          //      if isnan(isin(2))
              pEx = vector<Mat>(1,nn_examples[i]);                 //        tld.pex = x(:,i);
              continue;                                            //        continue;
          }                                                        //      end
          //pEx.insert(pEx.begin()+isin[1],nn_examples[i]);        //      tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; % add to model
          pEx.push_back(nn_examples[i]);
      }                                                            //    end
      if(y[i]==0 && conf>0.5)                                      //  if y(i) == 0 && conf1 > 0.5
        nEx.push_back(nn_examples[i]);                             //    tld.nex = [tld.nex x(:,i)];

  }                                                                 //  end
  acum++;
  printf("%d. Trained NN examples: %d positive %d negative\n",acum,(int)pEx.size(),(int)nEx.size());
}                                                                  //  end


void FerNNClassifier::NNConf(const Mat& example, vector<int>& isin,float& rsconf,float& csconf){
  /*Inputs:
   * -NN Patch
   * Outputs:
   * -Relative Similarity (rsconf), Conservative Similarity (csconf), In pos. set|Id pos set|In neg. set (isin)
   */
  isin=vector<int>(3,-1);
  if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
      rsconf = 0; //    conf1 = zeros(1,size(x,2));
      csconf=0;
      return;
  }
  if (nEx.empty()){ //if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
      rsconf = 1;   //    conf1 = ones(1,size(x,2));
      csconf=1;
      return;
  }
  Mat ncc(1,1,CV_32F);
  float nccP,csmaxP,maxP=0;
  bool anyP=false;
  int maxPidx,validatedPart = ceil(pEx.size()*valid);
  float nccN, maxN=0;
  bool anyN=false;
  for (int i=0;i<pEx.size();i++){
      matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples
      nccP=(((float*)ncc.data)[0]+1)*0.5;
      if (nccP>ncc_thesame)
        anyP=true;
      if(nccP > maxP){
          maxP=nccP;
          maxPidx = i;
          if(i<validatedPart)
            csmaxP=maxP;
      }
  }
  for (int i=0;i<nEx.size();i++){
      matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);     //measure NCC to negative examples
      nccN=(((float*)ncc.data)[0]+1)*0.5;
      if (nccN>ncc_thesame)
        anyN=true;
      if(nccN > maxN)
        maxN=nccN;
  }
  //set isin
  if (anyP) isin[0]=1;  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  if (anyN) isin[2]=1;  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
  //Measure Relative Similarity
  float dN=1-maxN;
  float dP=1-maxP;
  rsconf = (float)dN/(dN+dP);
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);
}

/*�Լ��Ϸ���������ÿһ�����Լ�nXT�����л����������ĺ�����ʵ�ƽ��ֵ�������thr_fern��0.6����
����Ϊ����ǰ��Ŀ�꣬Ȼ��ȡ����ƽ��ֵ������thr_fern����Ϊ�ü��Ϸ��������µ���ֵ��

   ������ڷ���������ÿһ�����Լ�nExT�����������ƶ��������nn_fern��0.65����
   ����Ϊ����ǰ��Ŀ�꣬Ȼ��ȡ�������������ƶȣ�����nn_fern����Ϊ������ڷ��������µ���ֵ��*/
void FerNNClassifier::evaluateTh(const vector<pair<vector<int>,int> >& nXT,const vector<cv::Mat>& nExT){
float fconf;
  for (int i=0;i<nXT.size();i++){
    fconf = (float) measure_forest(nXT[i].first)/nstructs;
    if (fconf>thr_fern)
      thr_fern=fconf;//���º��������ֵ
}
  vector <int> isin;
  float conf,dummy;
  for (int i=0;i<nExT.size();i++){
      NNConf(nExT[i],isin,conf,dummy);
      if (conf>thr_nn)
        thr_nn=conf;//����������ƶ�
  }
  if (thr_nn>thr_nn_valid)
    thr_nn_valid = thr_nn;
}

void FerNNClassifier::show(){
  Mat examples((int)pEx.size()*pEx[0].rows,pEx[0].cols,CV_8U);
  double minval;
  Mat ex(pEx[0].rows,pEx[0].cols,pEx[0].type());
  for (int i=0;i<pEx.size();i++){
    minMaxLoc(pEx[i],&minval);
    pEx[i].copyTo(ex);
    ex = ex-minval;
    Mat tmp = examples.rowRange(Range(i*pEx[i].rows,(i+1)*pEx[i].rows));
    ex.convertTo(tmp,CV_8U);
  }
  imshow("Examples",examples);
}
