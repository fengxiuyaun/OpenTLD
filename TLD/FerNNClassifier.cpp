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

/*TLD的分类器有三部分：方差分类器模块、集合分类器模块和最近邻分类器模块；
这三个分类器是级联的，每一个扫描窗口依次全部通过上面三个分类器，才被认为含有前景目标。
这里prepare这个函数主要是初始化集合分类器模块；
集合分类器（随机森林）基于n个基本分类器（共10棵树），
每个分类器（树）都是基于一个pixel comparisons（共13个像素比较集）的，
也就是说每棵树有13个判断节点（组成一个pixel comparisons），
输入的图像片与每一个判断节点（相应像素点）进行比较，产生0或者1，
然后将这13个0或者1连成一个13位的二进制码x（有2^13种可能），
每一个x对应一个后验概率P(y|x)= #p/(#p+#n) （也有2^13种可能），
#p和#n分别是正和负图像片的数目。那么整一个集合分类器（共10个基本分类器）就有10个后验概率了，
将10个后验概率进行平均，如果大于阈值（一开始设经验值0.65，后面再训练优化）的话，
就认为该图像片含有前景目标；
prepare这个函数主要是初始化集合分类器模块*/
void FerNNClassifier::prepare(const vector<Size>& scales){
  acum = 0;
  //Initialize test locations for features
  //nstructs为树木（由一个特征组构建，每组特征代表图像块的不同视图表示）的个数
  //structSize为每棵树的特征个数，也即每棵树的判断节点个数；树上每一个特征都作为一个决策节点；
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
		  //是两个随机分配的像素点坐标（就是由这两个像素点比较得到0或者1的）
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
	  //pCounter[i][j]标识第i棵树第j个叶子节点曾经有多个样本被分正样本
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));//后验概率初始化时，每个后验概率都得初始化为0；
	  pCounter.push_back(vector<int>(pow(2.0, structSize), 0)); //正样本数目
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));//负样本数目
  }
}

void FerNNClassifier::getFeatures(const cv::Mat& image,const int& scale_idx, vector<int>& fern){
  int leaf;
  for (int t=0;t<nstructs;t++){
      leaf=0;
      for (int f=0; f<structSize; f++){
          leaf = (leaf << 1) + features[scale_idx][t*nstructs+f](image);//判断当前分支节点(x1,x2)>(x2,y2)?
      }
      fern[t]=leaf;
  }
}

//返回该样本所有树的所有特征值对应的后验概率累加值，
float FerNNClassifier::measure_forest(vector<int> fern) {
  float votes = 0;
  for (int i = 0; i < nstructs; i++) {
      votes += posteriors[i][fern[i]];//n棵树的概率相加
  }
  return votes;
}

//运行时候以下面方式更新：将已知类别标签的样本（训练样本）通过n个分类器进行分类，
//如果分类结果错误，那么相应的#p和#n就会更新，这样P(y | x)也相应更新了。
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

/*对每一个样本ferns_data[i] ，如果样本是正样本标签，
先用measure_forest函数返回该样本所有树的所有特征值对应的后验概率累加值，
该累加值如果小于正样本阈值（0.6* nstructs，这就表示平均值需要大于0.6（0.6* nstructs / nstructs）
,0.6是程序初始化时定的集合分类器的阈值，为经验值，后面会用测试集来评估修改，找到最优），
也就是输入的是正样本，却被分类成负样本了，出现了分类错误，所以就把该样本添加到正样本库，
同时用update函数更新后验概率。对于负样本，同样，如果出现负样本分类错误，就添加到负样本库。*/
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
		  /*如果样本是正样本标签，先用measure_forest函数返回该样本所有树的所有特征值对应的后验概率累加值，
		  该累加值如果小于正样本阈值（0.6* nstructs，这就表示平均值需要大于0.6（0.6* nstructs / nstructs）,
		  0.6是程序初始化时定的集合分类器的阈值，为经验值，后面会用测试集来评估修改，找到最优），
		  也就是输入的是正样本，却被分类成负样本了，出现了分类错误，所以就把该样本添加到正样本库，
		  同时用update函数更新后验概率。*/
          if(ferns[i].second==1){                           //       if (Y[I] == 1) {
              if(measure_forest(ferns[i].first)<=thrP)      //         if (measure_forest(x) <= thrP)
                update(ferns[i].first,1,1);                 //             update(x,1,1);
		 //对于负样本，同样，如果出现负样本分类错误，就添加到负样本库。
          }else{                                            //        }else{
              if (measure_forest(ferns[i].first) >= thrN)   //         if (measure_forest(x) >= thrN)
                update(ferns[i].first,0,1);                 //             update(x,0,1);
          }
      }
  //}
}

/*  对每一个样本nn_data，如果标签是正样本，通过NNConf(nn_examples[i], isin, conf, dummy);
计算输入图像片与在线模型之间的相关相似度conf，如果相关相似度小于0.65 ，
则认为其不含有前景目标，也就是分类错误了；这时候就把它加到正样本库。
然后就通过pEx.push_back(nn_examples[i]);将该样本添加到pEx正样本库中；
同样，如果出现负样本分类错误，就添加到负样本库。*/
void FerNNClassifier::trainNN(const vector<cv::Mat>& nn_examples){
  float conf,dummy;
  vector<int> y(nn_examples.size(),0);
  y[0]=1;//第一个为正样本
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

/*对集合分类器，对每一个测试集nXT，所有基本分类器的后验概率的平均值如果大于thr_fern（0.6），
则认为含有前景目标，然后取最大的平均值（大于thr_fern）作为该集合分类器的新的阈值。

   对最近邻分类器，对每一个测试集nExT，最大相关相似度如果大于nn_fern（0.65），
   则认为含有前景目标，然后取最大的最大相关相似度（大于nn_fern）作为该最近邻分类器的新的阈值。*/
void FerNNClassifier::evaluateTh(const vector<pair<vector<int>,int> >& nXT,const vector<cv::Mat>& nExT){
float fconf;
  for (int i=0;i<nXT.size();i++){
    fconf = (float) measure_forest(nXT[i].first)/nstructs;
    if (fconf>thr_fern)
      thr_fern=fconf;//更新后验概率阈值
}
  vector <int> isin;
  float conf,dummy;
  for (int i=0;i<nExT.size();i++){
      NNConf(nExT[i],isin,conf,dummy);
      if (conf>thr_nn)
        thr_nn=conf;//更新相关相似度
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
