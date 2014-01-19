#include <matrix.h>
#include <mex.h>   
#include <math.h>

#define MAX_ITER 1000

using namespace std;

// Get the best weak classifier
int call_learner(double* hyp_weakclassifiers, const double* labels, double* Wi, const size_t size_classifiers, const size_t size_samples)
{
    int    best_classifier   = 0;
    double min_error         = size_samples;// initialize the minimum error
    for(unsigned int i=0; i< size_classifiers; i++)
    {
        double error = 0;
        for(unsigned int j=0; j< size_samples; j++)
        {
            if(hyp_weakclassifiers[i*size_samples+j] != labels[j])
               error += Wi[j];
        }
        if(error < min_error)
        {
            min_error = error;
            best_classifier = i;
        }
    }
    return best_classifier;
}

// AdaBoost
// Input: 
//        T                    - maximum iteration times
//        size_samples         - number of samples
//        size_classifiers     - number of classifiers
//        labels               - Nx1 vector, positive = 1, negative = -1
//        hyp_weakclassifiers  - hypothesis of weak classifiers. Colum vector of a NxK matrix where N is the number of samples, K is the number of weak classifiers.
// return:
//        strong classifier H, including two fields
//        'alpha': alpha of each weak classifier
//        'best_t': best weak classifier in each iteration
// Yoav Freund and Robert E. Schapire, “A decision-
// theoretic generalization of on-line learning and an ap-
// plication to boosting,” in Proceedings of the Second Eu-
// ropean Conference on Computational Learning Theory,
// 1995, pp. 23–37.

void AdaBoost(const int T, const size_t size_samples, const size_t size_classifiers, const double* labels, double* hyps, double* alpha, double* best_t)
{
    // weight of each sample
    double *Wi = (double *)mxCalloc(size_samples, sizeof(double));// weight of each sample
    // Initialize the weight
    for(unsigned int i=0; i< size_samples; i++)
        Wi[i] = 1.0f/size_samples;
    
    // Loop
    for(unsigned int t=0; t<T; t++)
    {
        // From the family of weak classifiers,
        // find the best weak classifer with minimum error
        int    bestclassifier   = 0;
        double min_error         = size_samples;// initialize the minimum error
        for(unsigned int i=0; i< size_classifiers; i++)
        {
            double error = 0;
            for(unsigned int j=0; j< size_samples; j++)
            {
                if(hyps[i*size_samples+j] != labels[j])
                    error += Wi[j];
            }
            if(error < min_error)
            {
                min_error = error;
                bestclassifier = i;
            }
        }
        
        // Stop condition
        if(min_error > 0.49999) // worse than random guess
            break;
        
        // Compute alpha
        alpha[t]  = 0.5*log((1.0f-min_error)/min_error);
        best_t[t] = bestclassifier + 1;
        
        // Update and normalize the weight of each sample
        for(unsigned int i=0; i< size_samples; i++)
        {
            bool predict = (labels[i]==hyps[bestclassifier*size_samples+i]);
            Wi[i] = Wi[i]*exp(2*alpha[t]*(!predict));
        }
        
        double sum_wi = 0;
        for(unsigned int i=0; i< size_samples; i++)
            sum_wi += Wi[i];
        
        for(unsigned int i=0; i< size_samples; i++)
            Wi[i] = Wi[i]/sum_wi;
        
        mexPrintf("\nIter: %d, error = %f, best classifier = %d alpha = %f", t, min_error, bestclassifier, alpha[t]);

    }
}

// Boosting for transfer learning
// Input: 
//        T                    - maximum iteration times
//        size_samples         - number of samples
//        size_classifiers     - number of classifiers
//        labels               - Nx1 vector, positive = 1, negative = -1
//        domains              - Nx1 vector, auxiliary = 0, target = 1
//        hyp_weakclassifiers  - hypothesis of weak classifiers. Colum vector of a NxK matrix where N is the number of samples, K is the number of weak classifiers.
//        Wi                   - Initial weight of the samples
// return:
//        strong classifier H, including two fields
//        'alpha': alpha of each weak classifier
//        'best_t': best weak classifier in each iteration
// Following ICML 2007 "Boosting for Transfer Learning"
void TrAdaBoost(const int T, const size_t size_samples, const size_t size_classifiers, const double* labels, const double* domains, double* hyps, double* Wi, double* alpha,  double* best_t, const bool dynamic_cost = false)
{
    int num_aux = 0;
    // Initialize the weight
    for(unsigned int i=0; i< size_samples; i++)
        if(domains[i]==0)
            num_aux++;

    // Loop
    for(unsigned int t=0; t<T; t++)
    {
        // Step 1: Compute probabilities of each sample
        double sum_wi = 0;
        for(unsigned int i=0; i< size_samples; i++)
            sum_wi += Wi[i];
        
        for(unsigned int i=0; i< size_samples; i++)
            Wi[i] = Wi[i]/sum_wi;
        
        // Step 2: Find the best weak classifier with minimum error
        int bestclassifier = call_learner(hyps, labels, Wi, size_classifiers, size_samples);
        
        // Step 3: Compute the error on target domain samples
        double error_tar = 0;
        for(unsigned int i=0; i< size_samples; i++)
        {
            if(domains[i]==1)// target domain
            {
                if(hyps[bestclassifier*size_samples+i]!=labels[i])
                    error_tar += Wi[i];
            }
        }

        // Stop condition
        if(error_tar > 0.49999) // worse than random guess
            break;
        
        // Step 4: Compute alpha
        alpha[t]            = 0.5*log((1.0f-error_tar)/error_tar);
        double Ct           = 1;
        if(dynamic_cost)
            Ct = 2*(1 - error_tar);// dynamic cost
        
        double alpha_aux    = Ct*0.5*log(1 + sqrt(2*log(num_aux/T)));
        best_t[t] = bestclassifier + 1;// add 1 for matlab index

        // Step 5: update the weight of each sample
        for(unsigned int i=0; i< size_samples; i++)
        {
            bool predict = (labels[i]==hyps[bestclassifier*size_samples+i]);
            if(domains[i]==0)// update auxiliary weight
                Wi[i] = Wi[i]*exp(-alpha_aux*(!predict));
            else// target domain
                Wi[i] = Wi[i]*exp(alpha[t]*(!predict));
        }
        mexPrintf("\nIter: %d, error = %f, best classifier = %d alpha = %f", t, error_tar, bestclassifier, alpha[t]);
	}
}

// Boosting for transfer learning using dynamic updating
// Input: 
//        T                    - maximum iteration times
//        size_samples         - number of samples
//        size_classifiers     - number of classifiers
//        labels               - Nx1 vector, positive = 1, negative = -1
//        domains              - Nx1 vector, auxiliary = 0, target = 1
//        hyp_weakclassifiers  - hypothesis of weak classifiers. Colum vector of a NxK matrix where N is the number of samples, K is the number of weak classifiers.
// return:
//        strong classifier H, including two fields
//        'alpha': alpha of each weak classifier
//        'best_t': best weak classifier in each iteration
// Reference: ECMPKDD'11: "Adaptive Boosting for Transfer Learning using Dynamic Updates"
void DTrAdaBoost(const int T, const size_t size_samples, const size_t size_classifiers, const double* labels, const double* domains, double* hyps, double* Wi, double* alpha, double* best_t)
{
    TrAdaBoost(T, size_samples, size_classifiers, labels, domains, hyps, Wi, alpha, best_t, true);
}

// matlab entry point
// 	H = mex_boosting(param, labels, domains, hyp_weakclassifiers);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  if (nrhs != 5)
    mexErrMsgTxt("Wrong number of inputs"); 
  //if (nlhs != 2)
  //  mexErrMsgTxt("Wrong number of outputs");
  
  // Parse the parameter 
  double  *alg;
  mxArray *tmp;
  tmp = mxGetField(prhs[0],0,"NAME_ALGORITHM");
  if(tmp)
  	alg = mxGetPr(tmp);
  
  // Input variables
  // Maximum iteration times
  double* max_iter;
  tmp = mxGetField(prhs[0],0,"MAX_ITERATION");
  if(tmp)
      max_iter = mxGetPr(tmp);
  else
      *max_iter = MAX_ITER;
  const int T = (int)*max_iter;
  
  // Labels
  double *data_labels  = mxGetPr(prhs[1]);
  // Domains
  double *data_domains = mxGetPr(prhs[2]);
  // Hypothesis of weak classifiers
  double *data_hyps                 = mxGetPr(prhs[3]);
  // Initial weights
  double *data_wi                   = mxGetPr(prhs[4]);
  const int *dims                   = mxGetDimensions(prhs[3]);
  size_t  size_samples              = dims[0];//number of samples
  size_t  size_classifiers          = dims[1];//number of weak classifiers
  
  // Return variable: alpha
  plhs[0] = mxCreateDoubleMatrix(1, T, mxREAL);
  double *alpha = mxGetPr(plhs[0]);
  // Return variable: best classifiers in each iteration
  plhs[1] = mxCreateDoubleMatrix(1, T, mxREAL);
  double *best_t = mxGetPr(plhs[1]);

  // run the algorithm
  switch((int)*alg)
  {
      case 0:
          mexPrintf("\nRun Algorithm = %d, AdaBoost",(int)*alg);
          mexPrintf("\nMax_Iter = %d, size_samples = %d, size_classifiers = %d",T, size_samples, size_classifiers);
          AdaBoost(T, size_samples, size_classifiers, data_labels, data_hyps, alpha, best_t);
          break;
      case 1:          
          mexPrintf("\nRun Algorithm = %d, TrAdaBoost",(int)*alg);
          mexPrintf("\nMax_Iter = %d, size_samples = %d, size_classifiers = %d",T, size_samples, size_classifiers);
          TrAdaBoost(T, size_samples, size_classifiers, data_labels, data_domains, data_hyps, data_wi, alpha, best_t);
          break;
      case 2:
          mexPrintf("\nRun Algorithm = %d, D-TrAdaBoost",(int)*alg);
          mexPrintf("\nMax_Iter = %d, size_samples = %d, size_classifiers = %d",T, size_samples, size_classifiers);
          DTrAdaBoost(T, size_samples, size_classifiers, data_labels, data_domains, data_hyps, data_wi, alpha, best_t);
          break;
      default:
          mexPrintf("\nAlgorithm %d does not exist!",*alg);
          break;
  }
}