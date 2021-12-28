/*
Compile with  gcc -o graph_align_algo.out graph_align_algo.c
The program takes as command line arguments  ER_l (lambda, the average degree of the graphs),  flag_s_one (set to 1 if s==1, set to 0 otherwise), ER_s (s, correlation parameter between the two graphs), N (number of nodes in each graph), depth (maximum number of iterations), seed (random number generator's seed) 
Execute for example with ./graph_align_algo.out 2.4 0 0.9 256 20 1234
*/
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#define twom12 (1.0/4096.0)
#include <stdio.h>
#include <stdlib.h>

// Global Variables/////////////////////////////////////////////
int N;
double ER_l, ER_s;
int depth; 
int flag_s_one;
int _;
double **matrix_A, **matrix_B;
int *degree_A, *degree_B, *degree_AB;
int max_degree_A, max_degree_B;

int **neighborhood_A, **neighborhood_B, **neighborhood_A_back, **neighborhood_B_back, **neighborhood_AB;
double ***msg_iip_to_jjp, ***msg_iip_from_jjp;
double *msg_marginals;
double* msg_probabilities;
int *tab_subset_A, *tab_subset_A_list, *tab_subset_B, *tab_subset_B_list;
int *tab_permu, permu_aux_int, *tab_permu_aux;
int* random_permu;
double argmax_score_error;
int* argmax_score;//max and argmax along target nodes
double* max_score;

//////////////////////////////////////////////////////////////////
int iseed[4],jseed,mr[4],lr[4];
//////////////////////////////////////////////////////////////////

// Routines //////////////////////////////////////////////////////
void read_params(int argc, char *argv[]);
void allocate_memory();
void free_memory();
void init_AB_ER();
void init_from_AB();
void init_msgs_uninformative();
void update();
void update_s_one();
void calc_observables(double * msg_marginals);
void subset_init(int, int, int *, int *);
int subset_next(int, int *, int *);
void permu_init(int);
int permu_next(int);
void max_argmax(double* array,int array_len, double* max, int* argmax);
void uniform_permutation(int* permu, int permu_len);
double log_sum_exp(double aa, double bb);
double error_perm(int* perm, int N); 


// Routines ///////
void init_random();
double rannyu();
double squeeze(int i[], int j);
void setrn( int iseed[] );
int nrannyu( int max );
int compare_doubles (const void *a, const void *b);
//////////////////////////////////////////////////////////////////
int main(int argc, char **argv ){ 
        
    int i;
    read_params(argc,argv);
    
    init_random();
  
    allocate_memory();

    init_AB_ER();    
    
    init_from_AB();

    init_msgs_uninformative();

    i=0;
    while(i<depth){
        if(flag_s_one==1){
            update_s_one(); 	
        }
        
        else{
            	update();
        }
        
        calc_observables(msg_marginals);
        printf("depth=%d overlap=%.5f\n",i+1,1-argmax_score_error);
        i++; 
    }

    free_memory();
}

void read_params(int argc, char *argv[]){
  if (argc>6){
      ER_l=atof(argv[1]);
      flag_s_one=atoi(argv[2]); // 1 if ER_s=1.
      ER_s=atof(argv[3]);
      N=atoi(argv[4]);
      depth=atoi(argv[5]);
      jseed=atoi(argv[6]);
  }
  else{
      printf("# Usage: ER_l flag_s_one ER_s N depth seed \n");
      exit(0);
  }

}

void allocate_memory(){
  int i,ip, Ns;

  Ns=N*N;

  matrix_A=(double **)calloc(N,sizeof(double*));
  matrix_B=(double **)calloc(N,sizeof(double*));
  for(i=0;i<N;i++)
    {
      matrix_A[i]=(double*)calloc(N,sizeof(double));
      matrix_B[i]=(double*)calloc(N,sizeof(double));
   }

  degree_A=(int*)calloc(N,sizeof(int));
  degree_B=(int*)calloc(N,sizeof(int));

  msg_marginals=(double*)calloc(Ns,sizeof(double));

  //tab_degree_AtoB=calloc(N,sizeof(int));
  //tab_degree_BtoA=calloc(N,sizeof(int));
  
  argmax_score=(int*)calloc(N,sizeof(int));
  max_score=(double*)calloc(N,sizeof(double));
  
  random_permu=(int*)calloc(N,sizeof(int));

}

void free_memory()
{
  int i, ip, j, index, di, dip;

  for(i=0;i<N;i++)
    {
      free(matrix_A[i]);
      free(matrix_B[i]);

    }
 
  free(matrix_A);
  free(matrix_B);

  for(i=0;i<N;i++)
    for(ip=0;ip<N;ip++)
      {
	index=(i*N)+ip;
	di=degree_A[i];
	dip=degree_B[ip];

	if((di>0)&&(dip>0))
	  {
	    for(j=0;j<di;j++)
	      {
		free(msg_iip_to_jjp[index][j]);
		free(msg_iip_from_jjp[index][j]);
	      }
	    
	    free(msg_iip_to_jjp[index]);
	    free(msg_iip_from_jjp[index]);
	  }
      }

  free(msg_iip_to_jjp);
  free(msg_iip_from_jjp);

  for(i=0;i<N;i++)
    if(degree_A[i]>0)
      {
	free(neighborhood_A[i]);
	free(neighborhood_A_back[i]);
      }
  free(neighborhood_A);
  free(neighborhood_A_back);
  for(i=0;i<N;i++)
    if(degree_B[i]>0)
      {
	free(neighborhood_B[i]);
	free(neighborhood_B_back[i]);
      }
  free(neighborhood_B);
  free(neighborhood_B_back);
  
  



  free(degree_A);
  free(degree_B);

  free(msg_marginals);


  free(tab_subset_A);
  free(tab_subset_A_list);
  free(tab_subset_B);
  free(tab_subset_B_list);

  free(tab_permu);
  free(tab_permu_aux);
  
 
  
  free(argmax_score);
  free(max_score);
  
  free(random_permu);
}

void init_AB_ER()
{
  int i,j;
  double proba;
  
  proba=ER_l/(ER_s*((double) N));

  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      {
	matrix_A[i][j]=0.;
	matrix_B[i][j]=0.;
      }

  for(i=1;i<N;i++)
    for(j=0;j<i;j++)
      {
	if(rannyu()<proba)
	  {
	    if(rannyu()<ER_s)
	      {
		matrix_A[i][j]=1.;
		matrix_A[j][i]=1.;
	      }
	    if(rannyu()<ER_s)
	      {
		matrix_B[i][j]=1.;
		matrix_B[j][i]=1.;
	      }
	  }
      }

}

void init_from_AB()
{
  int i, j, ip, index, di, dip;

  // from the matrix A :

  for(i=0;i<N;i++)
    degree_A[i]=0;

  for(i=1;i<N;i++)
    for(j=0;j<i;j++)
      if(matrix_A[i][j]!=0.)
	{
	  degree_A[i]++;
	  degree_A[j]++;
	}

  neighborhood_A=(int**)calloc(N,sizeof(int*));
  neighborhood_A_back=(int**)calloc(N,sizeof(int*));
  
  max_degree_A=0;

  for(i=0;i<N;i++)
    if(degree_A[i]>0)
      {
	neighborhood_A[i]=(int*)calloc(degree_A[i],sizeof(int));
	neighborhood_A_back[i]=(int*)calloc(degree_A[i],sizeof(int));
	if(degree_A[i]>max_degree_A)
	  max_degree_A=degree_A[i];
      }

  for(i=0;i<N;i++)
    degree_A[i]=0;

  for(i=1;i<N;i++)
    for(j=0;j<i;j++)
      if(matrix_A[i][j]!=0.)
	{
	  neighborhood_A[i][degree_A[i]]=j;
	  neighborhood_A[j][degree_A[j]]=i;
	  neighborhood_A_back[i][degree_A[i]]=degree_A[j];
	  neighborhood_A_back[j][degree_A[j]]=degree_A[i];
	  degree_A[i]++;
	  degree_A[j]++;
	}

  // from the matrix B :

  for(i=0;i<N;i++)
    degree_B[i]=0;

  for(i=1;i<N;i++)
    for(j=0;j<i;j++)
      if(matrix_B[i][j]!=0.)
	{
	  degree_B[i]++;
	  degree_B[j]++;
	}

  neighborhood_B=(int**)calloc(N,sizeof(int*));
  neighborhood_B_back=(int**)calloc(N,sizeof(int*));

  max_degree_B=0;

  for(i=0;i<N;i++)
    if(degree_B[i]>0)
      {
	neighborhood_B[i]=(int*)calloc(degree_B[i],sizeof(int));
	neighborhood_B_back[i]=(int*)calloc(degree_B[i],sizeof(int));
	if(degree_B[i]>max_degree_B)
	  max_degree_B=degree_B[i];
      }

  for(i=0;i<N;i++)
    degree_B[i]=0;

  for(i=1;i<N;i++)
    for(j=0;j<i;j++)
      if(matrix_B[i][j]!=0.)
	{
	  neighborhood_B[i][degree_B[i]]=j;
	  neighborhood_B[j][degree_B[j]]=i;
	  neighborhood_B_back[i][degree_B[i]]=degree_B[j];
	  neighborhood_B_back[j][degree_B[j]]=degree_B[i];
	  degree_B[i]++;
	  degree_B[j]++;
	}

		
  // memory allocation of the structures that depend on the degrees

  msg_iip_to_jjp=(double***)calloc((N*N),sizeof(double**));
  msg_iip_from_jjp=(double***)calloc((N*N),sizeof(double**));

  for(i=0;i<N;i++)
    for(ip=0;ip<N;ip++)
      {
	index=(i*N)+ip;
	di=degree_A[i];
	dip=degree_B[ip];

	if((di>0)&&(dip>0))
	  {
	    msg_iip_to_jjp[index]=(double**)calloc(di,sizeof(double*));
	    msg_iip_from_jjp[index]=(double**)calloc(di,sizeof(double*));
	    for(j=0;j<di;j++)
	      {
		msg_iip_to_jjp[index][j]=(double*)calloc(dip,sizeof(double));
		msg_iip_from_jjp[index][j]=(double*)calloc(dip,sizeof(double));
	      }
	  }
      }

  tab_subset_A=(int*)calloc(max_degree_A,sizeof(int));
  tab_subset_A_list=(int*)calloc(max_degree_A,sizeof(int));
  tab_subset_B=(int*)calloc(max_degree_B,sizeof(int));
  tab_subset_B_list=(int*)calloc(max_degree_B,sizeof(int));

  if(max_degree_A<max_degree_B)
    {
      tab_permu=(int*)calloc(max_degree_A,sizeof(int));
      tab_permu_aux=(int*)calloc(max_degree_A,sizeof(int));
    }
  else
    {
      tab_permu=(int*)calloc(max_degree_B,sizeof(int));
      tab_permu_aux=(int*)calloc(max_degree_B,sizeof(int));
    }
}


////////////////////////////////////////
void init_msgs_uninformative()
{
  int i, ip, index, di, dip, j, jp;

  for(i=0;i<N;i++)
    for(ip=0;ip<N;ip++)
      {
	index=(i*N)+ip;
	di=degree_A[i];
	dip=degree_B[ip];
	for(j=0;j<di;j++)
	  for(jp=0;jp<dip;jp++)
	    msg_iip_from_jjp[index][j][jp]=0.; //initialization corresponding to matching only the root of the tree.
      }

}


void update() // for the case ER_s < 1.
{
  int i, ip, index, di, dip, dmin, l, j, jp, m;
  int true_j, true_jp, index_jjp, back, backp;
  int done_A, done_B, done_permu;
  double sum_permu, prod_permu;
  

  // computation of the msg_marginals and msg_iip_to_jjp as a function of the msg_iip_from_jjp

  for(i=0;i<N;i++)
    for(ip=0;ip<N;ip++)
      {
	index=(i*N)+ip;
	di=degree_A[i];
	dip=degree_B[ip];
	
	if(di<dip)
	  dmin=di;
	else
	  dmin=dip;
	
	// contribution of l=0 :

	msg_marginals[index]=0;
	for(j=0;j<di;j++)
	  for(jp=0;jp<dip;jp++)
	    msg_iip_to_jjp[index][j][jp]=0;
    
	for(l=1;l<(dmin+1);l++)
	  {
	    done_A=0;
	    subset_init(di,l,tab_subset_A,tab_subset_A_list);
	    while(done_A==0) // sum over the subsets of A-neighbors containing l elements
	      {
		done_B=0;
		subset_init(dip,l,tab_subset_B,tab_subset_B_list);
		while(done_B==0) // sum over the subsets of B-neighbors containing l elements
		  {
		    sum_permu=-DBL_MAX;
		    done_permu=0;
		    permu_init(l);
		    while(done_permu==0) // sum over the permutations of l elements 
		      {
			prod_permu=l*log(ER_s/(ER_l*(1-ER_s)*(1-ER_s)));//1.;
			for(m=0;m<l;m++)
			  prod_permu+=msg_iip_from_jjp[index][tab_subset_A_list[m]][tab_subset_B_list[tab_permu[m]]];

			sum_permu=log_sum_exp(sum_permu,prod_permu);
			done_permu=permu_next(l);
		      }
            msg_marginals[index]=log_sum_exp(msg_marginals[index],sum_permu);
		    
		    for(j=0;j<di;j++)
		      for(jp=0;jp<dip;jp++)
			if((tab_subset_A[j]==0)&&(tab_subset_B[jp]==0))
			  msg_iip_to_jjp[index][j][jp]=log_sum_exp(msg_iip_to_jjp[index][j][jp],sum_permu);
		    
		    done_B=subset_next(dip,tab_subset_B,tab_subset_B_list);
		  }
		done_A=subset_next(di,tab_subset_A,tab_subset_A_list);
		
	      }
	  }
	  msg_marginals[index]+=ER_l*ER_s+(di+dip)*log(1-ER_s);
	  
	  for(j=0;j<di;j++)
	  for(jp=0;jp<dip;jp++)
	    msg_iip_to_jjp[index][j][jp]+=ER_l*ER_s+(di+dip-2)*log(1-ER_s);
      }

  // copy of the messages in preparation of the next iteration

  for(i=0;i<N;i++)
    for(ip=0;ip<N;ip++)
      {
	index=(i*N)+ip;
	di=degree_A[i];
	dip=degree_B[ip];

	for(j=0;j<di;j++)
	  for(jp=0;jp<dip;jp++)
	    {
	      true_j=neighborhood_A[i][j];
	      true_jp=neighborhood_B[ip][jp];

	      index_jjp=(true_j*N)+true_jp;
	      back=neighborhood_A_back[i][j];
	      backp=neighborhood_B_back[ip][jp];

	      msg_iip_from_jjp[index][j][jp]=msg_iip_to_jjp[index_jjp][back][backp];
	    }
      }

}
double log_sum_exp(double aa,double bb){
    double result;
    
    if(aa>bb){
        result=aa+log1p(exp(bb-aa));
        
    }
    
    else{
        result=bb+log1p(exp(aa-bb));
    }
    return result;

}


void update_s_one() // for the case ER_s=1
{
  int i, ip, index, di, dip, j, jp, m;
  int true_j, true_jp, index_jjp, back, backp;
  int done_permu;
  double sum_permu, prod_permu;
  

  // computation of the msg_marginals and msg_iip_to_jjp as a function of the msg_iip_from_jjp

  for(i=0;i<N;i++)
    for(ip=0;ip<N;ip++)
      {
	index=(i*N)+ip;
	di=degree_A[i];
	dip=degree_B[ip];
	
	if(di!=dip)
	  {
	    msg_marginals[index]=-DBL_MAX;
	    for(j=0;j<di;j++)
	      for(jp=0;jp<dip;jp++)
		msg_iip_to_jjp[index][j][jp]=-DBL_MAX;
	  }
	else
	  {
	    if(di==0)
	      msg_marginals[index]=ER_l;
	    else
	      {
		sum_permu=-DBL_MAX;
		done_permu=0;
		permu_init(di);
		while(done_permu==0) 
		  {
		    prod_permu=0;
		    for(m=0;m<di;m++)
              prod_permu+=msg_iip_from_jjp[index][m][tab_permu[m]];
		    
		    sum_permu=log_sum_exp(sum_permu,prod_permu);
		    done_permu=permu_next(di);
		  }
		msg_marginals[index]=ER_l-di*log(ER_l)+sum_permu;

	      }

	    if(di==1)
	      msg_iip_to_jjp[index][0][0]=ER_l;
	    else if(di>1)
	      {
		for(j=0;j<di;j++)
		  {
		    // the first di-1 elements of tab_subset_A_list contains all indices except j
		    for(m=0;m<(di-1);m++)
		      tab_subset_A_list[m]=m;
		    tab_subset_A_list[j]=di-1;

		    for(jp=0;jp<di;jp++)
		      {
		    // the first di-1 elements of tab_subset_B_list contains all indices except jp
			for(m=0;m<(di-1);m++)
			  tab_subset_B_list[m]=m;
			tab_subset_B_list[jp]=di-1;

			sum_permu=-DBL_MAX;
			done_permu=0;
			permu_init(di-1);
			while(done_permu==0) 
			  {
			    prod_permu=0.;
			    for(m=0;m<(di-1);m++)
			      prod_permu+=msg_iip_from_jjp[index][tab_subset_A_list[m]][tab_subset_B_list[tab_permu[m]]];
				
			    sum_permu=log_sum_exp(sum_permu,prod_permu);
			    done_permu=permu_next(di-1);
			  }
			msg_iip_to_jjp[index][j][jp]=ER_l+(1-di)*log(ER_l)+sum_permu;
		      }
		  }
	      }
	    
	  }
      }

  // copy of the messages in preparation of the next iteration

  for(i=0;i<N;i++)
    for(ip=0;ip<N;ip++)
      {
	index=(i*N)+ip;
	di=degree_A[i];
	dip=degree_B[ip];

	for(j=0;j<di;j++)
	  for(jp=0;jp<dip;jp++)
	    {
	      true_j=neighborhood_A[i][j];
	      true_jp=neighborhood_B[ip][jp];

	      index_jjp=(true_j*N)+true_jp;
	      back=neighborhood_A_back[i][j];
	      backp=neighborhood_B_back[ip][jp];

	      msg_iip_from_jjp[index][j][jp]=msg_iip_to_jjp[index_jjp][back][backp];
	    }
      }

}


void calc_observables(double * msg_marginals){

    int i;
    
    //argmax score approach
    for(i=0;i<N;i++){
        max_argmax((msg_marginals+N*i),N,& max_score[i],& argmax_score[i]);
    }
    argmax_score_error=error_perm(argmax_score,N);
    
}

// initialize for a subset of nb elements among length ones

void subset_init(int length, int nb, int *tab, int *tab_list){
  int i;

  for(i=0;i<nb;i++)
    tab[i]=1;
  for(i=nb;i<length;i++)
    tab[i]=0;

  for(i=0;i<nb;i++)
    tab_list[i]=i;
}

// returns -1 if the input is the last subset

int subset_next(int length, int *tab, int *tab_list) 
{
  int i, found, pos, res, nb_lastones, pos_zero, nb;

  if(length>1)
    {
      if(tab[length-1]==0)
	{
	  found=0;
	  for(i=length-2;(i>-1)&&(found==0);i--)
	    if(tab[i]==1)
	      {
		found=1;
		pos=i;
	      }
	  tab[pos]=0;
	  tab[pos+1]=1;
	  res=0; 
	}
      else
	{
	  nb_lastones=1;
	  found=0;
	  for(i=length-2;(i>-1)&&(found==0);i--)
	    if(tab[i]==0)
	      {
		found=1;
		pos_zero=i;
	      }
	    else
	      nb_lastones++;
	  if((found==1)&&(pos_zero>0))
	    {
	      found=0;
	      for(i=pos_zero-1;(i>-1)&&(found==0);i--)
		if(tab[i]==1)
		  {
		    found=1;
		    pos=i;
		  }
	      if(found==1)
		{
		  tab[pos]=0;
		  for(i=0;i<(nb_lastones+1);i++)
		    tab[pos+i+1]=1;
		  for(i=pos+nb_lastones+2;i<length;i++)
		    tab[i]=0;
		  res=0;
		}
	      else
		res=-1;
	    }
	  else
	    res=-1;

	}
    }
  else
    res=-1;

  // converts the information into a list of the elements in the subset

  if(res==0)
    {
      nb=0;
      for(i=0;i<length;i++)
	if(tab[i]==1)
	  {
	    tab_list[nb]=i;
	    nb++;
	  }
    }

  return res;
}

void permu_init(int length)
{
  int i;

  for(i=0;i<length;i++)
    {
      tab_permu[i]=i;
      tab_permu_aux[i]=0;
    }
  permu_aux_int=0;
}

int permu_next(int length) // returns -1 if the input is the last permutation
{
  int swapper;

  if(permu_aux_int>=length)
    return -1;

  while(permu_aux_int<length)
    {
      if(tab_permu_aux[permu_aux_int]<permu_aux_int)
	{
	  if((permu_aux_int%2)==0)
	    {
	      swapper=tab_permu[0];
	      tab_permu[0]=tab_permu[permu_aux_int];
	      tab_permu[permu_aux_int]=swapper;
	    }
	  else
	    {
	      swapper=tab_permu[tab_permu_aux[permu_aux_int]];
	      tab_permu[tab_permu_aux[permu_aux_int]]=tab_permu[permu_aux_int];
	      tab_permu[permu_aux_int]=swapper;
	    }
	  tab_permu_aux[permu_aux_int]++;
	  permu_aux_int=0;
	  return 0;
	}
      else
	{
	  tab_permu_aux[permu_aux_int]=0;
	  permu_aux_int++;
	}
    }
    
  return -1;
}


void uniform_permutation(int* permu, int permu_len){
//implements the Fisher-Yates algorithm
    int i;
    //actually there's no need to reinitialize it with the uniform permutation
    for(i=0;i<permu_len;i++){
        permu[i]=i;
    }
    
    
    int rand;
    int tmp;
    double r;

    for(i=permu_len-1;i>0;i--){
        r=rannyu();
        rand=(int)( (i+1)*r);//uniform in 0,...,i
        //printf("%f %d %d\n",r,i,rand);
        if(rand==i+1){
            rand=i; 
        }
        tmp=permu[i];
        permu[i]=permu[rand];
        permu[rand]=tmp;
    }
}

void max_argmax(double* array,int array_len, double* max, int* argmax){

  uniform_permutation(random_permu,N);
  
  int temp_argmax=random_permu[0];
  double temp_max=array[random_permu[0]];
  
  for(int i=1; i<array_len;i++)
    {
    if(*(array+random_permu[i])>temp_max)
      {
        temp_max=array[random_permu[i]];
        temp_argmax=random_permu[i];
    
      }
    
    }
    *max =temp_max;
    *argmax=temp_argmax;
}


double error_perm(int* perm, int N){
  double error=0;
  for(int i=0; i<N; i++){
    if(perm[i]!=i){
      error+=1;
      
    }
  }
  error/=N;
  return error;
}

/*******************/
/* Routines */
/*******************/

void init_random()
{
  int i;
  srand(jseed);
  for (i=0;i<4;i++)
    iseed[i] = rand() % 4096;
  setrn(iseed);
}

/***********************************************/

double rannyu()
{
  int i,j;
  
  for (i=0;i<4;i++)
    {
      lr[i] = lr[i]*mr[3];
      for (j=i+1;j<4;j++)
	lr[i] += lr[j]*mr[i+3-j];
    }
  lr[3]++;
  for (i=3;i>0;i--)
    {
      lr[i-1] += lr[i]/4096;
      lr[i] %= 4096;
    }
  lr[0] %= 4096;
  
  return squeeze(lr,0);
}

/***********************************************/

double squeeze( int i[], int j )
{
  if (j>3)
    return 0.0;
  else
    return (twom12*((double)i[j]+squeeze(i,j+1)));
}   
      
/***********************************************/

void setrn( int iseed[] )
{
  int i;

  for (i=0;i<4;i++)
    lr[i] = iseed[i];
  mr[0] = 0;
  mr[1] = 1;
  mr[2] = 3513;
  mr[3] = 821;
}

/*************************************************/
/* Return an integer in [0,max-1] */

int nrannyu( int max )
{
  return (int)(rannyu()*(double)max);
}

int compare_doubles (const void *a, const void *b)
{
  const double *da = (const double *) a;
  const double *db = (const double *) b;

  return (*da < *db) - (*da > *db);
}
