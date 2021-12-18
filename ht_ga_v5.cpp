//TODO 
//avg score correct and aligned vertices
//avg score neighbors of correct and aligned vertices
//output roc curve
//topological distance of errors (beware of nodes with high degree)

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#define twom12 (1.0/4096.0)
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

using namespace std;
#include <queue>
#include <vector>
#include <map>
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>
#include <complex>
#include <algorithm>
#include "lemon/list_graph.h"
#include "lemon/smart_graph.h"
#include "lemon/concepts/graph.h"
#include "lemon/tolerance.h"
#include "lemon/elevator.h"
#include "lemon/matching.h"
#include "lemon/dfs.h"
#include "lemon/core.h"
// #define N_QUANTILES 5

using namespace lemon;
using namespace std;

typedef ListGraph Graph;
typedef Graph::IncEdgeIt IncEdgeIt;
typedef MaxWeightedPerfectMatching<Graph, Graph::EdgeMap<double> > MWPM;
typedef Graph::EdgeIt EdgeIt;
typedef Graph::NodeIt NodeIt;

// Variables globales /////////////////////////////////////////////
int N;
double ER_l, ER_s;
int write_on_file; //if 1 results are written on a file
int verbose; //if 1 it prints a buch of stuff
int depth; //depth of the trees considered (to be chosen approximately of the order of log(N))
int flag_s_one;
int max_tree_depth_results;
int run_until_max_tree_depth;
int overflow_flag=0;
int failed_matching=0;
int failed_exp_matching=0;
int _;
double **matrix_A, **matrix_B;
int *degree_A, *degree_B, *degree_AB;
int max_degree_A, max_degree_B;
int **neighborhood_A, **neighborhood_B, **neighborhood_A_back, **neighborhood_B_back, **neighborhood_AB;
double ***msg_iip_to_jjp, ***msg_iip_from_jjp;
double *msg_marginals;
double* msg_probabilities;
double *exp_msg_marginals;
int *tab_subset_A, *tab_subset_A_list, *tab_subset_B, *tab_subset_B_list;
int *tab_permu, permu_aux_int, *tab_permu_aux;
int* random_permu;

clock_t start_time;
//quantities to be evaluated and outputted 
//double quantiles[]={0.05,0.25,0.5,0.75,0.95}; //quantiles for the distributio of the scores of the grond truth permutation edges (i.e. distrib of scores of i---i edges)
//double quantiles_true [N_QUANTILES]; //quantiles of the correct edges
//double quantiles_false [N_QUANTILES]; //quantiles of the incorrect edges
//double bad_ranks[N_QUANTILES]; //looking at the separation of good and bad edges quantiles does not give immadiate information about the error because bad edges are more numerous. To have info about the error we should insted look at ranks (N*0.05,N*0.25,...) of the bad edges
double max_bip_match_error;
double avg_max_match_score;
double exp_max_bip_match_error;
double log_avg_exp_max_match_exp_score;
double max_bip_match_edge_ovlap;
double exp_max_bip_match_edge_ovlap;
double argmax_score_error;
double avg_argmax_score;
double log_avg_argmax_exp_score;
double argmax_score_injectivity_violation;
double matrix_estimator_error;
double matrix_estimator_false_positives;
double matrix_estimator_false_negatives;
double matrix_estimator_frac_assigned_vertices;
double matrix_estimator_error_assigned_vertices;
double NTMA2_frac_good_matches;
double NTMA2_frac_bad_matches;
double NTMA2_score_good_matches;
double NTMA2_score_bad_matches;
double true_edge_overlap; //number fo edges in the true intersectino graph
double mean_tree_depth=0;
double mean_square_tree_depth=0;
double avg_prob_on_identity; //probability assigned to the identical permutation
double avg_prob_argmax;
double log_avg_exp_score_on_identity;
double log_avg_exp_score; 
double avg_score;
double avg_score_on_identity;
double avg_score_neighbors_B_planted; //symmetric wrt exchange A<--->B.  Look at L_{ij'} j'\sim i'
double avg_score_next_neighbors_A_B_planted; // Look at L_{jj'} j'\sim i' j\sim i
double avg_square_score_on_identity; 
double argmax_score_edge_ovlap;
double argmax_score_edge_set_error;
double argmax_score_avg_edge_set_overlap;
double avg_sum_square_marginals;
double avg_argmax_score_degeneracy;//take argmax_edge_set as a matrix and average over its entries
double frac_argmax_score_degen_coords;
double comp_size_rand_vertex; //given a vertex sampled uniformly at random, what's the size of the component it belongs to?
double comp_size_correct_vertex; //given a vertex which has been correctly matched, what is the size of the component it belongs to?
double tree_size_rand_vertex; //same as the two prevoius entries but looking at the maximal tree neighborhood
double tree_size_correct_vertex;
double tree_depth_rand_vertex;
double tree_depth_correct_vertex;
double avg_degree_correct_vertex;
double avg_degree_A_correct_vertex;
double avg_degree_minAB_correct_vertex;
double avg_degree_matrix_correct_vertex;
double avg_degree_A_matrix_correct_vertex;
double avg_degree_minAB_matrix_correct_vertex;
//
double avg_dist_incorrect_vertices_argmax_est_connect; // OK average distance between incorrectly classified vertices in the same component (i.e. the ground truth and the estimator are in the same component in B)
double frac_incorrect_vertices_argmax_est_disconnect; // OK fraction of incorrect vertices that are in a different component of B from the ground truth
double avg_dist_vertices_mindist_est_connect; // OK
double frac_vertices_mindist_est_disconnect;// OK
double error_mindist_est; // OK error=1-overlap of ther minimum distance estimator
double avg_dist_random_perm_connect; //OK
double frac_vertices_disconnect_rand_perm; //OK
double avg_est_dist_mindist_est_connect; // OK estimate of the distance of correctly classified vertices
double est_prob_disconnect_vertices_mindist_est; // OK estimate of the number of disconnected vertices given by the mindist estimator
double avg_cross_ent_argmax_estimator; //OK
double avg_brier_score_argmax_estimator; //OK
//auxiliary quantities
double* bad_edges_scores;
double* good_edges_scores;
int* max_bip_match_perm;
int* exp_max_bip_match_perm;
int* partial_ntma2_match;
int* max_match_perm;
int* argmax_score;//max and argmax along target nodes
double* max_score;
int* true_permutation;
double * score_normalization;
int ** argmax_edge_set;
int * min_avg_dist_estimator;
int* matrix_estimator;
double* expected_value_dist_tmp; //just a placeholder not to allocate memory at each call of the function
double* expected_mindist_estimator_distances_connect; // estimate (according to the scores) of the distance between the minimal distance estimator and the ground truth
double* expect_probs_disconnect_mindist;
//output related stuff
char out_folder_results []="./results/"; //relative path to the directory where final results files are to be saved
char identifier [150];
char fname_max_argmax_file[300];
FILE* outfile; //pointer to file object where data is written
FILE* outfile_A;
FILE* outfile_B;
FILE* outfile_max_argmax_scores;
//int *tab_degree_AtoB, *tab_degree_BtoA;
//int nb_matchings_correct, nb_matchings_incorrect, nb_constraints_A_violated, nb_constraints_B_violated;

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
void init_msgs_planted();
void update();
void update_s_one();
void calc_observables(double * msg_marginals);
FILE* open_output_file(FILE* outfile, const char * append_to_name);
void subset_init(int, int, int *, int *);
int subset_next(int, int *, int *);
void permu_init(int);
int permu_next(int);
int max_bipartite_matching(int N, double * weights, int* max_match);
void max_argmax(double* array,int array_len, double* max, int* argmax);
void find_rankings(int* ranks,int ranks_length, double* array ,int array_length, double* out_ranks);
void copy_array(double* source, double* destination, int len_source);
void NTMA2(int N, double* weights, int* partial_match, double threshold);//implements the NTMA-2 algorithm (algorithm 4 in 2002.01258)
void write_results_file(FILE* outfile, int current_depth);
void write_header(FILE* outfile);
void uniform_permutation(int* permu, int permu_len);
void graph_distance_matrix(int N, int ** neighborhood_A,int * degree_A, int**  distance_matrix_A); //-1=\intfy in the sense that if two vertices belong to different connected components then their distance will be -1
double log_sum_exp(double aa, double bb);
void argmax_set(double * array, int array_len, int* argmax); //returns a binary vector which is the indicator of the argmax of array
double error_perm(int* perm, int N); 
double error_edge_set(int ** edge_set, int N);//edge_set is an N x N 0/1 matrix. An error of 1/N is added each time  edge_set[i][i]=0
double avg_perm_overlap_edge_set(int ** edge_set, int N);//edge_set is an N x N 0/1 matrix. An overlap  1/(N*|S(i)|) is added every time edge_set[i][i]=1. S(i)={ip\in [N] s.t. edge_set[i][ip]=1} 
double injectivity_violation(int* perm, int N);
int edge_overlap(int * perm, int N, double** B, int ** neighborhood_A,int *degree_A); //computes the number of edges in the intersection graph 
void print_observables(int depth);
void compute_min_avg_dist_estimator(int N, int ** distance_matrix_B, int* min_avg_dist_estimator, double* expected_mindist_estimator_distances_connect,double* expect_probs_disconnect_mindist, double* msg_probabilities);
///tree neighborhood results related things//////
FILE* outfile_tree;
FILE* outfile_marginals;
int* tree_depths_A;
int* tree_depths_B;
int* tree_sizes_AB;
int* tree_depths_AB;
int* comp_sizes_AB;
int* comp_sizes_B;
int ** distance_matrix_B;
void max_tree_depth(int N, int ** neighborhood_A,int * degree_A, int* tree_depths_A, int* tree_sizes_A);
void components_sizes(int N, int ** neighborhood_A,int * degree_A, int* comp_sizes_A);
double * tree_msg_marginals;
int tree_depth_jjp;

// Routines Andrea ///////
void init_random();
double rannyu();
double squeeze(int i[], int j);
void setrn( int iseed[] );
int nrannyu( int max );
int compare_doubles (const void *a, const void *b);
//////////////////////////////////////////////////////////////////
int main(int argc, char **argv ){ 
    
    start_time=clock();
    
    int i,j,ip,jp;
    int done_tree_vertices=0; 
    int flag_written_tree_file=0;
    read_params(argc, argv);
    if (write_on_file==1){
        outfile=open_output_file(outfile,"");
        write_header(outfile);
    }
    
    if(max_tree_depth_results==1){
        outfile_tree=open_output_file(outfile_tree,"_tree"); 
        write_header(outfile_tree);
    }
    
    init_random();
  
    allocate_memory();

    init_AB_ER(); 
    
     //trying different matrices B to test the distnce function
    /*
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            matrix_B[i][j]=(i==(j+1) ||i==(j-1)) ? 1:0;
            //matrix_B[i][j]=(i==j) ? 0:1;
            //matrix_B[i][j]=0;
        }
    }
    matrix_B[0][N-1]=1;
    matrix_B[N-1][0]=1;
    */
    /*comment the following  to disable outputting the adjacency matrices*/
    
    char fname_A [300];
    char fname_B [300];
    _=sprintf(fname_A,"./graphs/graph_A_file_v5_l%.2f_s%.2f_N%d_seed%d_%s.dat",ER_l,ER_s,N,jseed,identifier);
    _=sprintf(fname_B,"./graphs/graph_B_file_v5_l%.2f_s%.2f_N%d_seed%d_%s.dat",ER_l,ER_s,N,jseed,identifier);
    
    outfile_A=fopen(fname_A,"w");
    outfile_B=fopen(fname_B,"w");
    if(outfile_A==NULL || outfile_B==NULL){
        fprintf(stderr,"ERROR: could not open graph file\n");
        exit(EXIT_FAILURE);
    }
    for(int ii=0;ii<N;ii++){
        for(int jj=0;jj<N;jj++){
            fprintf(outfile_A,"%.0f ",matrix_A[ii][jj]); //printing the log of the normalized probabilities, this way I can use them as priors in algorithms such as BP.
            fprintf(outfile_B,"%.0f ",matrix_B[ii][jj]);
        }
        fprintf(outfile_A,"\n");
        fprintf(outfile_B,"\n");
    }
    fclose(outfile_A);
    fclose(outfile_B);
    
    
    init_from_AB();

    /*
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            printf("%d ",distance_matrix_B[i][j]);
        }
        printf("\n");
    }
    */
    if(max_tree_depth_results==1){
        max_tree_depth( N, neighborhood_A, degree_A, tree_depths_A,tree_sizes_AB);//tree_sizes_AB here is only a placeholder
        max_tree_depth( N, neighborhood_B, degree_B, tree_depths_B,tree_sizes_AB);
    }
    
    max_tree_depth(N, neighborhood_AB, degree_AB, tree_depths_AB, tree_sizes_AB);
    components_sizes(N,neighborhood_AB,degree_AB,comp_sizes_AB);
    components_sizes(N,neighborhood_B,degree_B,comp_sizes_B);
    
    comp_size_rand_vertex=0;
    tree_size_rand_vertex=0;
    tree_depth_rand_vertex=0;
    for(i=0;i<N;i++){
        comp_size_rand_vertex+=comp_sizes_AB[i];
        tree_size_rand_vertex+=tree_sizes_AB[i];
        tree_depth_rand_vertex+=tree_depths_AB[i];
    }
    comp_size_rand_vertex= (double)comp_size_rand_vertex/N;
    tree_size_rand_vertex= (double)tree_size_rand_vertex/N;
    tree_depth_rand_vertex= (double)tree_depth_rand_vertex/N;
    
    //computing distance properties of random permutations
    avg_dist_random_perm_connect=0;
    frac_vertices_disconnect_rand_perm=1; 
    for(ip=0;ip<N;ip++){
        for(jp=0;jp<N;jp++){
            if(distance_matrix_B[ip][jp]>0){
                avg_dist_random_perm_connect+=((double)distance_matrix_B[ip][jp]/(N*comp_sizes_B[ip]));
            }
        }
        frac_vertices_disconnect_rand_perm-=((double) comp_sizes_B[ip]/(N*N));
    }
    
    init_msgs_uninformative();
    //init_msgs_planted();

    for(i=0;i<N;i++){
        true_permutation[i]=i;
    }
    true_edge_overlap = edge_overlap(true_permutation, N, matrix_B, neighborhood_A, degree_A); //computes the number of edges in the intersection graph 
    i=0;
    while(i<depth || (max_tree_depth_results==1 && flag_written_tree_file==0 && run_until_max_tree_depth==1)){
    
        if(flag_s_one==1)
            	update_s_one();

        else{
            	update();
        }
        
        calc_observables(msg_marginals);
        
        /*
        if(max_tree_depth_results==1){
            for(j=0;j<N;j++){
                for(jp=0;jp<N;jp++){
                    tree_depth_jjp = (tree_depths_A[j] < tree_depths_B[jp]) ? tree_depths_A[j] : tree_depths_B[jp];
                    //the array tree_marginals is initialized everywhere with the value of the isolated vertices
                    if(i==0){
                        mean_tree_depth+=((double)tree_depth_jjp/(N*N));
                        mean_square_tree_depth+=((double)tree_depth_jjp*tree_depth_jjp/(N*N));
                        if(tree_depth_jjp==0){
                            done_tree_vertices++;
                        }
                    }
                    if(i+1==tree_depth_jjp){
                        tree_msg_marginals[N*j+jp]=msg_marginals[N*j+jp];
                        
                        done_tree_vertices++;
                    }
                }
            }
        }
        */
        
        //to print the  argmax_score[i], msg_probabilities[N*i+argmax_score[i]] , msg_probabilities[N*i+i] for all i (at each depth). A file is opened at each iterations, the results are writetn
        /*
        _=sprintf(fname_max_argmax_file,"./max_argmax/max_argmax_v5_l%.2f_s%.2f_N%d_seed%d_d%d_%s.dat",ER_l,ER_s,N,jseed,i+1,identifier);
        outfile_max_argmax_scores=fopen(fname_max_argmax_file,"w");
        for(j=0;j<N;j++){
            fprintf(outfile_max_argmax_scores,"%.7f %d %.5f\n",msg_probabilities[N*j+argmax_score[j]],argmax_score[j],msg_probabilities[N*j+j]);
        }
        fclose(outfile_max_argmax_scores);
        */
        //to print all the marginals: comment the following not to print all the marginals
        /*
        if(i+1==10){
            char marg_fname [300];
            _=sprintf(marg_fname,"./marginals/log_marg_file_v5_l%.2f_s%.2f_N%d_seed%d_d%d_%s.dat",ER_l,ER_s,N,jseed,i+1,identifier);
            outfile_marginals=fopen(marg_fname,"w");
            if(outfile_marginals==NULL){
                fprintf(stderr,"ERROR: could not open marginal file\n");
                exit(EXIT_FAILURE);
            }
            for(int ii=0;ii<N;ii++){
                for(int jj=0;jj<N;jj++){
                    fprintf(outfile_marginals,"%.4f ",(msg_marginals[N*ii+jj]-score_normalization[ii])); //printing the log of the normalized probabilities, this way I can use them as priors in algorithms such as BP.
                }
                fprintf(outfile_marginals,"\n");
            }
            fclose(outfile_marginals);
        }
        */
        if(verbose==1){
            print_observables(i+1);
        }

        if(write_on_file && (i+1<=depth)){
            write_results_file(outfile,i+1);
        }
        if(max_tree_depth_results==1 && done_tree_vertices==N*N && flag_written_tree_file==0){
            calc_observables(tree_msg_marginals);
            
            write_results_file(outfile_tree,i+1);
            flag_written_tree_file=1;
            
            if(verbose==1){
                printf("\nMAX TREE NEIGHBORHOOD RESULTS\n");
                print_observables(i+1);
            }
        }
        i++; 
    }

    free_memory();
    if(max_tree_depth_results==1 && done_tree_vertices<N*N){
        fprintf(stderr,"WARNING: insufficient depth to explore tree neighborhoods");
    }
    if(max_tree_depth_results==1){
        fclose(outfile_tree);
    }
    

  // this part gives examples of the subset enumeration routines
  /*
  
  
  int tab_dummy[10], tab_dummy_list[10], i, l, nb, done;

  nb=3;
  l=5;
  
  subset_init(l,nb,tab_dummy,tab_dummy_list);


  done=0;
  while(done==0)
    {
      for(i=0;i<l;i++)
	printf("%i ",tab_dummy[i]);

      printf("          ");

      for(i=0;i<nb;i++)
	printf("%i ",tab_dummy_list[i]);
      

     printf("\n");
      done=subset_next(l,tab_dummy,tab_dummy_list);
    }
  */

  // this part gives examples of the permutation enumeration routine (Heap's algorithm)
  /*
  int l, i, done;

  l=5;
  tab_permu=calloc(l,sizeof(int));
  tab_permu_aux=calloc(l,sizeof(int));


  permu_init(l);
  done=0;
  while(done==0)
    {
      for(i=0;i<l;i++)
	printf("%i ",tab_permu[i]);
      printf("\n");
      done=permu_next(l);
    }

  free(tab_permu);
  free(tab_permu_aux);
  */
}

void read_params(int argc, char *argv[]){
  if (argc>10){
      ER_l=atof(argv[1]);
      flag_s_one=atoi(argv[2]); // 1 if ER_s=1.
      ER_s=atof(argv[3]);
      N=atoi(argv[4]);
      depth=atoi(argv[5]);
      jseed=atoi(argv[6]);
      write_on_file=atoi(argv[7]);
      verbose=atoi(argv[8]);
      strcpy(identifier,argv[9]);
      max_tree_depth_results=atoi(argv[10]); //if 1 a separate output file is opened and results are written for 
      run_until_max_tree_depth=atoi(argv[11]);
  }
  else{
      printf("# Usage: ER_l flag_s_one ER_s N depth seed write_on_file verbose identifier max_tree_depth run_until_max_tree_depth\n");
      exit(0);
  }

}

void allocate_memory(){
  int i,ip, Ns;

  Ns=N*N;

  matrix_A=(double **)calloc(N,sizeof(double*));
  matrix_B=(double **)calloc(N,sizeof(double*));
  argmax_edge_set=(int **)calloc(N,sizeof(int*));
  distance_matrix_B=(int **) calloc(N,sizeof(int*));
  for(i=0;i<N;i++)
    {
      distance_matrix_B[i]=(int*)calloc(N,sizeof(int));
      matrix_A[i]=(double*)calloc(N,sizeof(double));
      matrix_B[i]=(double*)calloc(N,sizeof(double));
      argmax_edge_set[i]=(int*)calloc(N,sizeof(int));
   }

  degree_A=(int*)calloc(N,sizeof(int));
  degree_B=(int*)calloc(N,sizeof(int));
  degree_AB=(int*)calloc(N,sizeof(int));

  msg_marginals=(double*)calloc(Ns,sizeof(double));
  msg_probabilities=(double*)calloc(Ns,sizeof(double));

  tree_msg_marginals=(double*)calloc(Ns,sizeof(double));

  exp_msg_marginals=(double*)calloc(Ns,sizeof(double));

  //tab_degree_AtoB=calloc(N,sizeof(int));
  //tab_degree_BtoA=calloc(N,sizeof(int));
  
  bad_edges_scores=(double*)calloc((int) N*(N-1),sizeof(double));
  good_edges_scores=(double*)calloc(N,sizeof(double));
  max_bip_match_perm=(int*)calloc(N,sizeof(int));
  exp_max_bip_match_perm=(int*)calloc(N,sizeof(int));
  argmax_score=(int*)calloc(N,sizeof(int));
  max_score=(double*)calloc(N,sizeof(double));
  partial_ntma2_match=(int*)calloc(N,sizeof(int));
  true_permutation=(int*)calloc(N,sizeof(int));
  random_permu=(int*)calloc(N,sizeof(int));
  tree_depths_A=(int*)calloc(N,sizeof(int));
  tree_depths_B=(int*)calloc(N,sizeof(int));
  tree_depths_AB=(int*)calloc(N,sizeof(int));
  tree_sizes_AB=(int*)calloc(N,sizeof(int));
  comp_sizes_AB=(int*)calloc(N,sizeof(int));
  comp_sizes_B=(int*)calloc(N,sizeof(int));
  score_normalization=(double*) calloc(N,sizeof(double));
  min_avg_dist_estimator= (int *) calloc(N,sizeof(int));
  expected_value_dist_tmp=(double*)calloc(N,sizeof(double));
  expected_mindist_estimator_distances_connect=(double*)calloc(N,sizeof(double));
  expect_probs_disconnect_mindist=(double*)calloc(N,sizeof(double));

  
  
}

void free_memory()
{
  int i, ip, j, index, di, dip;

  for(i=0;i<N;i++)
    {
      free(matrix_A[i]);
      free(matrix_B[i]);
      free(argmax_edge_set[i]);
      free(distance_matrix_B[i]);
    }
  free(distance_matrix_B);
  free(matrix_A);
  free(matrix_B);
  free(argmax_edge_set);

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
  
  for(i=0;i<N;i++)
    if(degree_AB[i]>0)
      {
	free(neighborhood_AB[i]);
      }
  free(neighborhood_AB);



  free(degree_A);
  free(degree_B);
  free(degree_AB);

  free(msg_marginals);
  free(msg_probabilities);
  free(tree_msg_marginals);
  free(exp_msg_marginals);

  free(tab_subset_A);
  free(tab_subset_A_list);
  free(tab_subset_B);
  free(tab_subset_B_list);

  free(tab_permu);
  free(tab_permu_aux);
  
  free(bad_edges_scores);
  free(good_edges_scores);
  free(max_bip_match_perm);
  free(exp_max_bip_match_perm);
  
  free(argmax_score);
  free(max_score);
  free(partial_ntma2_match);
  
  free(true_permutation);
  
  free(tree_depths_A);
  free(tree_depths_B);
  free(tree_depths_AB);
  free(tree_sizes_AB);
  free(comp_sizes_AB);
  free(comp_sizes_B);
  free(score_normalization);
  free(random_permu);
  free(min_avg_dist_estimator);
  free(expected_value_dist_tmp);
  free(expected_mindist_estimator_distances_connect);
  free(expect_probs_disconnect_mindist);

  //free(tab_degree_AtoB);
  //free(tab_degree_BtoA);
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
	graph_distance_matrix(N, neighborhood_B, degree_B, distance_matrix_B); //-1=\intfy in the sense that if two vertices belong to different connected components then their distance will be -1

	// for the matrix A*B
  for(i=0;i<N;i++)
    degree_AB[i]=0;

  for(i=1;i<N;i++)
    for(j=0;j<i;j++)
      if(matrix_B[i][j]!=0. && matrix_A[i][j]!=0.)
	{
	  degree_AB[i]++;
	  degree_AB[j]++;
	}

  neighborhood_AB=(int**)calloc(N,sizeof(int*));

  for(i=0;i<N;i++)
    if(degree_AB[i]>0)
      {
	neighborhood_AB[i]=(int*)calloc(degree_AB[i],sizeof(int));
      }

  for(i=0;i<N;i++)
    degree_AB[i]=0;

  for(i=1;i<N;i++)
    for(j=0;j<i;j++)
      if(matrix_B[i][j]!=0. && matrix_A[i][j]!=0.)
	{
	  neighborhood_AB[i][degree_AB[i]]=j;
	  neighborhood_AB[j][degree_AB[j]]=i;
	  degree_AB[i]++;
	  degree_AB[j]++;
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
    
    for(i=0;i<N;i++){
      for(ip=0;ip<N;ip++){
          if(flag_s_one==1){
              tree_msg_marginals[N*i+ip]=ER_l;
          }
          else{
              tree_msg_marginals[N*i+ip]=ER_l*ER_s+(degree_A[i]+degree_B[ip])*log(1-ER_s);
          }
      }
  }

}


/////////////////////////////////////////
//given a graph with N vertices represented by its adjaceny list neighborhood_A and degree list degree_A the algorithm finds, for every vertex, the depth of the maximal tree neighborhood andd stores it into tree_depths_A
//return also tree sizes
void max_tree_depth(int N, int ** neighborhood_A,int * degree_A, int* tree_depths_A, int* tree_sizes_A){
    int i,j,k,desc_j;
    int flag_cycle;
    int * dist_from_i;
    int * parent;
    int size_tree_rooted_i;
    dist_from_i=(int *)calloc(N,sizeof(int));
    parent=(int *)calloc(N,sizeof(int));
    std::deque <int> active;
    
    for(i=0;i<N;i++){ //root node
        //printf("\nroot=%d\n",i);
        
        //resetting variables
        active.clear();
        tree_depths_A[i]=0;
        tree_sizes_A[i]=1;
        for(j=0;j<N;j++){
            dist_from_i[j]=-1;
            parent[j]=-1;
        }
        //initializing
        dist_from_i[i]=0;
        flag_cycle=0; 
        active.push_back(i);
            
        while(!active.empty() && flag_cycle==0){
            j=active.front();
            //printf("  look at %d \n",j);
            for(k=0;k<degree_A[j];k++){
                desc_j=neighborhood_A[j][k];//descendant of j
                //printf("    desc %d=%d ",j,desc_j);
                parent[desc_j]=j;
                if(dist_from_i[desc_j]==-1){//in this case desc_j is an undiscovered vertex
                     tree_sizes_A[i]+=1;
                     dist_from_i[desc_j]=dist_from_i[j]+1;
                     
                     //printf("new d=%d\n",dist_from_i[desc_j]);
                     active.push_back(desc_j);
                }
                
                else if (desc_j!=parent[j]){
                    //printf("cycle\n");
                    flag_cycle=1;
                    tree_depths_A[i]=dist_from_i[j];
                    //printf("  tree depth at %d=%d\n",i,tree_depths_A[i]);
                    break;
                }
                else{
                   // printf("parent %d\n",parent[j]);
                }
            }
            //printf("  remove %d\n",j);
            active.pop_front();
        }
        if(flag_cycle==0){//if there are no cycles the maximal depth is just the distance of i from the furthest vertex in the connected component
            //printf("f\n");
            for(j=0;j<N;j++){
                tree_depths_A[i]= (tree_depths_A[i] > dist_from_i[j]) ? tree_depths_A[i] : dist_from_i[j];
            }
            //printf("  tree depth at %d=%d\n",i,tree_depths_A[i]);
        }
    }
    free(dist_from_i);
    free(parent);
}


void graph_distance_matrix(int N, int ** neighborhood_A,int * degree_A, int**  distance_matrix_A){ //-1=\intfy in the sense that if two vertices belong to different connected components then their distance will be -1
    int i,j,k,desc_j;
    int * dist_from_i;
    int * parent;
    dist_from_i=(int *)calloc(N,sizeof(int));
    parent=(int *)calloc(N,sizeof(int));
    std::deque <int> active;
    
    for(i=0;i<N;i++){ //root node
        //printf("\nroot=%d\n",i);
        //resetting variables
        active.clear();
        //tree_depths_A[i]=0;
        //tree_sizes_A[i]=1;
        for(j=0;j<N;j++){
            dist_from_i[j]=-1;
            parent[j]=-1;
        }
        //initializing
        dist_from_i[i]=0;
        active.push_back(i);
            
        while(!active.empty()){
            j=active.front();
            //printf("  look at %d \n",j);
            for(k=0;k<degree_A[j];k++){
                desc_j=neighborhood_A[j][k];//descendant of j
                //printf("    desc %d=%d ",j,desc_j);
                parent[desc_j]=j;
                if(dist_from_i[desc_j]==-1){//in this case desc_j is an undiscovered vertex
                     dist_from_i[desc_j]=dist_from_i[j]+1;
                     
                     //printf("new d=%d\n",dist_from_i[desc_j]);
                     active.push_back(desc_j);
                }
            }
            //printf("  remove %d\n",j);
            active.pop_front();
        }
        for(j=0;j<N;j++){
            distance_matrix_A[i][j]=dist_from_i[j];
        }
    }
    free(dist_from_i);
    free(parent);
}


void components_sizes(int N, int ** neighborhood_A,int * degree_A, int* comp_sizes_A){
    int i,j,k,desc_j;
    int * dist_from_i;
    int * discovered;
    int comp_size_i;
    
    discovered=(int *)calloc(N,sizeof(int));
    std::deque <int> active;
    std::deque <int> vertices;
    for(i=0;i<N;i++){
        vertices.push_back(i);
        comp_sizes_A[i]=0;
    }
    while(!vertices.empty()){
    
        //resetting variables
        comp_size_i=1;
        active.clear();
        for(j=0;j<N;j++){
            discovered[j]=0;
        }
        
        i=vertices.front(); //starting point to build the cluster
        
        //root node
        //printf("\nroot=%d\n",i);
        
        
        //initializing
        discovered[i]=1;
        active.push_back(i);
            
        while(!active.empty()){
            j=active.front();
            //printf("  look at %d \n",j);
            for(k=0;k<degree_A[j];k++){
                desc_j=neighborhood_A[j][k];// kth descendant of j
                //printf("    desc %d=%d ",j,desc_j);
                if(discovered[desc_j]==0){//in this case desc_j is an undiscovered vertex
                     comp_size_i+=1;
                     discovered[desc_j]=1;
                     active.push_back(desc_j);
                     //printf("    new vertex %d\n",desc_j);
                }
            }
            //printf("  remove %d\n",j);
            active.pop_front();
        }
        
        
        for(j=0;j<N;j++){
            if(discovered[j]==1){
                comp_sizes_A[j]=comp_size_i; //assigning the size to all points in the cluster
                //printf("  remove %d\n",j);
                vertices.erase(std::remove(vertices.begin(), vertices.end(), j), vertices.end()); //removing reached vertices from queue
            }
        }
        /*
        cout<<vertices.empty()<<endl;
        std::deque <int> copy=vertices;
        while(!copy.empty()){
            cout<<copy.front()<<" ";
            copy.pop_front();
        }
        
        cout<<endl;
        */
    }
    free(discovered);
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

void init_msgs_planted()
{
  /*
  int i, ip, index, di, dip, j, jp;
  double false_infinity, msg;

  
  false_infinity=20.*ER_l*ER_l;
  if(false_infinity<60.)
    false_infinity=60.;

  for(i=0;i<N;i++)
    for(ip=0;ip<N;ip++)
      {
	if(i==ip)
	  msg=false_infinity;
	else
	  msg=1./false_infinity;

	index=(i*N)+ip;
	di=degree_A[i];
	dip=degree_B[ip];
	msg_iip_to_A[index]=msg;
	msg_iip_to_B[index]=msg;
	for(j=0;j<di;j++)
	  for(jp=0;jp<dip;jp++)
	    msg_iip_to_jjp[index][j][jp]=msg;
      }
  */
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

	//msg_marginals[index]=ER_l*ER_s+(di+dip)*log(1-ER_s);//exp(ER_l*ER_s)*pow(1.-ER_s,di+dip);
	msg_marginals[index]=0;
	for(j=0;j<di;j++)
	  for(jp=0;jp<dip;jp++)
	    msg_iip_to_jjp[index][j][jp]=0;
	   // msg_iip_to_jjp[index][j][jp]=ER_l*ER_s+(di+dip-2)*log(1-ER_s);//exp(ER_l*ER_s)*pow(1.-ER_s,di+dip-2);
    
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
		  	  //prod_permu+=msg_iip_from_jjp[index][tab_subset_A_list[m]][tab_subset_B_list[tab_permu[m]]];

			sum_permu=log_sum_exp(sum_permu,prod_permu);
			//sum_permu+=prod_permu;
			done_permu=permu_next(l);
		      }
            msg_marginals[index]=log_sum_exp(msg_marginals[index],sum_permu);
		    //msg_marginals[index]+=(sum_permu*exp(ER_l*ER_s)*pow(ER_s/ER_l,l)*pow(1.-ER_s,di+dip-(2*l)));
		    
		    for(j=0;j<di;j++)
		      for(jp=0;jp<dip;jp++)
			if((tab_subset_A[j]==0)&&(tab_subset_B[jp]==0))
			  msg_iip_to_jjp[index][j][jp]=log_sum_exp(msg_iip_to_jjp[index][j][jp],sum_permu);
			  //msg_iip_to_jjp[index][j][jp]+=(sum_permu*exp(ER_l*ER_s)*pow(ER_s/ER_l,l)*pow(1.-ER_s,di+dip-2-(2*l)));
		    
		    done_B=subset_next(dip,tab_subset_B,tab_subset_B_list);
		  }
		done_A=subset_next(di,tab_subset_A,tab_subset_A_list);
		
		  //if(index==0)
		    //  printf("msg_part_a=%3.5f\n",msg_iip_to_jjp[index][0][0]);
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
	      //msg_marginals[index]=exp(ER_l);
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
		      //prod_permu*=msg_iip_from_jjp[index][m][tab_permu[m]];
		    
		    sum_permu=log_sum_exp(sum_permu,prod_permu);
		    done_permu=permu_next(di);
		  }
		msg_marginals[index]=ER_l-di*log(ER_l)+sum_permu;

		//msg_marginals[index]=exp(ER_l)*pow(ER_l,-di)*sum_permu;
	      }

	    if(di==1)
	      msg_iip_to_jjp[index][0][0]=ER_l;
	      //msg_iip_to_jjp[index][0][0]=exp(ER_l);
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
			//msg_iip_to_jjp[index][j][jp]=(exp(ER_l)*pow(ER_l,1-di)*sum_permu);
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

    int i, ip, j, jp;
    int k,kk;
    double tmp;
    double score;
    
    NTMA2_frac_good_matches=0;
    NTMA2_frac_bad_matches=0;
    NTMA2_score_good_matches=0;
    NTMA2_score_bad_matches=0;
    
    int count_bad=0;
    //int ranks [N_QUANTILES];
    
    for(i=0;i<N;i++){
        for(ip=0;ip<N;ip++){
            score=msg_marginals[(i*N)+ip];
            exp_msg_marginals[N*i+ip]=exp(msg_marginals[N*i+ip]);
            if(i==ip){
                good_edges_scores[i]=score;
            }
            else{
                bad_edges_scores[count_bad]=score;
                count_bad++;
            } 
        }
    }
    //computing normalizations to turn scores into probabilities
    for(i=0;i<N;i++){
        score_normalization[i]=-DBL_MAX;
        for(ip=0;ip<N;ip++){
            score_normalization[i]=log_sum_exp(score_normalization[i],msg_marginals[N*i+ip]);
        }
        for(ip=0;ip<N;ip++){
            msg_probabilities[N*i+ip]=exp(msg_marginals[N*i+ip]-score_normalization[i]);
        }
    }
   
    //max bipartite matching 
    failed_matching=max_bipartite_matching(N,msg_marginals,max_bip_match_perm);
    failed_exp_matching=max_bipartite_matching(N,exp_msg_marginals,exp_max_bip_match_perm);
    
    max_bip_match_error=error_perm(max_bip_match_perm,N);
    exp_max_bip_match_error=error_perm(exp_max_bip_match_perm,N);
    max_bip_match_edge_ovlap=edge_overlap(max_bip_match_perm,N,matrix_B,neighborhood_A,degree_A);
    exp_max_bip_match_edge_ovlap=edge_overlap(exp_max_bip_match_perm,N,matrix_B,neighborhood_A,degree_A);
    
    argmax_score_edge_ovlap=edge_overlap(argmax_score,N,matrix_B,neighborhood_A,degree_A);
    
    //argmax score approach
    for(i=0;i<N;i++){
        max_argmax((msg_marginals+N*i),N,& max_score[i],& argmax_score[i]);
        argmax_set((msg_marginals+N*i),N, argmax_edge_set[i]);
        if(max_score[i]>DBL_MAX/1e8){
            overflow_flag=1;   
        }
        max_score[i]=exp(max_score[i]-score_normalization[i]); //turning it into a probability
    }
    argmax_score_error=error_perm(argmax_score,N);
    argmax_score_injectivity_violation=injectivity_violation(argmax_score,N);
    
    
    argmax_score_edge_set_error=error_edge_set(argmax_edge_set,N);
    argmax_score_avg_edge_set_overlap=avg_perm_overlap_edge_set(argmax_edge_set,N);
    avg_argmax_score_degeneracy=0.;
    int degen_i;
    frac_argmax_score_degen_coords=0;
    for (i=0;i<N;i++){
        degen_i=0;
        for(ip=0;ip<N;ip++){
            degen_i+=argmax_edge_set[i][ip];
        }
        avg_argmax_score_degeneracy+=degen_i;
        if(degen_i>=2){
            frac_argmax_score_degen_coords+=1;
        }
        
    }
    avg_argmax_score_degeneracy/=(N*N);
    frac_argmax_score_degen_coords/=N;
    
    //computing size of correctly matched vertices
    comp_size_correct_vertex=0;
    tree_size_correct_vertex=0;
    tree_depth_correct_vertex=0;
    avg_degree_correct_vertex=0;
    avg_degree_minAB_correct_vertex=0;
    avg_degree_A_correct_vertex=0;
    for(i=0;i<N;i++){
        if(argmax_score[i]==i){
            comp_size_correct_vertex+=comp_sizes_AB[i];
            tree_size_correct_vertex+=tree_sizes_AB[i];
            tree_depth_correct_vertex+=tree_depths_AB[i];
            avg_degree_correct_vertex+=degree_AB[i];
            avg_degree_A_correct_vertex+=degree_A[i];
            avg_degree_minAB_correct_vertex+= degree_A[i]>degree_B[i] ? degree_B[i] : degree_A[i];
        }
        
    }
    comp_size_correct_vertex = (double)comp_size_correct_vertex/(N*(1-argmax_score_error));
    tree_size_correct_vertex = (double)tree_size_correct_vertex/(N*(1-argmax_score_error));
    tree_depth_correct_vertex= (double)tree_depth_correct_vertex/(N*(1-argmax_score_error));
    avg_degree_correct_vertex= (double)avg_degree_correct_vertex/(N*(1-argmax_score_error));
    avg_degree_A_correct_vertex= (double)avg_degree_A_correct_vertex/(N*(1-argmax_score_error));
    avg_degree_minAB_correct_vertex=(double) avg_degree_minAB_correct_vertex/(N*(1-argmax_score_error));
    
    
    
    //computing mean scores 
    avg_prob_on_identity=0; 
    avg_score_on_identity=0;
    avg_square_score_on_identity=0; 
    log_avg_exp_score_on_identity=-DBL_MAX;  //this is log(1/N*\sum_i L_{ii}})
    
    avg_prob_argmax=0;
    avg_argmax_score=0; 
    log_avg_argmax_exp_score=-DBL_MAX; 
    
    avg_max_match_score=0; 
    log_avg_exp_max_match_exp_score=-DBL_MAX; 
    
    log_avg_exp_score=-DBL_MAX;
    avg_score=0;
    
    //computing average probabilities
    for(i=0;i<N;i++){
        avg_prob_on_identity+=exp(msg_marginals[N*i+i]-score_normalization[i])/N;
        avg_score_on_identity+=msg_marginals[N*i+i]/N;
        avg_square_score_on_identity+=((msg_marginals[N*i+i]*msg_marginals[N*i+i])/N);
        log_avg_exp_score_on_identity=log_sum_exp(log_avg_exp_score_on_identity,msg_marginals[N*i+i]);

        avg_prob_argmax+=msg_probabilities[N*i+argmax_score[i]]/N;
        avg_argmax_score+=msg_marginals[N*i+argmax_score[i]]/N;
        log_avg_argmax_exp_score=log_sum_exp(log_avg_argmax_exp_score,msg_marginals[N*i+argmax_score[i]]);

        avg_max_match_score+=msg_marginals[N*i+max_bip_match_perm[i]]/N;
        log_avg_exp_max_match_exp_score=log_sum_exp(log_avg_exp_max_match_exp_score,msg_marginals[N*i+exp_max_bip_match_perm[i]]);
        
        log_avg_exp_score=log_sum_exp(log_avg_exp_score,score_normalization[i]);
        for(ip=0;ip<N;ip++){
            avg_score+=msg_marginals[N*i+ip]/(N*N);
        }
    }
    
    log_avg_argmax_exp_score=log_avg_argmax_exp_score-log(N);
    log_avg_exp_score_on_identity=log_avg_exp_score_on_identity-log(N);
    log_avg_exp_score=log_avg_exp_score-2*log(N);
    log_avg_exp_max_match_exp_score=log_avg_exp_max_match_exp_score-log(N);
    
    //matrix estimator properties
    matrix_estimator_false_positives=0;
    matrix_estimator_false_negatives=0;
    matrix_estimator_error=0;
    matrix_estimator_frac_assigned_vertices=0;
    matrix_estimator_error_assigned_vertices=0;
    avg_degree_matrix_correct_vertex=0;
    avg_degree_A_matrix_correct_vertex=0;
    avg_degree_minAB_matrix_correct_vertex=0;
    
    int flag_assigned_vertex;
    int num_correct_assigned_vertices=0;
    double prob_iip;
    for(i=0;i<N;i++){
        flag_assigned_vertex=0;
        for(ip=0;ip<N;ip++){
        
            prob_iip=msg_probabilities[N*i+ip];

            if(prob_iip>=0.5){
                flag_assigned_vertex=1;
            }
            
            if(i==ip && prob_iip<0.5){
                matrix_estimator_false_negatives+=1;
            }   
             
            if(i!=ip && prob_iip>=0.5){
                    matrix_estimator_false_positives+=1;
            }   
            
        }
        matrix_estimator_frac_assigned_vertices+=flag_assigned_vertex;
        if(flag_assigned_vertex==1 && msg_probabilities[N*i+i]<0.5){
            matrix_estimator_error_assigned_vertices+=1;
        }
        if(flag_assigned_vertex==1 && msg_probabilities[N*i+i]>=0.5){
            avg_degree_matrix_correct_vertex+=degree_AB[i];
            avg_degree_A_matrix_correct_vertex+=degree_A[i];
            avg_degree_minAB_matrix_correct_vertex+=degree_A[i]>degree_B[i] ? degree_B[i] : degree_A[i];
            num_correct_assigned_vertices+=1;
        }
    }
    matrix_estimator_false_positives/=N;
    matrix_estimator_false_negatives/=N;
    matrix_estimator_error=(matrix_estimator_false_positives+matrix_estimator_false_negatives);
    matrix_estimator_error_assigned_vertices/=matrix_estimator_frac_assigned_vertices;
    matrix_estimator_frac_assigned_vertices/=N;
    
    avg_degree_matrix_correct_vertex/=num_correct_assigned_vertices;
    avg_degree_A_matrix_correct_vertex/=num_correct_assigned_vertices;
    avg_degree_minAB_matrix_correct_vertex/=num_correct_assigned_vertices;
    
    
    //computing scores of neighbors of correct vertices
    avg_score_neighbors_B_planted=0;
    avg_score_next_neighbors_A_B_planted=0;
    double temp_avg_score_next_neighs;
    int count_neighs;
    for(i=0;i<N;i++){
        count_neighs=0;
        temp_avg_score_next_neighs=0;
        for(k=0;k<degree_B[i];k++){
            jp=neighborhood_B[i][k];
            avg_score_neighbors_B_planted+=(msg_marginals[N*i+jp]/degree_B[i]);
            
            for(kk=0;kk<degree_A[i];kk++){
                j=neighborhood_A[i][kk];
                if(j!=jp){
                    temp_avg_score_next_neighs+=msg_marginals[N*j+jp];
                    count_neighs+=1;
                }
            }
        }
        if(count_neighs>0){
            avg_score_next_neighbors_A_B_planted+=((double) temp_avg_score_next_neighs/count_neighs);
        }
        
    }
    avg_score_neighbors_B_planted/=N;
    avg_score_next_neighbors_A_B_planted/=N;

    //NTMA2 performance
    NTMA2(N,msg_marginals,partial_ntma2_match,0);//how to set the threshold?
    for(i=0;i<N;i++){
        if(partial_ntma2_match[i]!=-1 && partial_ntma2_match[i]==i){
            NTMA2_frac_good_matches+=1;
            NTMA2_score_good_matches+=msg_marginals[N*i+i];
        }
        else if (partial_ntma2_match[i]!=-1 && partial_ntma2_match[i]!=i){
             NTMA2_frac_bad_matches+=1;
             NTMA2_score_bad_matches+=msg_marginals[N*i+partial_ntma2_match[i]];
        }
    }
    NTMA2_score_good_matches/=NTMA2_frac_good_matches;
    NTMA2_score_bad_matches/=NTMA2_frac_bad_matches;//average score per edge computed both for good and bad matches

    NTMA2_frac_good_matches/=N;
    NTMA2_frac_bad_matches/=N;
    
    //computing minimal distance estimator
    compute_min_avg_dist_estimator( N, distance_matrix_B, min_avg_dist_estimator,  expected_mindist_estimator_distances_connect, expect_probs_disconnect_mindist, msg_probabilities);
    error_mindist_est=error_perm(min_avg_dist_estimator, N);
    
    //stuff with distances
    frac_incorrect_vertices_argmax_est_disconnect=0;
    avg_dist_incorrect_vertices_argmax_est_connect=0;
    avg_est_dist_mindist_est_connect=0;
    est_prob_disconnect_vertices_mindist_est=0;
    avg_dist_vertices_mindist_est_connect=0;
    frac_vertices_mindist_est_disconnect=0;
    for(i=0;i<N;i++){
        //argmax estimator
        if(distance_matrix_B[i][argmax_score[i]]>0){//automatically selects nodes which are in the same component but distinct from i
            avg_dist_incorrect_vertices_argmax_est_connect+=distance_matrix_B[i][argmax_score[i]];
        }
        else if(distance_matrix_B[i][argmax_score[i]]<0){
            frac_incorrect_vertices_argmax_est_disconnect+=1;
        }
        
        //mindist estimator
        if(distance_matrix_B[i][min_avg_dist_estimator[i]]>=0){//automatically selects nodes which are in the same component (this time they can coincide with i)
            avg_dist_vertices_mindist_est_connect+=distance_matrix_B[i][min_avg_dist_estimator[i]];
            
        }
        else if(distance_matrix_B[i][argmax_score[i]]<0){
            frac_vertices_mindist_est_disconnect+=1.;
        }
        
        avg_est_dist_mindist_est_connect+=expected_mindist_estimator_distances_connect[i]/N;
        est_prob_disconnect_vertices_mindist_est+=expect_probs_disconnect_mindist[i]/N;
    }
    frac_incorrect_vertices_argmax_est_disconnect/=(N * argmax_score_error);//normalizing by the number of incorrectly classified vertices. From this I can also get the fraction of incorrectly classified vertices which are in the same component
    avg_dist_incorrect_vertices_argmax_est_connect/=(N*(1-frac_incorrect_vertices_argmax_est_disconnect));
    avg_dist_vertices_mindist_est_connect/=(N-frac_vertices_mindist_est_disconnect);
    frac_vertices_mindist_est_disconnect/=N;
    
    //brier and cross entropy scores
    avg_cross_ent_argmax_estimator=0;
    avg_brier_score_argmax_estimator=0;
    for(i=0;i<N;i++){
        if(argmax_score[i]==i){
            avg_brier_score_argmax_estimator+=(msg_probabilities[N*i+i]-1)*(msg_probabilities[N*i+i]-1);
            avg_cross_ent_argmax_estimator+=-log(msg_probabilities[N*i+argmax_score[i]]);
        }
        else{
            avg_brier_score_argmax_estimator+=(msg_probabilities[N*i+argmax_score[i]])*(msg_probabilities[N*i+argmax_score[i]]);
            avg_cross_ent_argmax_estimator+=-log(1-msg_probabilities[N*i+argmax_score[i]]);
        }
        
    }
    avg_cross_ent_argmax_estimator/=N;
    avg_brier_score_argmax_estimator/=N;
    
    //summing square marginals and averaging them over nodes
    avg_sum_square_marginals=0;
    for(i=0;i<N;i++){
        for(ip=0;ip<N;ip++){
            avg_sum_square_marginals+=(msg_probabilities[N*i+ip]*msg_probabilities[N*i+ip]);
        }
    }
    avg_sum_square_marginals/=N;
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

FILE* open_output_file(FILE* ofile,const char * append_to_name){
    char result_file_name[300];
    sprintf(result_file_name,"%sHT_N%d_l%.3f_depth%d_%s%s.dat",out_folder_results,N,ER_l,depth,identifier,append_to_name);
    ofile=fopen(result_file_name,"a");//append so multiple processes can write there at the same time
    if(ofile==NULL){
        fprintf(stderr,"ERROR:unable to open %s\n",result_file_name);
        
    }
    else if (verbose==1){
        printf("opened %s\n",result_file_name);
    }
    return ofile;
}

//this function takes as input the number of nodes in each partition (N), the edge
// weights, and the array where the best matching should be stored
//best_matching[i] is the node in B to which i\in A is mapped in the max match
int max_bipartite_matching(int N,double * weights, int* max_match){
    int failed_matching=0;
    int i,ip;
    int Nnodi=2*N;
    if(Nnodi <4 || Nnodi%2!=0){
	    cout << "Numero di nodi inaccettabile"<<endl;
	    exit(EXIT_FAILURE);
    }

    int nedges=0;
    Graph g;
    Graph::EdgeMap<double> edgeMap(g);
    
    
    MWPM matching(g,edgeMap);
    
    vector<Graph::Node> nodi;
    vector<Graph::Edge> lati;
    g.clear();
    lati.clear();
    nedges=0;
    nodi.clear();
    /*Creo i nodi*/

    for(int i=0;i<Nnodi;i++){
	    nodi.push_back(g.addNode());
	    if(g.id(nodi[nodi.size()-1])!=nodi.size()-1) {cout<< "Problemi con indicizzazione vertici."<< endl; exit(-1);}
    }
    /*Creo gli archi*/ 
    
    for(int i=0;i<N;i++){
        for (int ip=0;ip<N;ip++){
        Graph::Edge e=g.addEdge(nodi[2*i],nodi[2*ip+1]); //Adding edge between even and odd nodes
    			edgeMap[e]=weights[N*i+ip];//-generator.Log();//Aggiungo peso su edge CAMBIATO DI SEGNO
        }
    }

    bool ok=matching.run();//running the algorithm
    if(!ok){ 
        cerr << "WARNING: matching not found!"<<endl;
        for(i=0;i<N;i++){
            max_match[i]=0;//returning a default constant value
            failed_matching=1;
        }
    }
    if(failed_matching==0){
        for(NodeIt n(g); n!=INVALID; ++n){
            if(g.id(n)%2==0){
                max_match[(int)round(g.id(n)/2)]=(int)round((g.id(matching.mate(n))-1)/2);
            }
        }
    }
    /*
    for(NodeIt n(g); n!=INVALID; ++n){
        if(g.id(n)%2==0){
            cout<<round(g.id(n)/2)<<" "<<round((g.id(matching.mate(n))-1)/2)<< endl;
        }
    }
    for(EdgeIt e(g); e!=INVALID; ++e){
	    if(matching.matching(e)){
		    cout << "Accoppiati " << g.id(g.u(e)) << " e " << g.id(g.v(e)) <<" a costo " << -edgeMap[e]<< endl;
	    }
    }
    //cout << "Costo match "<< -double(matching.matchingWeight())<< endl;
    */
    return failed_matching;
}


//this function takes 'array' sorts it and then returns the values corresponding to the set of indices in 'ranks'. THe values are then stored in 'out_ranks' in the same order as the respective element in 'ranks'
void find_rankings(int* ranks,int ranks_length, double* array ,int array_length, double* out_ranks){
    double* temp_array;
    temp_array=(double*) calloc(array_length,sizeof(double));
    copy_array(array,temp_array,array_length);
    std::sort(temp_array,temp_array+array_length);
    for(int i=0;i<ranks_length;i++){
        out_ranks[i]=temp_array[ranks[i]];
    }
    free(temp_array);
}

void copy_array(double* source, double* destination, int len_source){
    for(int i=0;i<len_source;i++){
        destination[i]=source[i];
    }
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
void argmax_set(double * array,int array_len, int* argmax){//argmax is a binary array with the same dimension as array. argmax[i]=1 if array[i]>=array[j] for all j. 
    double temp_max;
    int i;
    temp_max=array[0];
    for(i=0;i<N;i++){
        argmax[i]=0;
        if(array[i]>temp_max){
            temp_max=array[i];
        }
    }
    for(i=0;i<N;i++){
        if(array[i]==temp_max){
            argmax[i]=1;
        }
    }
}

double error_edge_set(int ** edge_set, int N){
    int i;
    double error=0;
    for (i=0;i<N;i++){
        if(edge_set[i][i]==0){
            error+=1.;
        }
    }
    error=error/N;
    return error;
}

double avg_perm_overlap_edge_set(int ** edge_set, int N){//edge_set is an N x N 0/1 matrix . An error of 1/N is added each time  edge_set[i][i]=0. An error (1-1/|S(i)|)/N is counted every time edge_set[i][i]=1. S(i)={ip\in [N] s.t. edge_set[i][ip]=1} 
    int i,ip,cardinality;
    double overlap=0;
    for (i=0;i<N;i++){
        if(edge_set[i][i]==1){
            cardinality=0;
            for(ip=0;ip<N;ip++){
                cardinality+=edge_set[i][ip];
            }
            overlap+=(1./cardinality);
        }
    }
    overlap=overlap/N;
    return overlap;
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

//finds all the max and argmax in the array and stores them in the two pointers given as arguments. degeneracy indicates the number of velues inthe array which attain the maximum value
//the arrays max and argmax should be appropirately initialized to recognize when a value is not assigned
int maxes_argmaxes(double* array,int array_len, double* max, int* argmax){
  int degeneracy=0;
  double temp_max=array[0];
  for(int i=1; i<array_len;i++){
    if(*(array+i)>temp_max)
      {
        temp_max=array[i];
      }
   }
   for(int i=1; i<array_len;i++){
     if(*(array+i)==temp_max){
       argmax[degeneracy]=i;
       degeneracy++;
     }
   } 
    *max = temp_max;
    return degeneracy;
}
//this could be optimized if necessary 
void NTMA2(int N, double* weights, int* partial_match, double threshold){//implements the NTMA-2 algorithm (algorithm 4 in 2002.01258, threshold=gamma^(d-1))
    int i,ip,j,jp;
    int flag;
    int flag_duplicate_i;
    int flag_duplicate_ip;
    int count_edges=0; //number fo edges added so far to the match
    double max;
    
    int* temp_match [2];
    temp_match[0]=(int*) calloc(N*N,sizeof(int));//in the worst case I add all edges to the list
    temp_match[1]=(int*) calloc(N*N,sizeof(int));
    
    
    for(i=0;i<N*N;i++){//initializing to the empty set
        temp_match[0][i]=-1;
        temp_match[1][i]=-1;
    }
    
    for(i=0;i<N;i++){
        partial_match[i]=-1;//initializing the match: -1 means the node is not matched
        for(ip=0;ip<N;ip++){
        
            if(weights[N*i+ip]>=threshold){
            
                flag=0;//set to one if weigth[N*i+ip] is not the max along i
                
                for(j=0;j<N;j++){
                    if(weights[N*i+ip]<weights[N*j+ip] || weights[N*i+ip]<weights[N*i+j]){//check if maximum over row and column
                        flag=1;
                        break;
                    }
                }
                if(flag==0){
                    temp_match[0][count_edges]=i;
                    temp_match[1][count_edges]=ip;
                    count_edges++;
                }
            }
        }
    }
    //now remove edges in excess to obtain a matching
    for (i=0;i<count_edges;i++){
        flag_duplicate_i=0;
        flag_duplicate_ip=0;
        
        for(j=0;j<count_edges;j++){
            //detecting duplicates across i
            if(j!=i && temp_match[0][i]==temp_match[0][j] && temp_match[0][i]!=-1){
                flag_duplicate_i=1;
                temp_match[0][j]=-1;
            }
            //detecting duplicates across ip
            if(j!=i && temp_match[1][i]==temp_match[1][j] && temp_match[1][i]!=-1){
                flag_duplicate_ip=1;
                temp_match[1][j]=-1; 
            }
        }
        //it's important to eliminate separately the duplicates in the base and target nodes
        if(flag_duplicate_i==1){
            temp_match[0][i]=-1;
        }
        if(flag_duplicate_ip==1){
            temp_match[1][i]=-1;
        }
        
    }
    for(i=0;i<count_edges;i++){
        if (temp_match[0][i]==-1 || temp_match[1][i]==-1){
            temp_match[1][i]=-1;
            temp_match[0][i]=-1;
        }
    }
    //...finished removing duplicates
    
    //assigning the remaining edges
    for(i=0;i<count_edges;i++){
        if(temp_match[0][i]!=-1){
            partial_match[temp_match[0][i]]=temp_match[1][i];
        }
    }
    free(temp_match[0]);
    free(temp_match[1]);
}

void compute_min_avg_dist_estimator(int N, int ** distance_matrix_B, int* min_avg_dist_estimator, double* expected__mindist_estimator_distances_connect,double* expect_probs_disconnect_mindist, double* msg_probabilities){
    //disconnected vertices count as placed at infinity, so the optimization is carried out in two steps. First the vertces with minimum probability outside of the componet are selected, then among these those with minimal average distance within the component are taken.
    //problem with this is that it's very rare that will be ties among vertices after optimizing in the first step. So it will only look at the probability of falling outside the component and try to minimize that one
    //
    int i,jp,ip;
    double min_expect_dist_connect;
    double min_prob_disconnect;
    int argmin_expect_dist;
    
    double tmp_expect_dist_connect;
    double tmp_prob_disconnect;
    
    for(i=0;i<N;i++){
        min_expect_dist_connect=DBL_MAX;
        min_prob_disconnect=DBL_MAX;
        for(jp=0;jp<N;jp++){ //in the end we minimize over jp
            tmp_expect_dist_connect=0;
            tmp_prob_disconnect=0;
            for(ip=0;ip<N;ip++){
                if(distance_matrix_B[jp][ip]>=0){
                    tmp_expect_dist_connect+=distance_matrix_B[jp][ip]*msg_probabilities[N*i+ip];
                }
                else{
                    tmp_prob_disconnect+=msg_probabilities[N*i+ip];
                }
            }
            //condition for updating the minimum
            if((tmp_prob_disconnect<min_prob_disconnect) || (tmp_prob_disconnect==min_prob_disconnect && tmp_expect_dist_connect<min_expect_dist_connect)){
                min_expect_dist_connect=tmp_expect_dist_connect;
                min_prob_disconnect=tmp_prob_disconnect;
                argmin_expect_dist=jp;
            }
        }
        min_avg_dist_estimator[i]=argmin_expect_dist;
        expected_mindist_estimator_distances_connect[i]=min_expect_dist_connect/(1.-min_prob_disconnect); //normalizing to get an expectation conditional on the fact that they belong to the same component
        expect_probs_disconnect_mindist[i]=min_prob_disconnect;
    }  
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

double injectivity_violation(int* perm, int N){
    double inj_viol=0;
    int i,j;
    for(i=0;i<N;i++){
        for(j=0;j<i;j++){
            if(perm[i]==perm[j]){
                inj_viol++;
                break;
            }
        }
    }
    return inj_viol/N;
}

int edge_overlap(int* perm,int N,double** B, int **neighborhood_A, int* degree_A){//computes the number of edges in the intersection graph. Works also if perm is not a permutation, in that case it outputs the number of common  
    int perm_edge_overlap=0;
    for(int i=0; i<N; i++){ 
        for(int j=0;j<degree_A[i];j++){//looping over neighbours of i
             perm_edge_overlap+=B[perm[i]][perm[neighborhood_A[i][j]]];
        }
    }
    return perm_edge_overlap/2;//this is the total number of common edges
}

void print_observables(int depth){
    int j;
    printf("depth=%d lambda=%.2f time=%.3fs overflow=%d fail_match=%d %d max_match_err=%.3f %.3f avg_max_match_score=%1.3e %1.3e\n",depth,ER_l,((double)(clock()-start_time))/CLOCKS_PER_SEC,overflow_flag,failed_matching,failed_exp_matching,max_bip_match_error,exp_max_bip_match_error, avg_max_match_score, log_avg_exp_max_match_exp_score);
    printf("argmax_score_err=%.3f argmax_score_inj_viol=%.3f avg_argmax_score(exp)=%1.3e %1.3e argmax_edge_ovlap=%.3f\n", argmax_score_error, argmax_score_injectivity_violation, avg_argmax_score, log_avg_argmax_exp_score,argmax_score_edge_ovlap);
    printf("prob_on_ident= %.3f avg_ident_score=%1.3e avg_score=%1.3e\n",avg_prob_on_identity, avg_score_on_identity, avg_score);
    printf("ntma2_frac_good/bad=%.3f %.3f ntma2_score_good/bad=%3.5e %3.5e\n",NTMA2_frac_good_matches, NTMA2_frac_bad_matches,NTMA2_score_good_matches, NTMA2_score_bad_matches);
    //print max match
    printf("max_match     %d %d %d %d %d\n", max_bip_match_perm[0], max_bip_match_perm[1], max_bip_match_perm[2], max_bip_match_perm[3], max_bip_match_perm[4]);
    printf("exp_max_match %d %d %d %d %d\n",exp_max_bip_match_perm[0], exp_max_bip_match_perm[1], exp_max_bip_match_perm[2], exp_max_bip_match_perm[3], exp_max_bip_match_perm[4]);
    printf("argmax score  %d %d %d %d %d\n", argmax_score[0], argmax_score[1], argmax_score[2], argmax_score[3], argmax_score[4]);
    printf("max_score %.2f %.2f %.2f %.2f %.2f\n",max_score[0],max_score[1], max_score[2], max_score[3], max_score[4]);
    printf("R_00 -- R_04 %1.3e %1.3e %1.3e %1.3e %1.3e\n",msg_marginals[0],msg_marginals[1],msg_marginals[2],msg_marginals[3],msg_marginals[4]);
    printf("rand_ovlap=%.3f set_ovlap=%.3f unif_ovlap=%.3f argmax_degen=%.3f frac_argmax_score_degen_coords=%.5f\n",1-argmax_score_error,1-argmax_score_edge_set_error,argmax_score_avg_edge_set_overlap,avg_argmax_score_degeneracy,frac_argmax_score_degen_coords);
    printf("rand_vertex_comp/tree_size= %.3f %.3f correct_vertex_comp/tree_size= %.3f %.3f\n",comp_size_rand_vertex,tree_size_rand_vertex,comp_size_correct_vertex,tree_size_correct_vertex);
    //good quantiles
    /*
    printf("quant true\n");
    for(j=0;j<N_QUANTILES;j++){
        printf("%1.4e ",quantiles_true[j]);
    }
    printf("\n");
    
    printf("rank bad\n");
    for(j=0;j<N_QUANTILES;j++){
        printf("%1.4e ",bad_ranks[j]);
    }
    printf("\n");
    
    printf("quant false\n");
    for(j=0;j<N_QUANTILES;j++){
        printf("%1.4e ",quantiles_false[j]);
    }
    printf("\n"); 
    */
    printf("\n");  
    fflush(stdout);
}
void write_header(FILE* ofile){
    int i;
    
    fseek(ofile, 0L, SEEK_END);
    int size = ftell(ofile);
    if(size==0){
        fprintf(ofile,"N lambda s seed depth time overflow_flag fail_match fail_exp_match max_match_err exp_max_match_err max_match_edge_overlap exp_max_match_edge_overlap argmax_score_error argmax_score_inj_viol ntma2_frac_good_matches ntma2_frac_bad_matches ntma2_score_good_matches ntma2_score_bad_matches avg_max_match_score log_avg_exp_max_match_exp_score avg_argmax_score log_avg_argmax_exp_score avg_prob_argmax argmax_score_edge_ovlap mean_tree_depth std_tree_depth avg_prob_on_identity log_avg_exp_score_on_identity avg_score_on_identity log_avg_exp_score avg_score std_score_on_identity argmax_score_edge_set_error argmax_score_avg_edge_set_overlap avg_argmax_score_degeneracy frac_argmax_score_degen_coords comp_size_rand_vertex tree_size_rand_vertex comp_size_correct_vertex tree_size_correct_vertex tree_depth_rand_vertex tree_depth_correct_vertex avg_degree_correct_vertex max_degree_A max_degree_B matrix_estimator_false_positives matrix_estimator_false_negatives matrix_estimator_error matrix_estimator_frac_assigned_vertices matrix_estimator_error_assigned_vertices avg_degree_A_correct_vertex avg_degree_minAB_correct_vertex avg_degree_matrix_correct_vertex avg_degree_A_matrix_correct_vertex avg_degree_minAB_matrix_correct_vertex avg_score_neighbors_B_planted avg_score_next_neighbors_A_B_planted avg_dist_incorrect_vertices_argmax_est_connect frac_incorrect_vertices_argmax_est_disconnect avg_dist_vertices_mindist_est_connect frac_vertices_mindist_est_disconnect error_mindist_est avg_dist_random_perm_connect frac_vertices_disconnect_rand_perm avg_est_dist_mindist_est_connect est_prob_disconnect_vertices_mindist_est avg_cross_ent_argmax_estimator avg_brier_score_argmax_estimator avg_sum_square_marginals");
        /*
        for(i=0;i<N_QUANTILES;i++){
            fprintf(ofile," q_true_%.2f",quantiles[i]);
        }
        for(i=0;i<N_QUANTILES;i++){
            fprintf(ofile," ranks_false_%.2f",quantiles[i]);
        }
        for(i=0;i<N_QUANTILES;i++){
            fprintf(ofile," q_false_%.2f",quantiles[i]);
        }
        */
        fprintf(ofile,"\n");
    }
    fflush(ofile);
}
    

void write_results_file(FILE* ofile, int current_depth){
    int j;
    double std_tree_depth;
    double std_score_on_identity;
    std_tree_depth=sqrt( (double) (N*N)/(N*N-1)*(mean_square_tree_depth-(mean_tree_depth*mean_tree_depth))  ); 
    std_score_on_identity=sqrt( (double) (N)/(N-1)*(avg_square_score_on_identity-(avg_score_on_identity*avg_score_on_identity)));                                                                                                                                                                                                                                                       //%d  %.5lf  %.5lf  %d      %d                    %.3lf                                     %d                %d              %d                  %.5lf                  %.5lf                         %.6lf                               %.6lf                                        %.5lf                       %.5lf                         %.5lf                     %.5lf                  %3.5e                       %3.5e                  %3.5e                    %3.5e                     %3.5e                  %3.5e                 %.5lf               %.5lf                                    %.5f              %.5f               %.5f                    %3.5e                     %3.5e                   %3.5e        %3.5e            %3.5e                     %.5f                           %.5f                            %.5f                        %.5f                        %.5f                   %.5f                     %.5f                    %.5f                      %.5f                      %.5f                        %.5f                %d              %d                    %.5f                             %.5f                         %.5f                               %.5f                                    %.5f                                %.5f                        %.5f                               %.5f                                  5.5f                                 %.5f                              %.5f                              %.5f                                    %.5f                                          %.5f                                        %.5f                                    %.5f                          %.5f                    %.5f                          %.5f                                   %.5f                               %.5f                                   %.5f                            %.5f                            %.5f
    fprintf(ofile,"%d %.5lf %.5lf %d %d %.3lf %d %d %d %.5lf %.5lf %.6lf %.6lf %.5lf %.5lf %.5lf %.5lf %3.5e %3.5e %3.5e %3.5e %3.5e %3.5e %.5lf %.5lf %.5f %.5f %.5f %3.5e %3.5e %3.5e %3.5e %3.5e %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %d %d %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f", N, ER_l, ER_s, jseed, current_depth,((double)(clock()-start_time))/CLOCKS_PER_SEC, overflow_flag, failed_matching, failed_exp_matching, max_bip_match_error, exp_max_bip_match_error, max_bip_match_edge_ovlap/true_edge_overlap, exp_max_bip_match_edge_ovlap/true_edge_overlap, argmax_score_error, argmax_score_injectivity_violation, NTMA2_frac_good_matches, NTMA2_frac_bad_matches,NTMA2_score_good_matches, NTMA2_score_bad_matches, avg_max_match_score, log_avg_exp_max_match_exp_score, avg_argmax_score, log_avg_argmax_exp_score, avg_prob_argmax, argmax_score_edge_ovlap/true_edge_overlap, mean_tree_depth, std_tree_depth, avg_prob_on_identity, log_avg_exp_score_on_identity, avg_score_on_identity, log_avg_exp_score, avg_score,std_score_on_identity,argmax_score_edge_set_error, argmax_score_avg_edge_set_overlap, avg_argmax_score_degeneracy, frac_argmax_score_degen_coords, comp_size_rand_vertex, tree_size_rand_vertex, comp_size_correct_vertex, tree_size_correct_vertex, tree_depth_rand_vertex, tree_depth_correct_vertex, avg_degree_correct_vertex, max_degree_A, max_degree_B, matrix_estimator_false_positives, matrix_estimator_false_negatives, matrix_estimator_error, matrix_estimator_frac_assigned_vertices, matrix_estimator_error_assigned_vertices, avg_degree_A_correct_vertex, avg_degree_minAB_correct_vertex, avg_degree_matrix_correct_vertex, avg_degree_A_matrix_correct_vertex, avg_degree_minAB_matrix_correct_vertex, avg_score_neighbors_B_planted, avg_score_next_neighbors_A_B_planted, avg_dist_incorrect_vertices_argmax_est_connect, frac_incorrect_vertices_argmax_est_disconnect, avg_dist_vertices_mindist_est_connect, frac_vertices_mindist_est_disconnect, error_mindist_est, avg_dist_random_perm_connect, frac_vertices_disconnect_rand_perm, avg_est_dist_mindist_est_connect, est_prob_disconnect_vertices_mindist_est, avg_cross_ent_argmax_estimator, avg_brier_score_argmax_estimator, avg_sum_square_marginals);         
                                                                                                                                                                                                    // "N lambda s seed depth time overflow_flag fail_match fail_exp_match max_match_err exp_max_match_err max_match_edge_overlap exp_max_match_edge_overlap argmax_score_error argmax_score_inj_viol ntma2_frac_good_matches ntma2_frac_bad_matches ntma2_score_good_matches ntma2_score_bad_matches avg_max_match_score log_avg_exp_max_match_exp_score avg_argmax_score log_avg_argmax_exp_score avg_prob_argmax argmax_score_edge_ovlap mean_tree_depth std_tree_depth avg_prob_on_identity log_avg_exp_score_on_identity avg_score_on_identity log_avg_exp_score avg_score
    /*
    for(j=0;j<N_QUANTILES;j++){
        fprintf(outfile," %.6f",log(quantiles_true[j]));
    }
    
    for(j=0;j<N_QUANTILES;j++){
        fprintf(outfile," %.6f",log(bad_ranks[j]));
    }

    for(j=0;j<N_QUANTILES;j++){
        fprintf(outfile," %.6f",log(quantiles_false[j]));
    }
    fprintf(outfile,"\n"); 
    */
    /*
    for(j=0;j<N_QUANTILES;j++){
        fprintf(ofile," %3.5e",quantiles_true[j]);
    }
    
    for(j=0;j<N_QUANTILES;j++){
        fprintf(ofile," %3.5e",bad_ranks[j]);
    }

    for(j=0;j<N_QUANTILES;j++){
        fprintf(ofile," %3.5e",quantiles_false[j]);
    }
    */
    fprintf(ofile,"\n"); 
    fflush(ofile);
}

/*******************/
/* Routines Andrea */
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
