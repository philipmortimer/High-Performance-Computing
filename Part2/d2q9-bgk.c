/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>
#include<string.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

#define ALIGN 64
#define GRID_EL_DIV 128

#define TIMESTEPLOOP(ii, jj) \
    do { \
      int ind = ii + jj*params.nx; \
      int y_n = jj + 1; \
      int x_e = (ii + 1 < params.nx)? (ii + 1) : (0); \
      int y_s = jj - 1; \
      int x_w = (ii > 0) ? (ii - 1) : (params.nx - 1); \
      tmp_cells_speed0[ind] = cells_speed0[ind]; /* central cell, no movement */ \
      tmp_cells_speed1[ind] = cells_speed1[x_w + jj*params.nx]; /* east */ \
      tmp_cells_speed2[ind] = cells_speed2[ii + y_s*params.nx]; /* north */ \
      tmp_cells_speed3[ind] = cells_speed3[x_e + jj*params.nx]; /* west */ \
      tmp_cells_speed4[ind] = cells_speed4[ii + y_n*params.nx]; /* south */ \
      tmp_cells_speed5[ind] = cells_speed5[x_w + y_s*params.nx]; /* north-east */ \
      tmp_cells_speed6[ind] = cells_speed6[x_e + y_s*params.nx]; /* north-west */ \
      tmp_cells_speed7[ind] = cells_speed7[x_e + y_n*params.nx]; /* south-west */ \
      tmp_cells_speed8[ind] = cells_speed8[x_w + y_n*params.nx]; /* south-east */ \
      if (!obstacles[ind - params.nx]) \
      { \
        float local_density = tmp_cells_speed0[ind] + tmp_cells_speed1[ind] + tmp_cells_speed2[ind] + \
        tmp_cells_speed3[ind] + tmp_cells_speed4[ind] + tmp_cells_speed5[ind] + tmp_cells_speed6[ind] + \
        tmp_cells_speed7[ind] + tmp_cells_speed8[ind]; \
        float u_x = (tmp_cells_speed1[ind] \
                      + tmp_cells_speed5[ind] \
                      + tmp_cells_speed8[ind] \
                      - (tmp_cells_speed3[ind] \
                          + tmp_cells_speed6[ind] \
                          + tmp_cells_speed7[ind])) \
                      / local_density; \
        float u_y = (tmp_cells_speed2[ind] \
                      + tmp_cells_speed5[ind] \
                      + tmp_cells_speed6[ind] \
                      - (tmp_cells_speed4[ind] \
                          + tmp_cells_speed7[ind] \
                          + tmp_cells_speed8[ind])) \
                      / local_density; \
        float u_x_squared = u_x * u_x; \
        float u_y_squared = u_y * u_y; \
        float u_sq = u_x_squared + u_y_squared; \
        float u_5 =   u_x + u_y;  /* north-east */ \
        float u_6 = - u_x + u_y;  /* north-west */ \
        float u_7 = - u_x - u_y;  /* south-west */ \
        float u_8 =   u_x - u_y;  /* south-east */ \
        float d_equ_0 = w0 * local_density \
                    * (1.f - u_sq * rec_c_sq_2); \
        float w1_dens_div_2_c_sq_square_dens = w1_div_2_c_sq_square * local_density; \
        float acc = u_sq * neg_c_sq + c_sq_squared_2; \
        float acc_x = acc + u_x_squared; \
        float acc_y = acc + u_y_squared; \
        float u_x_c_sq_2 = u_x * c_sq_2; \
        float u_y_c_sq_2 = u_y * c_sq_2; \
        float d_equ_1 = w1_dens_div_2_c_sq_square_dens * (acc_x + u_x_c_sq_2); \
        float d_equ_2 = w1_dens_div_2_c_sq_square_dens * (acc_y + u_y_c_sq_2); \
        float d_equ_3 = w1_dens_div_2_c_sq_square_dens * (acc_x - u_x_c_sq_2); \
        float d_equ_4 = w1_dens_div_2_c_sq_square_dens * (acc_y - u_y_c_sq_2); \
        float w2_dens_div_2_c_sq_square_dens = w2_div_2_c_sq_square * local_density; \
        float acc_seq = acc + c_sq_2; \
        float d_equ_5 = w2_dens_div_2_c_sq_square_dens * (u_5 * (c_sq_2 + u_5) + acc); \
        float d_equ_6 = w2_dens_div_2_c_sq_square_dens * (u_6 * (c_sq_2 + u_6) + acc); \
        float d_equ_7 = w2_dens_div_2_c_sq_square_dens * (u_7 * (c_sq_2 + u_7) + acc); \
        float d_equ_8 = w2_dens_div_2_c_sq_square_dens * (u_8 * (c_sq_2 + u_8) + acc);  \
        tmp_cells_speed0[ind] = tmp_cells_speed0[ind] + params.omega * (d_equ_0 - tmp_cells_speed0[ind]); \
        tmp_cells_speed1[ind] = tmp_cells_speed1[ind] + params.omega * (d_equ_1 - tmp_cells_speed1[ind]); \
        tmp_cells_speed2[ind] = tmp_cells_speed2[ind] + params.omega * (d_equ_2 - tmp_cells_speed2[ind]); \
        tmp_cells_speed3[ind] = tmp_cells_speed3[ind] + params.omega * (d_equ_3 - tmp_cells_speed3[ind]); \
        tmp_cells_speed4[ind] = tmp_cells_speed4[ind] + params.omega * (d_equ_4 - tmp_cells_speed4[ind]); \
        tmp_cells_speed5[ind] = tmp_cells_speed5[ind] + params.omega * (d_equ_5 - tmp_cells_speed5[ind]); \
        tmp_cells_speed6[ind] = tmp_cells_speed6[ind] + params.omega * (d_equ_6 - tmp_cells_speed6[ind]); \
        tmp_cells_speed7[ind] = tmp_cells_speed7[ind] + params.omega * (d_equ_7 - tmp_cells_speed7[ind]); \
        tmp_cells_speed8[ind] = tmp_cells_speed8[ind] + params.omega * (d_equ_8 - tmp_cells_speed8[ind]); \
        tot_u += sqrtf(u_sq); \
        tot_cells++; \
      } else { /* if the cell contains an obstacle */ \
        float vals_0 = tmp_cells_speed1[ind]; \
        float vals_1 = tmp_cells_speed2[ind]; \
        float vals_2 = tmp_cells_speed3[ind]; \
        float vals_3 = tmp_cells_speed4[ind]; \
        float vals_4 = tmp_cells_speed5[ind]; \
        float vals_5 = tmp_cells_speed6[ind]; \
        float vals_6 = tmp_cells_speed7[ind]; \
        float vals_7 = tmp_cells_speed8[ind]; \
        tmp_cells_speed1[ind] = vals_2; \
        tmp_cells_speed2[ind] = vals_3; \
        tmp_cells_speed3[ind] = vals_0; \
        tmp_cells_speed4[ind] = vals_1; \
        tmp_cells_speed5[ind] = vals_6; \
        tmp_cells_speed6[ind] = vals_7; \
        tmp_cells_speed7[ind] = vals_4; \
        tmp_cells_speed8[ind] = vals_5; \
      } \
    }while(0)

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

typedef struct
{
  int noCharsPrev;
  int noCharsFromLowToUp;
} t_no_chars;

typedef struct
{
  float* restrict speed0;
  float* restrict speed1;
  float* restrict speed2;
  float* restrict speed3;
  float* restrict speed4;
  float* restrict speed5;
  float* restrict speed6;
  float* restrict speed7;
  float* restrict speed8;
} t_speed_vec;

typedef struct {
  int nprocs;
  int rank;
  int startY;
  int endYNonInc;
  int nrows;
} t_grid_bounds;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_vec* cells_ptr, t_speed_vec* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int** tot_cells_ptr, t_grid_bounds bounds,
               float** global_av_vels_ptr, int**  global_tot_cells_ptr);
int initParams(const char* paramfile, t_param* params);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep_align(const t_param params,
  float*restrict cells_speed0,float*restrict cells_speed1,float*restrict cells_speed2,float*restrict cells_speed3,float*restrict cells_speed4,float*restrict cells_speed5,float*restrict cells_speed6,float*restrict cells_speed7,float*restrict cells_speed8,
  float*restrict tmp_cells_speed0,float*restrict tmp_cells_speed1,float*restrict tmp_cells_speed2,float*restrict tmp_cells_speed3,float*restrict tmp_cells_speed4,float*restrict tmp_cells_speed5,float*restrict tmp_cells_speed6,float*restrict tmp_cells_speed7,float*restrict tmp_cells_speed8,
  const int* obstacles,
  float* av_vel_ret, int* tot_cells_ret, const t_grid_bounds bounds);

int accelerate_flow_align(const t_param params,float* speed1,float* speed3,float* speed5,float* speed6,float* speed7,float* speed8, const int* obstacles, const t_grid_bounds bounds);

int write_values(const t_param params, t_speed_vec cells, int* obstacles, float* av_vels, const t_grid_bounds bounds);
t_grid_bounds splitGrid(int nprocs, int rank, int ny);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed_vec* cells, t_speed_vec* tmp_cells,
             int** obstacles_ptr, float** av_vels_ptr,
            int** tot_cells_ptr, t_grid_bounds bounds,
            float** global_av_vels_ptr, int**  global_tot_cells_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed_vec cells, const t_grid_bounds bounds);

/* compute average velocity */
float av_velocity(const t_param params, t_speed_vec cells, int* obstacles, const t_grid_bounds bounds);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed_vec cells, int* obstacles, const t_grid_bounds bounds);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int nprocs, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */

  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  int* tot_cells = NULL;
  float* global_av_vels   = NULL;     // Used to store reduced results of av_vels
  int* global_tot_cells = NULL;
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  initParams(paramfile, &params);

  t_speed_vec* cells     = _mm_malloc(sizeof(t_speed_vec),ALIGN);    /* grid containing fluid densities */
  t_speed_vec* tmp_cells = _mm_malloc(sizeof(t_speed_vec),ALIGN);    /* scratch space */
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  t_grid_bounds bounds = splitGrid(nprocs, rank, params.ny);
  initialise(paramfile, obstaclefile, &params, cells, tmp_cells, &obstacles, &av_vels, &tot_cells, bounds, &global_av_vels, &global_tot_cells);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    timestep_align(params, 
      cells->speed0,cells->speed1,cells->speed2,cells->speed3,cells->speed4,cells->speed5,cells->speed6,cells->speed7,cells->speed8, 
      tmp_cells->speed0,tmp_cells->speed1,tmp_cells->speed2,tmp_cells->speed3,tmp_cells->speed4,tmp_cells->speed5,tmp_cells->speed6,tmp_cells->speed7,tmp_cells->speed8,
      obstacles,
      &av_vels[tt], &tot_cells[tt], bounds);
    // Swaps pointers
    t_speed_vec* swap = tmp_cells;
    tmp_cells = cells;
    cells = swap;
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, *cells, bounds));
#endif
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collates and calculates average data from ranks
  MPI_Reduce(av_vels, global_av_vels, params.maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(tot_cells, global_tot_cells, params.maxIters, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  if(rank == 0){
    // Computes averages
    #pragma omp simd aligned(global_av_vels:ALIGN, global_tot_cells:ALIGN)
    for(int j = 0; j < params.maxIters; j++) {
      global_av_vels[j] = global_av_vels[j] / ((float)(global_tot_cells[j]));
    } 
  }

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  if(rank == 0) {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, *cells, obstacles, bounds));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  }
  write_values(params, *cells, obstacles, global_av_vels, bounds);
  finalise(&params, cells, tmp_cells, &obstacles, &av_vels, &tot_cells, bounds, &global_av_vels, &global_tot_cells);

  MPI_Finalize();
}


float timestep_align(const t_param params,
  float*restrict cells_speed0,float*restrict cells_speed1,float*restrict cells_speed2,float*restrict cells_speed3,float*restrict cells_speed4,float*restrict cells_speed5,float*restrict cells_speed6,float*restrict cells_speed7,float*restrict cells_speed8,
  float*restrict tmp_cells_speed0,float*restrict tmp_cells_speed1,float*restrict tmp_cells_speed2,float*restrict tmp_cells_speed3,float*restrict tmp_cells_speed4,float*restrict tmp_cells_speed5,float*restrict tmp_cells_speed6,float*restrict tmp_cells_speed7,float*restrict tmp_cells_speed8,
  const int* obstacles,
  float* av_vel_ret, int* tot_cells_ret, const t_grid_bounds bounds)
{
  __assume_aligned(cells_speed0,ALIGN);__assume_aligned(cells_speed1,ALIGN);__assume_aligned(cells_speed2,ALIGN);
  __assume_aligned(cells_speed3,ALIGN);__assume_aligned(cells_speed4,ALIGN);__assume_aligned(cells_speed5,ALIGN);
  __assume_aligned(cells_speed6,ALIGN);__assume_aligned(cells_speed7,ALIGN);__assume_aligned(cells_speed8,ALIGN);

  __assume_aligned(tmp_cells_speed0,ALIGN);__assume_aligned(tmp_cells_speed1,ALIGN);__assume_aligned(tmp_cells_speed2,ALIGN);
  __assume_aligned(tmp_cells_speed3,ALIGN);__assume_aligned(tmp_cells_speed4,ALIGN);__assume_aligned(tmp_cells_speed5,ALIGN);
  __assume_aligned(tmp_cells_speed6,ALIGN);__assume_aligned(tmp_cells_speed7,ALIGN);__assume_aligned(tmp_cells_speed8,ALIGN);

  __assume_aligned(obstacles, ALIGN);
  __assume(params.nx%GRID_EL_DIV==0); __assume(params.ny%GRID_EL_DIV==0);
  __assume(params.nx>=128); __assume(params.ny>=128);

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  const float rec_c_sq = 1.f / c_sq;
  const float rec_c_sq_2 = 1.f / (2.f * c_sq);
  const float w1_div_2_c_sq_square = w1 / (2.f * c_sq * c_sq);
  const float w2_div_2_c_sq_square = w2 / (2.f * c_sq * c_sq);
  const float c_sq_squared_2 = 2.f * c_sq * c_sq;
  const float c_sq_2 = 2.f * c_sq;
  const float neg_c_sq = -c_sq;

  float tot_u = 0.f;
  int tot_cells = 0;

  // Splits grid for each process
  int nRows = bounds.nrows;
  int startY = bounds.startY;
  int endYNonInc = bounds.endYNonInc;
  int nprocs = bounds.nprocs;
  int rank = bounds.rank;

  // Accelerate flow only needed if rank deals with y = params.ny - 2
  if (startY <= params.ny - 2 && endYNonInc > params.ny - 2){
    accelerate_flow_align(params,cells_speed1,cells_speed3,cells_speed5,cells_speed6,cells_speed7,cells_speed8, obstacles, bounds);
  }

  // Halo exchange
  int rankBelow = (rank + 1) % nprocs;
  int rankAbove = rank - 1 >= 0? rank - 1 : nprocs - 1;
  int ghostRowAboveStartIndex = 0;
  int ghostRowBelowStartIndex = (bounds.nrows + 1) * params.nx;
  MPI_Request receive_requests[6];
  MPI_Request send_requests[6];
  // Sends speed2, speed5 and speed6 to rank below
  MPI_Isend(&cells_speed2[nRows*params.nx], params.nx, MPI_FLOAT, rankBelow, 2,
              MPI_COMM_WORLD, &send_requests[0]);
  MPI_Isend(&cells_speed5[nRows*params.nx], params.nx, MPI_FLOAT, rankBelow, 5,
              MPI_COMM_WORLD, &send_requests[1]);
  MPI_Isend(&cells_speed6[nRows*params.nx], params.nx, MPI_FLOAT, rankBelow, 6,
              MPI_COMM_WORLD, &send_requests[2]);
  // Sends speed4, speed7 and speed8 to rank above
  MPI_Isend(&cells_speed4[params.nx], params.nx, MPI_FLOAT, rankAbove, 4,
              MPI_COMM_WORLD, &send_requests[3]);
  MPI_Isend(&cells_speed7[params.nx], params.nx, MPI_FLOAT, rankAbove, 7,
              MPI_COMM_WORLD, &send_requests[4]);
  MPI_Isend(&cells_speed8[params.nx], params.nx, MPI_FLOAT, rankAbove, 8,
              MPI_COMM_WORLD, &send_requests[5]);
  // Recieves speed2, speed5 and speed6 from rank above
  MPI_Irecv(&cells_speed2[ghostRowAboveStartIndex], params.nx, MPI_FLOAT, rankAbove,
              2, MPI_COMM_WORLD, &receive_requests[0]);
  MPI_Irecv(&cells_speed5[ghostRowAboveStartIndex], params.nx, MPI_FLOAT, rankAbove,
              5, MPI_COMM_WORLD, &receive_requests[1]);
  MPI_Irecv(&cells_speed6[ghostRowAboveStartIndex], params.nx, MPI_FLOAT, rankAbove,
              6, MPI_COMM_WORLD, &receive_requests[2]);
  // Receives speed4, speed7 and speed8 from rank below
  MPI_Irecv(&cells_speed4[ghostRowBelowStartIndex], params.nx, MPI_FLOAT, rankBelow,
              4, MPI_COMM_WORLD, &receive_requests[3]);
  MPI_Irecv(&cells_speed7[ghostRowBelowStartIndex], params.nx, MPI_FLOAT, rankBelow,
              7, MPI_COMM_WORLD, &receive_requests[4]);
  MPI_Irecv(&cells_speed8[ghostRowBelowStartIndex], params.nx, MPI_FLOAT, rankBelow,
              8, MPI_COMM_WORLD, &receive_requests[5]);
 
  // Computes grid cells not impacted by Halo cells 
  for (int jj = 2; jj < bounds.nrows; jj++)
  {
    #pragma omp simd aligned(cells_speed0:ALIGN,cells_speed1:ALIGN,cells_speed2:ALIGN,cells_speed3:ALIGN,cells_speed4:ALIGN,cells_speed5:ALIGN,cells_speed6:ALIGN,cells_speed7:ALIGN,cells_speed8:ALIGN,tmp_cells_speed0:ALIGN,tmp_cells_speed1:ALIGN,tmp_cells_speed2:ALIGN,tmp_cells_speed3:ALIGN,tmp_cells_speed4:ALIGN,tmp_cells_speed5:ALIGN,tmp_cells_speed6:ALIGN,tmp_cells_speed7:ALIGN,tmp_cells_speed8:ALIGN,obstacles:ALIGN)
    for (int ii = 0; ii < params.nx; ii++)
    {
      TIMESTEPLOOP(ii, jj);
    }
  } 
  // Waits to receive all data
  MPI_Waitall(6, receive_requests,
                MPI_STATUSES_IGNORE);
  // Computes rows that need halo rows
  #pragma omp simd aligned(cells_speed0:ALIGN,cells_speed1:ALIGN,cells_speed2:ALIGN,cells_speed3:ALIGN,cells_speed4:ALIGN,cells_speed5:ALIGN,cells_speed6:ALIGN,cells_speed7:ALIGN,cells_speed8:ALIGN,tmp_cells_speed0:ALIGN,tmp_cells_speed1:ALIGN,tmp_cells_speed2:ALIGN,tmp_cells_speed3:ALIGN,tmp_cells_speed4:ALIGN,tmp_cells_speed5:ALIGN,tmp_cells_speed6:ALIGN,tmp_cells_speed7:ALIGN,tmp_cells_speed8:ALIGN,obstacles:ALIGN)
  for (int ii = 0; ii < params.nx; ii++)
  {
    TIMESTEPLOOP(ii, bounds.nrows);
  }
  
  if(bounds.nrows != 1) {
    #pragma omp simd aligned(cells_speed0:ALIGN,cells_speed1:ALIGN,cells_speed2:ALIGN,cells_speed3:ALIGN,cells_speed4:ALIGN,cells_speed5:ALIGN,cells_speed6:ALIGN,cells_speed7:ALIGN,cells_speed8:ALIGN,tmp_cells_speed0:ALIGN,tmp_cells_speed1:ALIGN,tmp_cells_speed2:ALIGN,tmp_cells_speed3:ALIGN,tmp_cells_speed4:ALIGN,tmp_cells_speed5:ALIGN,tmp_cells_speed6:ALIGN,tmp_cells_speed7:ALIGN,tmp_cells_speed8:ALIGN,obstacles:ALIGN)
    for (int ii = 0; ii < params.nx; ii++)
    {
      TIMESTEPLOOP(ii, 1);
    }
  }

  *av_vel_ret = tot_u;
  *tot_cells_ret = tot_cells;

  // Ensures that all data has been sent to other processes before the pointer swap occurrs
  MPI_Waitall(6, send_requests,
                MPI_STATUSES_IGNORE);
}


int accelerate_flow_align(const t_param params,float*restrict speed1,float*restrict speed3,float*restrict speed5,float*restrict speed6,float*restrict speed7,float*restrict speed8, const int* obstacles, const t_grid_bounds bounds)
{
  __assume_aligned(speed1,ALIGN);__assume_aligned(speed3,ALIGN);__assume_aligned(speed5,ALIGN);
  __assume_aligned(speed6,ALIGN);__assume_aligned(speed7,ALIGN);__assume_aligned(speed8,ALIGN);

  __assume_aligned(obstacles, ALIGN);
  __assume(params.nx%GRID_EL_DIV==0); __assume(params.ny%GRID_EL_DIV==0);

  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  const int yCoord = params.ny - 2 - bounds.startY + 1;
  const int startInd = yCoord * params.nx;
  const int endInd = startInd + params.nx;
  #pragma omp simd aligned(speed1:ALIGN,speed3:ALIGN,speed5:ALIGN,speed6:ALIGN,speed7:ALIGN,speed8:ALIGN,obstacles:ALIGN)
  for (int ind = startInd; ind < endInd; ind++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ind - params.nx]
        && (speed3[ind] - w1) > 0.f
        && (speed6[ind] - w2) > 0.f
        && (speed7[ind] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      speed1[ind] += w1;
      speed5[ind] += w2;
      speed8[ind] += w2;
      /* decrease 'west-side' densities */
      speed3[ind] -= w1;
      speed6[ind] -= w2;
      speed7[ind] -= w2;
    }
  }
  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed_vec cells, int* obstacles, const t_grid_bounds bounds)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 1; jj < bounds.nrows + 1; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int ind = ii + jj*params.nx;
      /* ignore occupied cells */
      if (!obstacles[ind - params.nx])
      {
        float local_density = cells.speed0[ind] + cells.speed1[ind] + cells.speed2[ind] +
        cells.speed3[ind] + cells.speed4[ind] + cells.speed5[ind] + cells.speed6[ind] +
        cells.speed7[ind] + cells.speed8[ind];

        /* compute x velocity component */
        float u_x = (cells.speed1[ind]
                      + cells.speed5[ind]
                      + cells.speed8[ind]
                      - (cells.speed3[ind]
                         + cells.speed6[ind]
                         + cells.speed7[ind]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells.speed2[ind]
                      + cells.speed5[ind]
                      + cells.speed6[ind]
                      - (cells.speed4[ind]
                         + cells.speed7[ind]
                         + cells.speed8[ind]))
                     / local_density;
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initParams(const char* paramfile, t_param* params) {
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);  
  return EXIT_SUCCESS;
}

t_grid_bounds splitGrid(int nprocs, int rank, int ny){
  t_grid_bounds bounds;
  // Splits rows. Note that  for case when more ranks than rows, nRows = 0.
  bounds.nprocs = nprocs;
  bounds.rank = rank;
  int spareRows = ny % nprocs;
  int nRows = (ny / nprocs) + (spareRows > rank? 1 : 0);
  bounds.nrows = nRows;
  int startY = nRows * rank + (rank >= spareRows? spareRows : 0);
  bounds.startY = startY;
  bounds.endYNonInc = startY + nRows; 
  return bounds;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_vec* cells_ptr, t_speed_vec* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr,
              int** tot_cells_ptr, t_grid_bounds bounds, float** global_av_vels_ptr, int**  global_tot_cells_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  cells_ptr->speed0 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  cells_ptr->speed1 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  cells_ptr->speed2 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  cells_ptr->speed3 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  cells_ptr->speed4 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  cells_ptr->speed5 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  cells_ptr->speed6 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  cells_ptr->speed7 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  cells_ptr->speed8 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);

  /* 'helper' grid, used as scratch space */
  tmp_cells_ptr->speed0 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  tmp_cells_ptr->speed1 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  tmp_cells_ptr->speed2 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  tmp_cells_ptr->speed3 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  tmp_cells_ptr->speed4 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  tmp_cells_ptr->speed5 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  tmp_cells_ptr->speed6 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  tmp_cells_ptr->speed7 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);
  tmp_cells_ptr->speed8 = _mm_malloc((bounds.nrows + 2) * params->nx * sizeof(float),ALIGN);


  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int) * (bounds.nrows * params->nx),ALIGN);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;
  for(int jj = 0; jj < bounds.nrows + 2; jj++){
    for(int ii = 0; ii < params->nx; ii++) {
      int ind = ii + jj*params->nx;
      /* centre */
      cells_ptr->speed0[ind] = w0;
      /* axis directions */
      cells_ptr->speed1[ind] = w1;
      cells_ptr->speed2[ind] = w1;
      cells_ptr->speed3[ind] = w1;
      cells_ptr->speed4[ind] = w1;
      /* diagonals */
      cells_ptr->speed5[ind] = w2;
      cells_ptr->speed6[ind] = w2;
      cells_ptr->speed7[ind] = w2;
      cells_ptr->speed8[ind] = w2;

      /* first set all cells in obstacle array to zero */
      if (jj >= 1 && jj < bounds.nrows + 1){
        (*obstacles_ptr)[ind - params->nx] = 0;
      }
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    if (yy >= bounds.startY && yy < bounds.endYNonInc){
      (*obstacles_ptr)[(xx + yy*params->nx) - (bounds.startY *params->nx)] = blocked;
    }
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)_mm_malloc(sizeof(float) * params->maxIters,ALIGN);
  *tot_cells_ptr = (int*)_mm_malloc(sizeof(int) * params->maxIters,ALIGN);

  if(bounds.rank == 0) {
    *global_av_vels_ptr = (float*)_mm_malloc(sizeof(float) * params->maxIters,ALIGN);
    *global_tot_cells_ptr = (int*)_mm_malloc(sizeof(int) * params->maxIters,ALIGN);
  }

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed_vec* cells, t_speed_vec* tmp_cells,
             int** obstacles_ptr, float** av_vels_ptr,
             int** tot_cells_ptr, t_grid_bounds bounds, float** global_av_vels_ptr, int**  global_tot_cells_ptr)
{
  /*
  ** free up allocated memory
  */
  _mm_free(cells->speed0);
  _mm_free(cells->speed1);
  _mm_free(cells->speed2);
  _mm_free(cells->speed3);
  _mm_free(cells->speed4);
  _mm_free(cells->speed5);
  _mm_free(cells->speed6);
  _mm_free(cells->speed7);
  _mm_free(cells->speed8);

  _mm_free(cells);

  _mm_free(tmp_cells->speed0);
  _mm_free(tmp_cells->speed1);
  _mm_free(tmp_cells->speed2);
  _mm_free(tmp_cells->speed3);
  _mm_free(tmp_cells->speed4);
  _mm_free(tmp_cells->speed5);
  _mm_free(tmp_cells->speed6);
  _mm_free(tmp_cells->speed7);
  _mm_free(tmp_cells->speed8);

  _mm_free(tmp_cells);


  _mm_free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  _mm_free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  _mm_free(*tot_cells_ptr);
  *tot_cells_ptr = NULL;

  if(bounds.rank == 0) {
    _mm_free(*global_av_vels_ptr);
    global_av_vels_ptr = NULL;
    _mm_free(*global_tot_cells_ptr);
    global_tot_cells_ptr = NULL;
  }

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed_vec cells, int* obstacles, const t_grid_bounds bounds)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles, bounds) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed_vec cells, const t_grid_bounds bounds)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 1; jj < bounds.nrows + 1; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int ind = ii + jj * params.nx;
      total += cells.speed0[ind] + cells.speed1[ind] + cells.speed2[ind] +
      cells.speed3[ind] + cells.speed4[ind] + cells.speed5[ind] + cells.speed6[ind] +
      cells.speed7[ind] + cells.speed8[ind];
    }
  }

  return total;
}

int write_values(const t_param params, t_speed_vec cells, int* obstacles, float* av_vels, const t_grid_bounds bounds)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  const int ii_width = snprintf(NULL, 0, "%d", params.nx);
  const int jj_width = snprintf(NULL, 0, "%d", params.ny);
  const int record_width = 80 + jj_width + ii_width; // Each row has 80 characters + the number of digits in the biggest coordinate
  // Loads data into char array
  const int buff_length = record_width * bounds.nrows * params.nx;
  char file_contents[buff_length + 1]; // Plus one for null sentinel
  int idx = 0;
  for(int jj = 1; jj < bounds.nrows + 1; jj++) {
    for(int ii = 0; ii < params.nx; ii++) {
      int ind = ii + jj*params.nx;
      /* an occupied cell */
      if (obstacles[ind - params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        
        local_density = cells.speed0[ind] + cells.speed1[ind] + cells.speed2[ind] +
        cells.speed3[ind] + cells.speed4[ind] + cells.speed5[ind] + cells.speed6[ind] +
        cells.speed7[ind] + cells.speed8[ind];

        /* compute x velocity component */
        float u_x = (cells.speed1[ind]
                      + cells.speed5[ind]
                      + cells.speed8[ind]
                      - (cells.speed3[ind]
                         + cells.speed6[ind]
                         + cells.speed7[ind]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells.speed2[ind]
                      + cells.speed5[ind]
                      + cells.speed6[ind]
                      - (cells.speed4[ind]
                         + cells.speed7[ind]
                         + cells.speed8[ind]))
                     / local_density;

        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      // Writes to char array
      sprintf(&file_contents[idx], "%0*d %0*d %.12E %.12E %.12E %.12E %d\n", ii_width, ii, jj_width, (jj - 1) + bounds.startY, u_x, u_y, u, pressure, obstacles[ind - params.nx]);
      idx += record_width; 
    }
  }
  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, FINALSTATEFILE, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  MPI_File_set_size(fh, record_width * params.ny * params.nx); // Ensures file is correct size
  MPI_Offset offset = record_width * bounds.startY * params.nx;
  MPI_File_write_at_all(fh, offset, file_contents, buff_length, MPI_CHAR, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);

  if(bounds.rank == 0) {

    fp = fopen(AVVELSFILE, "w");

    if (fp == NULL)
    {
      die("could not open file output file", __LINE__, __FILE__);
    }

    for (int ii = 0; ii < params.maxIters; ii++)
    {
      fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
    }

    fclose(fp);
  }

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}