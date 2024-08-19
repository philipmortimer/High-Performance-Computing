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

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

#define ALIGN_SMALL 1024
#define ALIGN_MID 512
#define ALIGN_LARGE 64
#define SMALL_MID_THRESH 256 * 256
#define MID_LARGE_THRESH 1024 * 1024
#define GRID_EL_DIV 128

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

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_vec* cells_ptr, t_speed_vec* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int alignSize);
int initParams(const char* paramfile, t_param* params);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep_large_align(const t_param params,
  float*restrict cells_speed0,float*restrict cells_speed1,float*restrict cells_speed2,float*restrict cells_speed3,float*restrict cells_speed4,float*restrict cells_speed5,float*restrict cells_speed6,float*restrict cells_speed7,float*restrict cells_speed8,
  float*restrict tmp_cells_speed0,float*restrict tmp_cells_speed1,float*restrict tmp_cells_speed2,float*restrict tmp_cells_speed3,float*restrict tmp_cells_speed4,float*restrict tmp_cells_speed5,float*restrict tmp_cells_speed6,float*restrict tmp_cells_speed7,float*restrict tmp_cells_speed8,
  const int* obstacles);
float timestep_mid_align(const t_param params,
  float*restrict cells_speed0,float*restrict cells_speed1,float*restrict cells_speed2,float*restrict cells_speed3,float*restrict cells_speed4,float*restrict cells_speed5,float*restrict cells_speed6,float*restrict cells_speed7,float*restrict cells_speed8,
  float*restrict tmp_cells_speed0,float*restrict tmp_cells_speed1,float*restrict tmp_cells_speed2,float*restrict tmp_cells_speed3,float*restrict tmp_cells_speed4,float*restrict tmp_cells_speed5,float*restrict tmp_cells_speed6,float*restrict tmp_cells_speed7,float*restrict tmp_cells_speed8,
  const int* obstacles);
float timestep_small_align(const t_param params,
  float*restrict cells_speed0,float*restrict cells_speed1,float*restrict cells_speed2,float*restrict cells_speed3,float*restrict cells_speed4,float*restrict cells_speed5,float*restrict cells_speed6,float*restrict cells_speed7,float*restrict cells_speed8,
  float*restrict tmp_cells_speed0,float*restrict tmp_cells_speed1,float*restrict tmp_cells_speed2,float*restrict tmp_cells_speed3,float*restrict tmp_cells_speed4,float*restrict tmp_cells_speed5,float*restrict tmp_cells_speed6,float*restrict tmp_cells_speed7,float*restrict tmp_cells_speed8,
  const int* obstacles);

int accelerate_flow_small_align(const t_param params,float* speed1,float* speed3,float* speed5,float* speed6,float* speed7,float* speed8, const int* obstacles);
int accelerate_flow_mid_align(const t_param params,float* speed1,float* speed3,float* speed5,float* speed6,float* speed7,float* speed8, const int* obstacles);
int accelerate_flow_large_align(const t_param params,float* speed1,float* speed3,float* speed5,float* speed6,float* speed7,float* speed8, const int* obstacles);

int write_values(const t_param params, t_speed_vec cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed_vec* cells_ptr, t_speed_vec* tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed_vec cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed_vec cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed_vec cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{

  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */

  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
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
  const int alignSmall = params.nx * params.ny <= SMALL_MID_THRESH;
  const int alignMid = !alignSmall && params.nx * params.ny <= MID_LARGE_THRESH;
  const int alignLarge = !alignSmall && !alignMid && params.nx * params.ny > MID_LARGE_THRESH;
  const int alignSize = alignSmall ? (ALIGN_SMALL) : (alignMid? ALIGN_MID : ALIGN_LARGE);

  t_speed_vec* cells     = _mm_malloc(sizeof(t_speed_vec),alignSize);    /* grid containing fluid densities */
  t_speed_vec* tmp_cells = _mm_malloc(sizeof(t_speed_vec),alignSize);    /* scratch space */
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, cells, tmp_cells, &obstacles, &av_vels, alignSize);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    if(alignSmall) av_vels[tt] = timestep_small_align(params, 
      cells->speed0,cells->speed1,cells->speed2,cells->speed3,cells->speed4,cells->speed5,cells->speed6,cells->speed7,cells->speed8, 
      tmp_cells->speed0,tmp_cells->speed1,tmp_cells->speed2,tmp_cells->speed3,tmp_cells->speed4,tmp_cells->speed5,tmp_cells->speed6,tmp_cells->speed7,tmp_cells->speed8,
      obstacles);
    else if (alignMid) av_vels[tt] = timestep_mid_align(params, 
      cells->speed0,cells->speed1,cells->speed2,cells->speed3,cells->speed4,cells->speed5,cells->speed6,cells->speed7,cells->speed8, 
      tmp_cells->speed0,tmp_cells->speed1,tmp_cells->speed2,tmp_cells->speed3,tmp_cells->speed4,tmp_cells->speed5,tmp_cells->speed6,tmp_cells->speed7,tmp_cells->speed8,
      obstacles);
    else av_vels[tt] = timestep_large_align(params, 
      cells->speed0,cells->speed1,cells->speed2,cells->speed3,cells->speed4,cells->speed5,cells->speed6,cells->speed7,cells->speed8, 
      tmp_cells->speed0,tmp_cells->speed1,tmp_cells->speed2,tmp_cells->speed3,tmp_cells->speed4,tmp_cells->speed5,tmp_cells->speed6,tmp_cells->speed7,tmp_cells->speed8,
      obstacles);
    // Swaps pointers
    t_speed_vec* swap = tmp_cells;
    tmp_cells = cells;
    cells = swap;
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, *cells));
#endif
  }
  //av_vels[params.maxIters - 1] = av_velocity(params, cells, obstacles);
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, *cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, *cells, obstacles, av_vels);
  finalise(&params, cells, tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

// Same timestep and accelerate function is essentially defined thrice. A macro may be a better
// way of doing this if readability and maintainability was more of an issue, but the tradeoff between the two is non-trivial.

float timestep_large_align(const t_param params,
  float*restrict cells_speed0,float*restrict cells_speed1,float*restrict cells_speed2,float*restrict cells_speed3,float*restrict cells_speed4,float*restrict cells_speed5,float*restrict cells_speed6,float*restrict cells_speed7,float*restrict cells_speed8,
  float*restrict tmp_cells_speed0,float*restrict tmp_cells_speed1,float*restrict tmp_cells_speed2,float*restrict tmp_cells_speed3,float*restrict tmp_cells_speed4,float*restrict tmp_cells_speed5,float*restrict tmp_cells_speed6,float*restrict tmp_cells_speed7,float*restrict tmp_cells_speed8,
  const int* obstacles)
{
 __assume_aligned(cells_speed0,ALIGN_LARGE);__assume_aligned(cells_speed1,ALIGN_LARGE);__assume_aligned(cells_speed2,ALIGN_LARGE);
  __assume_aligned(cells_speed3,ALIGN_LARGE);__assume_aligned(cells_speed4,ALIGN_LARGE);__assume_aligned(cells_speed5,ALIGN_LARGE);
  __assume_aligned(cells_speed6,ALIGN_LARGE);__assume_aligned(cells_speed7,ALIGN_LARGE);__assume_aligned(cells_speed8,ALIGN_LARGE);

  __assume_aligned(tmp_cells_speed0,ALIGN_LARGE);__assume_aligned(tmp_cells_speed1,ALIGN_LARGE);__assume_aligned(tmp_cells_speed2,ALIGN_LARGE);
  __assume_aligned(tmp_cells_speed3,ALIGN_LARGE);__assume_aligned(tmp_cells_speed4,ALIGN_LARGE);__assume_aligned(tmp_cells_speed5,ALIGN_LARGE);
  __assume_aligned(tmp_cells_speed6,ALIGN_LARGE);__assume_aligned(tmp_cells_speed7,ALIGN_LARGE);__assume_aligned(tmp_cells_speed8,ALIGN_LARGE);

  __assume_aligned(obstacles, ALIGN_LARGE);
  __assume(params.nx%GRID_EL_DIV==0); __assume(params.ny%GRID_EL_DIV==0);

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

  accelerate_flow_large_align(params,cells_speed1,cells_speed3,cells_speed5,cells_speed6,cells_speed7,cells_speed8, obstacles);


  float tot_u = 0.f;
  int tot_cells = 0;

  int xPow2 = round(log2(params.nx));

  // Loops over all cells
  #pragma omp parallel for simd reduction(+:tot_u,tot_cells) aligned(cells_speed0:ALIGN_LARGE,cells_speed1:ALIGN_LARGE,cells_speed2:ALIGN_LARGE,cells_speed3:ALIGN_LARGE,cells_speed4:ALIGN_LARGE,cells_speed5:ALIGN_LARGE,cells_speed6:ALIGN_LARGE,cells_speed7:ALIGN_LARGE,cells_speed8:ALIGN_LARGE,tmp_cells_speed0:ALIGN_LARGE,tmp_cells_speed1:ALIGN_LARGE,tmp_cells_speed2:ALIGN_LARGE,tmp_cells_speed3:ALIGN_LARGE,tmp_cells_speed4:ALIGN_LARGE,tmp_cells_speed5:ALIGN_LARGE,tmp_cells_speed6:ALIGN_LARGE,tmp_cells_speed7:ALIGN_LARGE,tmp_cells_speed8:ALIGN_LARGE,obstacles:ALIGN_LARGE)
  for(int ind = 0; ind < params.nx * params.ny; ind++){
    int ii = ind % params.nx;
    int jj = (ind - ii) >> xPow2; // assuming it's a pow of 2. If not it should be (ind - ii) / params.nx
    // PROPOGATE
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int y_n = (jj + 1 < params.ny)? (jj + 1) : (0);
    int x_e = (ii + 1 < params.nx)? (ii + 1) : (0);
    int y_s = (jj > 0) ? (jj - 1): (jj + params.ny - 1);
    int x_w = (ii > 0) ? (ii - 1) : (ii + params.nx - 1);
    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    tmp_cells_speed0[ind] = cells_speed0[ind]; /* central cell, no movement */
    tmp_cells_speed1[ind] = cells_speed1[x_w + jj*params.nx]; /* east */
    tmp_cells_speed2[ind] = cells_speed2[ii + y_s*params.nx]; /* north */
    tmp_cells_speed3[ind] = cells_speed3[x_e + jj*params.nx]; /* west */
    tmp_cells_speed4[ind] = cells_speed4[ii + y_n*params.nx]; /* south */
    tmp_cells_speed5[ind] = cells_speed5[x_w + y_s*params.nx]; /* north-east */
    tmp_cells_speed6[ind] = cells_speed6[x_e + y_s*params.nx]; /* north-west */
    tmp_cells_speed7[ind] = cells_speed7[x_e + y_n*params.nx]; /* south-west */
    tmp_cells_speed8[ind] = cells_speed8[x_w + y_n*params.nx]; /* south-east */

    // COLLISION
    if (!obstacles[ind])
    {
      /* compute local density total */
      float local_density = tmp_cells_speed0[ind] + tmp_cells_speed1[ind] + tmp_cells_speed2[ind] +
      tmp_cells_speed3[ind] + tmp_cells_speed4[ind] + tmp_cells_speed5[ind] + tmp_cells_speed6[ind] +
      tmp_cells_speed7[ind] + tmp_cells_speed8[ind];

      /* compute x velocity component */
      float u_x = (tmp_cells_speed1[ind]
                    + tmp_cells_speed5[ind]
                    + tmp_cells_speed8[ind]
                    - (tmp_cells_speed3[ind]
                        + tmp_cells_speed6[ind]
                        + tmp_cells_speed7[ind]))
                    / local_density;
      /* compute y velocity component */
      float u_y = (tmp_cells_speed2[ind]
                    + tmp_cells_speed5[ind]
                    + tmp_cells_speed6[ind]
                    - (tmp_cells_speed4[ind]
                        + tmp_cells_speed7[ind]
                        + tmp_cells_speed8[ind]))
                    / local_density;

      /* velocity squared */
      float u_x_squared = u_x * u_x;
      float u_y_squared = u_y * u_y;
      float u_sq = u_x_squared + u_y_squared;

      
      /* directional velocity components */
      float u_5 =   u_x + u_y;  /* north-east */
      float u_6 = - u_x + u_y;  /* north-west */
      float u_7 = - u_x - u_y;  /* south-west */
      float u_8 =   u_x - u_y;  /* south-east */

      //float d_equ[NSPEEDS];
      /* zero velocity density: weight w0 */
      float d_equ_0 = w0 * local_density
                  * (1.f - u_sq * rec_c_sq_2);
      /* axis speeds: weight w1 */
      float w1_dens_div_2_c_sq_square_dens = w1_div_2_c_sq_square * local_density;
      float acc = u_sq * neg_c_sq + c_sq_squared_2;
      float acc_x = acc + u_x_squared;
      float acc_y = acc + u_y_squared;
      float u_x_c_sq_2 = u_x * c_sq_2;
      float u_y_c_sq_2 = u_y * c_sq_2;
      float d_equ_1 = w1_dens_div_2_c_sq_square_dens * (acc_x + u_x_c_sq_2);
      float d_equ_2 = w1_dens_div_2_c_sq_square_dens * (acc_y + u_y_c_sq_2);
      float d_equ_3 = w1_dens_div_2_c_sq_square_dens * (acc_x - u_x_c_sq_2);
      float d_equ_4 = w1_dens_div_2_c_sq_square_dens * (acc_y - u_y_c_sq_2);
      /* diagonal speeds: weight w2 */
      float w2_dens_div_2_c_sq_square_dens = w2_div_2_c_sq_square * local_density;
      float acc_seq = acc + c_sq_2;
      float d_equ_5 = w2_dens_div_2_c_sq_square_dens * (u_5 * (c_sq_2 + u_5) + acc);
      float d_equ_6 = w2_dens_div_2_c_sq_square_dens * (u_6 * (c_sq_2 + u_6) + acc);
      float d_equ_7 = w2_dens_div_2_c_sq_square_dens * (u_7 * (c_sq_2 + u_7) + acc);
      float d_equ_8 = w2_dens_div_2_c_sq_square_dens * (u_8 * (c_sq_2 + u_8) + acc); 

      /* relaxation step */
      tmp_cells_speed0[ind] = tmp_cells_speed0[ind] + params.omega * (d_equ_0 - tmp_cells_speed0[ind]);
      tmp_cells_speed1[ind] = tmp_cells_speed1[ind] + params.omega * (d_equ_1 - tmp_cells_speed1[ind]);
      tmp_cells_speed2[ind] = tmp_cells_speed2[ind] + params.omega * (d_equ_2 - tmp_cells_speed2[ind]);
      tmp_cells_speed3[ind] = tmp_cells_speed3[ind] + params.omega * (d_equ_3 - tmp_cells_speed3[ind]);
      tmp_cells_speed4[ind] = tmp_cells_speed4[ind] + params.omega * (d_equ_4 - tmp_cells_speed4[ind]);
      tmp_cells_speed5[ind] = tmp_cells_speed5[ind] + params.omega * (d_equ_5 - tmp_cells_speed5[ind]);
      tmp_cells_speed6[ind] = tmp_cells_speed6[ind] + params.omega * (d_equ_6 - tmp_cells_speed6[ind]);
      tmp_cells_speed7[ind] = tmp_cells_speed7[ind] + params.omega * (d_equ_7 - tmp_cells_speed7[ind]);
      tmp_cells_speed8[ind] = tmp_cells_speed8[ind] + params.omega * (d_equ_8 - tmp_cells_speed8[ind]);

      // Calculates velocity
      tot_u += sqrtf(u_sq);
      tot_cells++;
    } else { /* if the cell contains an obstacle */
      // REBOUND
      /* called after propagate, so taking values from scratch space
      ** mirroring, and writing into main grid */
      float vals_0 = tmp_cells_speed1[ind];
      float vals_1 = tmp_cells_speed2[ind];
      float vals_2 = tmp_cells_speed3[ind];
      float vals_3 = tmp_cells_speed4[ind];
      float vals_4 = tmp_cells_speed5[ind];
      float vals_5 = tmp_cells_speed6[ind];
      float vals_6 = tmp_cells_speed7[ind];
      float vals_7 = tmp_cells_speed8[ind];

      tmp_cells_speed1[ind] = vals_2;
      tmp_cells_speed2[ind] = vals_3;
      tmp_cells_speed3[ind] = vals_0;
      tmp_cells_speed4[ind] = vals_1;
      tmp_cells_speed5[ind] = vals_6;
      tmp_cells_speed6[ind] = vals_7;
      tmp_cells_speed7[ind] = vals_4;
      tmp_cells_speed8[ind] = vals_5;
    }
  } 
  return tot_u / (float)tot_cells;
}

float timestep_mid_align(const t_param params,
  float*restrict cells_speed0,float*restrict cells_speed1,float*restrict cells_speed2,float*restrict cells_speed3,float*restrict cells_speed4,float*restrict cells_speed5,float*restrict cells_speed6,float*restrict cells_speed7,float*restrict cells_speed8,
  float*restrict tmp_cells_speed0,float*restrict tmp_cells_speed1,float*restrict tmp_cells_speed2,float*restrict tmp_cells_speed3,float*restrict tmp_cells_speed4,float*restrict tmp_cells_speed5,float*restrict tmp_cells_speed6,float*restrict tmp_cells_speed7,float*restrict tmp_cells_speed8,
  const int* obstacles)
{
 __assume_aligned(cells_speed0,ALIGN_MID);__assume_aligned(cells_speed1,ALIGN_MID);__assume_aligned(cells_speed2,ALIGN_MID);
  __assume_aligned(cells_speed3,ALIGN_MID);__assume_aligned(cells_speed4,ALIGN_MID);__assume_aligned(cells_speed5,ALIGN_MID);
  __assume_aligned(cells_speed6,ALIGN_MID);__assume_aligned(cells_speed7,ALIGN_MID);__assume_aligned(cells_speed8,ALIGN_MID);

  __assume_aligned(tmp_cells_speed0,ALIGN_MID);__assume_aligned(tmp_cells_speed1,ALIGN_MID);__assume_aligned(tmp_cells_speed2,ALIGN_MID);
  __assume_aligned(tmp_cells_speed3,ALIGN_MID);__assume_aligned(tmp_cells_speed4,ALIGN_MID);__assume_aligned(tmp_cells_speed5,ALIGN_MID);
  __assume_aligned(tmp_cells_speed6,ALIGN_MID);__assume_aligned(tmp_cells_speed7,ALIGN_MID);__assume_aligned(tmp_cells_speed8,ALIGN_MID);

  __assume_aligned(obstacles, ALIGN_MID);
  __assume(params.nx%GRID_EL_DIV==0); __assume(params.ny%GRID_EL_DIV==0);

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

  accelerate_flow_mid_align(params,cells_speed1,cells_speed3,cells_speed5,cells_speed6,cells_speed7,cells_speed8, obstacles);


  float tot_u = 0.f;
  int tot_cells = 0;

  int xPow2 = round(log2(params.nx));

  // Loops over all cells
  #pragma omp parallel for simd reduction(+:tot_u,tot_cells) aligned(cells_speed0:ALIGN_MID,cells_speed1:ALIGN_MID,cells_speed2:ALIGN_MID,cells_speed3:ALIGN_MID,cells_speed4:ALIGN_MID,cells_speed5:ALIGN_MID,cells_speed6:ALIGN_MID,cells_speed7:ALIGN_MID,cells_speed8:ALIGN_MID,tmp_cells_speed0:ALIGN_MID,tmp_cells_speed1:ALIGN_MID,tmp_cells_speed2:ALIGN_MID,tmp_cells_speed3:ALIGN_MID,tmp_cells_speed4:ALIGN_MID,tmp_cells_speed5:ALIGN_MID,tmp_cells_speed6:ALIGN_MID,tmp_cells_speed7:ALIGN_MID,tmp_cells_speed8:ALIGN_MID,obstacles:ALIGN_MID)
  for(int ind = 0; ind < params.nx * params.ny; ind++){
    int ii = ind % params.nx;
    int jj = (ind - ii) >> xPow2; // assuming it's a pow of 2. If not it should be (ind - ii) / params.nx
    // PROPOGATE
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int y_n = (jj + 1 < params.ny)? (jj + 1) : (0);
    int x_e = (ii + 1 < params.nx)? (ii + 1) : (0);
    int y_s = (jj > 0) ? (jj - 1): (jj + params.ny - 1);
    int x_w = (ii > 0) ? (ii - 1) : (ii + params.nx - 1);
    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    tmp_cells_speed0[ind] = cells_speed0[ind]; /* central cell, no movement */
    tmp_cells_speed1[ind] = cells_speed1[x_w + jj*params.nx]; /* east */
    tmp_cells_speed2[ind] = cells_speed2[ii + y_s*params.nx]; /* north */
    tmp_cells_speed3[ind] = cells_speed3[x_e + jj*params.nx]; /* west */
    tmp_cells_speed4[ind] = cells_speed4[ii + y_n*params.nx]; /* south */
    tmp_cells_speed5[ind] = cells_speed5[x_w + y_s*params.nx]; /* north-east */
    tmp_cells_speed6[ind] = cells_speed6[x_e + y_s*params.nx]; /* north-west */
    tmp_cells_speed7[ind] = cells_speed7[x_e + y_n*params.nx]; /* south-west */
    tmp_cells_speed8[ind] = cells_speed8[x_w + y_n*params.nx]; /* south-east */

    // COLLISION
    if (!obstacles[ind])
    {
      /* compute local density total */
      float local_density = tmp_cells_speed0[ind] + tmp_cells_speed1[ind] + tmp_cells_speed2[ind] +
      tmp_cells_speed3[ind] + tmp_cells_speed4[ind] + tmp_cells_speed5[ind] + tmp_cells_speed6[ind] +
      tmp_cells_speed7[ind] + tmp_cells_speed8[ind];

      /* compute x velocity component */
      float u_x = (tmp_cells_speed1[ind]
                    + tmp_cells_speed5[ind]
                    + tmp_cells_speed8[ind]
                    - (tmp_cells_speed3[ind]
                        + tmp_cells_speed6[ind]
                        + tmp_cells_speed7[ind]))
                    / local_density;
      /* compute y velocity component */
      float u_y = (tmp_cells_speed2[ind]
                    + tmp_cells_speed5[ind]
                    + tmp_cells_speed6[ind]
                    - (tmp_cells_speed4[ind]
                        + tmp_cells_speed7[ind]
                        + tmp_cells_speed8[ind]))
                    / local_density;

      /* velocity squared */
      float u_x_squared = u_x * u_x;
      float u_y_squared = u_y * u_y;
      float u_sq = u_x_squared + u_y_squared;

      
      /* directional velocity components */
      float u_5 =   u_x + u_y;  /* north-east */
      float u_6 = - u_x + u_y;  /* north-west */
      float u_7 = - u_x - u_y;  /* south-west */
      float u_8 =   u_x - u_y;  /* south-east */

      //float d_equ[NSPEEDS];
      /* zero velocity density: weight w0 */
      float d_equ_0 = w0 * local_density
                  * (1.f - u_sq * rec_c_sq_2);
      /* axis speeds: weight w1 */
      float w1_dens_div_2_c_sq_square_dens = w1_div_2_c_sq_square * local_density;
      float acc = u_sq * neg_c_sq + c_sq_squared_2;
      float acc_x = acc + u_x_squared;
      float acc_y = acc + u_y_squared;
      float u_x_c_sq_2 = u_x * c_sq_2;
      float u_y_c_sq_2 = u_y * c_sq_2;
      float d_equ_1 = w1_dens_div_2_c_sq_square_dens * (acc_x + u_x_c_sq_2);
      float d_equ_2 = w1_dens_div_2_c_sq_square_dens * (acc_y + u_y_c_sq_2);
      float d_equ_3 = w1_dens_div_2_c_sq_square_dens * (acc_x - u_x_c_sq_2);
      float d_equ_4 = w1_dens_div_2_c_sq_square_dens * (acc_y - u_y_c_sq_2);
      /* diagonal speeds: weight w2 */
      float w2_dens_div_2_c_sq_square_dens = w2_div_2_c_sq_square * local_density;
      float acc_seq = acc + c_sq_2;
      float d_equ_5 = w2_dens_div_2_c_sq_square_dens * (u_5 * (c_sq_2 + u_5) + acc);
      float d_equ_6 = w2_dens_div_2_c_sq_square_dens * (u_6 * (c_sq_2 + u_6) + acc);
      float d_equ_7 = w2_dens_div_2_c_sq_square_dens * (u_7 * (c_sq_2 + u_7) + acc);
      float d_equ_8 = w2_dens_div_2_c_sq_square_dens * (u_8 * (c_sq_2 + u_8) + acc); 

      /* relaxation step */
      tmp_cells_speed0[ind] = tmp_cells_speed0[ind] + params.omega * (d_equ_0 - tmp_cells_speed0[ind]);
      tmp_cells_speed1[ind] = tmp_cells_speed1[ind] + params.omega * (d_equ_1 - tmp_cells_speed1[ind]);
      tmp_cells_speed2[ind] = tmp_cells_speed2[ind] + params.omega * (d_equ_2 - tmp_cells_speed2[ind]);
      tmp_cells_speed3[ind] = tmp_cells_speed3[ind] + params.omega * (d_equ_3 - tmp_cells_speed3[ind]);
      tmp_cells_speed4[ind] = tmp_cells_speed4[ind] + params.omega * (d_equ_4 - tmp_cells_speed4[ind]);
      tmp_cells_speed5[ind] = tmp_cells_speed5[ind] + params.omega * (d_equ_5 - tmp_cells_speed5[ind]);
      tmp_cells_speed6[ind] = tmp_cells_speed6[ind] + params.omega * (d_equ_6 - tmp_cells_speed6[ind]);
      tmp_cells_speed7[ind] = tmp_cells_speed7[ind] + params.omega * (d_equ_7 - tmp_cells_speed7[ind]);
      tmp_cells_speed8[ind] = tmp_cells_speed8[ind] + params.omega * (d_equ_8 - tmp_cells_speed8[ind]);

      // Calculates velocity
      tot_u += sqrtf(u_sq);
      tot_cells++;
    } else { /* if the cell contains an obstacle */
      // REBOUND
      /* called after propagate, so taking values from scratch space
      ** mirroring, and writing into main grid */
      float vals_0 = tmp_cells_speed1[ind];
      float vals_1 = tmp_cells_speed2[ind];
      float vals_2 = tmp_cells_speed3[ind];
      float vals_3 = tmp_cells_speed4[ind];
      float vals_4 = tmp_cells_speed5[ind];
      float vals_5 = tmp_cells_speed6[ind];
      float vals_6 = tmp_cells_speed7[ind];
      float vals_7 = tmp_cells_speed8[ind];

      tmp_cells_speed1[ind] = vals_2;
      tmp_cells_speed2[ind] = vals_3;
      tmp_cells_speed3[ind] = vals_0;
      tmp_cells_speed4[ind] = vals_1;
      tmp_cells_speed5[ind] = vals_6;
      tmp_cells_speed6[ind] = vals_7;
      tmp_cells_speed7[ind] = vals_4;
      tmp_cells_speed8[ind] = vals_5;
    }
  } 
  return tot_u / (float)tot_cells;
}


float timestep_small_align(const t_param params,
  float*restrict cells_speed0,float*restrict cells_speed1,float*restrict cells_speed2,float*restrict cells_speed3,float*restrict cells_speed4,float*restrict cells_speed5,float*restrict cells_speed6,float*restrict cells_speed7,float*restrict cells_speed8,
  float*restrict tmp_cells_speed0,float*restrict tmp_cells_speed1,float*restrict tmp_cells_speed2,float*restrict tmp_cells_speed3,float*restrict tmp_cells_speed4,float*restrict tmp_cells_speed5,float*restrict tmp_cells_speed6,float*restrict tmp_cells_speed7,float*restrict tmp_cells_speed8,
  const int* obstacles)
{
  __assume_aligned(cells_speed0,ALIGN_SMALL);__assume_aligned(cells_speed1,ALIGN_SMALL);__assume_aligned(cells_speed2,ALIGN_SMALL);
  __assume_aligned(cells_speed3,ALIGN_SMALL);__assume_aligned(cells_speed4,ALIGN_SMALL);__assume_aligned(cells_speed5,ALIGN_SMALL);
  __assume_aligned(cells_speed6,ALIGN_SMALL);__assume_aligned(cells_speed7,ALIGN_SMALL);__assume_aligned(cells_speed8,ALIGN_SMALL);

  __assume_aligned(tmp_cells_speed0,ALIGN_SMALL);__assume_aligned(tmp_cells_speed1,ALIGN_SMALL);__assume_aligned(tmp_cells_speed2,ALIGN_SMALL);
  __assume_aligned(tmp_cells_speed3,ALIGN_SMALL);__assume_aligned(tmp_cells_speed4,ALIGN_SMALL);__assume_aligned(tmp_cells_speed5,ALIGN_SMALL);
  __assume_aligned(tmp_cells_speed6,ALIGN_SMALL);__assume_aligned(tmp_cells_speed7,ALIGN_SMALL);__assume_aligned(tmp_cells_speed8,ALIGN_SMALL);

  __assume_aligned(obstacles, ALIGN_SMALL);
  __assume(params.nx%GRID_EL_DIV==0); __assume(params.ny%GRID_EL_DIV==0);

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

  accelerate_flow_small_align(params,cells_speed1,cells_speed3,cells_speed5,cells_speed6,cells_speed7,cells_speed8, obstacles);


  float tot_u = 0.f;
  int tot_cells = 0;

  int xPow2 = round(log2(params.nx));

  // Loops over all cells
  #pragma omp parallel for simd reduction(+:tot_u,tot_cells) aligned(cells_speed0:ALIGN_SMALL,cells_speed1:ALIGN_SMALL,cells_speed2:ALIGN_SMALL,cells_speed3:ALIGN_SMALL,cells_speed4:ALIGN_SMALL,cells_speed5:ALIGN_SMALL,cells_speed6:ALIGN_SMALL,cells_speed7:ALIGN_SMALL,cells_speed8:ALIGN_SMALL,tmp_cells_speed0:ALIGN_SMALL,tmp_cells_speed1:ALIGN_SMALL,tmp_cells_speed2:ALIGN_SMALL,tmp_cells_speed3:ALIGN_SMALL,tmp_cells_speed4:ALIGN_SMALL,tmp_cells_speed5:ALIGN_SMALL,tmp_cells_speed6:ALIGN_SMALL,tmp_cells_speed7:ALIGN_SMALL,tmp_cells_speed8:ALIGN_SMALL,obstacles:ALIGN_SMALL)
  for(int ind = 0; ind < params.nx * params.ny; ind++){
    int ii = ind % params.nx;
    int jj = (ind - ii) >> xPow2; // assuming it's a pow of 2. If not it should be (ind - ii) / params.nx
    // PROPOGATE
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int y_n = (jj + 1 < params.ny)? (jj + 1) : (0);
    int x_e = (ii + 1 < params.nx)? (ii + 1) : (0);
    int y_s = (jj > 0) ? (jj - 1): (jj + params.ny - 1);
    int x_w = (ii > 0) ? (ii - 1) : (ii + params.nx - 1);
    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    tmp_cells_speed0[ind] = cells_speed0[ind]; /* central cell, no movement */
    tmp_cells_speed1[ind] = cells_speed1[x_w + jj*params.nx]; /* east */
    tmp_cells_speed2[ind] = cells_speed2[ii + y_s*params.nx]; /* north */
    tmp_cells_speed3[ind] = cells_speed3[x_e + jj*params.nx]; /* west */
    tmp_cells_speed4[ind] = cells_speed4[ii + y_n*params.nx]; /* south */
    tmp_cells_speed5[ind] = cells_speed5[x_w + y_s*params.nx]; /* north-east */
    tmp_cells_speed6[ind] = cells_speed6[x_e + y_s*params.nx]; /* north-west */
    tmp_cells_speed7[ind] = cells_speed7[x_e + y_n*params.nx]; /* south-west */
    tmp_cells_speed8[ind] = cells_speed8[x_w + y_n*params.nx]; /* south-east */

    // COLLISION
    if (!obstacles[ind])
    {
      /* compute local density total */
      float local_density = tmp_cells_speed0[ind] + tmp_cells_speed1[ind] + tmp_cells_speed2[ind] +
      tmp_cells_speed3[ind] + tmp_cells_speed4[ind] + tmp_cells_speed5[ind] + tmp_cells_speed6[ind] +
      tmp_cells_speed7[ind] + tmp_cells_speed8[ind];

      /* compute x velocity component */
      float u_x = (tmp_cells_speed1[ind]
                    + tmp_cells_speed5[ind]
                    + tmp_cells_speed8[ind]
                    - (tmp_cells_speed3[ind]
                        + tmp_cells_speed6[ind]
                        + tmp_cells_speed7[ind]))
                    / local_density;
      /* compute y velocity component */
      float u_y = (tmp_cells_speed2[ind]
                    + tmp_cells_speed5[ind]
                    + tmp_cells_speed6[ind]
                    - (tmp_cells_speed4[ind]
                        + tmp_cells_speed7[ind]
                        + tmp_cells_speed8[ind]))
                    / local_density;

      /* velocity squared */
      float u_x_squared = u_x * u_x;
      float u_y_squared = u_y * u_y;
      float u_sq = u_x_squared + u_y_squared;

      
      /* directional velocity components */
      float u_5 =   u_x + u_y;  /* north-east */
      float u_6 = - u_x + u_y;  /* north-west */
      float u_7 = - u_x - u_y;  /* south-west */
      float u_8 =   u_x - u_y;  /* south-east */

      //float d_equ[NSPEEDS];
      /* zero velocity density: weight w0 */
      float d_equ_0 = w0 * local_density
                  * (1.f - u_sq * rec_c_sq_2);
      /* axis speeds: weight w1 */
      float w1_dens_div_2_c_sq_square_dens = w1_div_2_c_sq_square * local_density;
      float acc = u_sq * neg_c_sq + c_sq_squared_2;
      float acc_x = acc + u_x_squared;
      float acc_y = acc + u_y_squared;
      float u_x_c_sq_2 = u_x * c_sq_2;
      float u_y_c_sq_2 = u_y * c_sq_2;
      float d_equ_1 = w1_dens_div_2_c_sq_square_dens * (acc_x + u_x_c_sq_2);
      float d_equ_2 = w1_dens_div_2_c_sq_square_dens * (acc_y + u_y_c_sq_2);
      float d_equ_3 = w1_dens_div_2_c_sq_square_dens * (acc_x - u_x_c_sq_2);
      float d_equ_4 = w1_dens_div_2_c_sq_square_dens * (acc_y - u_y_c_sq_2);
      /* diagonal speeds: weight w2 */
      float w2_dens_div_2_c_sq_square_dens = w2_div_2_c_sq_square * local_density;
      float acc_seq = acc + c_sq_2;
      float d_equ_5 = w2_dens_div_2_c_sq_square_dens * (u_5 * (c_sq_2 + u_5) + acc);
      float d_equ_6 = w2_dens_div_2_c_sq_square_dens * (u_6 * (c_sq_2 + u_6) + acc);
      float d_equ_7 = w2_dens_div_2_c_sq_square_dens * (u_7 * (c_sq_2 + u_7) + acc);
      float d_equ_8 = w2_dens_div_2_c_sq_square_dens * (u_8 * (c_sq_2 + u_8) + acc); 

      /* relaxation step */
      tmp_cells_speed0[ind] = tmp_cells_speed0[ind] + params.omega * (d_equ_0 - tmp_cells_speed0[ind]);
      tmp_cells_speed1[ind] = tmp_cells_speed1[ind] + params.omega * (d_equ_1 - tmp_cells_speed1[ind]);
      tmp_cells_speed2[ind] = tmp_cells_speed2[ind] + params.omega * (d_equ_2 - tmp_cells_speed2[ind]);
      tmp_cells_speed3[ind] = tmp_cells_speed3[ind] + params.omega * (d_equ_3 - tmp_cells_speed3[ind]);
      tmp_cells_speed4[ind] = tmp_cells_speed4[ind] + params.omega * (d_equ_4 - tmp_cells_speed4[ind]);
      tmp_cells_speed5[ind] = tmp_cells_speed5[ind] + params.omega * (d_equ_5 - tmp_cells_speed5[ind]);
      tmp_cells_speed6[ind] = tmp_cells_speed6[ind] + params.omega * (d_equ_6 - tmp_cells_speed6[ind]);
      tmp_cells_speed7[ind] = tmp_cells_speed7[ind] + params.omega * (d_equ_7 - tmp_cells_speed7[ind]);
      tmp_cells_speed8[ind] = tmp_cells_speed8[ind] + params.omega * (d_equ_8 - tmp_cells_speed8[ind]);

      // Calculates velocity
      tot_u += sqrtf(u_sq);
      tot_cells++;
    } else { /* if the cell contains an obstacle */
      // REBOUND
      /* called after propagate, so taking values from scratch space
      ** mirroring, and writing into main grid */
      float vals_0 = tmp_cells_speed1[ind];
      float vals_1 = tmp_cells_speed2[ind];
      float vals_2 = tmp_cells_speed3[ind];
      float vals_3 = tmp_cells_speed4[ind];
      float vals_4 = tmp_cells_speed5[ind];
      float vals_5 = tmp_cells_speed6[ind];
      float vals_6 = tmp_cells_speed7[ind];
      float vals_7 = tmp_cells_speed8[ind];

      tmp_cells_speed1[ind] = vals_2;
      tmp_cells_speed2[ind] = vals_3;
      tmp_cells_speed3[ind] = vals_0;
      tmp_cells_speed4[ind] = vals_1;
      tmp_cells_speed5[ind] = vals_6;
      tmp_cells_speed6[ind] = vals_7;
      tmp_cells_speed7[ind] = vals_4;
      tmp_cells_speed8[ind] = vals_5;
    }
  } 
  return tot_u / (float)tot_cells;
}

int accelerate_flow_large_align(const t_param params,float*restrict speed1,float*restrict speed3,float*restrict speed5,float*restrict speed6,float*restrict speed7,float*restrict speed8, const int* obstacles)
{
  __assume_aligned(speed1,ALIGN_LARGE);__assume_aligned(speed3,ALIGN_LARGE);__assume_aligned(speed5,ALIGN_LARGE);
  __assume_aligned(speed6,ALIGN_LARGE);__assume_aligned(speed7,ALIGN_LARGE);__assume_aligned(speed8,ALIGN_LARGE);

  __assume_aligned(obstacles, ALIGN_LARGE);
  __assume(params.nx%GRID_EL_DIV==0); __assume(params.ny%GRID_EL_DIV==0);

  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  const int startInd = (params.ny - 2) * params.nx;
  const int endInd = startInd + params.nx;
  #pragma omp simd aligned(speed1:ALIGN_LARGE,speed3:ALIGN_LARGE,speed5:ALIGN_LARGE,speed6:ALIGN_LARGE,speed7:ALIGN_LARGE,speed8:ALIGN_LARGE,obstacles:ALIGN_LARGE)
  for (int ind = startInd; ind < endInd; ind++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ind]
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

int accelerate_flow_mid_align(const t_param params,float*restrict speed1,float*restrict speed3,float*restrict speed5,float*restrict speed6,float*restrict speed7,float*restrict speed8, const int* obstacles)
{
  __assume_aligned(speed1,ALIGN_MID);__assume_aligned(speed3,ALIGN_MID);__assume_aligned(speed5,ALIGN_MID);
  __assume_aligned(speed6,ALIGN_MID);__assume_aligned(speed7,ALIGN_MID);__assume_aligned(speed8,ALIGN_MID);

  __assume_aligned(obstacles, ALIGN_MID);
  __assume(params.nx%GRID_EL_DIV==0); __assume(params.ny%GRID_EL_DIV==0);

  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  const int startInd = (params.ny - 2) * params.nx;
  const int endInd = startInd + params.nx;
  #pragma omp simd aligned(speed1:ALIGN_MID,speed3:ALIGN_MID,speed5:ALIGN_MID,speed6:ALIGN_MID,speed7:ALIGN_MID,speed8:ALIGN_MID,obstacles:ALIGN_MID)
  for (int ind = startInd; ind < endInd; ind++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ind]
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

int accelerate_flow_small_align(const t_param params,float*restrict speed1,float*restrict speed3,float*restrict speed5,float*restrict speed6,float*restrict speed7,float*restrict speed8, const int* obstacles)
{
  __assume_aligned(speed1,ALIGN_SMALL);__assume_aligned(speed3,ALIGN_SMALL);__assume_aligned(speed5,ALIGN_SMALL);
  __assume_aligned(speed6,ALIGN_SMALL);__assume_aligned(speed7,ALIGN_SMALL);__assume_aligned(speed8,ALIGN_SMALL);

  __assume_aligned(obstacles, ALIGN_SMALL);
  __assume(params.nx%GRID_EL_DIV==0); __assume(params.ny%GRID_EL_DIV==0);

  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  const int startInd = (params.ny - 2) * params.nx;
  const int endInd = startInd + params.nx;
  #pragma omp simd aligned(speed1:ALIGN_SMALL,speed3:ALIGN_SMALL,speed5:ALIGN_SMALL,speed6:ALIGN_SMALL,speed7:ALIGN_SMALL,speed8:ALIGN_SMALL,obstacles:ALIGN_SMALL)
  for (int ind = startInd; ind < endInd; ind++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ind]
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


float av_velocity(const t_param params, t_speed_vec cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int ind = ii + jj*params.nx;
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
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

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_vec* cells_ptr, t_speed_vec* tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int alignSize)
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
  cells_ptr->speed0 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  cells_ptr->speed1 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  cells_ptr->speed2 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  cells_ptr->speed3 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  cells_ptr->speed4 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  cells_ptr->speed5 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  cells_ptr->speed6 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  cells_ptr->speed7 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  cells_ptr->speed8 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);


  /* 'helper' grid, used as scratch space */
  tmp_cells_ptr->speed0 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  tmp_cells_ptr->speed1 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  tmp_cells_ptr->speed2 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  tmp_cells_ptr->speed3 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  tmp_cells_ptr->speed4 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  tmp_cells_ptr->speed5 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  tmp_cells_ptr->speed6 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  tmp_cells_ptr->speed7 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);
  tmp_cells_ptr->speed8 = _mm_malloc(params->ny * params->nx * sizeof(float),alignSize);

  /* the map of obstacles */
  *obstacles_ptr = _mm_malloc(sizeof(int) * (params->ny * params->nx),alignSize);

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for(int ind = 0; ind < params->ny * params->nx; ind++) {
    int ii = ind % params->ny;
    int jj = (ind - ii) / params->nx;
    /* centre */
    cells_ptr->speed0[ii + jj*params->nx] = w0;
    /* axis directions */
    cells_ptr->speed1[ii + jj*params->nx] = w1;
    cells_ptr->speed2[ii + jj*params->nx] = w1;
    cells_ptr->speed3[ii + jj*params->nx] = w1;
    cells_ptr->speed4[ii + jj*params->nx] = w1;
    /* diagonals */
    cells_ptr->speed5[ii + jj*params->nx] = w2;
    cells_ptr->speed6[ii + jj*params->nx] = w2;
    cells_ptr->speed7[ii + jj*params->nx] = w2;
    cells_ptr->speed8[ii + jj*params->nx] = w2;

    /* first set all cells in obstacle array to zero */
    (*obstacles_ptr)[ii + jj*params->nx] = 0;

    // Inits temp cells
    tmp_cells_ptr->speed0[ii + jj*params->nx] = w0;
    /* axis directions */
    tmp_cells_ptr->speed1[ii + jj*params->nx] = w1;
    tmp_cells_ptr->speed2[ii + jj*params->nx] = w1;
    tmp_cells_ptr->speed3[ii + jj*params->nx] = w1;
    tmp_cells_ptr->speed4[ii + jj*params->nx] = w1;
    /* diagonals */
    tmp_cells_ptr->speed5[ii + jj*params->nx] = w2;
    tmp_cells_ptr->speed6[ii + jj*params->nx] = w2;
    tmp_cells_ptr->speed7[ii + jj*params->nx] = w2;
    tmp_cells_ptr->speed8[ii + jj*params->nx] = w2;
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
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)_mm_malloc(sizeof(float) * params->maxIters,alignSize);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed_vec* cells, t_speed_vec* tmp_cells,
             int** obstacles_ptr, float** av_vels_ptr)
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

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed_vec cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed_vec cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
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

int write_values(const t_param params, t_speed_vec cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        int ind = ii + jj*params.nx;
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

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

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