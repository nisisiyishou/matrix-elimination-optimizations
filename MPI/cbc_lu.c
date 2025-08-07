/******************************************************************************
 * FILE: gepp_3.c
 * DESCRIPTION:
 * The C program for Gaussian elimination with partial pivoting
 * Try to use loop unrolling to improve the performance - third attempt
 * Unroll both j and k loops in rank 1 updating for trailing submatrix
 *   with unrolling factor = 4
 * The performance is better than the first two attempts as data loaded into
 *   registers can be used multiple times before being replaced
 * We can see a big performance improvement when compiling the program
 * without using optimization options
 * However, if we use "gcc -O3", this loop unrolling program only chieved
 * around 10% performance improvement
 * Therefore, the program needs a further revision to enhance the performance
 * AUTHOR: Bing Bing Zhou
 * LAST REVISED: 1/06/2023
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

void print_matrix(double **T, int rows, int cols);
int test(double **t1, double **t2, int rows);

int main(int argc, char *argv[])
{
    double *a0; // auxiliary 1D for 2D matrix a
    double **a; // 2D matrix for sequential computation
    double *d0; // auxiliary 1D for 2D matrix d
    double **d; // 2D matrix, same initial data as a for computation with loop unrolling
    double *dK0;
    double **dK;
    double *dW0;
    double **dW;
    int n, K; // input size
    int n0;
    int i, j, k, l;
    int b;
    int ib, kb;

    int indk;
    double amax;
    register double di00, di10, di20, di30;
    register double dj00, dj01, dj02, dj03;
    register double ai00, ai10, ai20, ai30;
    register double ai01, ai11, ai21, ai31;
    register double ai02, ai12, ai22, ai32;
    register double ai03, ai13, ai23, ai33;

    register double bj00, bj10, bj20, bj30;
    register double bj01, bj11, bj21, bj31;
    register double bj02, bj12, bj22, bj32;
    register double bj03, bj13, bj23, bj33;
    double c;
    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed;

    int myid, numprocs;
    int Nb, bm;
    int q, r;
    int currproc, end;
    int cycle, pivot;
    int after_blocks_start;
    MPI_Status status;
    MPI_Datatype col_t, col_sub, col_dW;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0)
        if (argc != 2)
        {
            if (myid == 0)
            {
                printf("Wrong number of arguments.\n");
                printf("Please enter the command in the following format:\n");
                printf("mpirun –np [proc num] main [matrix size n]\n");
                printf("SAMPLE: mpirun –np 3 main 20\n");
            }

            MPI_Finalize();
            return 0;
        }

    if (numprocs == 1)
    {
        printf("\n\nThe number of processes created is just 1 - a trivial problem!\n\n");

        MPI_Finalize();
        return 0;
    }

    n = atoi(argv[1]); // n = 14
    b = 8;

    MPI_Type_vector(n, b, n, MPI_DOUBLE, &col_t);
    MPI_Type_commit(&col_t);

    //    printf("matrix a: \n");
    //    print_matrix(a, n, n);
    //    printf("matrix d: \n");
    //    print_matrix(d, n, n);

    if (myid == 0)
    {
        printf("Creating and initializing matrices...\n\n");
        /*** Allocate contiguous memory for 2D matrices ***/
        a0 = (double *)malloc(n * n * sizeof(double));
        a = (double **)malloc(n * sizeof(double *));
        for (i = 0; i < n; i++)
        {
            a[i] = a0 + i * n;
        }
        d0 = (double *)malloc(n * n * sizeof(double));
        d = (double **)malloc(n * sizeof(double *));
        for (i = 0; i < n; i++)
        {
            d[i] = d0 + i * n;
        }

        srand(time(0));
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n; j++)
            {
                a[i][j] = (double)rand() / RAND_MAX;
                d[i][j] = a[i][j];
            }
        }
        printf("Starting sequential computation...\n\n");
        /**** Sequential computation *****/
        gettimeofday(&start_time, 0);
        for (i = 0; i < n - 1; i++)
        {
            // find and record k where |a(k,i)|=𝑚ax|a(j,i)|
            amax = a[i][i];
            indk = i;
            for (k = i + 1; k < n; k++)
            {
                if (fabs(a[k][i]) > fabs(amax))
                {
                    amax = a[k][i];
                    indk = k;
                }
            }

            // exit with a warning that a is singular
            if (amax == 0)
            {
                printf("matrix is singular!\n");
                exit(1);
            }
            else if (indk != i) // swap row i and row k
            {
                for (j = 0; j < n; j++)
                {
                    c = a[i][j];
                    a[i][j] = a[indk][j];
                    a[indk][j] = c;
                }
            }

            // store multiplier in place of A(j,i)
            for (k = i + 1; k < n; k++)
            {
                a[k][i] = a[k][i] / a[i][i];
            }

            // subtract multiple of row a(i,:) to zero out a(j,i)
            for (k = i + 1; k < n; k++)
            {
                c = a[k][i];
                for (j = i + 1; j < n; j++)
                {
                    a[k][j] -= c * a[i][j];
                }
            }
        }
        gettimeofday(&end_time, 0);

        // print the running time
        seconds = end_time.tv_sec - start_time.tv_sec;
        microseconds = end_time.tv_usec - start_time.tv_usec;
        elapsed = seconds + 1e-6 * microseconds;
        printf("sequential calculation time: %f\n\n", elapsed);
        // printf("matrix a: \n");
        // print_matrix(a, n, n);
    }

    if (myid == 0)
    {
        printf("Starting column block cyclic partitioning computation...\n\n");
        gettimeofday(&start_time, 0);
    }
    Nb = n / b;        // the total number of column blocks 14/2 = 7
    bm = b * n;        // column block size 2*14 = 28
    q = Nb / numprocs; // each process gets at least q column blocks q = 7/3 = 2
    r = Nb % numprocs; // remaining column blocks 7%3 = 1
    if (myid < r)      // one more column block for each of the first r processes
        kb = q + 1;    // if myid == 0, kb = 3 -> rank 0 has 3 blocks
    else
        kb = q; // else myid == 1 or 2, kb = 2 -> rank 1 and 2 have 2 blocks
    K = kb * b; // number of columns in submatrix and may be different for different processes
    // K = 2 * 2 = 4 or K = 2 * 3 = 6

    MPI_Type_vector(n, b, K, MPI_DOUBLE, &col_sub);
    MPI_Type_commit(&col_sub);
    MPI_Type_vector(n, b, b, MPI_DOUBLE, &col_dW);
    MPI_Type_commit(&col_dW);

    dK0 = malloc(n * K * sizeof(double)); // K = 4 or 6, N = 14，submatrix memory allocation
    dK = malloc(n * sizeof(double *));
    if (dK == NULL)
    {
        fprintf(stderr, "**dK out of memory\n");
        exit(1);
    }
    for (i = 0; i < n; i++)  // K = 4 or K = 6
        dK[i] = &dK0[K * i]; // pointer to start of each row in column block submatrix

    dW0 = malloc(n * b * sizeof(double)); //
    dW = malloc(n * sizeof(double *));
    if (dW == NULL)
    {
        fprintf(stderr, "**dW out of memory\n");
        exit(1);
    }
    for (i = 0; i < n; i++)
    {
        dW[i] = &dW0[b * i]; // pointer to start of each row in multiplier buffer
    }

    if (myid == 0)
    {
        ib = 0;
        for (j = 0; j < q; j++) // q = 2, j < 2
        {
            for (i = 1; i < numprocs; i++) // numprocs = 3, i < 3
            {
                ib += b; // b = 2, A[2][0], bn = 28, i is the rank of the destination process
                // process (i = 1 in this case), the next 1 is the tag
                MPI_Send(&d[0][ib], 1, col_t, i, 1, MPI_COMM_WORLD); //
            }
            ib += b; // leave one row block for process 0 ib = 2 + 2 = 4
        }
        if (r > 1) // send remaining blocks, one block to each processe with myid < r
            for (i = 1; i < r; i++)
            {
                ib += b;
                MPI_Send(&d[0][ib], 1, col_t, i, 1, MPI_COMM_WORLD);
            }
    }
    else
    { // all other processes 1 and 2 receive a submatrix from process 0
        ib = 0;
        for (i = 0; i < kb; i++) // kb = 2 or 3, i < 2 or 3
        {                        // 0 means receiving from process 0, dK[0][0] and dK[0][2]
            MPI_Recv(&dK[0][ib], 1, col_sub, 0, 1, MPI_COMM_WORLD, &status);
            ib += b; // ib = 2
        }
    }

    // copy process 0's submatrix to dK
    if (myid == 0)
    {
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < K; j++)
            {
                cycle = j / b;                        // 4th column in submatrix, cycle = (4 - 1) / 2 = 1
                pivot = cycle * b * numprocs + j % b; // pivot = 1 * 2 * 3 + 3 % 2 = 7, 7th column in matrix
                dK[i][j] = d[i][pivot];
            }
        }
    }

    for (j = 0; j < n - 1; j += b)
    {
        end = j + b;
        for (i = j; i < end; i++)
        {
            currproc = j / b % numprocs;
            cycle = i / (b * numprocs); // cycle = 7 / (2*3) = 1
            pivot = cycle * b + i % b;  // pivot = 1 * 2 + 7 % 2 = 3, 4th column in submatrix
            if (myid == currproc)
            {
                amax = dK[i][pivot];
                indk = i;
                for (k = i + 1; k < n; k++)
                {
                    if (fabs(dK[k][pivot]) > fabs(amax))
                    {
                        amax = dK[k][pivot];
                        indk = k;
                    }
                }

                if (amax == 0)
                {
                    printf("matrix is singular!\n");
                    exit(1);
                }
            }
            MPI_Bcast(&indk, 1, MPI_INT, currproc, MPI_COMM_WORLD); // informing other processes new indk
            if (indk != i)                                          // swap row i and row k
            {
                for (k = 0; k < K; k++)
                {
                    c = dK[i][k];
                    dK[i][k] = dK[indk][k];
                    dK[indk][k] = c;
                }
            }

            // store multiplier in place of A(k,i)
            if (myid == currproc)
            {
                for (k = i + 1; k < n; k++)
                {
                    dK[k][pivot] = dK[k][pivot] / dK[i][pivot];
                }
                for (k = i + 1; k < n; k++) // brown upper triangle in column block
                {
                    c = dK[k][pivot];
                    for (l = pivot + 1; l < cycle * b + b; l++)
                    {
                        dK[k][l] -= c * dK[i][l];
                    }
                }
            }
        }

        if (myid == currproc)
        {
            for (i = 0; i < n; i++) // copying current column block multipliers (U) to dW
            {
                for (k = 0; k < b; k++)
                {
                    dW[i][k] = dK[i][cycle * b + k];
                }
            }
        }
        MPI_Bcast(&dW[0][0], 1, col_dW, currproc, MPI_COMM_WORLD); // broadcasting dW to all processes

        // pink part update
        
        if (myid > currproc)
        {
            after_blocks_start = (j / (b * numprocs)) * b;
        }
        else
        {
            after_blocks_start = (j / (b * numprocs)) * b + b;
        }

        for (i = 0; i < b; i++)
        {
            for (k = 0; k < i; k++)
            {
                di00 = dW[i + j][k];
                for (l = after_blocks_start; l < K; l+=4) // only update the columns after the current column block
                {
                    dK[i + j][l] -= di00 * dK[k + j][l];
                    dK[i + j][l+1] -= di00 * dK[k + j][l+1];
                    dK[i + j][l+2] -= di00 * dK[k + j][l+2];
                    dK[i + j][l+3] -= di00 * dK[k + j][l+3];
                }
            }
        }

        // // green part update
        for (i = end; i < n; i+=4)
        {
            for (k = 0; k < b; k+=4)
            {
                ai00 = dW[i][k];   ai01 = dW[i][k+1];   ai02 = dW[i][k+2];   ai03 = dW[i][k+3];
                ai10 = dW[i+1][k]; ai11 = dW[i+1][k+1]; ai12 = dW[i+1][k+2]; ai13 = dW[i+1][k+3];
                ai20 = dW[i+2][k]; ai21 = dW[i+2][k+1]; ai22 = dW[i+2][k+2]; ai23 = dW[i+2][k+3];
                ai30 = dW[i+3][k]; ai31 = dW[i+3][k+1]; ai32 = dW[i+3][k+2]; ai33 = dW[i+3][k+3];
                for (l = after_blocks_start; l < K; l+=4) // only update the columns after the current column block
                {
                    bj00 = dK[k+j][l];   bj01 = dK[k+j][l+1];   bj02 = dK[k+j][l+2];   bj03 = dK[k+j][l+3];
                    bj10 = dK[k+j+1][l]; bj11 = dK[k+j+1][l+1]; bj12 = dK[k+j+1][l+2]; bj13 = dK[k+j+1][l+3];
                    bj20 = dK[k+j+2][l]; bj21 = dK[k+j+2][l+1]; bj22 = dK[k+j+2][l+2]; bj23 = dK[k+j+2][l+3];
                    bj30 = dK[k+j+3][l]; bj31 = dK[k+j+3][l+1]; bj32 = dK[k+j+3][l+2]; bj33 = dK[k+j+3][l+3];
                    dK[i][l] = dK[i][l] - ai00*bj00 - ai01*bj10 - ai02*bj20 - ai03*bj30;
                    dK[i][l+1] = dK[i][l+1] - ai00*bj01 - ai01*bj11 - ai02*bj21 - ai03*bj31;
                    dK[i][l+2] = dK[i][l+2] - ai00*bj02 - ai01*bj12 - ai02*bj22 - ai03*bj32;
                    dK[i][l+3] = dK[i][l+3] - ai00*bj03 - ai01*bj13 - ai02*bj23 - ai03*bj33;

                    dK[i+1][l] = dK[i+1][l] - ai10*bj00 - ai11*bj10 - ai12*bj20 - ai13*bj30;
                    dK[i+1][l+1] = dK[i+1][l+1] - ai10*bj01 - ai11*bj11 - ai12*bj21 - ai13*bj31;
                    dK[i+1][l+2] = dK[i+1][l+2] - ai10*bj02 - ai11*bj12 - ai12*bj22 - ai13*bj32;
                    dK[i+1][l+3] = dK[i+1][l+3] - ai10*bj03 - ai11*bj13 - ai12*bj23 - ai13*bj33;

                    dK[i+2][l] = dK[i+2][l] - ai20*bj00 - ai21*bj10 - ai22*bj20 - ai23*bj30;
                    dK[i+2][l+1] = dK[i+2][l+1] - ai20*bj01 - ai21*bj11 - ai22*bj21 - ai23*bj31;
                    dK[i+2][l+2] = dK[i+2][l+2] - ai20*bj02 - ai21*bj12 - ai22*bj22 - ai23*bj32;
                    dK[i+2][l+3] = dK[i+2][l+3] - ai20*bj03 - ai21*bj13 - ai22*bj23 - ai23*bj33;

                    dK[i+3][l] = dK[i+3][l] - ai30*bj00 - ai31*bj10 - ai32*bj20 - ai33*bj30;
                    dK[i+3][l+1] = dK[i+3][l+1] - ai30*bj01 - ai31*bj11 - ai32*bj21 - ai33*bj31;
                    dK[i+3][l+2] = dK[i+3][l+2] - ai30*bj02 - ai31*bj12 - ai32*bj22 - ai33*bj32;
                    dK[i+3][l+3] = dK[i+3][l+3] - ai30*bj03 - ai31*bj13 - ai32*bj23 - ai33*bj33;
                    
                }
            }
        }
    }

    /* process 0 receives column blocks from other processes */
    if (myid == 0)
    {
        ib = 0;
        for (j = 0; j < q; j++) // q = 2
        {
            for (i = 1; i < numprocs; i++)
            {
                ib += b;
                MPI_Recv(&d[0][ib], 1, col_t, i, 2, MPI_COMM_WORLD, &status);
            }
            ib += b; // leave one block for process 0
        }
        if (r > 1)
            for (i = 1; i < r; i++)
            {
                ib += b;
                MPI_Recv(&d[0][ib], 1, col_t, i, 2, MPI_COMM_WORLD, &status);
            }
    }
    else
    {           // other processes send the submatrix back to process 0
        ib = 0; // starting address
        for (i = 0; i < kb; i++)
        {
            MPI_Send(&dK[0][ib], 1, col_sub, 0, 2, MPI_COMM_WORLD); // submatrix
            ib += b;
        }
    }

    // copy process 0's submatrix back to process 0 d
    if (myid == 0)
    {
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < K; j++)
            {
                cycle = j / b;                        // 4th column in submatrix, cycle = (4 - 1) / 2 = 1
                pivot = cycle * b * numprocs + j % b; // pivot = 1 * 2 * 3 + 3 % 2 = 7, 7th column in matrix
                d[i][pivot] = dK[i][j];
            }
        }
        gettimeofday(&end_time, 0);
        seconds = end_time.tv_sec - start_time.tv_sec;
        microseconds = end_time.tv_usec - start_time.tv_usec;
        elapsed = seconds + 1e-6 * microseconds;
        printf("sequential calculation with column block cyclic partitioning time: %f\n\n", elapsed);

        printf("Starting comparison...\n\n");
        int cnt;
        cnt = test(a, d, n);
        if (cnt == 0)
            printf("Done. There are no differences!\n");
        else
            printf("Results are incorrect! The number of different elements is %d\n", cnt);

        // printf("matrix a: \n");
        // print_matrix(a, n, n);
        // printf("matrix d: \n");
        // print_matrix(d, n, n);
    }

    MPI_Finalize();
    return 0;
}

void print_matrix(double **T, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.2f   ", T[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
    return;
}

int test(double **t1, double **t2, int rows)
{
    int i, j;
    int cnt;
    cnt = 0;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < rows; j++)
        {
            if ((t1[i][j] - t2[i][j]) * (t1[i][j] - t2[i][j]) > 1.0e-16)
            {
                cnt += 1;
            }
        }
    }

    return cnt;
}