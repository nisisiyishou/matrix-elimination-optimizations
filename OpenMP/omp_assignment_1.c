#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

void print_matrix(double **T, int rows, int cols);
int test(double **t1, double **t2, int rows);

int main(int agrc, char *agrv[])
{
    double *a0; // auxiliary 1D for 2D matrix a
    double **a; // 2D matrix for sequential computation
    double *d0; // auxiliary 1D for 2D matrix d
    double **d; // 2D matrix, same initial data as a for computation with loop unrolling
    int n;      // input size
    int n0;
    int i, j, k;
    int x;
    int indk;
    double amax;
    register double di00, di10, di20, di30;
    register double dj00, dj01, dj02, dj03;
    double c;
    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed;
    int end;
    int block_size = 4;
    register double ai00, ai10, ai20, ai30;
    register double ai01, ai11, ai21, ai31;
    register double ai02, ai12, ai22, ai32;
    register double ai03, ai13, ai23, ai33;

    register double bj00, bj10, bj20, bj30;
    register double bj01, bj11, bj21, bj31;
    register double bj02, bj12, bj22, bj32;
    register double bj03, bj13, bj23, bj33;

    int nthreads;
    double tt;

    if (agrc == 3)
    {
        n = atoi(agrv[1]);
        printf("The matrix size:  %d * %d \n", n, n);
        nthreads = atoi(agrv[2]);
        printf("The number of threads nthreads = %d\n\n", nthreads);
    }
    else
    {
        printf("Usage: %s n\n\n"
               " n: the matrix size\n"
               " nthreads: the number of threads\n\n", agrv[0]);
        return 1;
    }

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
    //    printf("matrix a: \n");
    //    print_matrix(a, n, n);
    //    printf("matrix d: \n");
    //    print_matrix(d, n, n);

    printf("Starting sequential computation...\n\n");
    /**** Sequential computation *****/
    srand(time(0));
    tt = omp_get_wtime();
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
    tt = omp_get_wtime() - tt;
    printf("sequential calculation time: %f\n\n", tt);

    omp_set_num_threads(nthreads);
    printf("Starting sequential computation with loop unrolling and blocking...\n\n");

    /***sequential computation with loop unrolling and blocking***/
    tt = omp_get_wtime();
    for (x = 0; x < n - 1; x += block_size)
    {
        if (x + block_size < n)
            end = x + block_size;
        else
            end = n;

        for (i = x; i < end; i++)
        {
            amax = d[i][i];
            indk = i;
            for (k = i + 1; k < n; k++)
                if (fabs(d[k][i]) > fabs(amax))
                {
                    amax = d[k][i];
                    indk = k;
                }

            if (amax == 0.0)
            {
                printf("the matrix is singular\n");
                exit(1);
            }
            else if (indk != i) // swap row i and row k
            {
                for (j = 0; j < n; j++)
                {
                    c = d[i][j];
                    d[i][j] = d[indk][j];
                    d[indk][j] = c;
                }
            }

            for (k = i + 1; k < n; k++)
                d[k][i] = d[k][i] / d[i][i];

            n0 = (end - (i+1))/4 * 4 + i+1;

            for (k = i + 1; k < n0; k += 4)
            {
                di00 = d[k][i];
                di10 = d[k + 1][i];
                di20 = d[k + 2][i];
                di30 = d[k + 3][i];
                
                #pragma omp parallel private(j, c, dj00, dj01, dj02, dj03)
                #pragma omp for nowait
                for (j = i + 1; j < n0; j += 4)
                {
                    dj00 = d[i][j];
                    dj01 = d[i][j + 1];
                    dj02 = d[i][j + 2];
                    dj03 = d[i][j + 3];
                    d[k][j] -= di00 * dj00;
                    d[k][j + 1] -= di00 * dj01;
                    d[k][j + 2] -= di00 * dj02;
                    d[k][j + 3] -= di00 * dj03;
                    d[k + 1][j] -= di10 * dj00;
                    d[k + 1][j + 1] -= di10 * dj01;
                    d[k + 1][j + 2] -= di10 * dj02;
                    d[k + 1][j + 3] -= di10 * dj03;
                    d[k + 2][j] -= di20 * dj00;
                    d[k + 2][j + 1] -= di20 * dj01;
                    d[k + 2][j + 2] -= di20 * dj02;
                    d[k + 2][j + 3] -= di20 * dj03;
                    d[k + 3][j] -= di30 * dj00;
                    d[k + 3][j + 1] -= di30 * dj01;
                    d[k + 3][j + 2] -= di30 * dj02;
                    d[k + 3][j + 3] -= di30 * dj03;
                }

                for (j = n0; j < end; j++)
                {
                    c = d[i][j];
                    d[k][j] -= di00 * c;
                    d[k + 1][j] -= di10 * c;
                    d[k + 2][j] -= di20 * c;
                    d[k + 3][j] -= di30 * c;
                }
            }

            for (k = n0; k < n; k++)
            {
                c = d[k][i];
                for (j = i + 1; j < end; j++)
                    d[k][j] -= c * d[i][j];
            }
        }
        // do delayed update on d[x:end][end+1:n]
        n0 = (n - end)/4 * 4 + end;
        for (k = x; k < end; k+=1) //column
        {
            for (i = x; i < k; i+=1) // blue row
            {
                c = d[k][i];
                #pragma omp parallel shared(d,i,k) private(j)
                {
                    #pragma omp for nowait
                    for (j = end; j < n0; j+=4) // pink row
                    {
                        d[k][j] -= c*d[i][j];
                        d[k][j+1] -= c*d[i][j+1];
                        d[k][j+2] -= c*d[i][j+2];
                        d[k][j+3] -= c*d[i][j+3];
                    }
                    #pragma omp for nowait
                    for (j=n0; j<n; j++)
                    {
                        d[k][j] -= c*d[i][j];
                    }
                }
            }
        }
        
        #pragma omp parallel shared(d) private(i, j, k,\
        ai00, ai01, ai02, ai03, ai10, ai11, ai12, ai13, \
        ai20, ai21, ai22, ai23, ai30, ai31, ai32, ai33, \
        bj00, bj01, bj02, bj03, bj10, bj11, bj12, bj13, \
        bj20, bj21, bj22, bj23, bj30, bj31, bj32, bj33)
        {
            #pragma omp for nowait
            for (i = end; i < n0; i+=4) // column
            {
                for (k = x; k < end; k+=4) // iterator
                {
                    ai00 = d[i][k];   ai01 = d[i][k+1];   ai02 = d[i][k+2];   ai03 = d[i][k+3];
                    ai10 = d[i+1][k]; ai11 = d[i+1][k+1]; ai12 = d[i+1][k+2]; ai13 = d[i+1][k+3];
                    ai20 = d[i+2][k]; ai21 = d[i+2][k+1]; ai22 = d[i+2][k+2]; ai23 = d[i+2][k+3];
                    ai30 = d[i+3][k]; ai31 = d[i+3][k+1]; ai32 = d[i+3][k+2]; ai33 = d[i+3][k+3];
                    for (j = end; j < n0; j+=4) //row
                    {
                        bj00 = d[k][j];   bj01 = d[k][j+1];   bj02 = d[k][j+2];   bj03 = d[k][j+3];
                        bj10 = d[k+1][j]; bj11 = d[k+1][j+1]; bj12 = d[k+1][j+2]; bj13 = d[k+1][j+3];
                        bj20 = d[k+2][j]; bj21 = d[k+2][j+1]; bj22 = d[k+2][j+2]; bj23 = d[k+2][j+3];
                        bj30 = d[k+3][j]; bj31 = d[k+3][j+1]; bj32 = d[k+3][j+2]; bj33 = d[k+3][j+3];
                        d[i][j] = d[i][j] - ai00*bj00 - ai01*bj10 - ai02*bj20 - ai03*bj30;
                        d[i][j+1] = d[i][j+1] - ai00*bj01 - ai01*bj11 - ai02*bj21 - ai03*bj31;
                        d[i][j+2] = d[i][j+2] - ai00*bj02 - ai01*bj12 - ai02*bj22 - ai03*bj32;
                        d[i][j+3] = d[i][j+3] - ai00*bj03 - ai01*bj13 - ai02*bj23 - ai03*bj33;

                        d[i+1][j] = d[i+1][j] - ai10*bj00 - ai11*bj10 - ai12*bj20 - ai13*bj30;
                        d[i+1][j+1] = d[i+1][j+1] - ai10*bj01 - ai11*bj11 - ai12*bj21 - ai13*bj31;
                        d[i+1][j+2] = d[i+1][j+2] - ai10*bj02 - ai11*bj12 - ai12*bj22 - ai13*bj32;
                        d[i+1][j+3] = d[i+1][j+3] - ai10*bj03 - ai11*bj13 - ai12*bj23 - ai13*bj33;

                        d[i+2][j] = d[i+2][j] - ai20*bj00 - ai21*bj10 - ai22*bj20 - ai23*bj30;
                        d[i+2][j+1] = d[i+2][j+1] - ai20*bj01 - ai21*bj11 - ai22*bj21 - ai23*bj31;
                        d[i+2][j+2] = d[i+2][j+2] - ai20*bj02 - ai21*bj12 - ai22*bj22 - ai23*bj32;
                        d[i+2][j+3] = d[i+2][j+3] - ai20*bj03 - ai21*bj13 - ai22*bj23 - ai23*bj33;

                        d[i+3][j] = d[i+3][j] - ai30*bj00 - ai31*bj10 - ai32*bj20 - ai33*bj30;
                        d[i+3][j+1] = d[i+3][j+1] - ai30*bj01 - ai31*bj11 - ai32*bj21 - ai33*bj31;
                        d[i+3][j+2] = d[i+3][j+2] - ai30*bj02 - ai31*bj12 - ai32*bj22 - ai33*bj32;
                        d[i+3][j+3] = d[i+3][j+3] - ai30*bj03 - ai31*bj13 - ai32*bj23 - ai33*bj33;
                    }
                    // remaining case
                    for (j=n0; j<n; j++)
                    {
                        d[i][j] = d[i][j] - ai00*d[k][j] - ai01*d[k+1][j] - ai02*d[k+2][j] - ai03*d[k+3][j];
                        d[i+1][j] = d[i+1][j] - ai10*d[k][j] - ai11*d[k+1][j] - ai12*d[k+2][j] - ai13*d[k+3][j];
                        d[i+2][j] = d[i+2][j] - ai20*d[k][j] - ai21*d[k+1][j] - ai22*d[k+2][j] - ai23*d[k+3][j];
                        d[i+3][j] = d[i+3][j] - ai30*d[k][j] - ai31*d[k+1][j] - ai32*d[k+2][j] - ai33*d[k+3][j];
                    }
                }
            }
        }
        for (i=n0; i<n; i++)
        {
            for (k=x; k<end; k++)
            {
                for (j=end; j<n; j++)
                {
                    d[i][j] -= d[i][k]*d[k][j];
                }
            }
        }
    }

    tt = omp_get_wtime() - tt;
    printf("sequential calculation with loop unrolling time: %f\n\n", tt);

    printf("Starting comparison...\n\n");
    int cnt;
    cnt = test(a, d, n);
    if (cnt == 0)
        printf("Done. There are no differences!\n");
    else
        printf("Results are incorrect! The number of different elements is %d\n", cnt);
    // print_matrix(a, n, n);
    // print_matrix(d, n, n);
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
