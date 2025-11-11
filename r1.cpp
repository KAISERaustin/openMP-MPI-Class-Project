#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring> 
#include <unistd.h>

// fills the array x of size n with positive random numbers
static inline void fill_rand(double* x, int n, unsigned seed) {
    //seeds the random number generator
    srand(seed);
    // fills the array with random numbers
    for (int i = 0; i < n; i++) {
        // generates a random number and places that number into the array
        x[i] = (double)(rand() % 10000 + 1);
    }
}

// computes the checksum for two rows row0 and row1 of the result of A*B
static inline long double checksum(double* A, double* B, int N, int row0, int row1)
{
    // initializes the checksum sum
    long double sum = 0.0;
    // array of the two rows to compute
    int rows[2] = {row0, row1};
    // loop for the two rows
    for ( int i = 0; i < 2; ++i) {
        // retrieves the row index
        int r = rows[i];
        // loops over columns of B
        for ( int j = 0; j < N; ++j) {
            // computes the inner product of row r of A with column j of B
            long double inner_sum = 0.0;
            // loops over elements of the row/column
            for ( int k = 0; k < N; ++k) {
                // accumulates the product into inner_sum
                inner_sum += A[r * N + k] * B[k * N + j];
            }
            // accumulates inner_sum into the total sum
            sum += inner_sum;
        }
    }
    // returns the computed checksum
    return sum;
}

int main(int argc, char** argv) {
    // initializes the MPI environment
    MPI_Init(&argc, &argv);

    // gets the number of processes
    int p = 1;
    // gets the rank of the process
    int id = 0;

    // retrieves the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    // retrieves the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    // retrieves N or the matrix size
    int N = (argc >=2) ? atoi(argv[1]) : 256;
    // retrieves random seed from user or sets to default value
    int SD = (argc >=3) ? atoi(argv[2]) : 1234;

    // allocates space for hostname
    char host[256];
    // ensures null termination
    host[255] = '\0';
    // gets the hostname
    gethostname( host, 255 );

    // computes the number of rows_assigned_to_rank assigned to each process
    int total_rows_per_rank = N / p;
    // computes the remainder_of_rows of rows_assigned_to_rank to be distributed
    int remainder_of_rows = N % p;
    // computes the number of rows_assigned_to_rank for this process (giving some ranks and additional if id < remainder_of_rows)
    int local_rows = total_rows_per_rank + (id < remainder_of_rows ? 1 : 0);
    // computes the row offset for this process
    int row_offset = id * total_rows_per_rank + (id < remainder_of_rows ? id : remainder_of_rows);

    // prints out some information about this process
    printf("Host %s | Rank %d | Amount = %d | Rows [%d .. %d)\n", host, id, local_rows, row_offset, row_offset + local_rows);
    // ALWAYS FLUSH STDOUT AFTER PRINTING IN AN MPI PROGRAM
    fflush(stdout);

    // pointer for rank row count
    int *rows_assigned_to_rank = nullptr;
    // pointer for rank row displacements
    int *starting_row_index = nullptr;
    // pointer for send count
    int *elements_to_send = nullptr;
    // pointer for send displacements
    int *starting_element_offset = nullptr;

    // allocates local matrix
    double *A_local = (double*)malloc(local_rows * N * sizeof(double));
    // allocates for full matrix B
    double *B = (double*)malloc(N * N * sizeof(double));
    // allocates local result matrix C
    double *C_local = (double*)calloc(local_rows * N, sizeof(double));

    // pointer for full matrix A
    double *A = nullptr;


    // only allow rank 0 to allocate and fill rows_assigned_to_rank
    if(id == 0) {
        // allocate each ranks row counts
        rows_assigned_to_rank = (int*)malloc(p * sizeof(int));
        // allocate each ranks row index
        starting_row_index = (int*)malloc(p * sizeof(int));

        // fills in rows_assigned_to_rank and starting_row_index
        int row_offset = 0;
        for ( int i = 0; i < p; ++i)
        {
            // computes the number of rows assigned to rank i
            rows_assigned_to_rank[i] = total_rows_per_rank + (i < remainder_of_rows ? 1 : 0);
            // computes the starting row index for rank i
            starting_row_index[i] = row_offset;
            // advances the row offset for the next rank
            row_offset += rows_assigned_to_rank[i];
        }

        // allocates each ranks element counts
        elements_to_send = (int*)malloc(p * sizeof(int));
        // allocates each ranks element offsets
        starting_element_offset = (int*)malloc(p * sizeof(int));
        // fills in elements_to_send and starting_element_offset
        for ( int i = 0; i < p; ++i) 
        {
            // computers the number of elements to send to rank i
            elements_to_send[i] = rows_assigned_to_rank[i] * N;
            // computes the starting element offset for rank i
            starting_element_offset[i] = starting_row_index[i] * N;
        }

        // allocates full matrix A
        A = (double*)malloc(N * N * sizeof(double));
        // fills full matrix A with random numbers
        fill_rand(A, N * N, SD + 1);
        // fills matrix B with random numbers
        fill_rand(B, N * N, SD + 2);
    }

    // broadcasts matrix B to all processes
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // scatters each ranks matrix A to all processes
    MPI_Scatterv(A, elements_to_send, 
                starting_element_offset, MPI_DOUBLE, A_local, local_rows * N, 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // synchronizes all processes before starting the timer
    MPI_Barrier(MPI_COMM_WORLD);
    // starts the timer and stores the start time in t0
    double t0 = MPI_Wtime();
    double first_two_rows_sum = 0.0;

    // performs local matrix multiplication C_local = A_local * B
    for (int i = 0; i < local_rows; ++i) {
        // points to the beginning of the correct row of C and A
        //C[row_offset + i][0] and A[row_offset + i][0]
        double* C_row = C_local + i * N; 
        double* A_row = A_local + i * N;
        // performs the product of the row of A with columns of B
        for ( int k = 0; k < N; ++k) {
            // retrieves the value of A[row_offset + i][k]
            double A_row_k = A_row[k];
            // points to the beginning of row k of B
            //B[k][0]
            double* B_row = B + k * N;
            // performs the multiplication of A[i][k] with row k of B and accumulates into C[i][j]
            for( int j = 0; j < N; ++j) {
                // C[row_offset + i][j] += A[row_offset + i][k] * B[k][j];
                C_row[j] += A_row_k * B_row[j];
            }
        }
    }

    // synchronizes all processes before stopping the timer
    MPI_Barrier(MPI_COMM_WORLD);

    // stops the timer and stores the end time in t1
    double t1 = MPI_Wtime();
    // computes the elapsed time
    double elapsed = t1 - t0;

    // computes the sum of the first two rows of C_local if this rank has those rows
    if( row_offset == 0 ) {
        // only printing first two rows from rank 0
        int rows_to_print = 2;
        // loops over the rows to print
        for (int i = 0; i < rows_to_print && i < local_rows; ++i) {
            // points to the beginning of the correct row of C_local
            double* C_row = C_local + i * N;
            // loops over columns of the row
            for ( int j = 0; j < N; ++j) {
                // adds C[i][j] to the sum
                first_two_rows_sum += C_row[j];
            }
        }
    }

    // initializes the pointer for the full result matrix C
    double *C_full = nullptr;
    // only allow rank 0 to allocate full C matrix
    if(id == 0)
    {
        // allocates full result matrix C
        C_full = (double*)malloc(N * N * sizeof(double));
    }

    // gathers the local result matrices C_local into the full result matrix C_full
    MPI_Gatherv(C_local, local_rows * N,
                MPI_DOUBLE, C_full, elements_to_send, starting_element_offset, 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double first_two_rows_sum_global = 0.0;
    MPI_Reduce(&first_two_rows_sum, &first_two_rows_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // only allow rank 0 to compute and print the checksum and elapsed time
    if(id == 0) 
    {

        // computes the checksum for first two rows
        long double check = checksum(A, B, N, 0, 1);

        // prints out N, number of processors, elapsed time, and checksum
        printf("N = %d Processor = %d Elapsed time: %.6f sec.\n", N, p, elapsed);
        // ALWAYS FLUSH STDOUT AFTER PRINTING IN AN MPI PROGRAM
        fflush(stdout);
        // prints out the checksum comparison
        printf("Checksum of first two rows: %.6Lf\nChecksum of the MPI first two rows: %.6Lf\n", check, first_two_rows_sum_global);
        // ALWAYS FLUSH STDOUT AFTER PRINTING IN AN MPI PROGRAM
        fflush(stdout);

    }

    // frees matrix A_local
    free(A_local);
    // free matrix B
    free(B);
    // frees matrix C_local
    free(C_local);

    // only allow rank 0 to free everything else
    if(id == 0) 
    {
        // frees full matrix A
        free(A);
        // frees full matrix C
        free(C_full);
        // free rows_assigned_to_rank 
        free(rows_assigned_to_rank);
        // free displacements
        free(starting_row_index);
        // free send counts
        free(elements_to_send);
        // free send displacements
        free(starting_element_offset);
    }
    // finalizes the MPI environment
    MPI_Finalize();
    // returns success
    return 0;
    
}