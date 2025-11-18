#include <omp.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring> 

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

// parses the schedule string into an omp_sched_t enum value
static omp_sched_t parse_schedule(char* schedule){
    // checks to see if there is a schedule string
    if ( !schedule ) return omp_sched_static;
    // compares the schedule string to known schedule types
    if ( strcmp(schedule, "static") == 0 ) return omp_sched_static;
    if ( strcmp(schedule, "dynamic") == 0 ) return omp_sched_dynamic;
    if ( strcmp(schedule, "guided") == 0 ) return omp_sched_guided;
    if ( strcmp(schedule, "auto") == 0 ) return omp_sched_auto;
    // defaults to static scheduling
    return omp_sched_static;
}

int main(int argc, char** argv) {
    // retrieves N or the matrix size
    int N = (argc >= 2) ? atoi(argv[1]) : 256;
    // retrieves random seed from user or sets to default value
    int SD = (argc >=3) ? atoi(argv[2]) : 1234;
    // retrieves number of threads from user or sets to default value
    int T = (argc >=4) ? atoi(argv[3]) : 0;
    // retrieves schedule type from user or sets to default value of static 
    const char* schedule_string = (argc >=5) ? argv[4] : "static";
    // retrieves chunk size from user or sets to default value of 8
    int CHUNK = (argc >=6) ? atoi(argv[5]) : 8;

    // allocates space for hostname
    char host[256];
    // ensures null termination
    host[255] = '\0';
    // gets the hostname
    gethostname( host, 255 );

    // allocates the size of the matrices A, B, and C
    double SIZE = (double)N * (double)N;
    // allocates matrices A, B, and C
    double* A = (double*)malloc( SIZE * sizeof(double) );
    double* B = (double*)malloc( SIZE * sizeof(double) );
    double* C = (double*)calloc( SIZE,  sizeof(double) );

    // fills matrices A and B with random numbers
    fill_rand( A, (int)SIZE, SD + 1 );
    fill_rand( B, (int)SIZE, SD + 2 );

    // sets the number of threads if T > 0
    if ( T > 0 ) omp_set_num_threads(T);
    
    // sets the schedule
    omp_set_schedule(parse_schedule((char*)schedule_string), CHUNK);

    // initializes the thread count
    int thread_count = omp_get_max_threads();

    // starts the timer and stores the start time in t0
    double t0 = omp_get_wtime();

    // performs matrix multiplication C = A * B
    int j, k;
#pragma omp parallel for schedule(runtime) shared (A, B, C, N ) private ( j, k )
    // loops over rows of A
    for ( int i = 0; i < N; ++i ) {
        // points to the beginning of the correct row of C
        double* C_row = C + i * N;
        // points to the beginning of the correct row of A
        double* A_row = A + i * N;
        // performs the product of the row of A with columns of B
        for ( k = 0; k < N; ++ k ) {
            // retrieves the value of A[i][k]
            double A_row_k = A_row[k];
            // points to the beginning of row k of B
            double* B_row = B + k * N;
            // performs the multiplication of A[i][k] with row k of B and accumulates into C[i][j]
            for ( j = 0; j < N; j++ ) {
                // C[i][j] += A[i][k] * B[k][j];
                C_row[j] += A_row_k * B_row[j];
            }
        }
    }

    // stops the timer and stores the end time in t1
    double t1 = omp_get_wtime();

    // computes the sum of the first two rows of C
    double first_two_rows_sum = 0.0;
    // parallelizes the summation with a reduction
#pragma omp parallel for reduction (+:first_two_rows_sum) schedule(static)
    // loops over columns of the first two rows
    for ( int j = 0; j < N; ++j ) {
        // adds C[0][j] and C[1][j] to the sum
        first_two_rows_sum += C[0 * N + j];
        first_two_rows_sum += C[1 * N + j];
    }

    // computes the checksum for rows 0 and 1
    long double check = checksum( A, B, N, 0, 1);

    // prints out the results
    printf("Host: %s | Size: %d x %d | Threads: %d \nSchedule: %s | Chunk: %d | Time: %.6f s | \nChecksum: %.0Lf | First two rows sum: %.0f\n",
            host, N, N, thread_count, schedule_string, CHUNK, t1 - t0, check, first_two_rows_sum );

    // frees matrices A, B, and C
    free( A );
    free( B );
    free( C );
    // exits the program
    return 0;
}