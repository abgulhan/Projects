#include <cstdio>
#include <cstdlib>
#include <vector>
#include <bitset>
#include <algorithm>
#include <queue>
#include <chrono>


#include <mpi.h>
#include <omp.h>
#define INF 200
#define MASTER 0
#define MAX_RANK 32



void djikstra(int* d, int num_nodes, int src, int* output) {

    std::vector<int> dist(num_nodes, INF); // will hold shortest distance to all nodes
    dist[src] = 0;
    std::vector<bool> visits(num_nodes, false); // vector of bits
    //std::bitset<2000> visits;

    // running djikstra
    for (int count = 0; count < num_nodes; ++count) {
        // pick min dist node that was no visited. src always picked first assuming no negative edges
        int min = INF; 
        int min_index;
        for (int v = 0; v < num_nodes; ++v) {
            if (visits[v] == false && dist[v] <= min) {
                min = dist[v];
                min_index = v;
            }
        }
     
        visits[min_index] = true;
 
        // update dist of neighboring nodes of v
        for (int v = 0; v < num_nodes; ++v) {
            // Update dist[v] only if is not in visits and dist[min_index] + d[min_index][v] < dist[v]
            if (!visits[v] && dist[min_index] + d[min_index*num_nodes + v] < dist[v]) {
                dist[v] = dist[min_index] + d[min_index*num_nodes + v]; 
            }
        }
    } 
    
    std::copy(dist.begin(), dist.end(), output); // update vth row of output matrix
}

int main(int argc, char** argv) {
    auto start_time = std::chrono::steady_clock::now();
    
    int rc = MPI_Init(&argc, &argv);
    auto end_time = std::chrono::steady_clock::now();
    auto diff = (end_time - start_time)/std::chrono::milliseconds(1);
    printf("time for MPI_Init %ld\n", diff);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n, m, *d;
    std::bitset<MAX_RANK> neighbors; // init as all 0 by default

    // input
    start_time = std::chrono::steady_clock::now();
    FILE *infile = fopen(argv[1], "r");
    fscanf(infile, "%d %d", &n, &m);


    // use residual method of task splitting
    int base = (n+size-1)/size; // approx size of vertex split for each rank
    int start = rank*base; // start vertex number
    int end = std::min(start + base, n); // end vertex number, end value NOT included

    // allocate space for solutions
    int local_data_size = (end-start) * n;

    d = (int *) malloc(sizeof(int) * n * n);
    int* min_d = (int *) malloc(sizeof(int) * local_data_size);


    // continue input reading
    for (int i = 0; i < n * n; ++i) d[i] = INF;
    int a, b, w;
    for (int i = 0; i < m; ++i) {
        fscanf(infile, "%d %d %d", &a, &b, &w);
        if (a/base==rank) {
            d[a * n + b] = w;
            if (neighbors.count() < size) { // find which rank node a and b belong to and update neighbors
                neighbors[b/base] = 1;
            }            
        }
        if (b/base==rank) {
            //printf("rank=%d start=%d end=%d length=%d\n", rank, start, end, end-start);
            d[b * n + a] = w;
            if (neighbors.count() < size) {
                neighbors[a/base] = 1;
            }
        }
    }
    fclose(infile);
    end_time = std::chrono::steady_clock::now();
    diff = (end_time - start_time)/std::chrono::milliseconds(1);
    printf("rank %d, time for file read %ld\n", rank, diff);
    
    // define graph topology in MPI
    int* dest = (int *) malloc(sizeof(int) * size);
    MPI_Comm graph;

    int num_neighbors = 0;
    for (int i=0; i<size; ++i) {
        if (neighbors[i]) {
            dest[num_neighbors++] = i;
        }
    }
    /*
    for (int i=0; i<size; ++i) {
        printf("dest[%d]=%d  rank=%d, ", i, dest[i], rank);
    }
    printf("num_neighbors=%d\n", num_neighbors);
    */

    /*
    int MPI_Dist_graph_create(MPI_Comm comm_old, int n, const int sources[],
                          const int degrees[], const int destinations[],
                          const int weights[],
                          MPI_Info info, int reorder, MPI_Comm *comm_dist_graph)
    */
    MPI_Dist_graph_create(MPI_COMM_WORLD, 1, &rank, &num_neighbors, dest, MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &graph);
    free(dest);

    // share local data among each rank
    start_time = std::chrono::steady_clock::now();
    std::vector<int> counts(size); // send and receive sounts
    for (int r=0; r<size-1; ++r) {
        counts[r] = base*n;
    }
    counts[size-1] = (n - ((size-1)*base))*n;

    std::vector<int> displs(size);
    for (int r=0; r<size; ++r) {
        displs[r]=r*base*n;
    }

    /*
    int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, const int *recvcounts, const int *displs,
                   MPI_Datatype recvtype, MPI_Comm comm)
    */
    //MPI_Allgatherv(&d[n*start], end-start, MPI_INT, d, counts.data(), displs.data(), MPI_INT, graph);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, d, counts.data(), displs.data(), MPI_INT, graph);

    end_time = std::chrono::steady_clock::now();
    diff = (end_time - start_time)/std::chrono::milliseconds(1);
    printf("rank %d, time for AllGather %ld\n", rank, diff);
    
    //printf("starting parallel rank=%d start=%d end=%d length=%d\n", rank, start, end, end-start);

    start_time = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for (int v=start; v<end; ++v) {
        //printf("v=%d rank=%d start=%d end=%d length=%d\n", v, rank, start, end, end-start);
        djikstra(d, n, v, &min_d[(v-start)*n]); // update vth row of min distance matrix
    }
    //printf("finish parallel rank=%d start=%d end=%d length=%d\n", rank, start, end, end-start);
    end_time = std::chrono::steady_clock::now();
    diff = (end_time - start_time)/std::chrono::milliseconds(1);
    printf("rank %d, time for djikstra %ld\n", rank, diff);


    if (rank == MASTER) {
        start_time = std::chrono::steady_clock::now();

        int* results = (int *) malloc(sizeof(int)*n*n);
        //int* recvcounts = (int *) malloc(sizeof(int) * (size-1));
        //int* displs = (int *) malloc(sizeof(int) * (size-1));

        int new_rank;

        MPI_Comm_rank(graph, &new_rank);
        //printf("NEW RANK = %d\n", new_rank);

        /*
        int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, const int *recvcounts, const int *displs,
                    MPI_Datatype recvtype, int root, MPI_Comm comm)
        */

        MPI_Gatherv(min_d, n*(end-start), MPI_INT, results, counts.data(), displs.data(), MPI_INT, MASTER, graph);
        end_time = std::chrono::steady_clock::now();
        diff = (end_time - start_time)/std::chrono::milliseconds(1);
        printf("rank %d, time for gather %ld\n", rank, diff);


        // output
        start_time = std::chrono::steady_clock::now();

        FILE *outfile = fopen(argv[2], "w");
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                fprintf(outfile, "%d%s",
                    (i == j ? 0 : results[i * n + j]),
                    (j == n - 1 ? " \n" : " ")
                );
            }
        }
        end_time = std::chrono::steady_clock::now();
        diff = (end_time - start_time)/std::chrono::milliseconds(1);
        printf("rank %d, time for write %ld\n", rank, diff);
        free(results);
    }
    else {
        MPI_Gatherv(min_d, n*(end-start), MPI_INT, NULL, NULL, NULL, MPI_INT, MASTER, graph);
    }

    free(d);
    free(min_d);
    MPI_Finalize();
}