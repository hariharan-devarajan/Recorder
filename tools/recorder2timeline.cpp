#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "reader.h"
#include <string>
#include <assert.h>
#include <mpi.h>
#include <ostream>
#include <sstream>
#include <fstream>
static int do_mutex;
static pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
RecorderReader reader;

struct Writer{
    std::ofstream outFile;
    char* filename;
    int rank, my_rank, total_ranks;
};

void write_to_json(Record *record, void* arg) {
    
   int cat = recorder_get_func_type(&reader, record);
   if (record->level == 0 || (cat == 0 || cat == 1 || cat == 3)) {
        Writer *writer = (Writer *) arg;
        bool user_func = (record->func_id == RECORDER_USER_FUNCTION);
        const char *func_name = recorder_get_func_name(&reader, record);

        int arg_count = record->arg_count;
        if (user_func) {
            func_name = record->args[1];
            arg_count--;
        }

        uint64_t ts = uint64_t(record->tstart / reader.metadata.time_resolution);
        int tid = record->tid;
        uint64_t dur = uint64_t((record->tend - record->tstart) / reader.metadata.time_resolution);
        if (dur <= 0) dur = 0;
        std::stringstream ss;
        ss  << "{\"pid\":"      << writer->rank
            << ",\"tid\":"      << cat
            << ",\"ts\":"       << ts
            << ",\"name\":\""   << func_name
            << "\",\"cat\":\""  << cat;
            ss  << "\",\"ph\":\"X\""<< ""
                << ",\"dur\":"      << dur;
        ss  <<",\"args\":\"";
        for (int arg_id = 0; !arg_id < arg_count; arg_id++) {
            char *arg = record->args[arg_id];
            ss  << " " << arg;
        }
        ss  << " tend: "<< record->tend  <<"\"},\n";
        writer->outFile << ss.rdbuf();
    }
}

int min(int a, int b) { return a < b ? a : b; }
int max(int a, int b) { return a > b ? a : b; }

static void * execute(void * global_writer) {
    Writer* writer = (Writer*)global_writer;
    int n = max(reader.metadata.total_ranks/writer->total_ranks, 1);
    int start_rank = n * writer->my_rank;
    int end_rank   = min(reader.metadata.total_ranks, n*(writer->my_rank+1));
    Writer local;
    local.outFile.open(writer->filename,std::ofstream::out| std::ofstream::app);
    for(int rank = start_rank; rank < end_rank; rank++) {
        CST cst;
        CFG cfg;
        local.rank = rank;
        recorder_read_cst(&reader, rank, &cst);
        recorder_read_cfg(&reader, rank, &cfg);
        recorder_decode_records(&reader, &cst, &cfg, write_to_json, &local);
        printf("\r[Recorder] rank %d finished, unique call signatures: %d\n", rank, cst.entries);
        recorder_free_cst(&cst);
        recorder_free_cfg(&cfg);
    }
    local.outFile.close();
}

int main(int argc, char **argv) {

    char textfile_dir[256];
    char textfile_path[256];
    sprintf(textfile_dir, "%s/_chrome", argv[1]);
    recorder_init_reader(argv[1], &reader);
    int mpi_size, mpi_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if(mpi_rank == 0)
        mkdir(textfile_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    MPI_Barrier(MPI_COMM_WORLD);
    recorder_init_reader(argv[1], &reader);

    // Each rank will process n files (n ranks traces)
    int n = max(reader.metadata.total_ranks/mpi_size, 1);
    int start_rank = n * mpi_rank;
    int end_rank   = min(reader.metadata.total_ranks, n*(mpi_rank+1));
    sprintf(textfile_path, "%s/timeline_%d.json", textfile_dir, mpi_rank);
    Writer local;
    local.outFile.open(textfile_path, std::ofstream::trunc|std::ofstream::out);
    local.outFile << "{\"traceEvents\": [";
    for(int rank = start_rank; rank < end_rank; rank++) {
        CST cst;
        CFG cfg;
        local.rank = rank;
        recorder_read_cst(&reader, rank, &cst);
        recorder_read_cfg(&reader, rank, &cfg);
        recorder_decode_records(&reader, &cst, &cfg, write_to_json, &local);
        printf("\r[Recorder] rank %d finished, unique call signatures: %d %s\n", rank, cst.entries, textfile_path);
        recorder_free_cst(&cst);
        recorder_free_cfg(&cfg);
    }
    local.outFile << "],\"displayTimeUnit\": \"ms\",\"systemTraceEvents\": \"SystemTraceData\",\"otherData\": {\"version\": \"Taxonomy v1.0\" }, \"stackFrames\": {}, \"samples\": []}";
    local.outFile.close();
    recorder_free_reader(&reader);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
