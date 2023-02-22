#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "reader.h"
#include <parquet/arrow/writer.h>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <stdint-gcc.h>
#include <mpi.h>
#include <map>
#include <experimental/filesystem>
#include <iostream>
namespace fs = std::experimental::filesystem;

std::vector<std::string> split(const std::string& s, char delimiter) {
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

class Directory{
public:
    std::string directory;
    std::string hostname;
    std::string username;
    std::string app;
    int proc_id;
    double timestamp;
    Directory():hostname(), username(), app(), proc_id(0), timestamp(0.0) {} /* default constructor */
    Directory(const Directory &other)
            : hostname(other.hostname),
              username(other.username),
              app(other.app),
              proc_id(other.proc_id),
              timestamp(other.timestamp) {} /* copy constructor*/
    Directory(Directory &&other)
            : hostname(other.hostname),
              username(other.username),
              app(other.app),
              proc_id(other.proc_id),
              timestamp(other.timestamp){} /* move constructor*/

    Directory(std::string path){
        this->directory = path;
        std::vector<std::string> split_strings = split(path, '-');
        if (split_strings.size() >=4) {
            hostname = split_strings[0];
            username = split_strings[1];    
            app = split_strings[2];    
            for (int i = 3; i<split_strings.size() - 2;++i) {
                   app += "-" + split_strings[i];
            }
            proc_id = atoi(split_strings[split_strings.size() - 2].c_str());    
            timestamp = atof(split_strings[split_strings.size() - 1].c_str());    
        }
    }

    Directory& operator=(const Directory& other) {

        this->hostname = other.hostname;
        this->username = other.username;
        this->app = other.app;
        this->proc_id = other.proc_id;
        this->timestamp = other.timestamp;
        return *this;
    }


    bool operator==(const Directory& other) const{
        return this->timestamp == other.timestamp;
    }

    bool operator!=(const Directory& other) const{
        return this->timestamp != other.timestamp;
    }

    bool operator<(const Directory& other) const{
        return this->timestamp < other.timestamp;
    }

    bool operator>(const Directory& other) const{
        return this->timestamp > other.timestamp;
    }

    bool operator<=(const Directory& other) const{
        return this->timestamp <= other.timestamp;
    }

    bool operator>=(const Directory& other) const{
        return this->timestamp >= other.timestamp;
    }
};

struct ParquetWriter {
    Directory directory;
    int rank;
    char* base_file;
    int64_t index;
    arrow::Int32Builder categoryBuilder, rankBuilder, threadBuilder, arg_countBuilder, levelBuilder;
    arrow::Int64Builder procBuilder,indexBuilder;
    arrow::FloatBuilder tstartBuilder, tendBuilder;
    arrow::StringBuilder func_idBuilder, appBuilder, hostnameBuilder, args_Builder[10];

    std::shared_ptr<arrow::Array> indexArray, categoryArray, procArray, rankArray, threadArray, arg_countArray, levelArray, tstartArray, tendArray,
            func_idArray, appArray,hostnameArray, argsArray[10];

    std::shared_ptr<arrow::Schema> schema;
    const int64_t chunk_size = 1024;
    const int64_t NUM_ROWS = 1024*1024*16; // 1B
    int64_t row_group = 0;
    ParquetWriter(char* _path);
    void finish();
};

ParquetWriter::ParquetWriter(char* _path) {
    row_group = 0;
    base_file = _path;
    schema = arrow::schema(
            {arrow::field("index", arrow::int64()), arrow::field("proc", arrow::int64()),
             arrow::field("rank", arrow::int32()),
             arrow::field("thread_id", arrow::int32()), arrow::field("cat", arrow::int32()),
             arrow::field("tstart", arrow::float32()), arrow::field("tend", arrow::float32()),
             arrow::field("func_id", arrow::utf8()), arrow::field("level", arrow::int32()),
             arrow::field("hostname", arrow::utf8()), arrow::field("arg_count", arrow::int32()),
             arrow::field("app", arrow::utf8()), arrow::field("args_1", arrow::utf8()),
             arrow::field("args_2", arrow::utf8()), arrow::field("args_3", arrow::utf8()),
             arrow::field("args_4", arrow::utf8()), arrow::field("args_5", arrow::utf8()),
             arrow::field("args_6", arrow::utf8()), arrow::field("args_7", arrow::utf8()),
             arrow::field("args_8", arrow::utf8()), arrow::field("args_9", arrow::utf8()),
             arrow::field("args_10", arrow::utf8())});
    index = 0;

    indexBuilder = arrow::Int64Builder();
    indexArray.reset();
    procBuilder = arrow::Int64Builder();
    procArray.reset();
    rankBuilder = arrow::Int32Builder();
    rankArray.reset();
    threadBuilder = arrow::Int32Builder();
    threadArray.reset();
    categoryBuilder = arrow::Int32Builder();
    categoryArray.reset();
    arg_countBuilder = arrow::Int32Builder();
    arg_countArray.reset();
    levelBuilder = arrow::Int32Builder();
    levelArray.reset();
    hostnameBuilder = arrow::StringBuilder();
    hostnameArray.reset();

    tstartBuilder = arrow::FloatBuilder();
    tstartArray.reset();
    tendBuilder = arrow::FloatBuilder();
    tendArray.reset();


    func_idBuilder = arrow::StringBuilder();
    func_idArray.reset();
    appBuilder = arrow::StringBuilder();
    appArray.reset();
    for (int i =0; i< 10; i++){
        args_Builder[i] = arrow::StringBuilder();
        argsArray[i].reset();
    }
}

void ParquetWriter::finish(void) {
    indexBuilder.Finish(&indexArray);
    procBuilder.Finish(&procArray);
    rankBuilder.Finish(&rankArray);
    threadBuilder.Finish(&threadArray);
    tstartBuilder.Finish(&tstartArray);
    tendBuilder.Finish(&tendArray);
    func_idBuilder.Finish(&func_idArray);
    levelBuilder.Finish(&levelArray);
    hostnameBuilder.Finish(&hostnameArray);
    appBuilder.Finish(&appArray);
    categoryBuilder.Finish(&categoryArray);
    arg_countBuilder.Finish(&arg_countArray);
    for (int arg_id = 0; arg_id < 10; arg_id++) {
        args_Builder[arg_id].Finish(&argsArray[arg_id]);
    }

    auto table = arrow::Table::Make(schema, {indexArray, procArray, rankArray,threadArray,categoryArray,
                                             tstartArray, tendArray, func_idArray, levelArray,
                                             hostnameArray, arg_countArray, appArray,
                                             argsArray[0], argsArray[1], argsArray[2],
                                             argsArray[3], argsArray[4], argsArray[5],
                                             argsArray[6], argsArray[7], argsArray[8],
                                             argsArray[9] });
    char path[256];
    sprintf(path, "%s_%d.parquet" , base_file, row_group);
    PARQUET_ASSIGN_OR_THROW(auto outfile, arrow::io::FileOutputStream::Open(path));
    PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1024));
}

RecorderReader reader;

void handle_one_record(Record* record, void* arg) {
    ParquetWriter *writer = (ParquetWriter*) arg;
    writer->procBuilder.Append(writer->directory.proc_id);
    writer->rankBuilder.Append(writer->rank);
    writer->threadBuilder.Append(record->tid);
    writer->tstartBuilder.Append(record->tstart);
    writer->tendBuilder.Append(record->tend);
    int cat = recorder_get_func_type(&reader, record);
    if (cat == RECORDER_FTRACE){
        writer->func_idBuilder.Append(record->args[1]);
        record->arg_count = 1;
    }else {
        writer->func_idBuilder.Append(recorder_get_func_name(&reader, record));
    }
    writer->categoryBuilder.Append(cat);
    writer->levelBuilder.Append(record->level);
    writer->hostnameBuilder.Append(writer->directory.hostname);
    writer->appBuilder.Append(writer->directory.app);
    writer->arg_countBuilder.Append(record->arg_count);

    for (int arg_id = 0; arg_id < 10; arg_id++) {
        if(arg_id < record->arg_count)
            writer->args_Builder[arg_id].Append(record->args[arg_id]);
        else
            writer->args_Builder[arg_id].Append("");
    }
    writer->index ++;
    writer->indexBuilder.Append(writer->index);
    if(writer->index % writer->NUM_ROWS == 0) {
        writer->indexBuilder.Finish(&writer->indexArray);
        writer->procBuilder.Finish(&writer->procArray);
        writer->rankBuilder.Finish(&writer->rankArray);
        writer->threadBuilder.Finish(&writer->threadArray);
        writer->categoryBuilder.Finish(&writer->categoryArray);
        writer->tstartBuilder.Finish(&writer->tstartArray);
        writer->tendBuilder.Finish(&writer->tendArray);
        writer->func_idBuilder.Finish(&writer->func_idArray);
        writer->levelBuilder.Finish(&writer->levelArray);
        writer->hostnameBuilder.Finish(&writer->hostnameArray);
        writer->appBuilder.Finish(&writer->appArray);
        writer->arg_countBuilder.Finish(&writer->arg_countArray);
        for (int arg_id = 0; arg_id < 10; arg_id++) {
            writer->args_Builder[arg_id].Finish(&writer->argsArray[arg_id]);
        }

        auto table = arrow::Table::Make(writer->schema, {writer->indexArray,writer->procArray, writer->rankArray,writer->threadArray, writer->categoryArray,
                                                         writer->tstartArray, writer->tendArray,
                                                         writer->func_idArray, writer->levelArray, writer->hostnameArray, writer->arg_countArray  ,writer->appArray,
                                                         writer->argsArray[0], writer->argsArray[1], writer->argsArray[2],
                                                         writer->argsArray[3], writer->argsArray[4], writer->argsArray[5],
                                                         writer->argsArray[6], writer->argsArray[7], writer->argsArray[8],
                                                         writer->argsArray[9] });
        char path[256];
        sprintf(path, "%s_%d.parquet" , writer->base_file, writer->row_group);
        PARQUET_ASSIGN_OR_THROW(auto outfile, arrow::io::FileOutputStream::Open(path));
        PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1024*1024*128));

        writer->row_group++;
        writer->indexBuilder = arrow::Int64Builder();
        writer->indexArray.reset();
        writer->procBuilder = arrow::Int64Builder();
        writer->procArray.reset();
        writer->rankBuilder = arrow::Int32Builder();
        writer->rankArray.reset();
        writer->threadBuilder = arrow::Int32Builder();
        writer->threadArray.reset();
        writer->categoryBuilder = arrow::Int32Builder();
        writer->categoryArray.reset();
        writer->arg_countBuilder = arrow::Int32Builder();
        writer->arg_countArray.reset();
        writer->levelBuilder = arrow::Int32Builder();
        writer->levelArray.reset();
        writer->hostnameBuilder = arrow::StringBuilder();
        writer->hostnameArray.reset();

        writer->tstartBuilder = arrow::FloatBuilder();
        writer->tstartArray.reset();
        writer->tendBuilder = arrow::FloatBuilder();
        writer->tendArray.reset();


        writer->func_idBuilder = arrow::StringBuilder();
        writer->func_idArray.reset();
        writer->appBuilder = arrow::StringBuilder();
        writer->appArray.reset();
        for (int i =0; i< 10; i++){
            writer->args_Builder[i] = arrow::StringBuilder();
            writer->argsArray[i].reset();
        }
    }
}

int min(int a, int b) { return a < b ? a : b; }
int max(int a, int b) { return a > b ? a : b; }



void process_rank(char* parquet_file_dir, int rank, Directory dir,ParquetWriter* writer) {


    CST cst;
    CFG cfg;
    writer->directory = dir;
    writer->rank = rank;
    recorder_read_cst(&reader, rank, &cst);
    recorder_read_cfg(&reader, rank, &cfg);
    recorder_decode_records(&reader, &cst, &cfg, handle_one_record, writer);
    printf("\r[Recorder] rank %d finished, unique call signatures: %d\n", rank, cst.entries);
    recorder_free_cst(&cst);
    recorder_free_cfg(&cfg);


}

int main(int argc, char **argv) {

    char parquet_file_dir[256], parquet_file_path[256];
    sprintf(parquet_file_dir, "%s/_parquet", argv[1]);
    mkdir(parquet_file_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    int mpi_size, mpi_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    auto ordered_map = std::map<Directory, std::string>();
    if(mpi_rank == 0)
        mkdir(parquet_file_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    std::string base_path(argv[1]);
    for (const auto & entry : fs::directory_iterator(argv[1])){
        std::string  dir_string{entry.path().u8string()};
        const size_t last_slash_idx = dir_string.rfind('/');
        std::string directory_name;
        if (std::string::npos != last_slash_idx)
        {
            directory_name = dir_string.substr(last_slash_idx + 1, std::string::npos - 1);
        }

        auto directory = Directory(directory_name);
        if(directory_name != "_parquet") {
            ordered_map.insert({directory, dir_string});
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int step = 0;
    int num_steps = ordered_map.size();
    int n = max(num_steps/mpi_size, 1);
    int start_step = n * mpi_rank;
    int end_step   = min(num_steps, n*(mpi_rank+1));
    int completed = 0;
    int prev=0;
    char parquet_filename_path[256];
    sprintf(parquet_filename_path, "%s/%d" , parquet_file_dir, mpi_rank);
    ParquetWriter writer(parquet_filename_path);
    for (auto x : ordered_map){
        if (reader.metadata.total_ranks == 1) {

            if (step >= start_step && step < end_step) {
                //printf("[Recorder] Converting workflow step %d of %d of rank 0 in %s by rank %d\n", step+1, num_steps, x.first.directory.c_str(), mpi_rank);
                recorder_init_reader(x.second.c_str(), &reader);
                process_rank(parquet_file_dir, 0, x.first, &writer);
                recorder_free_reader(&reader);
                completed++;
            }
        } else {
            recorder_init_reader(x.second.c_str(), &reader);
            // Each rank will process n files (n ranks traces)
            int n = max(reader.metadata.total_ranks/mpi_size, 1);
            int start_rank = n * mpi_rank;
            int end_rank   = min(reader.metadata.total_ranks, n*(mpi_rank+1));
            for(int rank = start_rank; rank < end_rank; rank++) {
                //printf("[Recorder] Converting workflow step %d of %d of rank %d in %s by rank %d\n", step+1, num_steps, rank, x.first.directory.c_str(), mpi_rank);
                process_rank(parquet_file_dir, rank, x.first, &writer);
            }
            recorder_free_reader(&reader);
            completed++;
        }
        step++;

        if(prev != completed && completed % 1 == 0) {
            printf("[Recorder] Completed %d of %d by rank %d\n", completed, num_steps/mpi_size, mpi_rank);
        }
        if (prev != completed) {
            prev = completed;
        }
    }
    writer.finish();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}
